# SPDX-License-Identifier: Apache-2.0
"""Compare THD (model-packed) vs BSHD (padded) logprobs for Qwen3.5/Qwen3-VL
on NPU via MegatronEngine.

Runs the same synthetic batch through the engine twice (separate processes,
since megatron/MindSpeed state is not re-initializable in-process) and
compares the dumped logprobs.

Usage (inside the NPU container, from the repo root):

  # 1) BSHD reference (current default path)
  torchrun --nproc_per_node=1 tests/torchrun/compare_thd_vs_bshd_logp_npu.py \
      run --model_path /home/model/Qwen3.5-0.8B --mode bshd --out /tmp/logp_bshd.pt

  # 2) THD model-packed path
  torchrun --nproc_per_node=1 tests/torchrun/compare_thd_vs_bshd_logp_npu.py \
      run --model_path /home/model/Qwen3.5-0.8B --mode thd --out /tmp/logp_thd.pt

  # 3) Compare
  python tests/torchrun/compare_thd_vs_bshd_logp_npu.py \
      compare /tmp/logp_bshd.pt /tmp/logp_thd.pt

Add ``--vision`` to run a deterministic image prompt, or ``--train_step``
to exercise backward and the optimizer in either mode.
"""

import argparse
import os
import sys


def run(args):
    os.environ.setdefault("TASK_QUEUE_ENABLE", "1")

    import torch
    import torch_npu  # noqa: F401

    from areal.api import FinetuneSpec
    from areal.api.alloc_mode import ModelAllocation
    from areal.api.cli_args import (
        MegatronEngineConfig,
        MicroBatchSpec,
        MindSpeedEngineConfig,
        OptimizerConfig,
        TrainEngineConfig,
    )
    from areal.engine import MegatronEngine
    from areal.engine.core.model import SequencePackingMode

    rank = int(os.environ.get("RANK", "0"))
    use_thd = args.mode == "thd"

    mindspeed_cfg = MindSpeedEngineConfig()
    config = TrainEngineConfig(
        backend=args.backend,
        experiment_name="thd-vs-bshd",
        trial_name="cmp",
        path=args.model_path,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=args.max_tokens_per_mb),
        optimizer=OptimizerConfig() if args.train_step else None,
        megatron=MegatronEngineConfig(
            wrap_with_ddp=args.train_step,
            bridge_type=args.bridge_type,
        ),
        mindspeed=mindspeed_cfg,
        gradient_checkpointing=False,
    )
    alloc = ModelAllocation.from_str(args.backend)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=32, train_batch_size=2)

    engine = MegatronEngine(config)
    engine.create_process_group(parallel_strategy=alloc.parallel)
    engine.initialize(addr=None, ft_spec=ft_spec)
    if not use_thd:
        if engine.parallel_strategy.context_parallel_size > 1:
            raise ValueError("The diagnostic BSHD override only supports CP=1.")
        engine.sequence_packing_mode = SequencePackingMode.PADDED
        engine.use_model_packed_seq = False
        engine.use_padded_seq = True
    if args.train_step:
        engine.train()
    else:
        engine.eval()

    if rank == 0:
        print(
            f"[cmp] mode={args.mode} use_padded_seq={engine.use_padded_seq} "
            f"use_model_packed_seq={engine.use_model_packed_seq}"
        )
        expected_packed = use_thd
        assert engine.use_model_packed_seq == expected_packed, (
            engine.use_model_packed_seq,
            expected_packed,
        )

    device = engine.device
    if args.vision:
        import numpy as np
        from PIL import Image

        image = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8), mode="RGB")
        image_processor_type = (
            engine.processor.image_processor.image_processor_type.lower()
        )
        image_token = (
            "<|vision_start|><|image_pad|><|vision_end|>"
            if "qwen" in image_processor_type
            else getattr(engine.processor, "image_token", "<image>")
        )
        user_message = {
            "role": "user",
            "content": f"{image_token} Briefly describe this image.",
        }
        prompt_text = engine.tokenizer.apply_chat_template(
            [user_message],
            add_generation_prompt=True,
            tokenize=False,
        )
        full_text = engine.tokenizer.apply_chat_template(
            [
                user_message,
                {"role": "assistant", "content": "It is a uniform gray square."},
            ],
            add_generation_prompt=False,
            tokenize=False,
        )
        prompt_processed = engine.processor(
            text=[prompt_text], images=[image], return_tensors="pt", padding=False
        )
        processed = engine.processor(
            text=[full_text], images=[image], return_tensors="pt", padding=False
        )
        input_ids = processed["input_ids"].to(device)
        attention_mask = processed["attention_mask"].to(device=device, dtype=torch.bool)
        loss_mask = torch.zeros_like(attention_mask, dtype=torch.int32)
        prompt_len = int(prompt_processed["attention_mask"].sum().item())
        loss_mask[:, max(prompt_len - 1, 0) :] = 1
        multi_modal_input = [{"pixel_values": processed["pixel_values"].to(device)}]
        if "image_grid_thw" in processed:
            multi_modal_input[0]["image_grid_thw"] = processed["image_grid_thw"].to(
                device
            )
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "multi_modal_input": multi_modal_input,
        }
        seq_lens = attention_mask.sum(dim=-1).cpu().tolist()
    else:
        # Deterministic synthetic batch: sequences of different lengths.
        torch.manual_seed(1234)
        vocab = engine.tokenizer.vocab_size
        seq_lens = [int(s) for s in args.seq_lens.split(",")]
        bs = len(seq_lens)
        max_len = max(seq_lens)

        input_ids = torch.randint(
            100, min(vocab, 50000), (bs, max_len), dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(bs, max_len, dtype=torch.bool, device=device)
        loss_mask = torch.zeros(bs, max_len, dtype=torch.int32, device=device)
        for i, seq_len in enumerate(seq_lens):
            attention_mask[i, :seq_len] = True
            loss_mask[i, seq_len // 2 : seq_len] = 1

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }

    if args.train_step:
        result = engine.train_batch(
            input_=data,
            loss_fn=lambda logprobs,
            entropy,
            input_data,
            **kwargs: logprobs.float().mean(),
            loss_weight_fn=lambda input_data: input_data["cu_seqlens"][-1],
        )
        if rank == 0:
            torch.save(
                {
                    "mode": args.mode,
                    "model_path": args.model_path,
                    "seq_lens": seq_lens,
                    "train_result": result,
                },
                args.out,
            )
            print(f"[cmp] train step completed: {result}")
        return

    logp = engine.forward(
        input_=data,
        aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
    )

    if rank == 0:
        out = {
            "mode": args.mode,
            "model_path": args.model_path,
            "seq_lens": seq_lens,
            "logp": logp.detach().float().cpu(),
            "loss_mask": loss_mask.cpu(),
        }
        torch.save(out, args.out)
        print(f"[cmp] saved logp shape={tuple(logp.shape)} to {args.out}")


def compare(args):
    import torch

    a = torch.load(args.file_a, weights_only=False)
    b = torch.load(args.file_b, weights_only=False)
    assert a["seq_lens"] == b["seq_lens"], (a["seq_lens"], b["seq_lens"])
    la, lb = a["logp"], b["logp"]
    assert la.shape == lb.shape, (la.shape, lb.shape)
    diff = (la - lb).abs()
    if "loss_mask" in a and "loss_mask" in b:
        assert torch.equal(a["loss_mask"], b["loss_mask"])
        diff = diff[a["loss_mask"].bool()]
    print(
        f"[compare] {a['mode']} vs {b['mode']}: shape={tuple(la.shape)} "
        f"compared_tokens={diff.numel()} "
        f"max_abs={diff.max().item():.6e} mean_abs={diff.mean().item():.6e}"
    )
    # bf16 forward with different (but equivalent) kernel paths: allow loose
    # tolerance; parity failures show up orders of magnitude above this.
    if "loss_mask" in a:
        la = la[a["loss_mask"].bool()]
        lb = lb[b["loss_mask"].bool()]
    ok = torch.allclose(la, lb, atol=args.atol, rtol=args.rtol)
    print(f"[compare] PASS={ok} (atol={args.atol}, rtol={args.rtol})")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run")
    pr.add_argument("--model_path", required=True)
    pr.add_argument("--mode", choices=["bshd", "thd"], required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--backend", default="megatron:d1p1t1")
    pr.add_argument(
        "--bridge_type",
        default="megatron-bridge",
        choices=["megatron-bridge", "mbridge"],
        help="Bridge for model construction. mode=thd requires megatron-bridge; "
        "mbridge is for legacy-path reference runs.",
    )
    pr.add_argument("--max_tokens_per_mb", type=int, default=8192)
    pr.add_argument("--seq_lens", default="384,1024,640")
    pr.add_argument("--vision", action="store_true")
    pr.add_argument("--train_step", action="store_true")

    pc = sub.add_parser("compare")
    pc.add_argument("file_a")
    pc.add_argument("file_b")
    pc.add_argument("--atol", type=float, default=2e-1)
    pc.add_argument("--rtol", type=float, default=2e-1)

    args = p.parse_args()
    if args.cmd == "run":
        run(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
