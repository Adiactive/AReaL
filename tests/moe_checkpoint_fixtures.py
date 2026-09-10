"""Generate small, offline HF checkpoints for distributed MoE regression tests."""

import argparse
import re
from pathlib import Path

import torch

VOCAB_SIZE = 2048
SPECIAL_TOKENS = (
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
)


def _expert_layout(
    state: dict[str, torch.Tensor], *, packed: bool
) -> dict[str, torch.Tensor]:
    """Normalize Transformers expert layouts to each bridge's checkpoint format."""
    state = dict(state)
    if packed:
        pattern = re.compile(r"(.+\.experts)\.(\d+)\.gate_proj\.weight$")
        prefixes = {m[1] for name in state if (m := pattern.fullmatch(name))}
        for prefix in prefixes:
            expert_ids = sorted(
                int(m[2])
                for name in state
                if (m := pattern.fullmatch(name)) and m[1] == prefix
            )
            state[f"{prefix}.gate_up_proj"] = torch.stack(
                [
                    torch.cat(
                        [
                            state.pop(f"{prefix}.{i}.gate_proj.weight"),
                            state.pop(f"{prefix}.{i}.up_proj.weight"),
                        ]
                    )
                    for i in expert_ids
                ]
            )
            state[f"{prefix}.down_proj"] = torch.stack(
                [state.pop(f"{prefix}.{i}.down_proj.weight") for i in expert_ids]
            )
    else:
        for name in list(state):
            if not name.endswith(".experts.gate_up_proj"):
                continue
            prefix = name.removesuffix(".gate_up_proj")
            gate_up = state.pop(name)
            down = state.pop(f"{prefix}.down_proj")
            for i in range(gate_up.shape[0]):
                gate, up = gate_up[i].chunk(2, dim=0)
                state[f"{prefix}.{i}.gate_proj.weight"] = gate.clone()
                state[f"{prefix}.{i}.up_proj.weight"] = up.clone()
                state[f"{prefix}.{i}.down_proj.weight"] = down[i].clone()
    return {name: tensor.contiguous() for name, tensor in state.items()}


def create_moe_checkpoint(output_dir: Path, model_type: str) -> Path:
    """Write deterministic BF16 weights, a shard index, and local HF metadata."""
    from huggingface_hub import save_torch_state_dict
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen3_5MoeConfig,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
    )

    common = dict(
        vocab_size=VOCAB_SIZE,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        moe_intermediate_size=256,
        num_experts_per_tok=2,
        max_position_embeddings=2048,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        tie_word_embeddings=False,
        dtype="bfloat16",
    )
    if model_type == "qwen3moe":
        config = Qwen3MoeConfig(
            **common,
            head_dim=64,
            num_hidden_layers=4,
            num_experts=8,
            norm_topk_prob=True,
            rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
        )
        model_cls = Qwen3MoeForCausalLM
    elif model_type == "qwen3_5_moe":
        config = Qwen3_5MoeConfig(
            text_config=dict(
                **common,
                head_dim=256,
                num_hidden_layers=16,
                num_experts=4,
                shared_expert_intermediate_size=512,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_num_key_heads=4,
                linear_num_value_heads=8,
                linear_conv_kernel_dim=4,
                full_attention_interval=4,
                layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 4,
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": 10_000_000.0,
                    "partial_rotary_factor": 0.25,
                    "mrope_section": [11, 11, 10],
                },
            ),
            vision_config=dict(
                depth=1,
                hidden_size=256,
                intermediate_size=512,
                num_heads=4,
                out_hidden_size=256,
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
                num_position_embeddings=256,
                deepstack_visual_indexes=[],
            ),
            image_token_id=4,
            video_token_id=5,
            vision_start_token_id=6,
            vision_end_token_id=7,
            tie_word_embeddings=False,
            dtype="bfloat16",
        )
        model_cls = Qwen3_5MoeForConditionalGeneration
    else:
        raise ValueError(f"Unsupported MoE fixture: {model_type}")

    config.areal_test_fixture = True
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(42)
        model = model_cls(config).to(dtype=torch.bfloat16)
        state = _expert_layout(model.state_dict(), packed=model_type == "qwen3_5_moe")
        if model_type == "qwen3_5_moe":
            # Reuse a full-attention layer for the unbuilt frozen MTP head.
            config.text_config.mtp_num_hidden_layers = 1
            prefix = "model.language_model.layers.15."
            mtp = {
                name.replace(prefix, "mtp.layers.0."): tensor.clone()
                for name, tensor in state.items()
                if name.startswith(prefix)
            }
            state.update(_expert_layout(mtp, packed=False))
            state["mtp.norm.weight"] = torch.randn(256, dtype=torch.bfloat16)
            state["mtp.fc.weight"] = torch.randn(256, 512, dtype=torch.bfloat16)
            state["mtp.pre_fc_norm_embedding.weight"] = torch.ones(
                256, dtype=torch.bfloat16
            )
            state["mtp.pre_fc_norm_hidden.weight"] = torch.ones(
                256, dtype=torch.bfloat16
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    config.architectures = [model_cls.__name__]
    config.save_pretrained(output_dir)
    if model_type == "qwen3_5_moe":
        import json

        from safetensors.torch import save_file

        # A mixed shard guards against dropping live weights with frozen MTP.
        mixed = {
            name: state.pop(name)
            for name in list(state)
            if name.startswith("mtp.") or name == "lm_head.weight"
        }
        save_torch_state_dict(state, output_dir, max_shard_size="4MB")
        filename = "model-mtp.safetensors"
        save_file(mixed, output_dir / filename, metadata={"format": "pt"})
        index_path = output_dir / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"].update({name: filename for name in mixed})
        index["metadata"]["total_size"] += sum(
            tensor.numel() * tensor.element_size() for tensor in mixed.values()
        )
        index_path.write_text(json.dumps(index, indent=2) + "\n")
    else:
        save_torch_state_dict(state, output_dir, max_shard_size="4MB")

    vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    vocab.update({f"token_{i}": i for i in range(len(vocab), VOCAB_SIZE)})
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        additional_special_tokens=list(SPECIAL_TOKENS[4:]),
    )
    tokenizer.save_pretrained(output_dir)
    if model_type == "qwen3_5_moe":
        from transformers import (
            Qwen2VLImageProcessor,
            Qwen3VLProcessor,
            Qwen3VLVideoProcessor,
        )

        Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(
                patch_size=16, temporal_patch_size=2, merge_size=2
            ),
            tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor(
                patch_size=16, temporal_patch_size=2, merge_size=2
            ),
        ).save_pretrained(output_dir)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for model_type in ("qwen3moe", "qwen3_5_moe"):
        create_moe_checkpoint(args.output_dir / model_type, model_type)
