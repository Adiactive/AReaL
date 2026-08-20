# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from pathlib import Path

import torch
import torch.distributed as dist


def get_vllm_lora_target_modules(target_modules: list[str]) -> list[str]:
    if not target_modules or "all-linear" in target_modules:
        target_modules = [
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
        ]

    bridge_to_vllm_targets = {
        "linear_qkv": ["q_proj", "k_proj", "v_proj"],
        "linear_proj": ["o_proj"],
        "linear_fc1": ["gate_proj", "up_proj"],
        "linear_fc2": ["down_proj"],
    }
    targets: list[str] = []
    for module_name in target_modules:
        # Megatron-Bridge accepts qualified glob patterns such as
        # ``language_model.*.linear_qkv``.  vLLM only needs the canonical
        # module suffix when constructing the PEFT adapter configuration.
        canonical_name = module_name.rsplit(".", 1)[-1]
        mapped = bridge_to_vllm_targets.get(canonical_name)
        if mapped is None:
            raise NotImplementedError(
                f"LoRA target module '{module_name}' is not supported in MegatronEngine yet."
            )
        targets.extend(mapped)
    return sorted(set(targets))


def normalize_bridge_lora_name(name: str) -> str:
    """Normalize Megatron-Bridge adapter names to AReaL's PEFT convention."""
    if not name.startswith("base_model.model."):
        name = f"base_model.model.{name}"
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if name.endswith(suffix):
            return f"{name[: -len(suffix)]}{suffix[: -len('.weight')]}.default.weight"
    raise ValueError(f"Unsupported Megatron-Bridge LoRA parameter name: {name}")


def _infer_target_modules_from_adapter_weights(weight_keys: Iterable[str]) -> list[str]:
    """
    Infer PEFT target_modules from adapter weight parameter names.

    Extracts module names from HF LoRA weight keys like:
    - base_model.model.layers.0.self_attn.q_proj.lora_A.weight -> q_proj
    - base_model.model.layers.1.mlp.gate_proj.lora_B.weight -> gate_proj
    """
    target_modules = set()

    for key in weight_keys:
        # Remove PEFT prefix
        key = key.replace("base_model.model.", "")

        # Look for .lora_A.weight or .lora_B.weight pattern
        if ".lora_A.weight" in key:
            # Extract module name before .lora_A.weight
            base_name = key.replace(".lora_A.weight", "")
            module_name = base_name.split(".")[-1]
            target_modules.add(module_name)
        elif ".lora_B.weight" in key:
            # Extract module name before .lora_B.weight
            base_name = key.replace(".lora_B.weight", "")
            module_name = base_name.split(".")[-1]
            target_modules.add(module_name)

    return sorted(list(target_modules))


def _build_adapter_config_dict(
    peft_config,
    target_modules: list[str],
    base_model_name_or_path: str,
) -> dict:
    """
    Build PEFT adapter_config.json dictionary.

    Creates a config compatible with HuggingFace PEFT library.
    """
    return {
        "base_model_name_or_path": base_model_name_or_path,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": peft_config.dim,
        "lora_alpha": peft_config.alpha,
        "lora_dropout": peft_config.dropout,
        "target_modules": target_modules,
        "bias": "none",
        "fan_in_fan_out": False,
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
    }


def _monkey_patch_save_hf_adapter():
    """Add save_hf_adapter to AutoBridge when megatron-bridge does not provide it."""
    try:
        from megatron.bridge import AutoBridge
    except ImportError:
        # megatron-bridge is not installed (e.g. NPU environment); nothing to patch.
        return

    if hasattr(AutoBridge, "save_hf_adapter"):
        # Already exists, no need to patch
        return

    def save_hf_adapter(
        self,
        model,
        path: str | Path,
        peft_config,
        base_model_name_or_path: str | None = None,
        show_progress: bool = True,
    ) -> None:
        """
        Save LoRA adapter weights as a HuggingFace PEFT-compatible directory.

        The output directory contains adapter_config.json and adapter_model.safetensors
        and can be loaded directly with peft.PeftModel.from_pretrained(base_model, path).

        Args:
            model: Megatron model instance or list of instances.
            path: Directory path where the adapter files will be saved.
            peft_config: The LoRA config used during training (provides dim, alpha, dropout, etc.).
            base_model_name_or_path: HuggingFace model identifier or local path of the base model.
                If None, inferred from hf_pretrained.model_name_or_path.
            show_progress: Display progress bar during export.

        Example:
            >>> bridge.save_hf_adapter(
            ...     megatron_model,
            ...     "./my-lora-adapter",
            ...     peft_config=lora,
            ...     base_model_name_or_path="Qwen/Qwen3-4B",
            ... )
            >>> # Load with HuggingFace PEFT
            >>> from peft import PeftModel
            >>> from transformers import AutoModelForCausalLM
            >>> base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
            >>> model = PeftModel.from_pretrained(base, "./my-lora-adapter")

        Note:
            This method is collective -- all ranks must call it. Only rank 0 writes files.
        """
        import json

        from safetensors.torch import save_file

        # Synchronize at start
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Export adapter weights
        adapter_state: dict[str, torch.Tensor] = {}
        for name, tensor in self.export_adapter_weights(
            # cpu=True may reduce memory pressure but hangs for MoE models using slurm
            model,
            cpu=False,
            show_progress=False,
        ):
            adapter_state[f"base_model.model.{name}"] = tensor.clone().float()

        if not adapter_state:
            raise RuntimeError(
                "No adapter weights were found on the model. "
                "Ensure the model has PEFT adapters applied before calling save_hf_adapter()."
            )

        # Only rank 0 writes files
        is_rank0 = (
            not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
        )
        if is_rank0:
            save_dir = Path(path)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Infer base model path if not provided
            if base_model_name_or_path is None:
                base_model_name_or_path = str(
                    getattr(self.hf_pretrained, "model_name_or_path", "")
                    or getattr(self.hf_pretrained, "name_or_path", "")
                )

            # Build adapter config
            target_modules = _infer_target_modules_from_adapter_weights(
                adapter_state.keys()
            )
            adapter_config = _build_adapter_config_dict(
                peft_config,
                target_modules=target_modules,
                base_model_name_or_path=base_model_name_or_path,
            )

            # Save adapter config
            config_path = save_dir / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump(adapter_config, f, indent=2)

            # Save adapter weights
            weights_path = save_dir / "adapter_model.safetensors"
            save_file(adapter_state, str(weights_path))

            print(f"✓ Saved LoRA adapter to {save_dir}")
            print(f"  - Config: {config_path}")
            print(f"  - Weights: {weights_path} ({len(adapter_state)} parameters)")

        # Synchronize at end
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # Attach the method to the class
    AutoBridge.save_hf_adapter = save_hf_adapter


# Current: This monkey patch is needed as the current megatron-bridge 0.3.0 does not have a built-in method
# to save LoRA adapters in HuggingFace PEFT format, which is required for our use case.
# Future: This code is however present in main branch of megatron-bridge so this patch is temporary
# and can be removed later when we upgrade the megatron-bridge version.
_monkey_patch_save_hf_adapter()


def patch_mbridge_name_mapping(bridge):
    """
    Patch mbridge name mapping to handle unfused layernorms in GLM-4 and Qwen models.
    This patch is required for megatron lora where we use unfused layers.

    Handles explicit layernorm names for:
    - input_layernorm -> input_layernorm (not fused with qkv)
    - pre_mlp_layernorm -> post_attention_layernorm (not fused with mlp)
    - q_layernorm -> q_norm (QK layernorm)
    - k_layernorm -> k_norm (QK layernorm)
    """
    import re

    orig = bridge._weight_name_mapping_mcore_to_hf

    def new_mapping(name: str):
        # Handle unfused norms + q/k norms
        m = re.match(r"^decoder\.layers\.(\d+)\.(.+)$", name)
        if m:
            i = m.group(1)
            tail = m.group(2)

            if tail == "input_layernorm.weight":
                return [f"model.layers.{i}.input_layernorm.weight"]

            if tail == "pre_mlp_layernorm.weight":
                return [f"model.layers.{i}.post_attention_layernorm.weight"]

            if tail == "self_attention.q_layernorm.weight":
                return [f"model.layers.{i}.self_attn.q_norm.weight"]

            if tail == "self_attention.k_layernorm.weight":
                return [f"model.layers.{i}.self_attn.k_norm.weight"]

        # Fallback to the original implementation for everything else
        return orig(name)

    bridge._weight_name_mapping_mcore_to_hf = new_mapping
    return bridge
