# SPDX-License-Identifier: Apache-2.0

"""Runtime patches for megatron-bridge bugs not yet in a released version.

Each patch is keyed to an upstream PR. Patches are not version-gated; instead
each one's hot path becomes a no-op once the upstream fix is present (the patch
checks for the missing attribute/behavior before acting), and an idempotency
sentinel prevents double-application. Apply patches at import time via
``_apply_patches_on_import()`` at module bottom.
"""

from __future__ import annotations

from typing import Any

import areal.utils.logging as logging

logger = logging.getLogger("MegatronBridgePatches")


def _patch_qwen3vl_pr3143_word_embeddings() -> None:
    """megatron-bridge PR #3143: expose word_embeddings on MTP shadow embedding.

    Bug (issue #3112 / PR #3143): in ``Qwen3VLGPTModel.forward``, when
    ``mtp_process and sequence_parallel`` are both True, ``self.embedding`` is
    temporarily replaced with a plain closure ``_sp_scatter_embedding``. The
    closure lacks the ``word_embeddings`` attribute that
    ``shared_embedding_or_output_weight()`` accesses during ``_postprocess``
    when ``share_embeddings_and_output_weights=True`` — typical for the
    smaller Qwen3.5 dense models (0.8B/2B/4B).

    Failure mode:
        ``AttributeError: 'function' object has no attribute 'word_embeddings'``

    Affected versions: megatron-bridge 0.4.0 and 0.4.1. Fixed on ``main``
    by commit 20749b09 (PR #3143) but not in any non-alpha release yet.

    Strategy: wrap ``Qwen3VLGPTModel._postprocess`` so it lazily restores
    ``word_embeddings`` on the shadow embedding by inspecting its closure.
    Closure-based recovery is non-invasive — we don't touch ``forward``
    itself (~70 LoC method).
    """
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
            Qwen3VLGPTModel,
        )
    except ImportError:
        return

    if getattr(Qwen3VLGPTModel, "_areal_pr3143_applied", False):
        return

    _orig_postprocess = Qwen3VLGPTModel._postprocess

    def _patched_postprocess(self, *args, **kwargs):
        emb = self.__dict__.get("embedding")
        # Only intervene when the shadow closure is currently installed and
        # lacks the expected attribute.
        if (
            callable(emb)
            and not hasattr(emb, "word_embeddings")
            and emb.__closure__ is not None
        ):
            for cell in emb.__closure__:
                try:
                    target = cell.cell_contents
                except ValueError:
                    continue
                if hasattr(target, "word_embeddings"):
                    emb.word_embeddings = target.word_embeddings
                    break
        return _orig_postprocess(self, *args, **kwargs)

    Qwen3VLGPTModel._postprocess = _patched_postprocess
    Qwen3VLGPTModel._areal_pr3143_applied = True
    logger.info(
        "Applied megatron-bridge PR #3143 workaround: "
        "Qwen3VLGPTModel shadow embedding word_embeddings restoration."
    )


def patch_mindspeed_row_parallel_lora() -> None:
    """Teach Megatron-Bridge LoRA about MindSpeed's row-parallel linear."""
    try:
        import importlib

        bridge_lora = importlib.import_module("megatron.bridge.peft.lora")
        bridge_utils = importlib.import_module("megatron.bridge.peft.utils")
        from megatron.bridge.peft.utils import AdapterAttributes
    except ImportError:
        return

    if getattr(
        bridge_lora.get_adapter_attributes_from_linear,
        "_areal_mindspeed_row_lora_applied",
        False,
    ):
        return

    original = bridge_utils.get_adapter_attributes_from_linear

    def _get_adapter_attributes(module, is_expert=False):
        module_type = type(module)
        is_mindspeed_row_parallel = (
            module_type.__module__ == "mindspeed.te.pytorch.module.linear"
            and module_type.__name__ == "TERowParallelLinear"
        )
        if not is_mindspeed_row_parallel:
            return original(module, is_expert=is_expert)

        disable_tensor_parallel_comm = getattr(module, "explicit_expert_comm", False)
        return AdapterAttributes(
            input_is_parallel=True,
            in_features=module.input_size,
            out_features=module.output_size,
            disable_tensor_parallel_comm=disable_tensor_parallel_comm,
            disable_sequence_parallel_comm=(
                disable_tensor_parallel_comm or not module.config.sequence_parallel
            ),
            base_linear_is_parallel=True,
        )

    bridge_utils.get_adapter_attributes_from_linear = _get_adapter_attributes
    bridge_lora.get_adapter_attributes_from_linear = _get_adapter_attributes
    _get_adapter_attributes._areal_mindspeed_row_lora_applied = True
    logger.info(
        "Applied MindSpeed TERowParallelLinear compatibility for Megatron-Bridge LoRA."
    )


def patch_qwen35_hybrid_lora_specs() -> None:
    """Unfuse Qwen3.5 attention/MLP inputs without changing its GDN layout.

    Megatron-Bridge builds Qwen3.5 as a heterogeneous list containing both
    Gated DeltaNet and full-attention layers. Replacing that list with a Qwen3
    dense spec would drop GDN entirely. Instead, wrap the native block-spec
    builder and replace only the fused LayerNorm+ColumnParallelLinear modules
    that LoRA targets. GDN's fused ``in_proj`` is deliberately retained until
    it has matching adapter and export support.
    """
    try:
        from megatron.bridge.models.qwen_vl import qwen35_vl_provider
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear
        from mindspeed.core.megatron_basic.megatron_basic import PTNorm
    except ImportError:
        return

    original = qwen35_vl_provider.get_transformer_block_with_experimental_attention_variant_spec
    if getattr(original, "_areal_qwen35_hybrid_lora_applied", False):
        return

    def _unfused_block_spec(*args, **kwargs):
        block_spec = original(*args, **kwargs)
        for layer_spec in block_spec.layer_specs:
            submodules = layer_spec.submodules

            attention = submodules.self_attention
            attention_submodules = getattr(attention, "submodules", None)
            # Full-attention layers expose linear_qkv; GDN layers expose
            # in_proj instead and must keep their native fused implementation.
            if attention_submodules is not None and hasattr(
                attention_submodules, "linear_qkv"
            ):
                submodules.input_layernorm = PTNorm
                attention_submodules.linear_qkv = ColumnParallelLinear
                if getattr(attention, "metainfo", None) is not None:
                    attention.metainfo["fuse_input_layernorm"] = False

            mlp = submodules.mlp
            mlp_submodules = getattr(mlp, "submodules", None)
            if mlp_submodules is not None and hasattr(mlp_submodules, "linear_fc1"):
                submodules.pre_mlp_layernorm = PTNorm
                mlp_submodules.linear_fc1 = ColumnParallelLinear
                if getattr(mlp, "metainfo", None) is not None:
                    mlp.metainfo["fuse_pre_mlp_layernorm"] = False
        return block_spec

    _unfused_block_spec._areal_qwen35_hybrid_lora_applied = True
    qwen35_vl_provider.get_transformer_block_with_experimental_attention_variant_spec = _unfused_block_spec
    logger.info(
        "Using unfused full-attention QKV and MLP FC1 specs for Qwen3.5 LoRA; "
        "the native hybrid GDN layer layout is preserved."
    )


def patch_qwen3_moe_lora_spec(provider: Any) -> None:
    """Unfuse Qwen3-MoE attention QKV without replacing its MoE layer spec.

    MindSpeed's fused LayerNorm+ColumnParallelLinear does not return the
    normalized activation required by Megatron-Bridge's LoRA wrapper.  Keep the
    provider's router and expert specifications intact, and replace only the
    attention input norm and QKV projection after the native spec is built.
    """
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from mindspeed.core.megatron_basic.megatron_basic import PTNorm

    original = provider.transformer_layer_spec
    if getattr(original, "_areal_qwen3_moe_lora_applied", False):
        return

    def _unfused_layer_spec(*args, **kwargs):
        layer_spec = original(*args, **kwargs)
        layer_specs = getattr(layer_spec, "layer_specs", [layer_spec])
        for spec in layer_specs:
            submodules = spec.submodules
            attention = submodules.self_attention
            attention_submodules = attention.submodules

            submodules.input_layernorm = PTNorm
            attention_submodules.linear_qkv = ColumnParallelLinear
            if getattr(attention, "metainfo", None) is not None:
                attention.metainfo["fuse_input_layernorm"] = False
        return layer_spec

    _unfused_layer_spec._areal_qwen3_moe_lora_applied = True
    provider.transformer_layer_spec = _unfused_layer_spec
    logger.info(
        "Using unfused attention QKV spec for Qwen3-MoE LoRA; preserving the "
        "native router and expert specifications."
    )


def _apply_patches_on_import() -> None:
    _patch_qwen3vl_pr3143_word_embeddings()
    patch_mindspeed_row_parallel_lora()


_apply_patches_on_import()
