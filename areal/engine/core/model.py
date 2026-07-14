# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import torch

VALID_VISION_MODELS = [
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
    "gemma3",
]
# This registry is used to check if a model is a vision model that we have checked it works with AReaL.
# As different vision models vary in their image processing, special tokens and keys, etc.
# We will add models to this registry as we test them.
# If you want to add a new vision model, please make sure it works with AReaL.


def is_valid_vision_model(model_type: str) -> bool:
    return model_type in VALID_VISION_MODELS


def is_qwen2_vl_model(model_type: str) -> bool:
    return model_type in ["qwen2_vl", "qwen2_5_vl"]


def is_qwen3_vl_model(model_type: str) -> bool:
    """True for the Qwen3-VL family (dense and MoE).

    Existing call sites in ``fsdp_engine``, ``fsdp_utils/parallel``, and
    ``awex/fsdp_adapter`` gate family-level behaviour (mRoPE index,
    attention-mask handling) that is identical for dense and MoE, so this
    helper covers both. Use ``is_qwen3_vl_moe_model`` when the MoE-vs-dense
    distinction matters.
    """
    return model_type in ("qwen3_vl", "qwen3_vl_moe")


def is_qwen3_vl_moe_model(model_type: str) -> bool:
    return model_type == "qwen3_vl_moe"


def is_qwen_vl_model(model_type: str) -> bool:
    return is_qwen2_vl_model(model_type) or is_qwen3_vl_model(model_type)


def lang_config(hf_config):
    """Return the language-model side of a (possibly nested) HF config.

    Qwen3-VL and similar VLMs nest text-model attributes (vocab_size,
    num_attention_heads, num_key_value_heads, hidden_size, head_dim) under
    ``hf_config.text_config``. Qwen2.5-VL and pure text models keep them
    flat. Use this anywhere the caller wants a language-side attribute and
    doesn't know the model family up front.
    """
    return getattr(hf_config, "text_config", hf_config)


def is_gemma3_model(model_type: str) -> bool:
    return model_type in ["gemma3"]


VALID_MOE_MODELS = [
    "qwen3_moe",
    "qwen3_vl_moe",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "bailing_moe_v2",
    "bailing_moe_linear",
    "bailing_hybrid",
]
# This registry is used to check if a model is a MoE model that we have checked it works with AReaL.


def is_moe_model(model_type: str) -> bool:
    return model_type in VALID_MOE_MODELS


def is_qwen3_moe_model(model_type: str) -> bool:
    return model_type in ["qwen3_moe"]


def is_qwen3_5_model(model_type: str) -> bool:
    return model_type in ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"]


def supports_model_packed_seq(model_type: str, bridge_type: str) -> bool:
    """Whether the bridge model owns BSHD-to-THD packing internally."""
    return bridge_type == "megatron-bridge" and (
        is_qwen3_vl_model(model_type) or model_type in ("qwen3_5", "qwen3_5_moe")
    )


class SequencePackingMode(str, Enum):
    WRAPPER_THD = "wrapper_thd"
    MODEL_THD = "model_thd"
    PADDED = "padded"


def resolve_sequence_packing_mode(
    model_type: str, bridge_type: str
) -> SequencePackingMode:
    """Select one packing path from the model and bridge contract."""
    if supports_model_packed_seq(model_type, bridge_type):
        return SequencePackingMode.MODEL_THD
    if is_valid_vision_model(model_type) or is_qwen3_5_model(model_type):
        return SequencePackingMode.PADDED
    return SequencePackingMode.WRAPPER_THD


def validate_context_parallel_mode(
    mode: SequencePackingMode, context_parallel_size: int, model_type: str
) -> None:
    """Reject context parallelism only for models that require padded input."""
    if context_parallel_size > 1 and mode == SequencePackingMode.PADDED:
        raise NotImplementedError(
            f"Context parallel (CP > 1) is not supported for "
            f"model_type={model_type!r}, which requires the padded BSHD forward. "
            f"Got context_parallel_size={context_parallel_size}."
        )


def requires_padded_seq(model_type: str, bridge_type: str = "mbridge") -> bool:
    """Whether the model must run the padded (BSHD) forward instead of packed (THD).

    Model-owned THD is selected from the bridge contract. Other VLM and
    GDN/SSM models require padded input, while ordinary text models use the
    wrapper-owned THD path.
    """
    return (
        resolve_sequence_packing_mode(model_type, bridge_type)
        == SequencePackingMode.PADDED
    )


# Copied from trl
def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
