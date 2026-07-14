# SPDX-License-Identifier: Apache-2.0
import pytest

from areal.engine.core.model import (
    SequencePackingMode,
    requires_padded_seq,
    resolve_sequence_packing_mode,
    supports_model_packed_seq,
)


@pytest.mark.parametrize(
    "model_type", ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"]
)
def test_qwen3_5_defaults_to_padded(model_type):
    """Qwen3.5 family requires the padded BSHD forward by default."""
    assert requires_padded_seq(model_type) is True
    assert requires_padded_seq(model_type, bridge_type="mbridge") is True


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_qwen3_5_packed_with_megatron_bridge(model_type):
    """The root Qwen3.5 contract owns packing only with megatron-bridge."""
    assert requires_padded_seq(model_type, bridge_type="megatron-bridge") is False
    assert requires_padded_seq(model_type, bridge_type="mbridge") is True


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_moe", "llama"])
def test_non_gdn_never_padded(model_type):
    """Ordinary text models use wrapper THD."""
    assert requires_padded_seq(model_type) is False


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("qwen3_vl", True),
        ("qwen3_vl_moe", True),
        ("qwen3_5", True),
        ("qwen3_5_moe", True),
        ("qwen3_5_text", False),
        ("qwen3_5_moe_text", False),
        ("qwen2_5_vl", False),
        ("gemma3", False),
        ("qwen3", False),
    ],
)
def test_model_packed_seq_support_is_limited_to_qwen3_vl_family(model_type, expected):
    assert supports_model_packed_seq(model_type, "megatron-bridge") is expected
    assert supports_model_packed_seq(model_type, "mbridge") is False


@pytest.mark.parametrize(
    ("model_type", "bridge_type", "expected"),
    [
        ("qwen3_vl", "megatron-bridge", SequencePackingMode.MODEL_THD),
        ("qwen3_5", "megatron-bridge", SequencePackingMode.MODEL_THD),
        ("qwen3_vl", "mbridge", SequencePackingMode.PADDED),
        ("qwen3_5", "mbridge", SequencePackingMode.PADDED),
        ("qwen3_5_text", "megatron-bridge", SequencePackingMode.PADDED),
        ("qwen2_5_vl", "megatron-bridge", SequencePackingMode.PADDED),
        ("gemma3", "megatron-bridge", SequencePackingMode.PADDED),
        ("qwen3", "megatron-bridge", SequencePackingMode.WRAPPER_THD),
        ("llama", "mbridge", SequencePackingMode.WRAPPER_THD),
    ],
)
def test_sequence_packing_mode_follows_model_contract(
    model_type, bridge_type, expected
):
    assert resolve_sequence_packing_mode(model_type, bridge_type) == expected
