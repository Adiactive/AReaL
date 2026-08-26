# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

try:
    # Import the engine before raw megatron.core so NPU compatibility patches and
    # the MindSpeed adaptor are installed in AReaL's supported order. MindSpeed
    # also inspects argv while patching; pytest's CLI arguments are not Megatron
    # arguments and can enable an invalid patch combination.
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        from areal.engine import megatron_engine as megatron_engine_module
    finally:
        sys.argv = original_argv
except ImportError:
    megatron_engine_module = None

pytest.importorskip("megatron.core")

from areal.engine.core.model import (
    SequencePackingMode,
    resolve_sequence_packing_mode,
    validate_context_parallel_mode,
)
from areal.engine.megatron_utils import packed_context_parallel as packed_cp


class _RecordingModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output
        self.forward_kwargs = None

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return self.output


@pytest.mark.parametrize("context_parallel_size", [1, 2])
def test_context_parallel_uses_model_contract_route(context_parallel_size):
    mode = resolve_sequence_packing_mode("qwen3_5", "megatron-bridge")
    assert mode == SequencePackingMode.MODEL_THD
    validate_context_parallel_mode(mode, context_parallel_size, "qwen3_5")


def test_context_parallel_preserves_wrapper_thd():
    mode = resolve_sequence_packing_mode("qwen3", "megatron-bridge")
    assert mode == SequencePackingMode.WRAPPER_THD
    validate_context_parallel_mode(mode, 2, "qwen3")


def test_context_parallel_rejects_padded_models():
    mode = resolve_sequence_packing_mode("qwen2_5_vl", "megatron-bridge")
    assert mode == SequencePackingMode.PADDED
    with pytest.raises(NotImplementedError, match="requires the padded BSHD"):
        validate_context_parallel_mode(mode, 2, "qwen2_5_vl")


def test_model_packed_forward_cp2_delegates_split_to_model():
    """Model-owned THD keeps padded inputs global and accepts CP-local output."""
    local_output = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    model = _RecordingModel(local_output)
    input_ids = torch.arange(8, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)

    with (
        patch.object(
            packed_cp.mpu, "get_tensor_model_parallel_world_size", return_value=1
        ),
        patch.object(packed_cp.mpu, "get_context_parallel_world_size", return_value=2),
        patch.object(packed_cp.mpu, "is_pipeline_last_stage", return_value=True),
    ):
        output = packed_cp.packed_context_parallel_forward(
            model,
            {"input_ids": input_ids, "cu_seqlens": cu_seqlens},
            gather_cp_output=False,
            is_vision_model=True,
            use_model_packed_seq=True,
        )

    assert model.forward_kwargs is not None
    torch.testing.assert_close(
        model.forward_kwargs["input_ids"],
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        model.forward_kwargs["attention_mask"],
        torch.ones((2, 4), dtype=torch.bool),
        rtol=0,
        atol=0,
    )
    assert model.forward_kwargs["position_ids"] is None
    packed_seq_params = model.forward_kwargs["packed_seq_params"]
    assert packed_seq_params.qkv_format == "thd"
    torch.testing.assert_close(
        packed_seq_params.cu_seqlens_q, cu_seqlens, rtol=0, atol=0
    )
    torch.testing.assert_close(output, local_output.squeeze(0), rtol=0, atol=0)


def test_model_packed_cp_rejects_unaligned_sequence():
    """Model-owned packing rejects lengths that cannot use the CP zigzag split."""
    with (
        patch.object(
            packed_cp.mpu, "get_tensor_model_parallel_world_size", return_value=1
        ),
        patch.object(packed_cp.mpu, "get_context_parallel_world_size", return_value=2),
        pytest.raises(ValueError, match="divisible by 4"),
    ):
        packed_cp._build_thd_packed_seq_params(torch.tensor([0, 6], dtype=torch.int32))


def test_model_packed_forward_cp2_forwards_vision_inputs():
    """Model-owned CP keeps vision tensors global until the model merges them."""
    local_output = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    model = _RecordingModel(local_output)
    pixel_values = torch.randn(6, 16)
    image_grid_thw = torch.tensor([[1, 2, 3]])
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)

    with (
        patch.object(
            packed_cp.mpu, "get_tensor_model_parallel_world_size", return_value=1
        ),
        patch.object(packed_cp.mpu, "get_context_parallel_world_size", return_value=2),
        patch.object(packed_cp.mpu, "is_pipeline_last_stage", return_value=True),
    ):
        output = packed_cp.packed_context_parallel_forward(
            model,
            {
                "input_ids": torch.arange(8, dtype=torch.int32),
                "cu_seqlens": cu_seqlens,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            },
            gather_cp_output=False,
            is_vision_model=True,
            use_model_packed_seq=True,
        )

    assert model.forward_kwargs is not None
    torch.testing.assert_close(
        model.forward_kwargs["input_ids"],
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        model.forward_kwargs["attention_mask"],
        torch.ones((2, 4), dtype=torch.bool),
        rtol=0,
        atol=0,
    )
    assert model.forward_kwargs["position_ids"] is None
    assert model.forward_kwargs["pixel_values"] is pixel_values
    assert model.forward_kwargs["image_grid_thw"] is image_grid_thw
    packed_seq_params = model.forward_kwargs["packed_seq_params"]
    assert packed_seq_params.qkv_format == "thd"
    torch.testing.assert_close(
        packed_seq_params.cu_seqlens_q, cu_seqlens, rtol=0, atol=0
    )
    torch.testing.assert_close(output, local_output.squeeze(0), rtol=0, atol=0)


def test_forward_result_reassembles_cp_local_logprobs():
    """Forward-only inference restores CP-local logprobs to global order."""
    if megatron_engine_module is None:
        pytest.skip("MegatronEngine dependencies are unavailable")

    engine = object.__new__(megatron_engine_module.MegatronEngine)
    engine.config = SimpleNamespace(is_critic=False, temperature=1.0)
    engine.enable_tree_training = False

    output = torch.randn(4, 8)
    local_labels = torch.arange(4)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    local_logprobs = torch.arange(4, dtype=torch.float32)
    global_logprobs = torch.arange(8, dtype=torch.float32)
    unpadded_logprobs = global_logprobs[:7]
    inputs = {
        "input_ids": torch.arange(8),
        "_cp_local_labels": local_labels,
        "_cp_padded_cu_seqlens": cu_seqlens,
        "_cp_padding_length": 1,
        "_cp_old_cu_seqlens": torch.tensor([0, 3, 7], dtype=torch.int32),
    }

    with (
        patch.object(
            megatron_engine_module.mpu,
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch.object(
            megatron_engine_module,
            "gather_logprobs",
            return_value=local_logprobs,
        ) as gather_logprobs,
        patch.object(
            megatron_engine_module,
            "reassemble_cp_packed_logprobs",
            return_value=global_logprobs,
        ) as reassemble,
        patch.object(
            megatron_engine_module,
            "unpad_logits",
            return_value=unpadded_logprobs,
        ) as unpad,
    ):
        result = engine._compute_forward_result(output, inputs)

    assert gather_logprobs.call_args.args[1] is local_labels
    reassemble.assert_called_once_with(local_logprobs, cu_seqlens)
    unpad.assert_called_once_with(
        global_logprobs,
        1,
        cu_seqlens,
        inputs["_cp_old_cu_seqlens"],
    )
    torch.testing.assert_close(result, unpadded_logprobs, rtol=0, atol=0)
