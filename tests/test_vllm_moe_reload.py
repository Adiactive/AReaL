# SPDX-License-Identifier: Apache-2.0

"""CPU coverage of the NPU weight-layout helper without importing vLLM kernels."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture
def undo_postprocess():
    # Importing the worker also installs device-specific Ascend/AWEX patches.
    # Compile the actual helper alone so these tensor tests run in CPU CI.
    path = (
        Path(__file__).resolve().parents[1]
        / "areal/engine/vllm_ext/vllm_worker_extension.py"
    )
    helper = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "undo_moe_postprocess_for_reload"
    )
    platform = SimpleNamespace(device_type="npu")
    namespace = {"current_platform": platform}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[helper.name], platform


def _model():
    model = nn.Module()
    model.mlp = nn.Module()
    model.mlp.experts = nn.Module()
    experts = model.mlp.experts.routed_experts = nn.Module()
    experts.w13_weight = nn.Parameter(torch.arange(48.0).reshape(2, 6, 4))
    experts.w2_weight = nn.Parameter(torch.arange(24.0).reshape(2, 4, 3))
    # Non-expert weights and similarly named quantization parameters stay intact.
    model.mlp.dense = nn.Linear(4, 6, bias=False)
    experts.w13_weight_scale = nn.Parameter(torch.ones(2, 6, 4))
    return model


def test_undo_postprocess_npu_restores_expert_layout_for_repeated_reload(
    undo_postprocess,
):
    """Each postprocess/undo cycle restores both tensor values and loader layout."""
    undo, _ = undo_postprocess
    model = _model()
    weights = dict(model.named_parameters())
    expected = {name: param.detach().clone() for name, param in weights.items()}
    expert_names = (
        "mlp.experts.routed_experts.w13_weight",
        "mlp.experts.routed_experts.w2_weight",
    )
    loader = object()
    for name in expert_names:
        weights[name].weight_loader = loader

    for _ in range(2):
        # Ascend postprocessing transposes the checkpoint layout after each load.
        for name in expert_names:
            param = weights[name]
            param.data = param.data.transpose(1, 2).contiguous()
        undo(model)

        for name, param in model.named_parameters():
            assert param is weights[name]
            assert param.is_contiguous()
            torch.testing.assert_close(param, expected[name], rtol=0, atol=0)
        for name in expert_names:
            assert weights[name].weight_loader is loader


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_undo_postprocess_non_npu_leaves_weights_unchanged(
    undo_postprocess, device_type
):
    """The Ascend reload fix must not transpose CPU or CUDA parameters."""
    undo, platform = undo_postprocess
    platform.device_type = device_type
    model = _model()
    expected = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }

    undo(model)

    for name, param in model.named_parameters():
        torch.testing.assert_close(param, expected[name], rtol=0, atol=0)
