import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from areal.engine.patch_awex import _patch_vllm_runtime_weights


@pytest.fixture
def converter(monkeypatch):
    class Converter:
        def iter_model_parameters(self, model):
            return model.named_parameters()

        def post_update_weights(self):
            self.original_post_called = True

    module = ModuleType("awex.models.ascend.qwen3_5")
    module.VLLMToHFWeightConverterQwen3VL = Converter
    monkeypatch.setitem(sys.modules, module.__name__, module)
    _patch_vllm_runtime_weights()
    installed = Converter.iter_model_parameters
    _patch_vllm_runtime_weights()
    assert Converter.iter_model_parameters is installed
    return Converter()


def make_model(packed=True):
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.linear_attn = nn.Module()
    conv = model.layer.linear_attn.conv1d = nn.Conv1d(6, 6, 4, groups=6, bias=False)
    if packed:
        conv.register_parameter(
            "ascend_conv1d_weight", nn.Parameter(torch.zeros(4, 6), requires_grad=False)
        )
    return model, conv


def test_gdn_update_refreshes_derived_weight_without_replacing_storage(converter):
    """Exchange canonical weights and refresh captured kernel storage each time."""
    model, conv = make_model()
    pointer = conv.ascend_conv1d_weight.data_ptr()
    parameters = dict(converter.iter_model_parameters(model))
    assert list(parameters) == ["layer.linear_attn.conv1d.weight"]
    for value in (2.0, 5.0):
        with torch.no_grad():
            parameters["layer.linear_attn.conv1d.weight"].copy_(
                torch.arange(24).reshape(6, 1, 4) + value
            )
        converter.post_update_weights()
        assert converter.original_post_called
        assert conv.ascend_conv1d_weight.data_ptr() == pointer
        torch.testing.assert_close(
            conv.ascend_conv1d_weight, conv.weight.squeeze(1).t(), rtol=0, atol=0
        )


def test_gdn_without_derived_weight_preserves_parameters(converter):
    """Older vLLM layouts retain ordinary exchange behavior."""
    model, _ = make_model(packed=False)
    assert (
        dict(converter.iter_model_parameters(model)).keys()
        == dict(model.named_parameters()).keys()
    )
    converter.post_update_weights()
    assert converter.original_post_called


def test_gdn_unknown_derived_layout_is_rejected(converter):
    """Reject shape drift instead of exchanging a stale derived tensor."""
    model, conv = make_model()
    conv.ascend_conv1d_weight = nn.Parameter(torch.zeros(6, 4))
    with pytest.raises(ValueError, match="Unsupported Ascend GDN convolution layout"):
        list(converter.iter_model_parameters(model))


def test_router_update_refreshes_fp32_copy_in_place(converter):
    """Keep the FP32 routing cache consistent with received BF16 weights."""
    model, _ = make_model(packed=False)
    model.layer.mlp = nn.Module()
    gate = model.layer.mlp.gate = nn.Linear(6, 4, bias=False, dtype=torch.bfloat16)
    gate.register_parameter(
        "weight_fp32", nn.Parameter(torch.zeros(4, 6), requires_grad=False)
    )
    pointer = gate.weight_fp32.data_ptr()
    parameters = dict(converter.iter_model_parameters(model))
    assert "layer.mlp.gate.weight_fp32" not in parameters
    assert parameters["layer.mlp.gate.weight"] is gate.weight
    with torch.no_grad():
        gate.weight.fill_(3.25)
    converter.post_update_weights()
    assert gate.weight_fp32.data_ptr() == pointer
    torch.testing.assert_close(gate.weight_fp32, gate.weight.float(), rtol=0, atol=0)
