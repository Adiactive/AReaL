# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from areal.engine.megatron_utils.mindspeed_gdn_patch import (
    torch_varlen_causal_conv1d,
)


def _reference_conv(x, weight, bias, cu_seqlens):
    outputs = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        segment = x[:, start:end].transpose(1, 2)
        output = F.conv1d(
            segment,
            weight.unsqueeze(1),
            bias=bias,
            padding=weight.shape[-1] - 1,
            groups=weight.shape[0],
        )[..., : end - start]
        outputs.append(F.silu(output.transpose(1, 2)))
    return torch.cat(outputs, dim=1)


def test_torch_varlen_causal_conv1d_matches_per_sequence_reference():
    torch.manual_seed(0)
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    x = torch.randn(1, 8, 4, requires_grad=True)
    weight = torch.randn(4, 3, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)

    actual, state = torch_varlen_causal_conv1d(
        x,
        weight,
        bias=bias,
        activation="silu",
        cu_seqlens=cu_seqlens,
    )
    expected = _reference_conv(x, weight, bias, cu_seqlens)

    assert state is None
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    actual.sum().backward()
    for tensor in (x, weight, bias):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
