# SPDX-License-Identifier: Apache-2.0
"""Runtime fix-ups for MindSpeed's packed-capable GatedDeltaNet."""

import weakref

import torch
import torch.nn.functional as F

from areal.utils import logging

logger = logging.getLogger("MegatronEngine")

_cu_max_seqlen_cache: dict[int, tuple[weakref.ReferenceType, int]] = {}


def _max_seqlen(cu_seqlens: torch.Tensor) -> int:
    key = id(cu_seqlens)
    cached = _cu_max_seqlen_cache.get(key)
    if cached is not None and cached[0]() is cu_seqlens:
        return cached[1]

    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

    def remove(reference):
        current = _cu_max_seqlen_cache.get(key)
        if current is not None and current[0] is reference:
            _cu_max_seqlen_cache.pop(key, None)

    _cu_max_seqlen_cache[key] = (weakref.ref(cu_seqlens, remove), max_seqlen)
    return max_seqlen


def torch_varlen_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Depthwise causal conv supporting packed sequences with torch ops."""
    del kwargs
    if initial_state is not None or output_final_state:
        raise NotImplementedError("Stateful causal conv is not supported.")

    mask = None
    if cu_seqlens is not None:
        if x.shape[0] != 1:
            raise ValueError("Packed causal conv expects batch size 1.")
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = _max_seqlen(cu_seqlens)
        offsets = torch.arange(max_seqlen, device=x.device)
        mask = offsets.unsqueeze(0) < lengths.unsqueeze(1)
        indices = cu_seqlens[:-1].long().unsqueeze(1) + offsets.unsqueeze(0)
        padded = x.new_zeros((lengths.shape[0], max_seqlen, x.shape[-1]))
        padded[mask] = x[0, indices[mask]]
        x = padded

    channels = x.shape[-1]
    conv_out = F.conv1d(
        x.transpose(1, 2).contiguous(),
        weight.unsqueeze(1),
        bias=bias,
        padding=weight.shape[-1] - 1,
        groups=channels,
    )[..., : x.shape[1]]
    output = conv_out.transpose(1, 2)
    if activation in ("silu", "swish"):
        output = F.silu(output)
    elif activation is not None:
        raise ValueError(f"Unsupported activation: {activation}")

    if mask is not None:
        output = output[mask].unsqueeze(0)
    if residual is not None:
        output = output + residual
    return output, None


def ensure_mindspeed_gdn_model_class() -> bool:
    """Use MindSpeed's GDN class without enabling unrelated EOD patches."""
    try:
        import megatron.core.models.gpt.experimental_attention_variant_module_specs as specs
        import megatron.core.ssm.gated_delta_net as mcore_gdn
        from mindspeed.core.ssm.gated_delta_net import GatedDeltaNet
    except ImportError:
        return False

    if (
        mcore_gdn.GatedDeltaNet is GatedDeltaNet
        and specs.GatedDeltaNet is GatedDeltaNet
    ):
        return True

    mcore_gdn.GatedDeltaNet = GatedDeltaNet
    specs.GatedDeltaNet = GatedDeltaNet
    logger.info("Activated MindSpeed's packed-capable GatedDeltaNet model class.")
    return True


def ensure_mindspeed_gdn_conv1d() -> bool:
    """Bind the torch varlen causal conv into MindSpeed's GDN if needed.

    Returns True when a varlen conv implementation is available afterwards.
    """
    try:
        import mindspeed.core.ssm.gated_delta_net as ms_gdn
    except ImportError:
        return False

    if ms_gdn.causal_conv1d is not None:
        return True

    ms_gdn.causal_conv1d = torch_varlen_causal_conv1d
    logger.info(
        "Activated torch-native varlen causal_conv1d for MindSpeed GDN "
        "because fla_npu is not installed."
    )
    return True
