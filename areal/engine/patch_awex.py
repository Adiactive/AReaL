# SPDX-License-Identifier: Apache-2.0

from functools import wraps


def _patch_vllm_expert_names():
    from awex.converter.vllm_converter import VLLMToHFWeightConverter

    original = VLLMToHFWeightConverter._normalize_name
    if getattr(original, "_areal_expert_names", False):
        return

    @wraps(original)
    def normalize_name(self, name: str) -> str:
        # vLLM 0.26 nests fused expert parameters under RoutedExperts.
        name = name.replace(".experts.routed_experts.", ".experts.")
        return original(self, name)

    normalize_name._areal_expert_names = True
    VLLMToHFWeightConverter._normalize_name = normalize_name


def _patch_vllm_runtime_weights():
    import torch
    from awex.models.ascend.qwen3_5 import VLLMToHFWeightConverterQwen3VL

    converter = VLLMToHFWeightConverterQwen3VL
    original_iter = converter.iter_model_parameters
    original_post = converter.post_update_weights
    if getattr(original_iter, "_areal_runtime_weights", False):
        return

    @wraps(original_iter)
    def iter_model_parameters(self, model):
        self._areal_runtime_copies = []
        derived_names = set()
        for prefix, module in model.named_modules():
            if prefix.endswith(".mlp.gate"):
                destination = getattr(module, "weight_fp32", None)
                if destination is not None:
                    source = module.weight
                    if (
                        destination.shape != source.shape
                        or destination.dtype != torch.float32
                    ):
                        raise ValueError(f"Unsupported Ascend router layout: {prefix}")
                    self._areal_runtime_copies.append((destination, source))
                    derived_names.add(prefix + ".weight_fp32")
            if ".linear_attn.conv1d" not in prefix:
                continue
            destination = getattr(module, "ascend_conv1d_weight", None)
            if destination is None:
                continue
            source = module.weight
            if (
                source.ndim != 3
                or source.shape[1] != 1
                or destination.shape != (source.shape[2], source.shape[0])
            ):
                raise ValueError(f"Unsupported Ascend GDN convolution layout: {prefix}")
            self._areal_runtime_copies.append(
                (destination, source.squeeze(1).transpose(0, 1))
            )
            derived_names.add(prefix + ".ascend_conv1d_weight")
        for name, parameter in original_iter(self, model):
            if name not in derived_names:
                yield name, parameter

    @wraps(original_post)
    @torch.no_grad()
    def post_update_weights(self):
        original_post(self)
        for destination, source in getattr(self, "_areal_runtime_copies", ()):
            # Preserve addresses referenced by captured inference graphs.
            destination.copy_(source)

    iter_model_parameters._areal_runtime_weights = True
    converter.iter_model_parameters = iter_model_parameters
    converter.post_update_weights = post_update_weights


def patch_awex():
    _patch_vllm_expert_names()
    _patch_vllm_runtime_weights()
