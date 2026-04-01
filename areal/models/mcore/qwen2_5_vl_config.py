import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer import TransformerConfig
from transformers import PretrainedConfig

from areal.models.mcore.common import (
    check_and_construct_configs,
    hf_to_mcore_base_args,
)


def hf_to_mcore_config_qwen2_5_vl(
    hf_config: PretrainedConfig, dtype: torch.dtype
) -> TransformerConfig:
    args: dict = hf_to_mcore_base_args(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        add_qkv_bias=True,
    )
    if hasattr(hf_config, "rope_scaling") and "mrope_section" in hf_config.rope_scaling:
        args["mrope_section"] = hf_config.rope_scaling["mrope_section"]
        
    return check_and_construct_configs(args, TransformerConfig)


def make_mcore_layer_specs_qwen2_5_vl(
    tfconfig: TransformerConfig, use_te: bool = True
):
    return get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=use_te)
