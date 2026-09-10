"""Validate offline MoE fixtures before using them in NPU integration tests."""

import json

import pytest
import torch
from safetensors import safe_open

from tests.moe_checkpoint_fixtures import create_moe_checkpoint

pytestmark = [pytest.mark.slow, pytest.mark.ci]


@pytest.fixture(scope="module", params=["qwen3moe", "qwen3_5_moe"])
def checkpoint(request, tmp_path_factory):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        path = tmp_path_factory.mktemp("moe") / request.param
        yield request.param, create_moe_checkpoint(path, request.param)
    finally:
        torch.set_num_threads(threads)


def test_checkpoint_shards_match_index_and_bridge_layout(checkpoint):
    """All weights must be indexed, small, and in the bridge's expected layout."""
    model_type, path = checkpoint
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    assert len(set(weight_map.values())) > 1
    actual_map = {}
    total_bytes = 0
    for shard in path.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                assert name not in actual_map
                actual_map[name] = shard.name
                tensor = f.get_tensor(name)
                assert tensor.dtype == torch.bfloat16
                assert torch.isfinite(tensor).all()
                total_bytes += tensor.numel() * tensor.element_size()
    assert actual_map == weight_map
    assert total_bytes == index["metadata"]["total_size"]
    assert total_bytes < 150 * 1024**2

    if model_type == "qwen3moe":
        assert "model.layers.0.mlp.experts.7.gate_proj.weight" in weight_map
        assert not any(name.endswith("experts.gate_up_proj") for name in weight_map)
    else:
        assert "model.language_model.layers.0.mlp.experts.gate_up_proj" in weight_map
        assert weight_map["mtp.fc.weight"] == weight_map["lm_head.weight"]
        assert "mtp.layers.0.mlp.experts.3.gate_proj.weight" in weight_map


def test_checkpoint_loads_offline_and_runs_forward(checkpoint, monkeypatch):
    """The fixture must support actual forward execution, not just conversion."""
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3MoeForCausalLM,
    )

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    model_type, path = checkpoint
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model_cls = (
        Qwen3MoeForCausalLM
        if model_type == "qwen3moe"
        else Qwen3_5MoeForConditionalGeneration
    )
    model = model_cls.from_pretrained(
        path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        experts_implementation="eager",
    ).eval()
    config = getattr(model.config, "text_config", model.config)
    assert len(tokenizer) == config.vocab_size
    assert all(0 <= i < config.vocab_size for i in tokenizer.all_special_ids)
    if model_type == "qwen3_5_moe":
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        assert processor.image_processor is not None
        assert processor.video_processor is not None
        assert config.mtp_num_hidden_layers == 1
        assert config.layer_types == (["linear_attention"] * 3 + ["full_attention"]) * 4

    input_ids = torch.arange(8, 72).reshape(2, 32)
    with torch.no_grad():
        result = model(input_ids=input_ids, use_cache=False, output_router_logits=True)
    assert result.logits.shape == (2, 32, config.vocab_size)
    assert torch.isfinite(result.logits).all()
    assert result.router_logits
    ep_size = 4 if model_type == "qwen3moe" else 2
    experts_per_rank = config.num_experts // ep_size
    for logits in result.router_logits:
        selected = logits.topk(config.num_experts_per_tok, dim=-1).indices
        assert set((selected // experts_per_rank).flatten().tolist()) == set(
            range(ep_size)
        )
