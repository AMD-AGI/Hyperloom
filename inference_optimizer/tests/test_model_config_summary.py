# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``summarize_model_config`` (config.json -> structured model info)."""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.model_config_utils import summarize_model_config


def _write_config(model_dir: Path, payload: dict) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = model_dir / "config.json"
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    return model_dir


def test_summary_missing_config_returns_empty(tmp_path: Path) -> None:
    assert summarize_model_config(str(tmp_path / "nope")) == {}
    assert summarize_model_config("") == {}


def test_summary_mha(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 128000,
            "max_position_embeddings": 8192,
            "torch_dtype": "bfloat16",
        },
    )
    out = summarize_model_config(str(m))
    assert out["model_type"] == "llama"
    assert out["architectures"] == ["LlamaForCausalLM"]
    assert out["attention_type"] == "MHA"
    assert out["num_attention_heads"] == 32
    assert out["num_key_value_heads"] == 32
    assert out["head_dim"] == 128
    assert out["is_moe"] is False


def test_summary_gqa(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "qwen2",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        },
    )
    out = summarize_model_config(str(m))
    assert out["attention_type"] == "GQA"


def test_summary_mqa(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "num_attention_heads": 32,
            "num_key_value_heads": 1,
        },
    )
    out = summarize_model_config(str(m))
    assert out["attention_type"] == "MQA"


def test_summary_mla(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "deepseek_v3",
            "num_attention_heads": 128,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
        },
    )
    out = summarize_model_config(str(m))
    assert out["attention_type"] == "MLA"


def test_summary_moe(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "qwen3_moe",
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "num_experts": 128,
            "num_experts_per_tok": 8,
        },
    )
    out = summarize_model_config(str(m))
    assert out["is_moe"] is True
    assert out["num_experts"] == 128
    assert out["num_experts_per_tok"] == 8


def test_summary_moe_deepseek_alias(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "num_attention_heads": 128,
            "kv_lora_rank": 512,
        },
    )
    out = summarize_model_config(str(m))
    assert out["is_moe"] is True
    assert out["num_experts"] == 256
    assert out["attention_type"] == "MLA"


def test_summary_quantization(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
        },
    )
    out = summarize_model_config(str(m))
    assert out["quantization"] == "fp8"


def test_summary_nested_text_config(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "text_config": {
                "hidden_size": 8192,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "num_hidden_layers": 80,
            },
        },
    )
    out = summarize_model_config(str(m))
    assert out["attention_type"] == "GQA"
    assert out["hidden_size"] == 8192
    assert out["num_hidden_layers"] == 80
