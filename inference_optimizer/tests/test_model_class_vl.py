"""Tests for VL model class inference via nested text_config.

Verifies that _infer_model_class_from_config correctly classifies VL MoE
models (Qwen3-VL, Qwen2-VL) whose MoE fields are nested under text_config,
rather than misclassifying them as dense.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.coordinator_helpers import (
    _infer_model_class_from_config,
)


def _write_config(model_dir: Path, payload: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")


class TestInferModelClassVL:
    def test_qwen3_vl_moe_classified_as_moe(self, tmp_path):
        """Qwen3-VL-235B has MoE fields under text_config — must not be dense."""
        m = tmp_path / "qwen3vl"
        _write_config(m, {
            "architectures": ["Qwen3VLMoeForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "vision_config": {"hidden_size": 1280},
            "text_config": {
                "num_hidden_layers": 94,
                "hidden_size": 7168,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 2048,
            },
        })
        result = _infer_model_class_from_config(str(m))
        assert result != "dense", f"expected MoE class, got {result!r}"
        assert "moe" in result

    def test_qwen2_vl_moe_classified_as_moe(self, tmp_path):
        """Qwen2-VL variant with MoE under text_config is not dense."""
        m = tmp_path / "qwen2vl"
        _write_config(m, {
            "architectures": ["Qwen2VLForConditionalGeneration"],
            "model_type": "qwen2_vl",
            "vision_config": {"hidden_size": 1280},
            "text_config": {
                "num_hidden_layers": 80,
                "hidden_size": 8192,
                "num_experts": 64,
                "num_experts_per_tok": 4,
            },
        })
        result = _infer_model_class_from_config(str(m))
        assert "moe" in result

    def test_flat_llm_moe_still_classified_as_moe(self, tmp_path):
        """Flat (non-VL) MoE config must be unaffected by the text_config logic."""
        m = tmp_path / "qwen3moe"
        _write_config(m, {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "num_hidden_layers": 48,
            "hidden_size": 2048,
            "num_experts": 128,
            "num_experts_per_tok": 8,
        })
        result = _infer_model_class_from_config(str(m))
        assert "moe" in result

    def test_vl_dense_decoder_classified_as_dense(self, tmp_path):
        """A VL model with a dense text decoder must stay classified as dense."""
        m = tmp_path / "qwen2vl_dense"
        _write_config(m, {
            "architectures": ["Qwen2VLForConditionalGeneration"],
            "model_type": "qwen2_vl",
            "vision_config": {"hidden_size": 1280},
            "text_config": {
                "num_hidden_layers": 28,
                "hidden_size": 3584,
            },
        })
        result = _infer_model_class_from_config(str(m))
        assert result == "dense"

    def test_top_level_keys_win_over_text_config(self, tmp_path):
        """Top-level fields must take precedence over text_config on conflict."""
        m = tmp_path / "conflict"
        _write_config(m, {
            "model_type": "qwen3_vl",
            "num_experts": 0,          # top-level says dense (0 experts)
            "text_config": {
                "num_experts": 128,    # nested says MoE — top-level should win
                "num_experts_per_tok": 8,
            },
        })
        result = _infer_model_class_from_config(str(m))
        assert result == "dense"

    def test_missing_config_returns_dense(self, tmp_path):
        """Missing config.json gracefully returns dense (safe degrade)."""
        result = _infer_model_class_from_config(str(tmp_path / "no_model"))
        assert result == "dense"
