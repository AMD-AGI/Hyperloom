"""Tests for the unsupported-model preflight gate (whitelist approach).

Only text-generation (causal LM) models pass; others are rejected before the
baseline boot. A missing/invalid ``config.json`` does NOT hard-block — only a
positively-identified non-text-generation model is rejected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from inference_optimizer import cli


def _write_config(model_dir: Path, payload) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = model_dir / "config.json"
    if isinstance(payload, str):
        cfg.write_text(payload, encoding="utf-8")
    else:
        cfg.write_text(json.dumps(payload), encoding="utf-8")


def _args(model: str) -> argparse.Namespace:
    return argparse.Namespace(model=model)


def _seed_state(session_dir: Path, monkeypatch) -> None:
    """Create a minimal seeded session so the preflight can load/save state."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session_dir))
    from inference_optimizer.orchestrator.shared_state import SharedState

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "reports").mkdir(parents=True, exist_ok=True)
    SharedState(session_id="t", model_name="m", model_path="m").save(session_dir)


# 1. classifier — whitelist-based detection
def test_detect_gemma3_conditional_generation_rejected(tmp_path):
    """Gemma3ForConditionalGeneration is not a causal LM arch -> rejected."""
    m = tmp_path / "gemma3"
    _write_config(m, {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
    })
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert hit["architecture"] == "Gemma3ForConditionalGeneration"
    assert "unsupported architecture" in hit["signal"]


def test_detect_unknown_arch_rejected(tmp_path):
    """An unknown architecture not matching text-generation markers -> rejected."""
    m = tmp_path / "custom"
    _write_config(m, {"architectures": ["SomeCustomArch"], "model_type": "qwen2_vl"})
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert hit["architecture"] == "SomeCustomArch"


def test_detect_vision_encoder_rejected(tmp_path):
    """CLIPVisionModel is not a text-generation arch -> rejected."""
    m = tmp_path / "clip"
    _write_config(m, {"architectures": ["CLIPVisionModel"], "model_type": "clip"})
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None


def test_detect_seq2seq_rejected(tmp_path):
    """T5ForConditionalGeneration is not ForCausalLM -> rejected."""
    m = tmp_path / "t5"
    _write_config(m, {"architectures": ["T5ForConditionalGeneration"], "model_type": "t5"})
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None


def test_detect_unknown_model_type_no_arch_rejected(tmp_path):
    """No architectures field + unknown model_type -> rejected."""
    m = tmp_path / "weird"
    _write_config(m, {"model_type": "some_exotic_type"})
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert "allowlist" in hit["signal"]


def test_detect_empty_config_rejected(tmp_path):
    """A readable config without identity tags cannot prove text generation."""
    m = tmp_path / "empty"
    _write_config(m, {})
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert "neither architectures nor model_type" in hit["signal"]


def test_detect_phi3v_causal_lm_rejected(tmp_path):
    """Some VLM architectures still end with ForCausalLM and need a denylist."""
    m = tmp_path / "phi3v"
    _write_config(m, {"architectures": ["Phi3VForCausalLM"], "model_type": "phi3_v"})
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert "Phi3VForCausalLM" in hit["signal"]


def test_detect_causal_lm_allowed(tmp_path):
    """Standard ForCausalLM architectures pass through."""
    for i, arch in enumerate(
        ["MistralForCausalLM", "Qwen2ForCausalLM", "LlamaForCausalLM"]
    ):
        m = tmp_path / f"text{i}"
        _write_config(m, {"architectures": [arch], "model_type": "llama"})
        assert cli._detect_unsupported_model(str(m)) is None


def test_detect_causal_lm_variant_allowed(tmp_path):
    """Text-generation variants with suffixes after ForCausalLM pass through."""
    for i, arch in enumerate(["DeepseekV3ForCausalLMNextN", "LlamaForCausalLMEagle3"]):
        m = tmp_path / f"variant{i}"
        _write_config(m, {"architectures": [arch], "model_type": "llama"})
        assert cli._detect_unsupported_model(str(m)) is None


def test_detect_lm_head_model_allowed(tmp_path):
    """GPT2LMHeadModel style architectures pass through."""
    m = tmp_path / "gpt2"
    _write_config(m, {"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"})
    assert cli._detect_unsupported_model(str(m)) is None


def test_detect_known_model_type_no_arch_allowed(tmp_path):
    """No architectures field but known model_type -> allowed."""
    m = tmp_path / "known_type"
    _write_config(m, {"model_type": "mistral"})
    assert cli._detect_unsupported_model(str(m)) is None


def test_detect_known_text_model_type_with_nonstandard_arch_allowed(tmp_path):
    """Known text-generation model_type can allow non-ForCausalLM class names."""
    m = tmp_path / "chatglm"
    _write_config(m, {
        "architectures": ["ChatGLMForConditionalGeneration"],
        "model_type": "chatglm",
    })
    assert cli._detect_unsupported_model(str(m)) is None


def test_detect_causal_lm_with_vision_config_rejected(tmp_path):
    """Explicit vision keys win over a generic ForCausalLM architecture."""
    m = tmp_path / "visioncausal"
    _write_config(m, {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vision_config": {"hidden_size": 1024},
    })
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert "vision_config" in hit["signal"]


def test_detect_kimi_k25_text_compatible_exception_allowed(tmp_path):
    """Kimi-K2.6 carries vision_config but its text path is benchmark-compatible."""
    m = tmp_path / "kimi_k25"
    _write_config(m, {
        "architectures": ["KimiK25ForConditionalGeneration"],
        "model_type": "kimi_k25",
        "vision_config": {"hidden_size": 1024},
    })
    assert cli._detect_unsupported_model(str(m)) is None


def test_detect_missing_config_returns_none(tmp_path):
    assert cli._detect_unsupported_model(str(tmp_path / "nope")) is None


def test_detect_invalid_config_returns_none(tmp_path):
    m = tmp_path / "bad"
    _write_config(m, "{not valid json")
    assert cli._detect_unsupported_model(str(m)) is None


# 2. preflight gate — persists a canonical stop reason and signals exit
def test_preflight_blocks_gemma3(tmp_path, monkeypatch):
    model = tmp_path / "gemma3"
    _write_config(model, {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
    })
    sd = tmp_path / "session"
    _seed_state(sd, monkeypatch)

    blocked = cli._preflight_unsupported_model_arch(_args(str(model)), sd)

    assert blocked is True
    final = json.loads((sd / "reports" / "final.json").read_text())
    assert final["stop_reason"] == "unsupported_model_arch"
    detail = final["stop_detail"]
    assert "Gemma3ForConditionalGeneration" in detail
    assert "text-generation" in detail.lower()
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "unsupported_model_arch"
    breakdown = sd / "session_breakdown.json"
    assert breakdown.exists()


def test_preflight_blocks_unknown_arch(tmp_path, monkeypatch):
    model = tmp_path / "custom"
    _write_config(model, {"architectures": ["X"], "model_type": "qwen2_vl"})
    sd = tmp_path / "session_custom"
    _seed_state(sd, monkeypatch)
    assert cli._preflight_unsupported_model_arch(_args(str(model)), sd) is True
    final = json.loads((sd / "reports" / "final.json").read_text())
    assert final["stop_reason"] == "unsupported_model_arch"


def test_preflight_allows_plain_text(tmp_path, monkeypatch):
    model = tmp_path / "mistral"
    _write_config(model, {
        "architectures": ["MistralForCausalLM"],
        "model_type": "mistral",
    })
    sd = tmp_path / "session_text"
    _seed_state(sd, monkeypatch)
    assert cli._preflight_unsupported_model_arch(_args(str(model)), sd) is False
    assert not (sd / "reports" / "final.json").exists()


def test_preflight_allows_missing_config(tmp_path, monkeypatch):
    model = tmp_path / "no_config"
    model.mkdir()
    sd = tmp_path / "session_missing"
    _seed_state(sd, monkeypatch)
    assert cli._preflight_unsupported_model_arch(_args(str(model)), sd) is False
    assert not (sd / "reports" / "final.json").exists()


# 3. stop_reason vocabulary registration
def test_stop_reason_is_canonical_vocab():
    from inference_optimizer.orchestrator.phase_state import (
        STOP_REASON_VOCAB,
        is_valid_stop_reason,
    )

    assert "unsupported_model_arch" in STOP_REASON_VOCAB
    assert is_valid_stop_reason("unsupported_model_arch")


def test_preflight_persists_stop_reason_under_strict_env(tmp_path, monkeypatch):
    """Under strict mode the preflight must still persist the stop_reason."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_STRICT_STOP_REASON", "1")
    model = tmp_path / "gemma3strict"
    _write_config(model, {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
    })
    sd = tmp_path / "session_strict"
    _seed_state(sd, monkeypatch)
    assert cli._preflight_unsupported_model_arch(_args(str(model)), sd) is True
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "unsupported_model_arch"
