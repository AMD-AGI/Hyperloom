"""Tests for the unsupported-multimodal-model preflight gate.

Policy: Hyperloom only supports text-generation (decoder-only causal LM)
models. Multimodal / vision models (e.g. Gemma3ForConditionalGeneration,
qwen2_vl) occasionally leak past the upstream submission filter and only die
~5 minutes in with a cryptic vLLM ``OSError: Can't load image processor ...``,
wasting a full session slot. This Hyperloom-side safety net classifies the
model from its ``config.json`` BEFORE the baseline server is launched and
fails fast with a clear, actionable stop reason.

Best-effort contract: a missing / unreadable / invalid ``config.json`` does
NOT hard-block (the upstream filter + downstream loader still apply); only a
positively-identified multimodal model is rejected.
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


# ---------------------------------------------------------------------------
# 1. classifier — pure detection on config.json signals
# ---------------------------------------------------------------------------
def test_detect_gemma3_conditional_generation(tmp_path):
    m = tmp_path / "gemma3"
    _write_config(m, {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
    })
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert hit["architecture"] == "Gemma3ForConditionalGeneration"
    assert hit["model_type"] == "gemma3"


def test_detect_qwen2_vl_model_type(tmp_path):
    m = tmp_path / "qwen2vl"
    # model_type alone (architectures intentionally not a *_VL class) must trip.
    _write_config(m, {"architectures": ["SomeCustomArch"], "model_type": "qwen2_vl"})
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert hit["model_type"] == "qwen2_vl"


def test_detect_vision_config_key(tmp_path):
    m = tmp_path / "visioncfg"
    _write_config(m, {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vision_config": {"hidden_size": 1024},
    })
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert "vision_config" in hit["signal"]


def test_detect_image_token_index_key(tmp_path):
    m = tmp_path / "imgtoken"
    _write_config(m, {"architectures": ["FooForCausalLM"], "image_token_index": 32000})
    assert cli._detect_unsupported_model(str(m)) is not None


def test_detect_plain_text_models_allowed(tmp_path):
    for i, arch in enumerate(
        ["MistralForCausalLM", "Qwen2ForCausalLM", "LlamaForCausalLM"]
    ):
        m = tmp_path / f"text{i}"
        _write_config(m, {"architectures": [arch], "model_type": "llama"})
        assert cli._detect_unsupported_model(str(m)) is None


def test_detect_missing_config_returns_none(tmp_path):
    # No config.json at all -> best-effort: cannot classify -> do not block.
    assert cli._detect_unsupported_model(str(tmp_path / "nope")) is None


def test_detect_invalid_config_returns_none(tmp_path):
    m = tmp_path / "bad"
    _write_config(m, "{not valid json")
    assert cli._detect_unsupported_model(str(m)) is None


# ---------------------------------------------------------------------------
# 2. preflight gate — persists a canonical stop reason and signals exit
# ---------------------------------------------------------------------------
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
    assert "gemma3" in detail
    assert "multimodal" in detail.lower()
    assert "text-generation" in detail.lower()
    final_md = (sd / "reports" / "final.md").read_text(encoding="utf-8")
    assert "unsupported_model_arch" in final_md
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "unsupported_model_arch"
    # Delivery-artifact parity: fail-fast exits before coordinator.run()'s
    # finally, so it MUST emit session_breakdown.json itself.
    breakdown = sd / "session_breakdown.json"
    assert breakdown.exists()
    assert json.loads(breakdown.read_text(encoding="utf-8"))


def test_preflight_blocks_qwen2_vl_model_type(tmp_path, monkeypatch):
    model = tmp_path / "qwen2vl"
    _write_config(model, {"architectures": ["X"], "model_type": "qwen2_vl"})
    sd = tmp_path / "session_qwen"
    _seed_state(sd, monkeypatch)
    assert cli._preflight_unsupported_model_arch(_args(str(model)), sd) is True
    final = json.loads((sd / "reports" / "final.json").read_text())
    assert final["stop_reason"] == "unsupported_model_arch"


def test_preflight_blocks_vision_config(tmp_path, monkeypatch):
    model = tmp_path / "visioncfg"
    _write_config(model, {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vision_config": {"hidden_size": 1024},
    })
    sd = tmp_path / "session_vision"
    _seed_state(sd, monkeypatch)
    assert cli._preflight_unsupported_model_arch(_args(str(model)), sd) is True


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
    # Missing config.json -> best-effort: do NOT hard-block.
    assert cli._preflight_unsupported_model_arch(_args(str(model)), sd) is False
    assert not (sd / "reports" / "final.json").exists()


# ---------------------------------------------------------------------------
# 3. stop_reason must be a canonical STOP_REASON_VOCAB term so it round-trips
# through the validated set_stop_reason() writer (not mapped to 'unknown',
# not raising under strict mode) and the robustness monitor treats it as
# terminal.
# ---------------------------------------------------------------------------
def test_stop_reason_is_canonical_vocab():
    from inference_optimizer.orchestrator.phase_state import (
        STOP_REASON_VOCAB,
        is_valid_stop_reason,
    )

    assert "unsupported_model_arch" in STOP_REASON_VOCAB
    assert is_valid_stop_reason("unsupported_model_arch")


def test_preflight_persists_stop_reason_under_strict_env(tmp_path, monkeypatch):
    """Under ``INFERENCE_OPTIMIZER_STRICT_STOP_REASON=1`` the preflight must
    still persist the canonical stop_reason — proving the writer goes through
    the validated ``set_stop_reason()`` path AND that the term is registered in
    the vocab (an off-vocab term would raise in strict mode and abort the
    report)."""
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
