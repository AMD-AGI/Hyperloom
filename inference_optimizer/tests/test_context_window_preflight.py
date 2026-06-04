"""Tests for the context-window preflight.

Policy: do NOT stretch a small model context with --context-length (RoPE
extrapolation / CUDA-error risk). Instead, when ISL+OSL+headroom exceeds the
model's max_position_embeddings, fail fast with a persisted stop reason so the
run does not boot a server that 400s every request and dies on the watchdog.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from inference_optimizer import cli


def _write_config(model_dir: Path, **fields) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(fields), encoding="utf-8")


def _args(model: str, isl: int = 1024, osl: int = 1024) -> argparse.Namespace:
    return argparse.Namespace(model=model, isl=isl, osl=osl)


# ---------------------------------------------------------------------------
# max_position_embeddings loader
# ---------------------------------------------------------------------------
def test_loads_max_position_embeddings(tmp_path):
    m = tmp_path / "model"
    _write_config(m, model_type="llama", max_position_embeddings=2048)
    assert cli._load_model_max_position_embeddings(str(m)) == 2048


def test_loads_alias_and_nested_text_config(tmp_path):
    m = tmp_path / "alias"
    _write_config(m, n_positions=4096)
    assert cli._load_model_max_position_embeddings(str(m)) == 4096
    n = tmp_path / "nested"
    n.mkdir()
    (n / "config.json").write_text(
        json.dumps({"text_config": {"max_position_embeddings": 8192}}),
        encoding="utf-8",
    )
    assert cli._load_model_max_position_embeddings(str(n)) == 8192


def test_missing_config_returns_none(tmp_path):
    assert cli._load_model_max_position_embeddings(str(tmp_path / "nope")) is None


# ---------------------------------------------------------------------------
# headroom resolution
# ---------------------------------------------------------------------------
def test_headroom_default_and_override(monkeypatch):
    monkeypatch.delenv(cli._CONTEXT_HEADROOM_ENV, raising=False)
    assert cli._context_headroom_tokens() == cli._CONTEXT_HEADROOM_DEFAULT
    monkeypatch.setenv(cli._CONTEXT_HEADROOM_ENV, "1024")
    assert cli._context_headroom_tokens() == 1024
    monkeypatch.setenv(cli._CONTEXT_HEADROOM_ENV, "garbage")
    assert cli._context_headroom_tokens() == cli._CONTEXT_HEADROOM_DEFAULT


# ---------------------------------------------------------------------------
# preflight gate
# ---------------------------------------------------------------------------
def _seed_state(session_dir: Path, monkeypatch):
    """Create a minimal seeded session so the preflight can load/save state."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session_dir))
    from inference_optimizer.orchestrator.shared_state import SharedState

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "reports").mkdir(parents=True, exist_ok=True)
    SharedState(session_id="t", model_name="m", model_path="m").save(session_dir)


def test_preflight_fails_for_2048_model(tmp_path, monkeypatch):
    monkeypatch.delenv(cli._CONTEXT_HEADROOM_ENV, raising=False)
    model = tmp_path / "ctx2048"
    _write_config(model, model_type="llama", max_position_embeddings=2048)
    sd = tmp_path / "session"
    _seed_state(sd, monkeypatch)

    blocked = cli._preflight_context_window(_args(str(model)), sd)

    assert blocked is True
    final = json.loads((sd / "reports" / "final.json").read_text())
    assert final["stop_reason"] == "model_context_window_too_small"
    assert "max_position_embeddings=2048" in final["stop_detail"]
    final_md = (sd / "reports" / "final.md").read_text(encoding="utf-8")
    assert "model_context_window_too_small" in final_md
    assert "max_position_embeddings=2048" in final_md
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "model_context_window_too_small"
    # PR-review-1: fail-fast exits before cli's coordinator.run try/finally, so
    # it MUST emit session_breakdown.json itself — otherwise CI's delivery
    # contract turns a clean skip into "Missing artifacts".
    breakdown = sd / "session_breakdown.json"
    assert breakdown.exists()
    assert json.loads(breakdown.read_text(encoding="utf-8"))


def test_preflight_passes_for_4096_model(tmp_path, monkeypatch):
    monkeypatch.delenv(cli._CONTEXT_HEADROOM_ENV, raising=False)
    model = tmp_path / "ctx4096"
    _write_config(model, model_type="llama", max_position_embeddings=4096)
    sd = tmp_path / "session4096"
    _seed_state(sd, monkeypatch)

    assert cli._preflight_context_window(_args(str(model)), sd) is False
    assert not (sd / "reports" / "final.json").exists()


def test_preflight_skipped_when_maxpos_unknown(tmp_path, monkeypatch):
    model = tmp_path / "no_config"
    model.mkdir()
    sd = tmp_path / "session_unknown"
    _seed_state(sd, monkeypatch)
    # No config.json -> maxpos unknown -> gate skipped (do not block).
    assert cli._preflight_context_window(_args(str(model)), sd) is False


def test_preflight_2048_passes_when_headroom_lowered(tmp_path, monkeypatch):
    """A 2048 model with ISL+OSL=1024+1024 and headroom forced to 0 just fits
    the equality boundary (2048 >= 2048) — gate does not block."""
    monkeypatch.setenv(cli._CONTEXT_HEADROOM_ENV, "0")
    model = tmp_path / "ctx2048b"
    _write_config(model, max_position_embeddings=2048)
    sd = tmp_path / "session2048b"
    _seed_state(sd, monkeypatch)
    assert cli._preflight_context_window(_args(str(model), 1024, 1024), sd) is False


# ---------------------------------------------------------------------------
# MAX_MODEL_LEN resolution — clamp to the native window (no context stretch).
# ---------------------------------------------------------------------------
# Policy: MAX_MODEL_LEN = ISL+OSL+headroom, but never above the model's
# max_position_embeddings. sglang ignores MAX_MODEL_LEN (context_length stays
# None), but the vllm benchmark wires it into --max-model-len; an unclamped
# ISL+OSL+4096 would exceed a small native window and crash the server (or
# silently mis-size KV cache), defeating the "fail fast, don't stretch" policy.
def test_max_model_len_clamped_to_native_window(tmp_path):
    model = tmp_path / "ctx4096"
    _write_config(model, max_position_embeddings=4096)
    # desired = 1024 + 1024 + 4096 = 6144, native window = 4096 -> clamp to 4096.
    assert cli._resolve_max_model_len(1024, 1024, str(model)) == 4096


def test_max_model_len_uses_full_headroom_when_window_large(tmp_path):
    model = tmp_path / "ctx32768"
    _write_config(model, max_position_embeddings=32768)
    # native window is comfortably above desired -> keep ISL+OSL+headroom.
    assert (
        cli._resolve_max_model_len(1024, 1024, str(model))
        == 1024 + 1024 + cli._MAX_MODEL_LEN_HEADROOM
    )


def test_max_model_len_fallback_when_maxpos_unknown(tmp_path):
    model = tmp_path / "noconfig"
    model.mkdir()
    # No config.json -> cannot clamp -> keep the headroom default (prior behaviour).
    assert (
        cli._resolve_max_model_len(1024, 1024, str(model))
        == 1024 + 1024 + cli._MAX_MODEL_LEN_HEADROOM
    )
