# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the context-window preflight.

Policy: do NOT stretch a small model context; when ISL+OSL+headroom exceeds
max_position_embeddings, fail fast with a persisted stop reason.
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


# max_position_embeddings loader
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


# headroom resolution
def test_headroom_default_and_override(monkeypatch):
    monkeypatch.delenv(cli._CONTEXT_HEADROOM_ENV, raising=False)
    assert cli._context_headroom_tokens() == cli._CONTEXT_HEADROOM_DEFAULT
    monkeypatch.setenv(cli._CONTEXT_HEADROOM_ENV, "1024")
    assert cli._context_headroom_tokens() == 1024
    monkeypatch.setenv(cli._CONTEXT_HEADROOM_ENV, "garbage")
    assert cli._context_headroom_tokens() == cli._CONTEXT_HEADROOM_DEFAULT


# preflight gate
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
    # PR-review-1: fail-fast must emit session_breakdown.json itself (it exits before coordinator.run's try/finally).
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
    """A 2048 model with ISL+OSL=2048 and headroom 0 just fits the equality boundary (2048 >= 2048)."""
    monkeypatch.setenv(cli._CONTEXT_HEADROOM_ENV, "0")
    model = tmp_path / "ctx2048b"
    _write_config(model, max_position_embeddings=2048)
    sd = tmp_path / "session2048b"
    _seed_state(sd, monkeypatch)
    assert cli._preflight_context_window(_args(str(model), 1024, 1024), sd) is False


# MAX_MODEL_LEN resolution — clamp to the native window (no context stretch).
# Policy: MAX_MODEL_LEN = ISL+OSL+headroom, never above max_position_embeddings (else vllm's --max-model-len crashes the server).
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


# follow-up #1: the preflight stop_reason must be a canonical STOP_REASON_VOCAB term written via set_stop_reason().
def test_context_window_stop_reason_is_canonical_vocab():
    from inference_optimizer.orchestrator.phase_state import (
        STOP_REASON_VOCAB,
        is_valid_stop_reason,
    )

    assert "model_context_window_too_small" in STOP_REASON_VOCAB
    assert is_valid_stop_reason("model_context_window_too_small")


def test_preflight_persists_stop_reason_under_strict_env(tmp_path, monkeypatch):
    """Under ``INFERENCE_OPTIMIZER_STRICT_STOP_REASON=1`` the preflight still persists the canonical stop_reason (proving the validated writer path + vocab registration)."""
    monkeypatch.delenv(cli._CONTEXT_HEADROOM_ENV, raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_STRICT_STOP_REASON", "1")
    model = tmp_path / "ctx2048strict"
    _write_config(model, model_type="llama", max_position_embeddings=2048)
    sd = tmp_path / "session_strict"
    _seed_state(sd, monkeypatch)

    assert cli._preflight_context_window(_args(str(model)), sd) is True
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "model_context_window_too_small"
    final = json.loads((sd / "reports" / "final.json").read_text())
    assert final["stop_reason"] == "model_context_window_too_small"


def test_monitor_offline_vocab_includes_context_window():
    """The robustness monitor's offline STOP_REASON_VOCAB fallback must list the preflight stop_reason so it's treated as terminal."""
    import inference_optimizer

    package_root = Path(inference_optimizer.__file__).resolve().parent
    monitor = package_root / "launcher" / "robustness_monitor.sh.example"
    text = monitor.read_text(encoding="utf-8")
    assert "model_context_window_too_small" in text


def test_preflight_reason_suggests_lowering_headroom(tmp_path, monkeypatch):
    """follow-up #4: the fail-fast advice must tell operators to LOWER the headroom env (which shrinks `required`), not raise it."""
    monkeypatch.delenv(cli._CONTEXT_HEADROOM_ENV, raising=False)
    model = tmp_path / "ctx2048reason"
    _write_config(model, max_position_embeddings=2048)
    sd = tmp_path / "session_reason"
    _seed_state(sd, monkeypatch)

    assert cli._preflight_context_window(_args(str(model)), sd) is True
    detail = json.loads((sd / "reports" / "final.json").read_text())["stop_detail"]
    assert f"lower {cli._CONTEXT_HEADROOM_ENV}".lower() in detail.lower()
    assert f"raise {cli._CONTEXT_HEADROOM_ENV}".lower() not in detail.lower()
