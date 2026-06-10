# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the model-config compatibility preflight.

Policy: fail fast (with a persisted stop reason) when config.json is present
but statically known to crash vLLM/transformers at load — a corrupt config, or
a RoPE block without any max-position field (the DeepSeek-V3.2-Exp shape that
dies with "'PreTrainedConfig' object has no attribute 'max_position_embeddings'"
deep in engine init). A fully absent config is NOT blocked (soft-degrade).
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


def _args(model: str, *, gpu_type: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(model=model, isl=1024, osl=1024, gpu_type=gpu_type)


def _seed_state(session_dir: Path, monkeypatch):
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session_dir),
    )
    from inference_optimizer.orchestrator.shared_state import SharedState

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "reports").mkdir(parents=True, exist_ok=True)
    SharedState(session_id="t", model_name="m", model_path="m").save(session_dir)


@pytest.fixture(autouse=True)
def _default_non_amd_gpu(monkeypatch):
    """Keep config checks hermetic unless a test passes gpu_type explicitly."""
    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.setattr(cli, "_autodetect_gpu_type", lambda: None)


# ---------------------------------------------------------------------------
# _detect_incompatible_model_config
# ---------------------------------------------------------------------------
def test_detect_healthy_config_returns_none(tmp_path):
    m = tmp_path / "ok"
    _write_config(m, model_type="llama", max_position_embeddings=4096)
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_rope_with_maxpos_ok(tmp_path):
    m = tmp_path / "rope_ok"
    _write_config(
        m, model_type="llama", max_position_embeddings=8192,
        rope_scaling={"type": "yarn", "factor": 4.0},
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_amd_unsupported_arch_with_maxpos_blocks(tmp_path):
    m = tmp_path / "deepseek_v32_amd"
    _write_config(
        m,
        model_type="deepseek_v32",
        architectures=["DeepseekV32ForCausalLM"],
        max_position_embeddings=163840,
        rope_scaling={"type": "yarn", "factor": 40.0},
    )

    reason = cli._detect_incompatible_model_config(str(m), gpu_type="mi300x")
    assert reason is not None
    assert "AMD/ROCm" in reason


def test_detect_amd_unsupported_arch_not_blocked_off_amd(tmp_path):
    m = tmp_path / "deepseek_v32_non_amd"
    _write_config(
        m,
        model_type="deepseek_v32",
        architectures=["DeepseekV32ForCausalLM"],
        max_position_embeddings=163840,
        rope_scaling={"type": "yarn", "factor": 40.0},
    )

    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_rope_without_maxpos_blocks(tmp_path):
    m = tmp_path / "rope_no_maxpos"
    _write_config(
        m, model_type="deepseek_v32", rope_scaling={"factor": 4.0},
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "RoPE" in reason


def test_detect_corrupt_config_blocks(tmp_path):
    m = tmp_path / "corrupt"
    m.mkdir(parents=True, exist_ok=True)
    (m / "config.json").write_text("{not valid json", encoding="utf-8")
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "unparseable" in reason


def test_detect_absent_config_not_blocked(tmp_path):
    m = tmp_path / "no_config"
    m.mkdir(parents=True, exist_ok=True)
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_rope_in_nested_text_config(tmp_path):
    m = tmp_path / "nested"
    m.mkdir(parents=True, exist_ok=True)
    (m / "config.json").write_text(
        json.dumps({"text_config": {"rope_theta": 10000.0}}),
        encoding="utf-8",
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None


def test_detect_dual_chunk_blocks_on_amd(tmp_path):
    m = tmp_path / "dual_chunk_amd"
    _write_config(
        m,
        model_type="qwen2",
        max_position_embeddings=1010000,
        dual_chunk_attention_config={"chunk_size": 262144},
    )
    reason = cli._detect_incompatible_model_config(str(m), gpu_type="mi300x")
    assert reason is not None
    assert "dual_chunk" in reason


def test_detect_dual_chunk_not_blocked_off_amd(tmp_path):
    m = tmp_path / "dual_chunk_non_amd"
    _write_config(
        m,
        model_type="qwen2",
        max_position_embeddings=1010000,
        dual_chunk_attention_config={"chunk_size": 262144},
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_unregistered_custom_config_blocks(tmp_path):
    m = tmp_path / "kimi_k2"
    _write_config(
        m,
        model_type="kimi_k2",
        max_position_embeddings=131072,
        auto_map={"AutoConfig": "configuration_deepseek.DeepseekV3Config"},
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "kimi_k2" in reason


def test_detect_custom_automap_known_type_not_blocked(tmp_path):
    m = tmp_path / "known_automap"
    _write_config(
        m,
        model_type="llama",
        max_position_embeddings=8192,
        auto_map={"AutoConfig": "configuration_custom.CustomConfig"},
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


# ---------------------------------------------------------------------------
# _preflight_model_config_compat — persistence + return contract
# ---------------------------------------------------------------------------
def test_preflight_blocks_and_persists(tmp_path, monkeypatch):
    model = tmp_path / "bad"
    _write_config(model, model_type="x", rope_scaling={"factor": 2.0})
    sd = tmp_path / "session"
    _seed_state(sd, monkeypatch)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is True
    final = json.loads((sd / "reports" / "final.json").read_text())
    assert final["stop_reason"] == "model_config_incompatible"
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "model_config_incompatible"
    assert (sd / "session_breakdown.json").exists()


def test_preflight_passes_for_healthy_model(tmp_path, monkeypatch):
    model = tmp_path / "good"
    _write_config(model, model_type="llama", max_position_embeddings=8192)
    sd = tmp_path / "session_ok"
    _seed_state(sd, monkeypatch)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is False
    assert not (sd / "reports" / "final.json").exists()


def test_preflight_blocks_amd_unsupported_arch_from_args_gpu_type(
    tmp_path, monkeypatch,
):
    model = tmp_path / "deepseek_v32"
    _write_config(
        model,
        model_type="deepseek_v32",
        architectures=["DeepseekV32ForCausalLM"],
        max_position_embeddings=163840,
        rope_scaling={"type": "yarn", "factor": 40.0},
    )
    sd = tmp_path / "session_amd_arch"
    _seed_state(sd, monkeypatch)

    assert cli._preflight_model_config_compat(
        _args(str(model), gpu_type="mi300x"), sd,
    ) is True
    final = json.loads((sd / "reports" / "final.json").read_text())
    assert final["stop_reason"] == "model_config_incompatible"


def test_stop_reason_is_canonical_vocab():
    from inference_optimizer.orchestrator.phase_state import (
        STOP_REASON_VOCAB,
        is_valid_stop_reason,
    )

    assert "model_config_incompatible" in STOP_REASON_VOCAB
    assert is_valid_stop_reason("model_config_incompatible")


def test_preflight_persists_under_strict_env(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_STRICT_STOP_REASON", "1")
    model = tmp_path / "bad_strict"
    _write_config(model, model_type="x", rope_parameters={"factor": 2.0})
    sd = tmp_path / "session_strict"
    _seed_state(sd, monkeypatch)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is True
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "model_config_incompatible"
