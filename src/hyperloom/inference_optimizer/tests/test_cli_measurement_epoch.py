# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A seeded session records the build's measurement epoch, and only a session that carries it resumes."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.cli import bootstrap as cb
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.orchestrator.policy import gate as policy

    for key in ("HYPERLOOM_BENCHMARK_CONFIG", "HYPERLOOM_AGENTX", "HYPERLOOM_AGENTIC_BACKEND"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {})
    monkeypatch.setattr(cb, "_load_model_arch", lambda *_a, **_k: {})
    monkeypatch.setattr(policy, "detect_gpu_count", lambda: 8)
    monkeypatch.setattr(policy, "research_lane_ceiling", lambda: 16)


def _seed_and_reload(tmp_path) -> SharedState:
    cb._seed_shared_state(
        tmp_path,
        _build_parser().parse_args(["optimize", "--model", "/models/m"]),
        session_id="s",
        framework="sglang",
        settled=cb.SettledLaunch(compute_partition={}, gpu_power_settings={}, base=None),
    )
    return SharedState.load_or_init(tmp_path)


def test_a_synthetic_session_records_the_epoch_and_resumes(tmp_path):
    state = _seed_and_reload(tmp_path)
    assert state.measurement_epoch == cb.MEASUREMENT_EPOCH
    assert state.agentx_backend == ""
    assert cb.measurement_state_is_stale(state) == ""


def test_an_agentx_session_records_inferencex_and_resumes(tmp_path, monkeypatch, native_benchmark):
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    state = _seed_and_reload(tmp_path)
    assert (state.measurement_epoch, state.agentx_backend) == (cb.MEASUREMENT_EPOCH, "inferencex")
    assert cb.measurement_state_is_stale(state) == ""


def test_a_state_written_before_the_switch_is_refused(tmp_path):
    (tmp_path / "state.json").write_text('{"benchmark_mode": "synthetic", "agentx_epoch": 1}', encoding="utf-8")
    assert "epoch 0" in cb.measurement_state_is_stale(SharedState.load_or_init(tmp_path))
