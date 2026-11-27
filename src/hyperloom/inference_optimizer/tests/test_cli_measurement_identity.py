# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A seeded session records the workload it measures, and a resume that would measure another is refused."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.cli import _reset_state_file, bootstrap as cb
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.orchestrator.policy import gate as policy

    for key in ("HYPERLOOM_BENCHMARK_CONFIG", "HYPERLOOM_AGENTX", "HYPERLOOM_AGENTIC_BACKEND", "DATASET_PATH"):
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


def test_a_synthetic_session_records_no_backend_and_resumes(tmp_path):
    state = _seed_and_reload(tmp_path)
    assert state.agentx_backend == ""
    assert cb.measurement_state_is_stale(state) == ""


def test_an_agentx_session_records_inferencex_and_resumes(tmp_path, monkeypatch, native_benchmark):
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    state = _seed_and_reload(tmp_path)
    assert state.agentx_backend == "inferencex"
    assert cb.measurement_state_is_stale(state) == ""


@pytest.mark.parametrize(("seeded", "resumed"), [("/data/prompts.jsonl", ""), ("", "/data/prompts.jsonl")])
def test_a_resume_that_sends_another_prompt_dataset_than_the_seed_recorded_is_refused(
    tmp_path, monkeypatch, seeded, resumed
):
    monkeypatch.setenv("DATASET_PATH", seeded)
    state = _seed_and_reload(tmp_path)
    assert (state.prompt_dataset, cb.measurement_state_is_stale(state)) == (seeded, "")

    monkeypatch.setenv("DATASET_PATH", resumed)
    assert "prompt dataset" in cb.measurement_state_is_stale(state)


def test_an_agentx_session_records_no_prompt_dataset_its_rounds_do_not_send(tmp_path, monkeypatch, native_benchmark):
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    monkeypatch.setenv("DATASET_PATH", "/data/prompts.jsonl")
    state = _seed_and_reload(tmp_path)
    assert (state.prompt_dataset, cb.measurement_state_is_stale(state)) == ("", "")


def test_a_reset_session_still_resumes(tmp_path, monkeypatch, native_benchmark):
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    _reset_state_file(tmp_path, _seed_and_reload(tmp_path))
    state = SharedState.load_or_init(tmp_path)
    assert (state.baseline_tput, state.agentx_backend) == (0.0, "inferencex")
    assert cb.measurement_state_is_stale(state) == ""


def test_a_reset_session_keeps_the_prompt_dataset_it_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_PATH", "/data/prompts.jsonl")
    _reset_state_file(tmp_path, _seed_and_reload(tmp_path))
    assert SharedState.load_or_init(tmp_path).prompt_dataset == "/data/prompts.jsonl"
