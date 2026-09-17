# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the sweep's session deadline through the production grid and launch path."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.orchestrator.actions.executors import _ray_serving, _server_lifecycle, benchmark_backend
from hyperloom.orchestrator.actions.stop_attribution import SESSION_TIME_EXHAUSTED_CLASS
from hyperloom.orchestrator.kernel import conc_sweep
from hyperloom.orchestrator.rehearsal import LaunchAttempt, LaunchScenario, ScriptedLaunchBackend, VirtualClock
from hyperloom.orchestrator.rehearsal.clock import installed_clock
from hyperloom.orchestrator.state import shared_state


@pytest.fixture
def sweep_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch_backend):
    clock = VirtualClock()

    class ClockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(clock.wall(), tz=tz)

    monkeypatch.setattr(shared_state, "datetime", ClockDatetime)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(tmp_path / "leaks"))
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.setattr(_ray_serving, "maybe_serving_lease", lambda **kwargs: None)
    monkeypatch.setattr(_server_lifecycle, "teardown_lifecycle_server", lambda **kwargs: None)
    monkeypatch.setattr(benchmark_backend, "resolve_benchmark_interpreter", lambda: sys.executable)
    monkeypatch.setattr(gr, "build_benchmark_command", lambda **kwargs: [sys.executable, "scripted-benchmark"])
    monkeypatch.setattr(gr, "ensure_eval_probe_patched", lambda *args: True)
    monkeypatch.setattr(conc_sweep, "_build_roofline_ceiling", lambda *args, **kwargs: None)
    base = tmp_path / "baseline.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "/models/test-model",
                    "benchmark_script": "sglang_mi300x.sh",
                    "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(conc_sweep, "materialize_config_with_envs", lambda *args, **kwargs: base)
    state = shared_state.SharedState()
    state.baseline_tput = 100.0
    state.baseline_runtime_sec = 5.0
    state.isl = state.osl = 256
    state.tp = 1
    state.gpu_type = "mi300x"
    state.model_path = "/models/test-model"
    state.baseline_config_path = str(base)
    state.current_best = {"extra_server_args": "--enable-torch-compile"}
    state.max_minutes = 20
    state.closing_grace_sec = 60.0
    state.leg_anchor_unix = clock.wall()
    state.start_ts = datetime.fromtimestamp(clock.wall(), timezone.utc).isoformat()
    grid_calls = []
    bound_calls = []
    real_bounds = state.grid_session_deadline_sec

    def bounds(**kwargs):
        bound_calls.append(clock.monotonic())
        return real_bounds(**kwargs)

    monkeypatch.setattr(state, "grid_session_deadline_sec", bounds)

    async def record_grid(**kwargs):
        grid_calls.append(dict(kwargs))
        return await gr.run_grid(**kwargs)

    monkeypatch.setattr(conc_sweep, "run_grid", record_grid)

    def run(*, persistent, attempts, usable=40.0, concs=(8, 4, 2), bounded=True):
        monkeypatch.setattr(
            _server_lifecycle,
            "resolve_lifecycle_params",
            lambda _: {"eligible": persistent, "framework": "sglang", "port": 8888},
        )
        state.elapsed_charged_sec = 1200.0 - 60.0 - usable
        if not bounded:
            state.max_minutes = 0
        backend = launch_backend(ScriptedLaunchBackend(LaunchScenario(attempts=tuple(attempts)), clock=clock))
        with installed_clock(clock):
            payload = asyncio.run(
                conc_sweep.run_conc_sweep(
                    state, tmp_path / "session", concs=list(concs), total_budget_sec=None, write_reports=False
                )
            )
        return payload, backend, grid_calls, bound_calls, clock

    return run


def _measured(seconds=5.0):
    return LaunchAttempt(
        duration_sec=seconds,
        artifacts={
            "benchmark_sglang_test/benchmark_report.json": {
                "success": True,
                "framework": "sglang",
                "throughput": {
                    "output_throughput": 100.0,
                    "request_throughput": 1.0,
                    "completed_requests": 40,
                    "duration_seconds": seconds,
                },
            }
        },
    )


@pytest.mark.parametrize("persistent", [False, True], ids=["restart-rungs", "boot-and-reuse"])
def test_session_deadline_is_shared_across_rungs_and_anchor(sweep_run, persistent):
    payload, backend, calls, bounds, clock = sweep_run(persistent=persistent, attempts=[_measured()] * 6, usable=100.0)
    assert payload["status"] == "succeeded"
    assert len(backend.calls) == 6
    assert len(bounds) == 1
    assert [call["grid"][0].name for call in calls] == [
        f"{arm}_conc{conc}" for arm in ("optimized", "baseline") for conc in (8, 4, 2)
    ]
    assert [call.get("session_deadline_sec") for call in calls] == [10100.0] * 6
    assert [call.get("variant_expected_sec") for call in calls] == [5.0] * 6
    assert [call.session_deadline_sec for call in backend.calls] == [10100.0] * 6
    assert clock.elapsed == 30.0


@pytest.mark.parametrize("path", ["restart", "reuse", "boot-retry", "boot-timeout", "anchor", "fallback"])
def test_session_exhaustion_stops_ladder_and_preserves_attribution(sweep_run, path):
    if path == "boot-timeout":
        prefix = []
    elif path == "boot-retry":
        prefix = [LaunchAttempt(outcome="died_silently", duration_sec=10.0)]
    elif path == "anchor":
        prefix = [_measured(10.0)] * 3
    elif path == "fallback":
        prefix = [LaunchAttempt(outcome="died_silently", duration_sec=5.0)] * 3
    else:
        prefix = [_measured(10.0)]
    attempts = prefix + [LaunchAttempt(outcome="hang")] * 20
    payload, backend, calls, bounds, clock = sweep_run(persistent=path != "restart", attempts=attempts)
    assert clock.elapsed == 40.0
    assert len(backend.calls) == len(prefix) + 1
    assert len(calls) == len(backend.calls)
    assert len(bounds) == 1
    assert all(call.get("session_deadline_sec") == 10040.0 for call in calls)
    assert all(call.session_deadline_sec == 10040.0 for call in backend.calls)
    assert payload["budget_exhausted"] is True
    assert payload["budget_skip_reason"] == SESSION_TIME_EXHAUSTED_CLASS
    assert payload["budget_remaining_sec"] == 0.0
    points = payload["optimized"]["points"] + payload["baseline"]["points"]
    skipped = [point for point in points if point["status"] == "skipped"]
    assert len(points) == 6
    assert len(skipped) == {"boot-timeout": 6, "anchor": 3, "fallback": 6}.get(path, 5)
    assert all(point["error_class"] == SESSION_TIME_EXHAUSTED_CLASS for point in skipped)
    assert payload["status"] == ("failed" if path == "boot-retry" else "skipped")
    if path == "boot-retry":
        assert (
            next(point for point in points if point["conc"] == 8 and point["arm"] == "optimized")["status"] == "failed"
        )


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("usable", [0.0, 4.0], ids=["expired", "cannot-fit-measurement"])
def test_session_admission_skip_is_not_a_boot_failure(sweep_run, persistent, usable):
    payload, backend, calls, bounds, clock = sweep_run(persistent=persistent, attempts=[_measured()] * 6, usable=usable)
    assert not backend.calls
    assert len(calls) == 1
    assert len(bounds) == 1
    assert clock.elapsed == 0.0
    assert payload["status"] == "skipped"
    assert payload["budget_skip_reason"] == SESSION_TIME_EXHAUSTED_CLASS
    assert payload["budget_remaining_sec"] == usable
    assert all(
        point["error_class"] == SESSION_TIME_EXHAUSTED_CLASS
        for arm in ("optimized", "baseline")
        for point in payload[arm]["points"]
    )


def test_unbounded_session_keeps_unbounded_grid(sweep_run):
    payload, backend, calls, bounds, clock = sweep_run(persistent=True, attempts=[_measured()] * 6, bounded=False)
    assert payload["status"] == "succeeded"
    assert payload["budget_exhausted"] is False
    assert len(backend.calls) == 6
    assert all(call.get("session_deadline_sec") is None for call in calls)
    assert clock.elapsed == 30.0
