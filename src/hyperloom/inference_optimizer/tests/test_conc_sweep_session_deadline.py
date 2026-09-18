# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the sweep's session deadline through the production grid and launch path."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.orchestrator.actions.executors import _aiter_jit, _ray_serving, _server_lifecycle, benchmark_backend
from hyperloom.orchestrator.actions.executors._subprocess_kill import ORCHESTRATOR_CANCELLED_RETURNCODE
from hyperloom.orchestrator.actions.stop_attribution import (
    ORCHESTRATOR_CANCELLED_CLASS,
    SESSION_TIME_EXHAUSTED_CLASS,
    STOPPED_BY_THE_RUN,
    StoppedByTheRun,
)
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
    monkeypatch.setattr(_aiter_jit, "sweep_stale_aiter_locks_if_dead", lambda: {})
    monkeypatch.setattr(_server_lifecycle, "teardown_lifecycle_server", MagicMock())
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

    def run(
        *,
        persistent,
        attempts,
        usable=40.0,
        concs=(8, 4, 2),
        bounded=True,
        total_budget_sec=None,
        baseline_runtime_sec=5.0,
        max_minutes=20,
        closing_grace_sec=60.0,
        recorder=None,
        write_reports=False,
        benchmark_mode="",
    ):
        monkeypatch.setattr(
            _server_lifecycle,
            "resolve_lifecycle_params",
            lambda _: {"eligible": persistent, "framework": "sglang", "port": 8888},
        )
        state.max_minutes = max_minutes if bounded else 0
        state.closing_grace_sec = closing_grace_sec
        state.elapsed_charged_sec = max_minutes * 60.0 - closing_grace_sec - usable
        state.baseline_runtime_sec = baseline_runtime_sec
        state.benchmark_mode = benchmark_mode
        backend = launch_backend(ScriptedLaunchBackend(LaunchScenario(attempts=tuple(attempts)), clock=clock))
        with installed_clock(clock):
            payload = asyncio.run(
                conc_sweep.run_conc_sweep(
                    state,
                    tmp_path / "session",
                    concs=list(concs) if concs is not None else None,
                    total_budget_sec=total_budget_sec,
                    write_reports=write_reports,
                    recorder=recorder,
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
@pytest.mark.parametrize("sweep_budget", [False, True], ids=["session-deadline", "sweep-deadline"])
def test_budget_exhaustion_stops_ladder_and_preserves_attribution(sweep_run, path, sweep_budget):
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
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=path != "restart",
        attempts=attempts,
        bounded=not sweep_budget,
        total_budget_sec=40 if sweep_budget else None,
    )
    error_class = "budget_exhausted" if sweep_budget else SESSION_TIME_EXHAUSTED_CLASS
    assert clock.elapsed == 40.0
    assert len(backend.calls) == len(prefix) + 1
    assert len(calls) == len(backend.calls)
    assert len(bounds) == 1
    assert all(call.get("session_deadline_sec") == 10040.0 for call in calls)
    assert all(call.session_deadline_sec == 10040.0 for call in backend.calls)
    assert payload["budget_exhausted"] is True
    assert payload["budget_skip_reason"] == ("total_budget_exhausted" if sweep_budget else SESSION_TIME_EXHAUSTED_CLASS)
    assert payload["budget_remaining_sec"] == 0.0
    points = payload["optimized"]["points"] + payload["baseline"]["points"]
    skipped = [point for point in points if point["status"] == "skipped"]
    assert len(points) == 6
    assert len(skipped) == {"boot-timeout": 6, "anchor": 3, "fallback": 6}.get(path, 5)
    assert all(point["error_class"] == error_class for point in skipped)
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


@pytest.mark.parametrize("persistent", [False, True], ids=["restart-rungs", "boot-and-reuse"])
@pytest.mark.parametrize("baseline_runtime_sec", [5.0, None], ids=["measured", "unknown"])
def test_two_hour_sweep_admits_work_below_hard_cap(sweep_run, persistent, baseline_runtime_sec):
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=persistent,
        attempts=[_measured()] * 6,
        usable=7080.0,
        max_minutes=120,
        closing_grace_sec=120.0,
        total_budget_sec=7080,
        baseline_runtime_sec=baseline_runtime_sec,
    )
    assert payload["status"] == "succeeded"
    assert len(backend.calls) == 6
    assert clock.elapsed == 30.0
    assert [call.get("variant_expected_sec") for call in calls] == [baseline_runtime_sec] * 6
    assert [call.session_deadline_sec for call in backend.calls] == [17080.0] * 6
    assert all(call.timeout == 7800.0 for call in backend.calls)
    assert len(bounds) == 1
    assert payload["budget_exhausted"] is False


@pytest.mark.parametrize("persistent", [False, True], ids=["restart-rungs", "boot-and-reuse"])
def test_sweep_keeps_running_after_remaining_drops_below_hard_cap(sweep_run, persistent):
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=persistent,
        attempts=[_measured(1300.0)] + [_measured()] * 5,
        bounded=False,
        total_budget_sec=9000,
    )
    assert payload["status"] == "succeeded"
    assert len(backend.calls) == 6
    assert clock.elapsed == 1325.0
    assert [call.session_deadline_sec for call in backend.calls] == [19000.0] * 6


@pytest.mark.parametrize(
    "usable", [None, 7080.0, 9000.0, 21480.0], ids=["sweep-only", "session-earlier", "tied", "sweep-earlier"]
)
def test_boot_retries_share_total_budget_and_preserve_stop_source(sweep_run, usable):
    recorder = MagicMock()
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=True,
        attempts=[LaunchAttempt(outcome="hang", server_log="Loading model weights; not ready\n")] * 2,
        bounded=usable is not None,
        usable=usable or 40.0,
        max_minutes=360 if usable is not None and usable >= 9000.0 else 120,
        closing_grace_sec=120.0,
        total_budget_sec=9000,
        concs=(8, 4),
        recorder=recorder,
        write_reports=True,
    )
    limit = min(usable, 9000.0) if usable is not None else 9000.0
    session_first = usable is not None and usable <= 9000.0
    error_class = SESSION_TIME_EXHAUSTED_CLASS if session_first else "budget_exhausted"
    assert clock.elapsed == limit
    assert len(backend.calls) == (1 if limit < 7800.0 else 2)
    assert all(call.session_deadline_sec == 10000.0 + limit for call in backend.calls)
    assert all(call.timeout == 7800.0 for call in backend.calls)
    assert len(bounds) == 1
    assert payload["budget_exhausted"] is True
    assert payload["budget_skip_reason"] == (
        SESSION_TIME_EXHAUSTED_CLASS if session_first else "total_budget_exhausted"
    )
    assert payload["budget_remaining_sec"] == 0.0
    points = payload["optimized"]["points"] + payload["baseline"]["points"]
    assert len(points) == 4
    skipped = [point for point in points if point["status"] == "skipped"]
    assert len(skipped) == (4 if limit < 7800.0 else 3)
    assert {point["error_class"] for point in skipped} == {error_class}
    aborts = [
        json.loads(path.read_text(encoding="utf-8")) for path in Path(payload["workspace"]).rglob("abort_reason.json")
    ]
    assert any(item["error_class"] == error_class for item in aborts)
    final_abort = json.loads(
        (Path(backend.calls[-1].server_log_path).parent / "abort_reason.json").read_text(encoding="utf-8")
    )
    assert final_abort["error_class"] == error_class
    if not session_first:
        assert all(item["error_class"] != SESSION_TIME_EXHAUSTED_CLASS for item in aborts)
    report = json.loads(Path(payload["report_json_path"]).read_text(encoding="utf-8"))
    assert report["budget_skip_reason"] == payload["budget_skip_reason"]
    recorded_skips = [
        call.kwargs["point"]
        for call in recorder.record_variant.call_args_list
        if call.kwargs["point"]["status"] == "skipped"
    ]
    assert recorded_skips and {point["error_class"] for point in recorded_skips} == {error_class}
    recorder.finish.assert_called_once_with(payload, stop_reason="")


@pytest.mark.parametrize("persistent", [False, True], ids=["restart-rungs", "boot-and-reuse"])
@pytest.mark.parametrize("first_duration", [None, 6.0], ids=["initial-admission", "remaining-admission"])
def test_sweep_admission_prices_measurement_and_preserves_budget_fields(sweep_run, persistent, first_duration):
    attempts = [_measured(first_duration)] if first_duration is not None else [_measured()]
    launches = int(first_duration is not None)
    total = 10 if launches else 4
    recorder = MagicMock()
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=persistent,
        attempts=attempts,
        bounded=False,
        total_budget_sec=total,
        recorder=recorder,
        write_reports=True,
    )
    assert len(backend.calls) == launches
    assert clock.elapsed == (first_duration or 0.0)
    assert payload["status"] == "skipped"
    assert payload["was_skipped"] is True
    assert payload["skip_reason"] == "budget_exhausted_no_successful_pairs"
    assert conc_sweep.conc_sweep_declined_to_run(payload) is False
    assert payload["total_budget_sec"] == total
    assert payload["budget_exhausted"] is True
    assert payload["budget_skip_reason"] == "insufficient_remaining_for_variant"
    assert payload["budget_remaining_sec"] == 4.0
    points = payload["baseline"]["points"] + payload["optimized"]["points"]
    assert sum(point["status"] == "succeeded" for point in points) == launches
    skipped = [point for point in points if point["status"] == "skipped"]
    assert len(skipped) == 6 - launches
    assert {point["error_class"] for point in skipped} == {"budget_exhausted"}
    if persistent:
        _server_lifecycle.teardown_lifecycle_server.assert_called_once()
    recorded_skips = [
        call.kwargs for call in recorder.record_variant.call_args_list if call.kwargs["point"]["status"] == "skipped"
    ]
    assert recorded_skips
    assert all(call["stage"] == "budget_skip" and call["budget_remaining_sec"] == 4.0 for call in recorded_skips)
    recorder.record_budget.assert_called_once_with(
        declared_total_sec=total,
        granted_total_sec=total,
        rung_cost_sec=5.0,
        raised=False,
        gate_active=True,
        deadline=clock.wall() - clock.elapsed + total,
    )


def test_the_agentx_ladder_keeps_the_declared_budget(sweep_run):
    recorder = MagicMock()
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=False,
        attempts=[_measured(6660.0)],
        bounded=False,
        total_budget_sec=9000,
        baseline_runtime_sec=6660.0,
        benchmark_mode="agentx",
        concs=None,
        recorder=recorder,
    )
    assert len(backend.calls) == 1
    assert calls[0]["grid"][0].name == "optimized_conc28"
    assert payload["concs_requested"] == conc_sweep.AGENTX_DEFAULT_CONCS
    assert len(payload["baseline"]["points"]) == len(payload["optimized"]["points"]) == 7
    points = payload["baseline"]["points"] + payload["optimized"]["points"]
    assert sum(point["status"] == "succeeded" for point in points) == 1
    skipped = [point for point in points if point["status"] == "skipped"]
    assert len(skipped) == 13
    assert {point["error_class"] for point in skipped} == {"budget_exhausted"}
    assert payload["total_budget_sec"] == 9000
    assert payload["budget_remaining_sec"] == 2340.0
    assert payload["budget_exhausted"] is True
    assert payload["budget_skip_reason"] == "insufficient_remaining_for_variant"
    recorder.record_budget.assert_called_once_with(
        declared_total_sec=9000,
        granted_total_sec=9000,
        rung_cost_sec=6660.0,
        raised=False,
        gate_active=True,
        deadline=clock.wall() - clock.elapsed + 9000,
    )


@pytest.mark.parametrize("remaining", [0.0, 4.0], ids=["expired", "positive"])
def test_grid_unknown_estimate_does_not_use_hard_cap_for_admission(sweep_run, remaining):
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=False,
        attempts=[LaunchAttempt(outcome="hang")],
        usable=remaining,
        concs=(8,),
        baseline_runtime_sec=None,
    )
    assert len(backend.calls) == (1 if remaining > 0 else 0)
    assert clock.elapsed == remaining
    assert payload["budget_skip_reason"] == SESSION_TIME_EXHAUSTED_CLASS
    if backend.calls:
        assert backend.calls[0].timeout == 7800.0


@pytest.mark.parametrize("cancelled", [False, True], ids=["deadline", "cancelled"])
def test_grid_deadline_attribution_does_not_relabel_cancellation(sweep_run, monkeypatch, cancelled):
    sweep_stop = StoppedByTheRun(
        error_class="budget_exhausted",
        interrupted="conc_sweep total budget exhausted while this round was running",
        never_started="conc_sweep total budget exhausted before this round ran",
        ends_the_batch=True,
    )
    original_grid = conc_sweep.run_grid

    async def attributed_grid(**kwargs):
        kwargs["deadline_stop"] = sweep_stop
        return await original_grid(**kwargs)

    monkeypatch.setattr(conc_sweep, "run_grid", attributed_grid)
    attempt = (
        LaunchAttempt(outcome="died_silently", duration_sec=1.0, returncode=ORCHESTRATOR_CANCELLED_RETURNCODE)
        if cancelled
        else LaunchAttempt(outcome="hang")
    )
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=False, attempts=[attempt] * 2, usable=40.0, concs=(8,)
    )
    expected = STOPPED_BY_THE_RUN[ORCHESTRATOR_CANCELLED_CLASS] if cancelled else sweep_stop
    assert backend.calls
    point = payload["optimized"]["points"][0]
    assert point["error_class"] == expected.error_class
    assert point["error"] == expected.interrupted
    abort_path = Path(backend.calls[0].server_log_path).parent / "abort_reason.json"
    abort = json.loads(abort_path.read_text(encoding="utf-8"))
    assert abort["error_class"] == expected.error_class
    assert abort["error"] == f"{expected.interrupted}; tree reaped"
