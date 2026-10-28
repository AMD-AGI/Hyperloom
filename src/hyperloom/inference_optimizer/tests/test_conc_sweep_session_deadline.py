# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the sweep's session deadline through the production grid and launch path."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from hyperloom.inference_optimizer.breakdown.recorder.conc_sweep_event import SweepBudget
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
    monkeypatch.setattr(_aiter_jit, "sweep_stale_aiter_locks_if_dead", dict)
    monkeypatch.setattr(_server_lifecycle, "teardown_lifecycle_server", MagicMock())
    monkeypatch.setattr(benchmark_backend, "resolve_benchmark_interpreter", lambda: sys.executable)
    monkeypatch.setattr(gr, "build_benchmark_command", lambda **kwargs: [sys.executable, "scripted-benchmark"])
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._inferencex_patcher.ensure_eval_probe_patched", lambda *args: True
    )
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
    monkeypatch.setattr(conc_sweep, "prepare_grid_benchmark_config", lambda *args, **kwargs: base)
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
        benchmark_mode="",
        anchor_conc=0,
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
        state.conc = anchor_conc
        backend = launch_backend(ScriptedLaunchBackend(LaunchScenario(attempts=tuple(attempts)), clock=clock))
        with installed_clock(clock):
            payload = asyncio.run(
                conc_sweep.run_conc_sweep(
                    state,
                    tmp_path / "session",
                    concs=list(concs) if concs is not None else None,
                    total_budget_sec=total_budget_sec,
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


def _hang():
    return LaunchAttempt(outcome="hang")


def _points(payload, arm):
    return [(point["conc"], point["status"], point["error_class"]) for point in payload[arm]["points"]]


@pytest.mark.parametrize("persistent", [False, True], ids=["restart-rungs", "boot-and-reuse"])
def test_the_first_arms_later_rungs_run_on_half_the_deadline(sweep_run, persistent):
    payload, backend, calls, bounds, clock = sweep_run(persistent=persistent, attempts=[_measured()] * 6, usable=100.0)
    assert payload["status"] == "succeeded"
    assert len(backend.calls) == 6
    assert len(bounds) == 1
    assert [call["grid"][0].name for call in calls] == [
        f"{arm}_conc{conc}" for arm in ("baseline", "optimized") for conc in (2, 4, 8)
    ]
    # The baseline's first rung has no pair to leave room for; its later rungs get half of the 100s left when it
    # started, and the optimized arm runs to the session's own deadline.
    deadlines = [10100.0, 10050.0, 10050.0] + [10100.0] * 3
    assert [call["stop"].session_deadline_sec for call in calls] == deadlines
    assert [call.session_deadline_sec for call in backend.calls] == deadlines
    # A rung is held to its deadline, not admitted against an estimate of what it costs.
    assert all(call["stop"].variant_expected_sec is None for call in calls)
    assert clock.elapsed == 30.0


def test_a_long_top_rung_no_longer_takes_the_whole_budget(sweep_run):
    """OSL 8192 at CONC 16, 2869s granted, a 413s boot and a 655s anchor benchmark. The anchor rung sends the
    session's request window, so it costs what the session measured; it runs first, the cheaper rung next, and the
    top rung -- longer than the whole budget -- never starts, so the optimized arm pairs both."""
    recorder = MagicMock()
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=True,
        attempts=[_measured(1068.0), _measured(300.0), _measured(884.0), _measured(250.0)],
        bounded=False,
        total_budget_sec=2869,
        concs=(256, 16, 2),
        anchor_conc=16,
        recorder=recorder,
    )
    assert [call["grid"][0].name for call in calls] == [
        "baseline_conc16",
        "baseline_conc2",
        "optimized_conc16",
        "optimized_conc2",
    ]
    assert payload["status"] == "succeeded"
    assert payload["summary"]["successful_pairs"] == 2
    for arm in ("baseline", "optimized"):
        assert _points(payload, arm) == [
            (2, "succeeded", ""),
            (16, "succeeded", ""),
            (256, "skipped", "budget_exhausted"),
        ]
    assert clock.elapsed == 1068.0 + 300.0 + 884.0 + 250.0
    assert payload["budget_skip_reason"] == "first_arm_share_exhausted"
    recorder.record_budget.assert_called_once_with(
        SweepBudget(declared_total_sec=2869, deadline=clock.wall() - clock.elapsed + 2869)
    )


@pytest.mark.parametrize("persistent", [False, True], ids=["restart-rungs", "boot-and-reuse"])
def test_a_rung_that_cannot_outrun_the_one_before_it_is_never_started(sweep_run, persistent):
    """Past the anchor no rung benchmarks faster than the one before it, so with 3s of the baseline's half left after
    a 12s rung the next one is recorded as never run."""
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=persistent,
        attempts=[_measured(5.0), _measured(12.0), _measured(5.0), _measured(12.0)],
        bounded=False,
        total_budget_sec=40,
    )
    assert [call["grid"][0].name for call in calls] == [
        "baseline_conc2",
        "baseline_conc4",
        "optimized_conc2",
        "optimized_conc4",
    ]
    assert clock.elapsed == 34.0
    assert payload["summary"]["successful_pairs"] == 2
    for arm in ("baseline", "optimized"):
        assert _points(payload, arm)[2] == (8, "skipped", "budget_exhausted")
    assert payload["baseline"]["points"][2]["error"] == (
        "the conc_sweep budget left no time to measure this concurrency on both arms"
    )
    assert payload["budget_skip_reason"] == "first_arm_share_exhausted"
    assert payload["budget_remaining_sec"] == 3.0


def test_an_anchor_longer_than_half_the_budget_is_still_paired(sweep_run):
    """The first rung has no pair yet to leave room for, so a baseline anchor past half of the budget still runs to
    completion, and the faster optimized anchor fits in what it left."""
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=True,
        attempts=[_measured(1550.0), _measured(1150.0)],
        bounded=False,
        total_budget_sec=2869,
        concs=(16, 32),
        anchor_conc=16,
    )
    assert [call["grid"][0].name for call in calls] == ["baseline_conc16", "baseline_conc32", "optimized_conc16"]
    # Past its half by the time the anchor finished, the baseline arm starts no second rung.
    assert len(backend.calls) == 2
    assert clock.elapsed == 2700.0
    assert payload["status"] == "succeeded"
    assert payload["summary"]["successful_pairs"] == 1
    for arm in ("baseline", "optimized"):
        assert _points(payload, arm) == [(16, "succeeded", ""), (32, "skipped", "budget_exhausted")]


@pytest.mark.parametrize("sweep_budget", [False, True], ids=["session-deadline", "sweep-deadline"])
def test_the_first_arms_half_drops_whole_pairs(sweep_run, sweep_budget):
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=True,
        attempts=[_measured(10.0), _hang(), _measured(10.0)],
        bounded=not sweep_budget,
        total_budget_sec=40 if sweep_budget else None,
    )
    assert [call["grid"][0].name for call in calls] == ["baseline_conc2", "baseline_conc4", "optimized_conc2"]
    assert [call.session_deadline_sec for call in backend.calls] == [10040.0, 10020.0, 10040.0]
    assert clock.elapsed == 30.0
    assert payload["status"] == "succeeded"
    assert payload["summary"]["successful_pairs"] == 1
    for arm in ("baseline", "optimized"):
        assert _points(payload, arm) == [
            (2, "succeeded", ""),
            (4, "skipped", "budget_exhausted"),
            (8, "skipped", "budget_exhausted"),
        ]
    assert payload["optimized"]["points"][2]["error"] == (
        "the conc_sweep budget left no time to measure this concurrency on both arms"
    )
    assert payload["budget_exhausted"] is True
    assert payload["budget_skip_reason"] == "first_arm_share_exhausted"


def test_a_failed_rung_is_still_paired_so_the_failure_is_attributed(sweep_run):
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=False,
        attempts=[LaunchAttempt(outcome="died_silently", duration_sec=5.0), _measured(), _hang(), *[_measured()] * 2],
        total_budget_sec=40,
        bounded=False,
    )
    assert [call["grid"][0].name for call in calls] == [
        "baseline_conc2",
        "baseline_conc4",
        "baseline_conc8",
        "optimized_conc2",
        "optimized_conc4",
    ]
    assert [status for _, status, _ in _points(payload, "baseline")] == ["failed", "succeeded", "skipped"]
    assert [status for _, status, _ in _points(payload, "optimized")] == ["succeeded", "succeeded", "skipped"]
    assert payload["summary"]["successful_pairs"] == 1


def test_a_first_arm_that_reaches_nothing_leaves_the_second_nothing_to_pair(sweep_run):
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=True, attempts=[_hang()], bounded=False, total_budget_sec=40
    )
    assert [call["grid"][0].name for call in calls] == ["baseline_conc2"]
    assert clock.elapsed == 40.0
    assert payload["budget_skip_reason"] == "total_budget_exhausted"
    assert payload["status"] == "skipped"
    assert payload["was_skipped"] is True
    assert payload["skip_reason"] == "budget_exhausted_no_successful_pairs"
    assert conc_sweep.conc_sweep_declined_to_run(payload) is False
    for arm in ("baseline", "optimized"):
        assert {error_class for _, _, error_class in _points(payload, arm)} == {"budget_exhausted"}
    _server_lifecycle.teardown_lifecycle_server.assert_called()


def test_an_expired_session_launches_nothing_and_names_the_session(sweep_run):
    payload, backend, calls, bounds, clock = sweep_run(persistent=True, attempts=[_measured()] * 6, usable=0.0)
    assert not backend.calls
    assert len(calls) == 1
    assert len(bounds) == 1
    assert clock.elapsed == 0.0
    assert payload["status"] == "skipped"
    assert payload["budget_skip_reason"] == SESSION_TIME_EXHAUSTED_CLASS
    assert payload["budget_remaining_sec"] == 0.0
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
    assert all(call["stop"].session_deadline_sec is None for call in calls)
    assert clock.elapsed == 30.0


@pytest.mark.parametrize("persistent", [False, True], ids=["restart-rungs", "boot-and-reuse"])
def test_two_hour_sweep_runs_below_the_hard_cap(sweep_run, persistent):
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=persistent,
        attempts=[_measured()] * 6,
        usable=7080.0,
        max_minutes=120,
        closing_grace_sec=120.0,
        total_budget_sec=7080,
    )
    assert payload["status"] == "succeeded"
    assert len(backend.calls) == 6
    assert clock.elapsed == 30.0
    assert [call.session_deadline_sec for call in backend.calls] == [17080.0] + [13540.0] * 2 + [17080.0] * 3
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
    assert [call.session_deadline_sec for call in backend.calls] == [19000.0] + [14500.0] * 2 + [19000.0] * 3


@pytest.mark.parametrize(
    "usable", [None, 7080.0, 9000.0, 21480.0], ids=["sweep-only", "session-earlier", "tied", "sweep-earlier"]
)
def test_the_second_arm_records_who_owns_the_deadline_it_ran_out_on(sweep_run, usable):
    recorder = MagicMock()
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=True,
        attempts=[
            _measured(10.0),
            LaunchAttempt(outcome="hang", server_log="Loading model weights; not ready\n"),
            LaunchAttempt(outcome="hang", server_log="Loading model weights; not ready\n"),
        ],
        bounded=usable is not None,
        usable=usable or 40.0,
        max_minutes=360 if usable is not None and usable >= 9000.0 else 120,
        closing_grace_sec=120.0,
        total_budget_sec=9000,
        concs=(4, 8),
        recorder=recorder,
    )
    limit = min(usable, 9000.0) if usable is not None else 9000.0
    session_first = usable is not None and usable <= 9000.0
    error_class = SESSION_TIME_EXHAUSTED_CLASS if session_first else "budget_exhausted"
    assert clock.elapsed == limit
    assert [call.session_deadline_sec for call in backend.calls] == [
        10000.0 + limit,
        10000.0 + limit / 2,
        10000.0 + limit,
    ]
    assert all(call.timeout == 7800.0 for call in backend.calls)
    assert len(bounds) == 1
    assert payload["status"] == "skipped"
    assert payload["budget_exhausted"] is True
    assert payload["budget_skip_reason"] == (
        SESSION_TIME_EXHAUSTED_CLASS if session_first else "total_budget_exhausted"
    )
    assert payload["budget_remaining_sec"] == 0.0
    # The baseline rung reaped at its half names the split; the optimized rung names the deadline it ran into.
    assert _points(payload, "baseline") == [(4, "succeeded", ""), (8, "skipped", "budget_exhausted")]
    assert _points(payload, "optimized") == [(4, "skipped", error_class), (8, "skipped", "budget_exhausted")]
    final_abort = json.loads(
        (Path(backend.calls[-1].server_log_path).parent / "abort_reason.json").read_text(encoding="utf-8")
    )
    assert final_abort["error_class"] == error_class
    report = json.loads(Path(payload["report_json_path"]).read_text(encoding="utf-8"))
    assert report["budget_skip_reason"] == payload["budget_skip_reason"]
    recorded_skips = [
        call.args[1].point
        for call in recorder.record_variant.call_args_list
        if call.args[1].point["status"] == "skipped"
    ]
    assert {point["error_class"] for point in recorded_skips} == {"budget_exhausted", error_class}
    recorder.finish.assert_called_once_with(payload, stop_reason="")


def test_the_agentx_ladder_keeps_the_declared_budget_and_runs_its_cheap_end_first(sweep_run):
    """An agentic rung replays a fixed set of trajectories, so after the anchor the high-CONC rungs come first; the
    serial CONC-1 replay, the costliest, is the one a deadline drops."""
    payload, backend, calls, bounds, clock = sweep_run(
        persistent=False,
        attempts=[_measured(3000.0), _hang(), _measured(2000.0)],
        bounded=False,
        total_budget_sec=9000,
        baseline_runtime_sec=6660.0,
        benchmark_mode="agentx",
        concs=None,
        anchor_conc=14,
    )
    assert [call["grid"][0].name for call in calls] == ["baseline_conc14", "baseline_conc28", "optimized_conc14"]
    assert payload["concs_requested"] == conc_sweep.AGENTX_DEFAULT_CONCS
    assert len(payload["baseline"]["points"]) == len(payload["optimized"]["points"]) == 7
    for arm in ("baseline", "optimized"):
        assert [point["conc"] for point in payload[arm]["points"] if point["status"] == "succeeded"] == [14]
    assert payload["total_budget_sec"] == 9000
    assert clock.elapsed == 6500.0


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
        kwargs["stop"] = replace(kwargs["stop"], deadline_stop=sweep_stop)
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
    point = payload["baseline"]["points"][0]
    assert point["error_class"] == expected.error_class
    assert point["error"] == expected.interrupted
    abort_path = Path(backend.calls[0].server_log_path).parent / "abort_reason.json"
    abort = json.loads(abort_path.read_text(encoding="utf-8"))
    assert abort["error_class"] == expected.error_class
    assert abort["error"] == f"{expected.interrupted}; tree reaped"
