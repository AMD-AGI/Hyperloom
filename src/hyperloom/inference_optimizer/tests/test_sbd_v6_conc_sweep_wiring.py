# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The sweep's own wiring into the SBD V6 ``conc_sweep`` event.

The recorder is covered on its own in ``test_sbd_v6_conc_sweep_timeline``.
What is pinned here is that ``run_conc_sweep`` actually calls it at the points
the facts exist -- a recorder nothing drives records nothing, and the failure
looks exactly like a sweep that never ran.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.conc_sweep_event import (
    ARM_BASELINE,
    ARM_OPTIMIZED,
    GRID_REQUESTED,
    PRODUCER,
    STAGE_BOOT,
    STAGE_BOOT_ATTEMPT,
    STAGE_BUDGET_SKIP,
    STAGE_REUSE,
    STAGE_SERVER_RESTART,
    STRATEGY_SERVER_RESTART,
    STRATEGY_SINGLE_SERVER,
    conc_sweep_event_id,
    make_conc_sweep_recorder,
)
from hyperloom.inference_optimizer.breakdown.recorder.event_sink import make_sink
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.actions.executors._grid_runner import GridVariant, VariantResult
from hyperloom.orchestrator.kernel.conc_sweep import run_conc_sweep
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    sd = tmp_path / "Qwen-Test" / "20260101T000000Z"
    sd.mkdir(parents=True)
    (sd / "reports").mkdir()
    return sd


@pytest.fixture
def baseline_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.yaml"
    path.write_text("benchmark:\n  benchmark_script: bench.sh\n")
    return path


def _state(baseline_yaml: Path, **overrides: Any) -> SharedState:
    state = SharedState()
    state.baseline_tput = 100.0
    state.isl = 1024
    state.osl = 1024
    state.tp = 8
    state.phase = "SWEEP"
    state.macro_cycle = 2
    state.benchmark_mode = "synthetic"
    state.current_best = {
        "action": "explore",
        "variant_name": "env_tuning_3",
        "tput": 140.0,
        "extra_server_args": "--enable-torch-compile",
        "extra_envs": {"SGLANG_FOO": "1"},
    }
    state.baseline_config_path = str(baseline_yaml)
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _variant(name: str, *, throughput: float | None, status: str = "succeeded", envs: dict[str, str]) -> VariantResult:
    return VariantResult(
        name=name,
        extra_server_args="",
        extra_envs=dict(envs),
        status=status,
        output_throughput=throughput,
        request_throughput=throughput,
        total_token_throughput=throughput,
        duration_seconds=10.0 if status == "succeeded" else None,
        ttft_mean_ms=25.0 if status == "succeeded" else None,
        e2el_mean_ms=200.0 if status == "succeeded" else None,
        workspace=f"/tmp/{name}",
        error=None if status == "succeeded" else "boom",
        error_class="" if status == "succeeded" else "magpie_timeout",
        killed_overtime=False,
    )


def _materialize(src, out_dir, **_kw):
    out = Path(out_dir) / "conc_sweep_base.with_envs.yaml"
    out.write_text(Path(src).read_text())
    return out


def _recorder():
    recorder = make_conc_sweep_recorder(
        make_sink(conc_sweep_event_id(phase="SWEEP", macro_cycle=2), producer=PRODUCER),
        task_id="cs-1",
        task_kind="conc_sweep",
        reason="phase_entry",
        params={"concs": [4, 16], "variant_timeout_sec": 1800, "total_budget_sec": 9000},
    )
    assert recorder is not None
    return recorder


def _sweep_event(session_dir: Path) -> dict[str, Any]:
    events = [event for event in read_timeline_events(session_dir) if event.get("type") == "conc_sweep"]
    assert len(events) == 1
    return events[0]


def _run(state: SharedState, session_dir: Path, *, recorder: Any, run_grid: Any, concs: list[int] | None = None):
    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_materialize),
    ):
        return asyncio.run(
            run_conc_sweep(
                state,
                session_dir,
                concs=[4, 16] if concs is None else concs,
                recorder=recorder,
            )
        )


async def _all_succeed(*, grid: list[GridVariant], **_kw):
    return [
        _variant(
            variant.name,
            throughput=(130.0 if variant.name.startswith("optimized_") else 100.0) * int(variant.extra_envs["CONC"]),
            envs=variant.extra_envs,
        )
        for variant in grid
    ]


def test_sweep_executor_uses_shared_benchmark_deadline(session_dir: Path, baseline_yaml: Path, monkeypatch):
    from types import SimpleNamespace
    from hyperloom.orchestrator.actions.executors import conc_sweep as executor_module

    state = _state(baseline_yaml)
    monkeypatch.setattr(executor_module.SharedState, "load_or_init", lambda _: state)

    async def run_sweep(state, session_dir, *, concs, total_budget_sec, recorder):
        return {"status": "succeeded"}

    monkeypatch.setattr(executor_module, "run_conc_sweep", run_sweep)
    executor = executor_module.ConcSweepExecutor()
    monkeypatch.setattr(executor, "_make_recorder", lambda *args: None)
    ctx = SimpleNamespace(
        extra={"session_dir": str(session_dir)},
        task=SimpleNamespace(params={"variant_timeout_sec": 1}),
    )
    assert asyncio.run(executor._run(ctx))["status"] == "succeeded"


def test_a_sweep_records_both_arms_and_every_rung(session_dir: Path, baseline_yaml: Path):
    state = _state(baseline_yaml)
    with session_scope(session_dir):
        payload = _run(state, session_dir, recorder=_recorder(), run_grid=_all_succeed)
        event = _sweep_event(session_dir)

    assert payload["status"] == "succeeded"
    assert event["status"] == "succeeded"
    ext = event["ext"]
    for arm in (ARM_BASELINE, ARM_OPTIMIZED):
        assert [point["conc"] for point in ext["arms"][arm]["points"]] == [4, 16]
    # The optimized arm is the one that carries the session's server args.
    assert ext["arms"][ARM_OPTIMIZED]["extra_server_args"] == "--enable-torch-compile"
    assert ext["arms"][ARM_OPTIMIZED]["extra_envs"]["SGLANG_FOO"] == "1"
    assert ext["arms"][ARM_BASELINE]["extra_server_args"] == ""
    assert [pair["conc"] for pair in ext["comparison"]] == [4, 16]
    assert [pair["speedup"] for pair in ext["comparison"]] == pytest.approx([1.3, 1.3])
    assert ext["result"]["successful_pairs"] == 2


def test_the_sweep_records_the_plan_it_resolved(session_dir: Path, baseline_yaml: Path):
    state = _state(baseline_yaml)
    with session_scope(session_dir):
        _run(state, session_dir, recorder=_recorder(), run_grid=_all_succeed)
        ext = _sweep_event(session_dir)["ext"]

    assert ext["plan"]["grid_source"] == GRID_REQUESTED
    assert ext["plan"]["concs_requested"] == [4, 16]
    # The ladder runs descending so a single-server arm boots at its most
    # demanding rung.
    assert ext["plan"]["concs_ordered"] == [16, 4]
    assert ext["plan"]["arms_order"] == [ARM_OPTIMIZED, ARM_BASELINE]
    assert ext["workload"] == {
        "session_id": session_dir.name,
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "benchmark_mode": "synthetic",
    }
    assert ext["input_anchor"]["base_variant_id"] == "env_tuning_3"
    assert ext["input_anchor"]["base_action"] == "explore"
    assert ext["input_anchor"]["anchor_tput"] == 140.0
    assert ext["environment"]["sweep_task_id"].startswith("conc_sweep_")
    assert ext["environment"]["base_config_path"].endswith("conc_sweep_base.with_envs.yaml")


def test_each_rung_carries_the_load_and_the_cap_it_ran_under(session_dir: Path, baseline_yaml: Path):
    state = _state(baseline_yaml)
    with session_scope(session_dir):
        _run(state, session_dir, recorder=_recorder(), run_grid=_all_succeed)
        ext = _sweep_event(session_dir)["ext"]

    arm = ext["arms"][ARM_OPTIMIZED]
    # NUM_PROMPTS is CONC times the factor, and was never written down before.
    assert [(rung["conc"], rung["num_prompts"]) for rung in arm["grid"]] == [(16, 80), (4, 20)]
    for point in arm["points"]:
        assert point["num_prompts"] == point["conc"] * 5
        assert point["granted_cap_sec"] == 7800.0
        assert point["start_time"]
        assert point["wall_duration_sec"] is not None


def test_the_arm_says_which_execution_strategy_ran_the_ladder(session_dir: Path, baseline_yaml: Path):
    state = _state(baseline_yaml)
    with session_scope(session_dir):
        _run(state, session_dir, recorder=_recorder(), run_grid=_all_succeed)
        ext = _sweep_event(session_dir)["ext"]

    arm = ext["arms"][ARM_OPTIMIZED]
    assert arm["strategy"] == STRATEGY_SERVER_RESTART
    assert arm["strategy_reason"] == "framework_not_lifecycle_eligible"
    assert arm["lifecycle"]["eligible"] is False
    assert {point["stage"] for point in arm["points"]} == {STAGE_SERVER_RESTART}
    assert arm["status"] == "succeeded"


def test_a_lifecycle_capable_arm_boots_once_and_reuses(session_dir: Path, baseline_yaml: Path):
    state = _state(baseline_yaml)
    with (
        session_scope(session_dir),
        patch(
            "hyperloom.orchestrator.actions.executors._server_lifecycle.resolve_lifecycle_params",
            return_value={"eligible": True, "reason": "supported", "port": 8888, "framework": "sglang"},
        ),
    ):
        _run(state, session_dir, recorder=_recorder(), run_grid=_all_succeed)
        ext = _sweep_event(session_dir)["ext"]

    arm = ext["arms"][ARM_OPTIMIZED]
    assert arm["strategy"] == STRATEGY_SINGLE_SERVER
    assert arm["lifecycle"] == {"eligible": True, "reason": "supported", "port": 8888, "framework": "sglang"}
    # Booted at the top of the ladder, reused down it.
    assert [(point["conc"], point["stage"]) for point in arm["points"]] == [
        (4, STAGE_REUSE),
        (16, STAGE_BOOT),
    ]
    assert arm["boot"]["succeeded"] is True
    assert arm["boot"]["booted_conc"] == 16
    assert arm["boot"]["failed_concs"] == []


def test_the_concurrency_the_server_would_not_boot_at_is_recorded(session_dir: Path, baseline_yaml: Path):
    state = _state(baseline_yaml)

    async def _top_rung_will_not_boot(*, grid: list[GridVariant], **_kw):
        return [
            _variant(
                variant.name,
                throughput=None if int(variant.extra_envs["CONC"]) == 16 else 400.0,
                status="failed" if int(variant.extra_envs["CONC"]) == 16 else "succeeded",
                envs=variant.extra_envs,
            )
            for variant in grid
        ]

    with (
        session_scope(session_dir),
        patch(
            "hyperloom.orchestrator.actions.executors._server_lifecycle.resolve_lifecycle_params",
            return_value={"eligible": True, "reason": "supported", "port": 8888, "framework": "sglang"},
        ),
    ):
        _run(state, session_dir, recorder=_recorder(), run_grid=_top_rung_will_not_boot)
        ext = _sweep_event(session_dir)["ext"]

    boot = ext["arms"][ARM_OPTIMIZED]["boot"]
    assert boot["succeeded"] is True
    assert boot["booted_conc"] == 4
    assert boot["failed_concs"] == [16]
    assert [(row["conc"], row["status"]) for row in boot["attempts"]] == [(16, "failed"), (4, "succeeded")]
    # The failed boot counts toward the curve, since a lower rung did come up.
    stages = {point["conc"]: point["stage"] for point in ext["arms"][ARM_OPTIMIZED]["points"]}
    assert stages == {4: STAGE_BOOT, 16: STAGE_BOOT_ATTEMPT}


def test_a_budget_that_refuses_an_arm_records_the_gate(session_dir: Path, baseline_yaml: Path):
    state = _state(baseline_yaml, closing_phase=True)
    with session_scope(session_dir):
        _run(state, session_dir, recorder=_recorder(), run_grid=_all_succeed)
        ext = _sweep_event(session_dir)["ext"]

    for arm in (ARM_BASELINE, ARM_OPTIMIZED):
        assert ext["arms"][arm]["refused"] == {"reason": "session_deadline_reserve", "remaining_sec": 0.0}
        assert ext["arms"][arm]["points"] == []
    assert ext["runtime"]["budget_skip_reason"] == "session_deadline_reserve"


def test_a_rung_the_budget_refused_mid_ladder_is_recorded_as_such(session_dir: Path, baseline_yaml: Path):
    state = _state(baseline_yaml, baseline_runtime_sec=5.0)
    calls = []
    clock = {"now": 100.0}

    async def _refuse_reuse(
        *, grid: list[GridVariant], session_deadline_sec, variant_expected_sec, deadline_stop, **_kw
    ):
        calls.append((session_deadline_sec, variant_expected_sec, deadline_stop.error_class))
        if len(calls) == 1:
            clock["now"] += 6.0
            return await _all_succeed(grid=grid)
        assert len(calls) == 2
        assert 0 < session_deadline_sec - clock["now"] < variant_expected_sec
        return [
            VariantResult(
                name=variant.name,
                extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="skipped",
                error=deadline_stop.never_started,
                error_class=deadline_stop.error_class,
            )
            for variant in grid
        ]

    with (
        session_scope(session_dir),
        patch(
            "hyperloom.orchestrator.actions.executors._server_lifecycle.resolve_lifecycle_params",
            return_value={"eligible": True, "reason": "supported", "port": 8888, "framework": "sglang"},
        ),
        patch("hyperloom.orchestrator.kernel.conc_sweep.time.monotonic", side_effect=lambda: clock["now"]),
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_refuse_reuse),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_materialize),
    ):
        asyncio.run(
            run_conc_sweep(
                state,
                session_dir,
                concs=[4, 16],
                total_budget_sec=10,
                recorder=_recorder(),
            )
        )
        ext = _sweep_event(session_dir)["ext"]

    assert calls == [(110.0, 5.0, "budget_exhausted")] * 2
    # The optimized arm booted at 16 and was cut off before 4.
    stages = {point["conc"]: point["stage"] for point in ext["arms"][ARM_OPTIMIZED]["points"]}
    assert stages == {16: STAGE_BOOT, 4: STAGE_BUDGET_SKIP}
    refused = [point for point in ext["arms"][ARM_OPTIMIZED]["points"] if point["conc"] == 4][0]
    assert refused["status"] == "skipped"
    assert refused["error_class"] == "budget_exhausted"
    assert refused["budget_remaining_sec"] == 4.0
    assert refused["output_throughput"] is None
    # The arm that never got to start says so at the arm level instead.
    assert ext["arms"][ARM_BASELINE]["refused"]["reason"] == "insufficient_remaining_for_variant"
    assert ext["runtime"]["budget_skip_reason"] == "insufficient_remaining_for_variant"


@pytest.mark.parametrize(
    "override, reason",
    [
        ({"baseline_tput": 0.0}, "no_baseline_tput"),
        ({"isl": 0}, "missing_workload_shape"),
        ({"current_best": {"extra_server_args": "", "extra_envs": {}}}, "no_optimization_to_compare"),
    ],
)
def test_a_sweep_that_declined_still_reaches_the_timeline(
    session_dir: Path,
    baseline_yaml: Path,
    override: dict[str, Any],
    reason: str,
):
    state = _state(baseline_yaml, **override)
    with session_scope(session_dir):
        payload = _run(state, session_dir, recorder=_recorder(), run_grid=_all_succeed)
        event = _sweep_event(session_dir)

    assert payload["skip_reason"] == reason
    assert event["status"] == "skipped"
    assert event["ext"]["result"]["skip_reason"] == reason
    assert event["ext"]["result"]["declined"] is True


def test_an_empty_ladder_declines_rather_than_running_nothing(session_dir: Path, baseline_yaml: Path):
    state = _state(baseline_yaml)
    with session_scope(session_dir):
        _run(state, session_dir, recorder=_recorder(), run_grid=_all_succeed, concs=[])
        event = _sweep_event(session_dir)

    assert event["ext"]["result"]["skip_reason"] == "empty_conc_list"


def test_a_missing_config_declines_with_the_path_it_looked_for(session_dir: Path, tmp_path: Path):
    state = _state(tmp_path / "nope.yaml")
    with session_scope(session_dir):
        _run(state, session_dir, recorder=_recorder(), run_grid=_all_succeed)
        event = _sweep_event(session_dir)

    assert event["ext"]["result"]["skip_reason"] == "baseline_config_missing"


def test_a_sweep_with_no_recorder_writes_no_event(session_dir: Path, baseline_yaml: Path):
    state = _state(baseline_yaml)
    with session_scope(session_dir):
        payload = _run(state, session_dir, recorder=None, run_grid=_all_succeed)
        assert not [event for event in read_timeline_events(session_dir) if event.get("type") == "conc_sweep"]

    assert payload["status"] == "succeeded"
    assert len(payload["optimized"]["points"]) == 2


class _Task:
    task_id = "task-77"
    kind = "conc_sweep"
    params: dict[str, Any] = {"reason": "phase_entry", "concs": [4], "total_budget_sec": 9000}


class _Ctx:
    task = _Task()
    extra: dict[str, Any] = {}


def _ctx(session_dir: Path) -> _Ctx:
    ctx = _Ctx()
    ctx.extra = {"session_dir": str(session_dir)}
    return ctx


def test_the_dispatch_names_the_event_and_binds_the_session_itself(session_dir: Path, baseline_yaml: Path):
    from hyperloom.orchestrator.actions.executors.conc_sweep import ConcSweepExecutor

    state = _state(baseline_yaml)
    state.save(session_dir)

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_all_succeed),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_materialize),
    ):
        result = asyncio.run(ConcSweepExecutor()(_ctx(session_dir)))

    assert result["status"] == "succeeded"
    event = _sweep_event(session_dir)
    assert event["id"] == "sweep:2:conc_sweep"
    assert event["status"] == "succeeded"
    # The dispatch identifies itself, which the sweep's own payload never did.
    assert event["ext"]["request"] == {
        "task_id": "task-77",
        "task_kind": "conc_sweep",
        "reason": "phase_entry",
        "requested_concs": [4],
        "requested_variant_timeout_sec": None,
        "requested_total_budget_sec": 9000,
    }


def test_a_sweep_that_raised_leaves_a_closed_failed_event(session_dir: Path, baseline_yaml: Path):
    from hyperloom.orchestrator.actions.executors.conc_sweep import ConcSweepExecutor

    state = _state(baseline_yaml)
    state.save(session_dir)

    def _explode(*_a: Any, **_kw: Any):
        raise RuntimeError("the harness fell over")

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_all_succeed),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_explode),
        pytest.raises(RuntimeError),
    ):
        asyncio.run(ConcSweepExecutor()(_ctx(session_dir)))

    event = _sweep_event(session_dir)
    assert event["status"] == "failed"
    assert event["ext"]["failure"]["error_class"] == "RuntimeError"
    assert "the harness fell over" in event["ext"]["failure"]["message"]


def _cancelled_after(rungs: int) -> Any:
    """A ladder that measures ``rungs`` rungs and is then cancelled.

    ``CancelledError`` is a ``BaseException``, which is how a session shutdown
    reaches an arm: the handler around each rung catches ``Exception``, so the
    arm's own teardown is the first thing to see it.
    """
    calls = {"n": 0}

    async def _run_grid(*, grid: list[GridVariant], **_kw):
        calls["n"] += 1
        if calls["n"] > rungs:
            raise asyncio.CancelledError
        return await _all_succeed(grid=grid)

    return _run_grid


def test_a_sweep_that_could_not_write_its_report_records_the_step_that_broke(
    session_dir: Path,
    baseline_yaml: Path,
):
    state = _state(baseline_yaml)

    def _no_csv(*_a: Any, **_kw: Any):
        raise OSError("read-only file system")

    with (
        session_scope(session_dir),
        patch("hyperloom.orchestrator.kernel.conc_sweep._write_csv", side_effect=_no_csv),
    ):
        payload = _run(state, session_dir, recorder=_recorder(), run_grid=_all_succeed)
        event = _sweep_event(session_dir)

    # The measurement stands; only its persistence broke.
    assert payload["status"] == "succeeded"
    assert event["status"] == "succeeded"
    assert event["ext"]["failure"] is None
    assert event["ext"]["faults"] == [
        {"stage": "report_write", "error_class": "OSError", "message": "read-only file system"}
    ]


def test_a_lifecycle_that_would_not_resolve_is_recorded_rather_than_read_as_unsupported(
    session_dir: Path,
    baseline_yaml: Path,
):
    state = _state(baseline_yaml)
    with (
        session_scope(session_dir),
        patch(
            "hyperloom.orchestrator.actions.executors._server_lifecycle.resolve_lifecycle_params",
            side_effect=RuntimeError("lifecycle probe exploded"),
        ),
    ):
        _run(state, session_dir, recorder=_recorder(), run_grid=_all_succeed)
        ext = _sweep_event(session_dir)["ext"]

    # Both arms fall back to a server per rung, and each says why it had to.
    assert [fault["stage"] for fault in ext["faults"]] == [
        f"resolve_lifecycle_params:{ARM_OPTIMIZED}",
        f"resolve_lifecycle_params:{ARM_BASELINE}",
    ]
    assert {fault["error_class"] for fault in ext["faults"]} == {"RuntimeError"}
    for arm in (ARM_BASELINE, ARM_OPTIMIZED):
        assert ext["arms"][arm]["strategy"] == STRATEGY_SERVER_RESTART


def test_a_sweep_whose_arm_lost_a_rung_does_not_read_as_whole(session_dir: Path, baseline_yaml: Path):
    state = _state(baseline_yaml)
    calls = {"n": 0}

    async def _one_bad_rung(*, grid: list[GridVariant], **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return [_variant(v.name, throughput=None, status="failed", envs=v.extra_envs) for v in grid]
        return await _all_succeed(grid=grid)

    with session_scope(session_dir):
        _run(state, session_dir, recorder=_recorder(), run_grid=_one_bad_rung)
        event = _sweep_event(session_dir)

    assert event["ext"]["arms"][ARM_OPTIMIZED]["status"] == "degraded"
    assert event["status"] == "degraded"
    # Nothing raised, so nothing claims a failure.
    assert event["ext"]["failure"] is None
    assert event["ext"]["faults"] == []


def _cancelled_arm(session_dir: Path, baseline_yaml: Path, *, rungs: int, lifecycle: bool) -> dict[str, Any]:
    """Drive a sweep whose ladder is cancelled after ``rungs`` rungs, through
    the dispatch so the event is closed and its rows land."""
    from hyperloom.orchestrator.actions.executors.conc_sweep import ConcSweepExecutor

    state = _state(baseline_yaml)
    state.save(session_dir)
    ctx = _ctx(session_dir)
    ctx.task = _Task()
    ctx.task.params = {**_Task.params, "concs": [4, 16]}
    lifecycle_params = {"eligible": lifecycle, "reason": "supported", "port": 8888, "framework": "sglang"}
    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_cancelled_after(rungs)),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_materialize),
        patch(
            "hyperloom.orchestrator.actions.executors._server_lifecycle.resolve_lifecycle_params",
            return_value=lifecycle_params,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        asyncio.run(ConcSweepExecutor()(ctx))

    return _sweep_event(session_dir)["ext"]["arms"][ARM_OPTIMIZED]


@pytest.mark.parametrize("rungs, status", [(1, "degraded"), (0, "failed")])
def test_a_restart_ladder_cancelled_mid_way_reports_the_cancellation(
    session_dir: Path,
    baseline_yaml: Path,
    rungs: int,
    status: str,
):
    arm = _cancelled_arm(session_dir, baseline_yaml, rungs=rungs, lifecycle=False)

    assert arm["strategy"] == STRATEGY_SERVER_RESTART
    assert arm["status"] == status
    assert arm["failure"]["error_class"] == "CancelledError"
    assert arm["failure"]["stage"] == "arm_ladder"
    # Whatever it did measure is still the curve it produced.
    assert len(arm["points"]) == rungs


def test_a_reuse_ladder_cancelled_mid_way_does_not_report_the_rungs_it_reached_as_the_whole_ladder(
    session_dir: Path,
    baseline_yaml: Path,
):
    arm = _cancelled_arm(session_dir, baseline_yaml, rungs=1, lifecycle=True)

    assert arm["strategy"] == STRATEGY_SINGLE_SERVER
    assert arm["status"] == "degraded"
    assert arm["failure"]["error_class"] == "CancelledError"
    assert [point["conc"] for point in arm["points"]] == [16]


def test_a_dispatch_with_no_session_records_nothing_and_still_runs(session_dir: Path, baseline_yaml: Path):
    from hyperloom.orchestrator.actions.executors.conc_sweep import ConcSweepExecutor

    state = _state(baseline_yaml)
    state.save(session_dir)
    ctx = _ctx(session_dir)

    with (
        patch(
            "hyperloom.inference_optimizer.session.session_binding.session_is_bound",
            return_value=False,
        ),
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_all_succeed),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_materialize),
    ):
        result = asyncio.run(ConcSweepExecutor()(ctx))

    assert result["status"] == "succeeded"
    assert not [event for event in read_timeline_events(session_dir) if event.get("type") == "conc_sweep"]
