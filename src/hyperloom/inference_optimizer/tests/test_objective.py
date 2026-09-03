# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Objective + Coordinator.run() long-loop tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.roles import (
    MockBackend,
    ScriptedPlan,
)
from hyperloom.orchestrator.actions.executors import report_executor
from hyperloom.orchestrator.loop.coordinator import (
    Coordinator,
    effective_closing_grace_sec,
)
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.state.objective import (
    ObjectiveError,
    TargetBaselineObjective,
    TargetGainObjective,
    TargetTputObjective,
    TimeOnlyObjective,
    build_objective,
)
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.inference_optimizer.session.session_paths import target_baseline_json


# fixtures
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _backends_silent() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "critic": MockBackend(silent, name="c"),
        "robustness": MockBackend(silent, name="r"),
    }


# TargetGainObjective
def test_target_gain_basic_progress():
    obj = TargetGainObjective(target_gain_pct=10.0)
    s = SharedState(baseline_tput=1000.0, cumulative_gain_validated=0.0)
    assert obj.kind() == "gain_pct"
    assert obj.progress(s) == 0.0
    assert not obj.reached(s)

    s.cumulative_gain_validated = 5.0
    assert obj.progress(s) == 0.5
    assert not obj.reached(s)

    s.cumulative_gain_validated = 12.0
    assert obj.progress(s) == 1.0
    assert obj.reached(s)


def test_target_gain_gap_pct_counts_down_to_zero():
    obj = TargetGainObjective(target_gain_pct=15.0)
    s = SharedState(baseline_tput=1000.0, cumulative_gain_validated=0.0)
    assert obj.gap_pct(s) == pytest.approx(15.0)

    s.cumulative_gain_validated = 9.89
    assert obj.gap_pct(s) == pytest.approx(5.11)

    s.cumulative_gain_validated = 20.0
    assert obj.gap_pct(s) == 0.0


def test_target_gain_zero_or_negative_rejected():
    with pytest.raises(ObjectiveError, match="must be > 0"):
        TargetGainObjective(target_gain_pct=0)
    with pytest.raises(ObjectiveError, match="must be > 0"):
        TargetGainObjective(target_gain_pct=-3)


# TargetTputObjective
def test_target_tput_uses_current_best():
    obj = TargetTputObjective(target_tput_per_gpu=900.0)
    s = SharedState(baseline_tput=800.0, current_best={"tput": 850.0})
    assert obj.kind() == "tput"
    assert obj.progress(s) == pytest.approx(850.0 / 900.0)
    assert not obj.reached(s)

    s.current_best = {"tput": 950.0}
    assert obj.progress(s) == 1.0
    assert obj.reached(s)


def test_target_tput_falls_back_to_baseline_when_no_current_best():
    obj = TargetTputObjective(target_tput_per_gpu=900.0)
    s = SharedState(baseline_tput=750.0, current_best={})
    assert obj.progress(s) == pytest.approx(750.0 / 900.0)


def test_target_tput_gap_pct_is_relative_to_current():
    obj = TargetTputObjective(target_tput_per_gpu=1000.0)
    s = SharedState(baseline_tput=800.0, current_best={"tput": 800.0})
    assert obj.gap_pct(s) == pytest.approx(25.0)

    s.current_best = {"tput": 1000.0}
    assert obj.gap_pct(s) == 0.0

    # No measurement yet => no distance to report.
    assert obj.gap_pct(SharedState()) == 0.0


def test_target_tput_zero_rejected():
    with pytest.raises(ObjectiveError, match="must be > 0"):
        TargetTputObjective(target_tput_per_gpu=0)


# TargetBaselineObjective
def test_target_baseline_loads_ref_tput(tmp_path):
    workspace = tmp_path / "ref-baseline"
    workspace.mkdir()
    (workspace / "benchmark_report.json").write_text(
        json.dumps(
            {
                "throughput": {"output_throughput": 1234.5},
            }
        )
    )
    obj = TargetBaselineObjective(baseline_dir=str(workspace))
    s = SharedState(baseline_tput=1000.0, current_best={"tput": 1000.0})
    assert obj.kind() == "baseline"
    assert "1234" in obj.describe()
    assert obj.progress(s) == pytest.approx(1000.0 / 1234.5)
    assert not obj.reached(s)

    s.current_best = {"tput": 1300.0}
    assert obj.reached(s)


def test_target_baseline_missing_dir_rejected(tmp_path):
    with pytest.raises(ObjectiveError, match="not found"):
        TargetBaselineObjective(baseline_dir=str(tmp_path / "nope"))


def test_target_baseline_missing_report_rejected(tmp_path):
    workspace = tmp_path / "empty"
    workspace.mkdir()
    with pytest.raises(ObjectiveError, match="no benchmark_report"):
        TargetBaselineObjective(baseline_dir=str(workspace))


def _write_report(root, rel, tput):
    path = root / rel / "benchmark_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"throughput": {"output_throughput": tput}}))
    return path


def test_target_baseline_falls_back_to_a_warmup_only_reference(tmp_path):
    """A budget-dropped measure round leaves the warmup as the only report.

    Raising there refuses to start a session against a reference the earlier
    run really did produce.
    """
    workspace = tmp_path / "warmup-only"
    _write_report(workspace, "warmup_round/bench", 800.0)
    assert TargetBaselineObjective(baseline_dir=str(workspace))._ref_tput == pytest.approx(800.0)


def test_target_baseline_prefers_the_measured_round_over_the_warmup(tmp_path):
    workspace = tmp_path / "both"
    _write_report(workspace, "warmup_round/bench", 800.0)
    _write_report(workspace, "measured/bench", 1200.0)
    assert TargetBaselineObjective(baseline_dir=str(workspace))._ref_tput == pytest.approx(1200.0)


# TimeOnlyObjective
def test_time_only_never_reached():
    obj = TimeOnlyObjective()
    s = SharedState(baseline_tput=999.0, current_best={"tput": 9999.0}, cumulative_gain_validated=99.0)
    assert obj.kind() == "time_only"
    assert obj.progress(s) == 0.0
    assert not obj.reached(s)
    assert obj.gap_pct(s) == 0.0


# build_objective factory
def test_build_objective_target_gain():
    obj = build_objective({"MAX_HOURS": "2", "TARGET_GAIN_PCT": "10"})
    assert isinstance(obj, TargetGainObjective)
    assert obj.target_gain_pct == 10.0


def test_build_objective_target_tput():
    obj = build_objective({"MAX_HOURS": "1", "TARGET_TPUT_PER_GPU": "1500"})
    assert isinstance(obj, TargetTputObjective)


def test_build_objective_time_only_when_no_target():
    obj = build_objective({"MAX_HOURS": "0.5"})
    assert isinstance(obj, TimeOnlyObjective)


def test_build_objective_rejects_multiple_targets():
    with pytest.raises(ObjectiveError, match="at most one TARGET_"):
        build_objective(
            {
                "MAX_HOURS": "1",
                "TARGET_GAIN_PCT": "5",
                "TARGET_TPUT_PER_GPU": "100",
            }
        )


def test_build_objective_rejects_missing_max_hours():
    with pytest.raises(ObjectiveError, match="MAX_HOURS"):
        build_objective({"TARGET_GAIN_PCT": "5"})


def test_build_objective_rejects_zero_max_hours():
    with pytest.raises(ObjectiveError, match="must be > 0"):
        build_objective({"MAX_HOURS": "0"})


def test_build_objective_treats_empty_target_as_unset():
    obj = build_objective(
        {
            "MAX_HOURS": "1",
            "TARGET_GAIN_PCT": "",
            "TARGET_TPUT_PER_GPU": None,
            "TARGET_DIR": "",
        }
    )
    assert isinstance(obj, TimeOnlyObjective)


# Coordinator.run() long loop
@pytest.mark.asyncio
async def test_run_stops_on_max_ticks(session_dir):
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        reason = await c.run(max_ticks=3)
        assert reason == "max_ticks"
        assert c.shared_state.stop_reason == "max_ticks"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_stops_on_objective_reached(session_dir):
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        c.shared_state.baseline_tput = 1000.0
        c.shared_state.cumulative_gain_validated = 50.0
        c.shared_state.save(session_dir)
        reason = await c.run(
            objective=TargetGainObjective(target_gain_pct=10.0),
            max_ticks=10,
        )
        assert reason == "target_reached"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_does_not_stop_on_a_gain_no_rebench_confirmed(session_dir):
    """A current_best ahead of the last stack rebench is not evidence of the target."""
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        c.shared_state.baseline_tput = 1000.0
        c.shared_state.current_best = {"action": "explore", "tput": 1500.0}
        c.shared_state.cumulative_gain_validated = 2.0
        c.shared_state.save(session_dir)
        reason = await c.run(
            objective=TargetGainObjective(target_gain_pct=10.0),
            max_ticks=3,
        )
        assert reason == "max_ticks"
    finally:
        await c.stop()


def test_target_tput_reads_the_shared_grading_anchor():
    """A current_best carrying only ``output_throughput`` still counts as measured."""
    obj = TargetTputObjective(target_tput_per_gpu=900.0)
    s = SharedState(baseline_tput=750.0, current_best={"output_throughput": 950.0})
    assert obj.reached(s)


@pytest.mark.asyncio
async def test_run_stops_on_time_exhausted(session_dir):
    """A short wall-clock budget stops the run wherever it expires."""
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        reason = await c.run(max_minutes=0.001, max_ticks=1000)
        # The 60 ms budget can expire before or after PRELUDE depending on the
        # runner's speed; both reasons prove that the session bound stopped it.
        assert reason in {"time_exhausted", "time_exhausted_during_prelude"}
    finally:
        await c.stop()


def test_closing_grace_default_scales_with_max_hours():
    assert effective_closing_grace_sec(120.0, None) == pytest.approx(120.0)
    assert effective_closing_grace_sec(0.6, None) == pytest.approx(0.72)
    assert effective_closing_grace_sec(120.0, 30.0) == 30.0
    assert effective_closing_grace_sec(120.0, 0.0) == 0.0


def _write_marker_target_baseline(session_dir: Path) -> None:
    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "no_target",
                "reason": "no_target_gpu_configured",
                "row_count": 0,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_run_closing_phase_writes_report_on_time_exhausted(session_dir):
    _write_marker_target_baseline(session_dir)
    c = Coordinator(session_dir, backends=_backends_silent())
    c.sub.register_executor("report", report_executor)
    c.shared_state.baseline_tput = 100.0
    c.shared_state.save(session_dir)
    try:
        reason = await c.run(
            max_minutes=0.0001,
            max_ticks=50,
            closing_grace_sec=30.0,
            tick_interval_sec=0.0,
        )
        assert reason == "time_exhausted"
        assert (session_dir / "reports" / "final.md").exists()
        assert (session_dir / "reports" / "final.json").exists()
        state = json.loads((session_dir / "state.json").read_text())
        assert state["closing_phase"] is False
        assert state["closing_report_task_id"]
        assert state["closing_started_unix"] > 0
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_closing_phase_skips_reactor(session_dir):
    _write_marker_target_baseline(session_dir)

    class _SpyBackend(MockBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0

        async def run(self, prompt: str, **kwargs):
            self.calls += 1
            return await super().run(prompt, **kwargs)

    spy = _SpyBackend(ScriptedPlan(turns=[], default_intent=_heartbeat()), name="o")
    backends = {
        "orchestration": spy,
        "critic": MockBackend(ScriptedPlan(turns=[], default_intent=_heartbeat()), name="c"),
        "robustness": MockBackend(ScriptedPlan(turns=[], default_intent=_heartbeat()), name="r"),
    }
    c = Coordinator(session_dir, backends=backends)
    c.sub.register_executor("report", report_executor)
    c.shared_state.baseline_tput = 50.0
    c.shared_state.save(session_dir)
    calls_at_closing: list[int] = []
    real_enter = c._enter_closing_phase

    async def _enter_and_record(*, grace_sec: float) -> float:
        calls_at_closing.append(spy.calls)
        return await real_enter(grace_sec=grace_sec)

    c.phase_close._enter_closing_phase = _enter_and_record  # type: ignore[method-assign]
    try:
        await c.run(
            max_minutes=0.0001,
            max_ticks=30,
            closing_grace_sec=5.0,
            tick_interval_sec=0.0,
        )
        assert calls_at_closing, "expected closing phase to be entered"
        # A spent bound cancels phase-enter and skips reactors on the tick
        # that trips CLOSE. CLOSE itself must still not add LLM turns.
        assert spy.calls == calls_at_closing[0]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_stops_on_signal_via_stop_event(session_dir):
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        c._stop.set()
        reason = await c.run(max_ticks=10)
        assert reason == "signal"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_stops_on_emergency_crash_count(session_dir):
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        # Record crashes so the trailing-window rate sees them.
        for _ in range(30):
            c.shared_state.increment_crash_count()
        reason = await c.run(max_ticks=10)
        assert reason == "emergency"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_does_not_emergency_below_default_threshold(session_dir):
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        c.shared_state.crash_count = 5
        reason = await c.run(max_ticks=2)
        assert reason == "max_ticks"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_records_tick_exception_and_continues(session_dir, monkeypatch):
    c = Coordinator(session_dir, backends=_backends_silent())
    calls = {"n": 0}

    async def flaky_dispatcher_once() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("dispatcher boom")

    monkeypatch.setattr(c.dispatcher, "_pump_dispatcher_once", flaky_dispatcher_once)
    try:
        reason = await c.run(max_ticks=2)
        assert reason == "max_ticks"
        assert calls["n"] == 2
        assert c.shared_state.crash_count == 1
        assert c.shared_state.last_tick_exception["stage"] == "tick_body"
        assert c.shared_state.last_tick_exception["type"] == "RuntimeError"
        assert "dispatcher boom" in c.shared_state.last_tick_exception["message"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_repeated_tick_exceptions_stop_as_emergency(
    session_dir,
    monkeypatch,
):
    c = Coordinator(session_dir, backends=_backends_silent())

    async def broken_dispatcher_once() -> None:
        raise RuntimeError("persistent dispatcher boom")

    monkeypatch.setattr(c.dispatcher, "_pump_dispatcher_once", broken_dispatcher_once)
    try:
        reason = await c.run(max_ticks=10, crash_emergency_threshold=2)
        assert reason == "emergency"
        assert c.shared_state.crash_count == 2
        assert c.shared_state.stop_reason == "emergency"
        assert c.shared_state.last_tick_exception["type"] == "RuntimeError"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_finally_labels_residual_escape_as_coordinator_exception(
    session_dir,
    monkeypatch,
):
    c = Coordinator(session_dir, backends=_backends_silent())

    async def broken_dispatcher_once() -> None:
        raise RuntimeError("recorded tick failure")

    def broken_stop_when(_c) -> bool:
        raise ValueError("stop callback failed")

    monkeypatch.setattr(c.dispatcher, "_pump_dispatcher_once", broken_dispatcher_once)
    try:
        with pytest.raises(ValueError, match="stop callback failed"):
            await c.run(stop_when=broken_stop_when, max_ticks=10)
        persisted = SharedState.load_or_init(session_dir)
        assert persisted.stop_reason == "coordinator_exception"
        assert persisted.last_tick_exception["type"] == "RuntimeError"
        assert "recorded tick failure" in persisted.last_tick_exception["message"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_emergency_threshold_can_be_lowered_per_call(session_dir):
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        for _ in range(4):
            c.shared_state.increment_crash_count()
        reason = await c.run(max_ticks=10, crash_emergency_threshold=3)
        assert reason == "emergency"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_stops_on_custom_stop_when(session_dir):
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        ticks = {"n": 0}

        def custom_stop(_c) -> bool:
            ticks["n"] += 1
            return ticks["n"] >= 2

        reason = await c.run(stop_when=custom_stop, max_ticks=20)
        assert reason == "custom"
        assert ticks["n"] == 2
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_persists_stop_reason_to_state_json(session_dir):
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        await c.run(max_ticks=2)
    finally:
        await c.stop()
    on_disk = json.loads((session_dir / "state.json").read_text())
    assert on_disk["stop_reason"] == "max_ticks"


@pytest.mark.asyncio
async def test_run_persists_max_minutes_to_state(session_dir):
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        await c.run(max_ticks=1, max_minutes=42.0)
    finally:
        await c.stop()
    assert c.shared_state.max_minutes == 42


@pytest.mark.asyncio
async def test_run_async_stop_when_callback(session_dir):
    """stop_when can be an async function too."""
    c = Coordinator(session_dir, backends=_backends_silent())
    try:

        async def acb(_c):
            return True

        reason = await c.run(stop_when=acb, max_ticks=10)
        assert reason == "custom"
    finally:
        await c.stop()
