# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""P2-1 Objective + Coordinator.run() long-loop tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.action_executors import report_executor
from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
    effective_closing_grace_sec,
)
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.orchestrator.objective import (
    ObjectiveError,
    TargetBaselineObjective,
    TargetGainObjective,
    TargetTputObjective,
    TimeOnlyObjective,
    build_objective,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import target_baseline_json


# fixtures
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE,
                  payload={"topic": "heartbeat", "body_md": "ok"})


def _backends_silent() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }


# TargetGainObjective
def test_target_gain_basic_progress():
    obj = TargetGainObjective(target_gain_pct=10.0)
    s = SharedState(baseline_tput=1000.0, cumulative_gain=0.0)
    assert obj.kind() == "gain_pct"
    assert obj.progress(s) == 0.0
    assert obj.remaining_gap(s) == 10.0
    assert not obj.reached(s)

    s.cumulative_gain = 5.0
    assert obj.progress(s) == 0.5
    assert obj.remaining_gap(s) == 5.0
    assert not obj.reached(s)

    s.cumulative_gain = 12.0
    assert obj.progress(s) == 1.0
    assert obj.remaining_gap(s) == 0.0
    assert obj.reached(s)


def test_target_gain_zero_or_negative_rejected():
    with pytest.raises(ObjectiveError, match="must be > 0"):
        TargetGainObjective(target_gain_pct=0)
    with pytest.raises(ObjectiveError, match="must be > 0"):
        TargetGainObjective(target_gain_pct=-3)


def test_target_gain_pressure_zero_before_baseline():
    obj = TargetGainObjective(target_gain_pct=10.0)
    s = SharedState(baseline_tput=0.0, cumulative_gain=0.0)
    assert obj.pressure_input(s) == 0.0


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


def test_target_tput_zero_rejected():
    with pytest.raises(ObjectiveError, match="must be > 0"):
        TargetTputObjective(target_tput_per_gpu=0)


# TargetBaselineObjective
def test_target_baseline_loads_ref_tput(tmp_path):
    workspace = tmp_path / "ref-baseline"
    workspace.mkdir()
    (workspace / "benchmark_report.json").write_text(json.dumps({
        "throughput": {"output_throughput": 1234.5},
    }))
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


# TimeOnlyObjective
def test_time_only_never_reached():
    obj = TimeOnlyObjective()
    s = SharedState(baseline_tput=999.0, current_best={"tput": 9999.0},
                    cumulative_gain=99.0)
    assert obj.kind() == "time_only"
    assert obj.progress(s) == 0.0
    assert obj.remaining_gap(s) == float("inf")
    assert not obj.reached(s)


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
        build_objective({
            "MAX_HOURS": "1", "TARGET_GAIN_PCT": "5", "TARGET_TPUT_PER_GPU": "100",
        })


def test_build_objective_rejects_missing_max_hours():
    with pytest.raises(ObjectiveError, match="MAX_HOURS"):
        build_objective({"TARGET_GAIN_PCT": "5"})


def test_build_objective_rejects_zero_max_hours():
    with pytest.raises(ObjectiveError, match="must be > 0"):
        build_objective({"MAX_HOURS": "0"})


def test_build_objective_treats_empty_target_as_unset():
    obj = build_objective({
        "MAX_HOURS": "1",
        "TARGET_GAIN_PCT": "",
        "TARGET_TPUT_PER_GPU": None,
        "TARGET_DIR": "",
    })
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
        c.shared_state.cumulative_gain = 50.0
        c.shared_state.save(session_dir)
        reason = await c.run(
            objective=TargetGainObjective(target_gain_pct=10.0),
            max_ticks=10,
        )
        assert reason == "target_reached"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_defers_target_reached_for_kernel_work(session_dir, monkeypatch):
    """Explore-only target must not stop before KERNEL / kernel_opt."""
    monkeypatch.setenv("HYPERLOOM_AUTO_INTEGRATE", "0")
    _write_marker_target_baseline(session_dir)
    c = Coordinator(session_dir, backends=_backends_silent())
    c.shared_state.baseline_tput = 1000.0
    c.shared_state.cumulative_gain = 50.0
    c.shared_state.phase = "EXPLORE"
    c.shared_state.last_profile_trace = "/tmp/profile.trace"
    c.shared_state.save(session_dir)
    try:
        reason = await c.run(
            objective=TargetGainObjective(target_gain_pct=10.0),
            max_ticks=5,
            tick_interval_sec=0,
        )
        assert reason == "max_ticks"
        assert c.shared_state.stop_reason == "max_ticks"
        assert (c.shared_state.phase or "").upper() == "KERNEL"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_target_reached_finalizes_journal(session_dir, tmp_path, monkeypatch):
    """Early target_reached must populate optimization_journal totals."""
    monkeypatch.setenv("HYPERLOOM_AUTO_INTEGRATE", "0")
    c = Coordinator(session_dir, backends=_backends_silent())
    c.shared_state.baseline_tput = 1000.0
    c.shared_state.cumulative_gain = 50.0
    c.shared_state.current_best = {"tput": 1500.0, "action": "explore"}
    c.shared_state.kernel_enabled = False
    c.role_registry = {
        k: v for k, v in c.role_registry.items() if k != "kernel"
    }
    c.shared_state.save(session_dir)
    try:
        reason = await c.run(
            objective=TargetGainObjective(target_gain_pct=10.0),
            max_ticks=10,
            tick_interval_sec=0,
        )
        assert reason == "target_reached"
        journal_path = session_dir / "reports" / "optimization_journal.json"
        assert journal_path.is_file()
        data = json.loads(journal_path.read_text(encoding="utf-8"))
        assert data.get("final_throughput") == pytest.approx(1500.0)
        assert data.get("total_gain_pct") == pytest.approx(50.0)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_stops_on_time_exhausted(session_dir):
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        reason = await c.run(max_minutes=0.001, max_ticks=1000)
        assert reason == "time_exhausted"
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
        json.dumps({
            "status": "no_target",
            "reason": "no_target_gpu_configured",
            "row_count": 0,
        }),
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
        "kernel":        MockBackend(ScriptedPlan(turns=[], default_intent=_heartbeat()), name="k"),
        "critic":        MockBackend(ScriptedPlan(turns=[], default_intent=_heartbeat()), name="c"),
        "robustness":    MockBackend(ScriptedPlan(turns=[], default_intent=_heartbeat()), name="r"),
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

    c._enter_closing_phase = _enter_and_record  # type: ignore[method-assign]
    try:
        await c.run(
            max_minutes=0.0001,
            max_ticks=30,
            closing_grace_sec=5.0,
            tick_interval_sec=0.0,
        )
        assert spy.calls >= 1
        assert calls_at_closing, "expected closing phase to be entered"
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
        c.shared_state.crash_count = 30
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

    monkeypatch.setattr(c, "_pump_dispatcher_once", flaky_dispatcher_once)
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

    monkeypatch.setattr(c, "_pump_dispatcher_once", broken_dispatcher_once)
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

    monkeypatch.setattr(c, "_pump_dispatcher_once", broken_dispatcher_once)
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
        c.shared_state.crash_count = 4
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
