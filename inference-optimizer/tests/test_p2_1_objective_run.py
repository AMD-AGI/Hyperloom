"""P2-1 Objective + Conductor.run() long-loop tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    MockBackend,
    MockCriticBackend,
    MockKernelBackend,
    MockRobustnessBackend,
    MockTurn,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.conductor import Conductor
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.objective import (
    Objective,
    ObjectiveError,
    TargetBaselineObjective,
    TargetGainObjective,
    TargetTputObjective,
    TimeOnlyObjective,
    build_objective,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SESSION_ROOT", str(tmp_path))
    return make_session_dir("p2-1-test")


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


# ===========================================================================
# TargetGainObjective
# ===========================================================================
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


# ===========================================================================
# TargetTputObjective
# ===========================================================================
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


# ===========================================================================
# TargetBaselineObjective
# ===========================================================================
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


# ===========================================================================
# TimeOnlyObjective
# ===========================================================================
def test_time_only_never_reached():
    obj = TimeOnlyObjective()
    s = SharedState(baseline_tput=999.0, current_best={"tput": 9999.0},
                    cumulative_gain=99.0)
    assert obj.kind() == "time_only"
    assert obj.progress(s) == 0.0
    assert obj.remaining_gap(s) == float("inf")
    assert not obj.reached(s)


# ===========================================================================
# build_objective factory
# ===========================================================================
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


# ===========================================================================
# Conductor.run() long loop
# ===========================================================================
@pytest.mark.asyncio
async def test_run_stops_on_max_ticks(session_dir):
    c = Conductor(session_dir, backends=_backends_silent())
    try:
        reason = await c.run(max_ticks=3)
        assert reason == "max_ticks"
        assert c.shared_state.stop_reason == "max_ticks"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_stops_on_objective_reached(session_dir):
    c = Conductor(session_dir, backends=_backends_silent())
    try:
        # Pre-load state so objective is already reached on tick 1.
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
async def test_run_stops_on_time_exhausted(session_dir):
    c = Conductor(session_dir, backends=_backends_silent())
    try:
        # max_minutes = 0.001 = 60ms; each tick is fast, but the budget is
        # enforced AFTER each tick so we always run >=1 tick.
        reason = await c.run(max_minutes=0.001, max_ticks=1000)
        assert reason == "time_exhausted"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_stops_on_signal_via_stop_event(session_dir):
    c = Conductor(session_dir, backends=_backends_silent())
    try:
        c._stop.set()  # simulate SIGINT
        reason = await c.run(max_ticks=10)
        assert reason == "signal"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_stops_on_emergency_crash_count(session_dir):
    c = Conductor(session_dir, backends=_backends_silent())
    try:
        c.shared_state.crash_count = 5
        reason = await c.run(max_ticks=10)
        assert reason == "emergency"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_stops_on_custom_stop_when(session_dir):
    c = Conductor(session_dir, backends=_backends_silent())
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
    c = Conductor(session_dir, backends=_backends_silent())
    try:
        await c.run(max_ticks=2)
    finally:
        await c.stop()
    on_disk = json.loads((session_dir / "state.json").read_text())
    assert on_disk["stop_reason"] == "max_ticks"


@pytest.mark.asyncio
async def test_run_persists_max_minutes_to_state(session_dir):
    c = Conductor(session_dir, backends=_backends_silent())
    try:
        await c.run(max_ticks=1, max_minutes=42.0)
    finally:
        await c.stop()
    assert c.shared_state.max_minutes == 42


@pytest.mark.asyncio
async def test_run_async_stop_when_callback(session_dir):
    """stop_when can be an async function too."""
    c = Conductor(session_dir, backends=_backends_silent())
    try:
        async def acb(_c):
            return True
        reason = await c.run(stop_when=acb, max_ticks=10)
        assert reason == "custom"
    finally:
        await c.stop()
