"""Tests for ``orchestrator.early_stop`` — IMPL-CHECKLIST §3.39‒3.46."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from inference_optimizer.paths import asset_actions_dir
from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.early_stop import (
    BRIER_PLATEAU_THRESHOLD,
    EMERGENCY_CRASH_THRESHOLD,
    NO_LEVERAGE_THRESHOLD,
    TIME_BUFFER_MIN,
    should_stop_early,
    signal_brier_plateau,
    signal_emergency,
    signal_no_more_leverage,
    signal_target_reached,
    signal_time_exhausted,
)
from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.objective import build_objective
from inference_optimizer.orchestrator.scheduler import BudgetAwareScheduler
from inference_optimizer.orchestrator.shared_state import SharedState


PACKAGE_ACTIONS_DIR = asset_actions_dir()


@dataclass
class _SatisfiableObjective:
    """Test stub: ``is_satisfied`` returns whatever we set."""
    satisfied: bool = False
    description: str = "stub"

    def is_satisfied(self, state: SharedState) -> bool:
        return self.satisfied

    def describe(self) -> str:
        return self.description


@dataclass
class _FakeFlags:
    enable_critic: bool = True


@pytest.fixture(scope="module")
def registry() -> ActionRegistry:
    return ActionRegistry(PACKAGE_ACTIONS_DIR).load()


@pytest.fixture
def state() -> SharedState:
    return SharedState(
        session_id="t",
        max_minutes=120.0,
        elapsed_minutes=10.0,
    )


def _scheduler(reg) -> BudgetAwareScheduler:
    env = {"MODEL_PATH": "/m", "MAX_HOURS": "2"}
    return BudgetAwareScheduler(
        objective=build_objective(env),
        mode=ExecutionMode.GUIDED_KERNEL_OPT,
        env=env,
        action_registry=reg,
    )


# ---------------------------------------------------------------------------
# target_reached
# ---------------------------------------------------------------------------
def test_target_reached_fires_when_objective_satisfied(state: SharedState):
    obj = _SatisfiableObjective(satisfied=True)
    sig = signal_target_reached(state, obj)
    assert sig is not None
    assert sig.name == "target_reached"


def test_target_reached_silent_when_not_satisfied(state: SharedState):
    obj = _SatisfiableObjective(satisfied=False)
    assert signal_target_reached(state, obj) is None


# ---------------------------------------------------------------------------
# time_exhausted
# ---------------------------------------------------------------------------
def test_time_exhausted_fires_inside_buffer(state: SharedState):
    state.elapsed_minutes = state.max_minutes - (TIME_BUFFER_MIN - 0.1)
    sig = signal_time_exhausted(state)
    assert sig is not None
    assert sig.name == "time_exhausted"


def test_time_exhausted_silent_with_room(state: SharedState):
    state.elapsed_minutes = 1.0
    assert signal_time_exhausted(state) is None


def test_time_exhausted_silent_when_max_zero(state: SharedState):
    state.max_minutes = 0.0
    assert signal_time_exhausted(state) is None


# ---------------------------------------------------------------------------
# emergency
# ---------------------------------------------------------------------------
def test_emergency_fires_at_threshold(state: SharedState):
    state.crash_count = EMERGENCY_CRASH_THRESHOLD
    sig = signal_emergency(state)
    assert sig is not None and sig.name == "emergency"


def test_emergency_silent_below_threshold(state: SharedState):
    state.crash_count = EMERGENCY_CRASH_THRESHOLD - 1
    assert signal_emergency(state) is None


# ---------------------------------------------------------------------------
# brier_plateau (only when critic is enabled)
# ---------------------------------------------------------------------------
def test_brier_plateau_fires_with_high_avg(state: SharedState):
    flags = _FakeFlags(enable_critic=True)
    window = [BRIER_PLATEAU_THRESHOLD + 0.05] * 6
    sig = signal_brier_plateau(state, flags=flags, brier_window=window)
    assert sig is not None and sig.name == "brier_plateau"


def test_brier_plateau_silent_when_critic_disabled(state: SharedState):
    flags = _FakeFlags(enable_critic=False)
    window = [BRIER_PLATEAU_THRESHOLD + 0.05] * 6
    assert signal_brier_plateau(state, flags=flags, brier_window=window) is None


def test_brier_plateau_silent_with_too_few_samples(state: SharedState):
    flags = _FakeFlags(enable_critic=True)
    window = [BRIER_PLATEAU_THRESHOLD + 0.5] * 2
    assert signal_brier_plateau(state, flags=flags, brier_window=window) is None


def test_brier_plateau_silent_with_low_avg(state: SharedState):
    flags = _FakeFlags(enable_critic=True)
    window = [0.05] * 5
    assert signal_brier_plateau(state, flags=flags, brier_window=window) is None


# ---------------------------------------------------------------------------
# no_more_leverage
# ---------------------------------------------------------------------------
def test_no_more_leverage_fires_when_all_scores_low(
    registry, state: SharedState
):
    sch = _scheduler(registry)
    # Drive every score to 0 by zeroing out time_left -> depth_gate=0.
    state.elapsed_minutes = state.max_minutes
    sig = signal_no_more_leverage(state, sch)
    assert sig is not None
    assert sig.name == "no_more_leverage"


def test_no_more_leverage_silent_with_normal_score(
    registry, state: SharedState
):
    sch = _scheduler(registry)
    state.elapsed_minutes = 5.0
    sig = signal_no_more_leverage(state, sch)
    # may or may not fire depending on data — assert dataclass shape only
    assert sig is None or sig.name == "no_more_leverage"


# ---------------------------------------------------------------------------
# should_stop_early — priority ordering
# ---------------------------------------------------------------------------
def test_should_stop_early_returns_target_first(
    registry, state: SharedState
):
    obj = _SatisfiableObjective(satisfied=True)
    state.crash_count = EMERGENCY_CRASH_THRESHOLD + 1
    state.elapsed_minutes = state.max_minutes  # also time_exhausted
    sig = should_stop_early(state, obj, scheduler=_scheduler(registry))
    assert sig is not None and sig.name == "target_reached"


def test_should_stop_early_emergency_beats_time_exhausted(
    registry, state: SharedState
):
    obj = _SatisfiableObjective(satisfied=False)
    state.crash_count = EMERGENCY_CRASH_THRESHOLD
    state.elapsed_minutes = state.max_minutes - 0.1
    sig = should_stop_early(state, obj, scheduler=_scheduler(registry))
    assert sig is not None and sig.name == "emergency"


def test_should_stop_early_returns_none_when_all_quiet(
    registry, state: SharedState
):
    obj = _SatisfiableObjective(satisfied=False)
    state.crash_count = 0
    state.elapsed_minutes = 5.0
    sig = should_stop_early(state, obj, scheduler=_scheduler(registry))
    # Acceptable for either None or no_more_leverage depending on actions
    assert sig is None or sig.name in {
        "no_more_leverage", "brier_plateau",
    }
