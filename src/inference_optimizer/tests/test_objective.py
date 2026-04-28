"""Tests for orchestrator/objective.py — DESIGN §8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from inference_optimizer.orchestrator.objective import (
    TargetGainObjective,
    TargetTputObjective,
    TimeOnlyObjective,
    build_objective,
)


@dataclass
class FakeState:
    cumulative_gain: float = 0.0
    baseline_tput: Optional[float] = None
    current_tput: Optional[float] = None
    elapsed_minutes: float = 0.0
    max_minutes: float = 60.0


# ---------------------------------------------------------------------------
def test_target_gain_progress_and_pressure():
    obj = TargetGainObjective(target_gain_pct=20.0)

    s = FakeState(cumulative_gain=0.0)
    assert obj.progress(s) == 0.0
    assert obj.reached(s) is False
    assert obj.remaining_gap(s) == 20.0
    assert obj.pressure_input(s) == 1.0

    s = FakeState(cumulative_gain=10.0)
    assert obj.progress(s) == 0.5
    assert obj.pressure_input(s) == 0.5

    s = FakeState(cumulative_gain=25.0)
    assert obj.progress(s) == 1.0
    assert obj.reached(s) is True
    assert obj.remaining_gap(s) == 0.0
    assert obj.pressure_input(s) == 0.0


def test_target_tput_handles_missing_baseline():
    obj = TargetTputObjective(target_tput=700.0)
    assert obj.progress(FakeState()) == 0.0
    assert obj.reached(FakeState()) is False
    s = FakeState(baseline_tput=400.0, current_tput=550.0)
    # 150/300 = 0.5
    assert obj.progress(s) == pytest.approx(0.5)


def test_time_only_objective_never_reaches():
    obj = TimeOnlyObjective()
    assert obj.kind() == "time_only"
    assert obj.reached(FakeState(elapsed_minutes=120, max_minutes=60)) is False
    assert obj.pressure_input(FakeState()) == 0.0


# ---------------------------------------------------------------------------
def test_build_objective_factory_gain():
    env = {"MODEL_PATH": "/model", "MAX_HOURS": "4", "TARGET_GAIN_PCT": "30"}
    obj = build_objective(env)
    assert obj.kind() == "gain_pct"


def test_build_objective_factory_tput():
    env = {"MODEL_PATH": "/m", "MAX_HOURS": "4", "TARGET_TPUT_PER_GPU": "700"}
    obj = build_objective(env)
    assert obj.kind() == "tput"


def test_build_objective_factory_time_only():
    env = {"MODEL_PATH": "/m", "MAX_HOURS": "4"}
    obj = build_objective(env)
    assert obj.kind() == "time_only"


def test_build_objective_rejects_multiple_targets():
    env = {
        "MODEL_PATH": "/m",
        "MAX_HOURS": "4",
        "TARGET_GAIN_PCT": "30",
        "TARGET_TPUT_PER_GPU": "700",
    }
    with pytest.raises(ValueError, match="At most one"):
        build_objective(env)


def test_build_objective_rejects_missing_required():
    with pytest.raises(ValueError, match="MODEL_PATH"):
        build_objective({"MAX_HOURS": "4"})
    with pytest.raises(ValueError, match="MAX_HOURS"):
        build_objective({"MODEL_PATH": "/m"})
    with pytest.raises(ValueError):
        build_objective({"MODEL_PATH": "/m", "MAX_HOURS": "0"})
