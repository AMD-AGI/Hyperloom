# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Supplementary coverage for Objective pressure/describe + build factory."""

from __future__ import annotations

import json

import pytest

from inference_optimizer.orchestrator.objective import (
    ObjectiveError,
    TargetBaselineObjective,
    TargetGainObjective,
    TargetTputObjective,
    TimeOnlyObjective,
    build_objective,
)
from inference_optimizer.orchestrator.shared_state import SharedState


def test_target_gain_pressure_and_describe():
    obj = TargetGainObjective(target_gain_pct=10.0)
    s = SharedState(baseline_tput=1000.0, cumulative_gain=5.0)
    assert obj.pressure_input(s) == 0.5
    assert "target_gain_pct=10" in obj.describe()


def test_target_tput_pressure_describe_and_gap():
    obj = TargetTputObjective(target_tput_per_gpu=1000.0)
    s = SharedState(baseline_tput=400.0, current_best={"tput": 600.0})
    assert obj.remaining_gap(s) == 400.0
    assert obj.pressure_input(s) == pytest.approx(0.6)
    assert "target_tput_per_gpu=1000" in obj.describe()


def test_target_baseline_cur_fallback_and_pressure(tmp_path):
    ws = tmp_path / "ref"
    ws.mkdir()
    (ws / "benchmark_report.json").write_text(
        json.dumps({"throughput": {"output_throughput": 1000.0}}),
        encoding="utf-8",
    )
    obj = TargetBaselineObjective(baseline_dir=str(ws))
    # no current_best -> fall back to baseline_tput
    s = SharedState(baseline_tput=500.0, current_best={})
    assert obj.remaining_gap(s) == 500.0
    assert obj.pressure_input(s) == pytest.approx(0.5)


def test_target_baseline_invalid_throughput(tmp_path):
    ws = tmp_path / "ref"
    ws.mkdir()
    (ws / "benchmark_report.json").write_text(
        json.dumps({"throughput": {"output_throughput": 0}}),
        encoding="utf-8",
    )
    with pytest.raises(ObjectiveError, match="invalid output_throughput"):
        TargetBaselineObjective(baseline_dir=str(ws))


def test_time_only_pressure_and_describe():
    obj = TimeOnlyObjective()
    s = SharedState(baseline_tput=1.0)
    assert obj.pressure_input(s) == 0.0
    assert "no target" in obj.describe()


def test_build_objective_target_dir(tmp_path):
    ws = tmp_path / "ref"
    ws.mkdir()
    (ws / "benchmark_report.json").write_text(
        json.dumps({"throughput": {"output_throughput": 1234.0}}),
        encoding="utf-8",
    )
    obj = build_objective({"MAX_HOURS": "1", "TARGET_DIR": str(ws)})
    assert isinstance(obj, TargetBaselineObjective)


def test_build_objective_max_hours_not_float():
    with pytest.raises(ObjectiveError, match="not a float"):
        build_objective({"MAX_HOURS": "abc"})
