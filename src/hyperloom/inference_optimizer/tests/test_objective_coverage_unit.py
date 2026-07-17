# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Supplementary coverage for Objective describe + build factory."""

from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.state.objective import (
    ObjectiveError,
    TargetBaselineObjective,
    TargetGainObjective,
    TargetTputObjective,
    TimeOnlyObjective,
    build_objective,
)


def test_target_gain_describe():
    obj = TargetGainObjective(target_gain_pct=10.0)
    assert "target_gain_pct=10" in obj.describe()


def test_target_tput_describe():
    obj = TargetTputObjective(target_tput_per_gpu=1000.0)
    assert "target_tput_per_gpu=1000" in obj.describe()


def test_target_baseline_describe(tmp_path):
    ws = tmp_path / "ref"
    ws.mkdir()
    (ws / "benchmark_report.json").write_text(
        json.dumps({"throughput": {"output_throughput": 1000.0}}),
        encoding="utf-8",
    )
    obj = TargetBaselineObjective(baseline_dir=str(ws))
    assert "ref_tput=1000.0" in obj.describe()


def test_target_baseline_invalid_throughput(tmp_path):
    ws = tmp_path / "ref"
    ws.mkdir()
    (ws / "benchmark_report.json").write_text(
        json.dumps({"throughput": {"output_throughput": 0}}),
        encoding="utf-8",
    )
    with pytest.raises(ObjectiveError, match="invalid output_throughput"):
        TargetBaselineObjective(baseline_dir=str(ws))


def test_time_only_describe():
    obj = TimeOnlyObjective()
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
