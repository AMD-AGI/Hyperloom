# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.measure.socket import RoundPlan, TestCall


def _call(tmp_path: Path) -> TestCall:
    return TestCall(
        points=(tmp_path / "p.yaml",),
        output_dir=tmp_path,
        timeout_sec=60,
        cwd=str(tmp_path),
        task_id="t",
    )


@pytest.mark.parametrize("warmup", [2, -1])
def test_a_round_plan_allows_at_most_one_warmup_round(warmup):
    with pytest.raises(ValueError, match="at most one"):
        RoundPlan(warmup_rounds=warmup)


def test_a_round_plan_measures_one_round():
    with pytest.raises(ValueError, match="measures one round"):
        RoundPlan(measure_rounds=2)


def test_a_call_defaults_to_a_single_point_test_with_no_accuracy(tmp_path):
    call = _call(tmp_path)

    assert (call.kind, call.prep, call.accuracy, call.rounds) == ("test", "grid", False, RoundPlan())
