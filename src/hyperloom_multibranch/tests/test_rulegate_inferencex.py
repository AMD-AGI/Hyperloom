# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import math

import pytest

from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom_multibranch.contracts.judgement import Verdict
from hyperloom_multibranch.rulegate.inferencex import (
    FLOOR_PCT,
    GAIN_BELOW_THRESHOLD,
    GAIN_NOT_CONFIRMED,
    DispersionThreshold,
    compare,
    confirm,
)

NORMAL = DispersionThreshold((1000.0, 1010.0, 990.0))
PARENT = {GRADED_OUTPUT: 1000.0}


@pytest.mark.parametrize(
    ("baselines", "spread", "bar"),
    [
        ((1000.0, 1010.0, 990.0), 1.0, 2.0),
        ((1000.0, 1002.0, 998.0), 0.2, FLOOR_PCT),
        ((1000.0, 1050.0, 950.0), 5.0, 10.0),
    ],
)
def test_the_bar_is_twice_the_baseline_spread_above_a_floor(baselines, spread, bar):
    threshold = DispersionThreshold(baselines)
    assert threshold.spread_pct == pytest.approx(spread)
    assert threshold.threshold_pct == pytest.approx(bar)


def test_spot_checked_baselines_refine_the_spread():
    assert DispersionThreshold((1000.0, 1020.0, 990.0)).threshold_pct == pytest.approx(3.045, abs=1e-3)
    assert NORMAL.with_baseline(1000.0).spread_pct == pytest.approx(math.sqrt(2 / 3))


@pytest.mark.parametrize("baselines", [(1000.0,), (1000.0, 0.0), (1000.0, math.nan)])
def test_the_spread_refuses_too_few_or_unusable_baselines(baselines):
    with pytest.raises(ValueError):
        DispersionThreshold(baselines)


@pytest.mark.parametrize(
    ("throughput", "accuracy_ok", "verdict", "refused_by"),
    [
        (1030.0, True, Verdict.NEEDS_CONFIRM, ()),
        (1030.0, False, Verdict.NEEDS_CONFIRM, ()),
        (1020.0, True, Verdict.FLAT, (GAIN_BELOW_THRESHOLD,)),
        (1010.0, False, Verdict.FLAT, (GAIN_BELOW_THRESHOLD,)),
        (970.0, True, Verdict.REGRESSED, (GAIN_BELOW_THRESHOLD,)),
    ],
)
def test_a_first_measurement_against_its_parent(throughput, accuracy_ok, verdict, refused_by):
    judgement = compare(throughput, accuracy_ok, PARENT, NORMAL, "r1")
    assert judgement.verdict is verdict
    assert judgement.refused_by == refused_by
    assert judgement.accuracy_ok is accuracy_ok
    assert judgement.threshold_pct == pytest.approx(2.0)
    assert judgement.recorded_value == {GRADED_OUTPUT: throughput}
    assert judgement.request_ids == ("r1",)


@pytest.mark.parametrize(
    ("first_ok", "retest", "retest_ok", "verdict", "refused_by"),
    [
        (True, 1025.0, True, Verdict.IMPROVED, ()),
        (False, 1025.0, False, Verdict.BLOCKED, ()),
        (True, 1025.0, False, Verdict.INCONCLUSIVE, ()),
        (False, 1025.0, True, Verdict.INCONCLUSIVE, ()),
        (True, 1015.0, True, Verdict.FLAT, (GAIN_NOT_CONFIRMED,)),
        (True, 900.0, True, Verdict.REGRESSED, (GAIN_NOT_CONFIRMED,)),
    ],
)
def test_a_retest_settles_the_gain_and_the_accuracy_gate(first_ok, retest, retest_ok, verdict, refused_by):
    first = compare(1030.0, first_ok, PARENT, NORMAL, "r1")
    judgement = confirm(first, retest, retest_ok, PARENT, NORMAL, "r2")
    assert judgement.verdict is verdict
    assert judgement.refused_by == refused_by
    assert judgement.accuracy_ok is (first_ok and retest_ok)
    assert judgement.recorded_value == {GRADED_OUTPUT: (1030.0 + retest) / 2}
    assert judgement.request_ids == ("r1", "r2")


def test_both_measurements_are_held_to_the_bar_as_it_stands_at_the_retest():
    first = compare(1030.0, True, PARENT, NORMAL, "r1")
    raised = NORMAL.with_baseline(1040.0)
    assert raised.threshold_pct == pytest.approx(4.278, abs=1e-3)

    judgement = confirm(first, 1050.0, True, PARENT, raised, "r2")

    assert judgement.verdict is Verdict.FLAT
    assert judgement.refused_by == (GAIN_NOT_CONFIRMED,)


def test_only_a_provisional_judgement_is_retested():
    with pytest.raises(ValueError, match="provisional"):
        confirm(compare(1010.0, True, PARENT, NORMAL, "r1"), 1030.0, True, PARENT, NORMAL, "r2")


def test_a_failed_measurement_is_not_compared():
    with pytest.raises(ValueError, match="cannot compare"):
        compare(0.0, True, PARENT, NORMAL, "r1")
