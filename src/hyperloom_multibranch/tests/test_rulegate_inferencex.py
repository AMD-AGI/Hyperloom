# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import math

import pytest

from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom_multibranch.contracts.judgement import ACCURACY_SCORE, Verdict
from hyperloom_multibranch.rulegate.inferencex import (
    DROPPED_REQUESTS,
    FLOOR_PCT,
    GAIN_BELOW_THRESHOLD,
    GAIN_NOT_CONFIRMED,
    DispersionThreshold,
    InferenceXBaseline,
    InferenceXReading,
    compare,
    confirm,
)

NORMAL = DispersionThreshold((1000.0, 1010.0, 990.0))
PARENT = {GRADED_OUTPUT: 1000.0}
BASELINE = InferenceXBaseline(NORMAL.threshold_pct, 0.90)
SCORES = {True: 0.88, False: 0.80}


def _reading(throughput: float, accuracy_ok: bool = True) -> InferenceXReading:
    return InferenceXReading(throughput, SCORES[accuracy_ok])


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
    judgement = compare(_reading(throughput, accuracy_ok), BASELINE, PARENT, "r1")
    assert judgement.verdict is verdict
    assert judgement.refused_by == refused_by
    assert judgement.accuracy_ok is accuracy_ok
    assert judgement.threshold_pct == pytest.approx(2.0)
    assert judgement.recorded_value == {GRADED_OUTPUT: throughput, ACCURACY_SCORE: SCORES[accuracy_ok]}
    assert judgement.request_ids == ("r1",)


@pytest.mark.parametrize(("baseline_accuracy", "accuracy_ok"), [(0.90, True), (0.96, False)])
def test_the_eval_score_is_held_to_the_baseline_score(baseline_accuracy, accuracy_ok):
    judgement = compare(
        InferenceXReading(1030.0, 0.90), InferenceXBaseline(NORMAL.threshold_pct, baseline_accuracy), PARENT, "r1"
    )
    assert judgement.accuracy_ok is accuracy_ok


@pytest.mark.parametrize(
    ("first_ok", "retest", "retest_ok", "verdict", "refused_by"),
    [
        (True, 1025.0, True, Verdict.IMPROVED, ()),
        (False, 1025.0, False, Verdict.BLOCKED, ()),
        (True, 1025.0, False, Verdict.INCONCLUSIVE, ()),
        (False, 1025.0, True, Verdict.INCONCLUSIVE, ()),
        (True, 1015.0, True, Verdict.IMPROVED, ()),
        (True, 1005.0, True, Verdict.FLAT, (GAIN_NOT_CONFIRMED,)),
        (True, 900.0, True, Verdict.REGRESSED, (GAIN_NOT_CONFIRMED,)),
    ],
)
def test_a_retest_settles_the_gain_and_the_accuracy_gate(first_ok, retest, retest_ok, verdict, refused_by):
    first = compare(_reading(1030.0, first_ok), BASELINE, PARENT, "r1")
    judgement = confirm(first, _reading(retest, retest_ok), BASELINE, PARENT, "r2")
    assert judgement.verdict is verdict
    assert judgement.refused_by == refused_by
    assert judgement.accuracy_ok is (first_ok and retest_ok)
    assert judgement.recorded_value == {
        GRADED_OUTPUT: (1030.0 + retest) / 2,
        ACCURACY_SCORE: pytest.approx((SCORES[first_ok] + SCORES[retest_ok]) / 2),
    }
    assert judgement.request_ids == ("r1", "r2")


@pytest.mark.parametrize(
    ("throughput", "verdict", "refused_by"),
    [
        (1030.0, Verdict.NEEDS_CONFIRM, (DROPPED_REQUESTS,)),
        (1010.0, Verdict.REGRESSED, (DROPPED_REQUESTS, GAIN_BELOW_THRESHOLD)),
    ],
)
def test_a_measurement_that_dropped_requests_is_a_defect_whatever_the_parent_dropped(throughput, verdict, refused_by):
    judgement = compare(InferenceXReading(throughput, SCORES[True], dropped=3), BASELINE, PARENT, "r1")
    assert (judgement.verdict, judgement.refused_by, judgement.accuracy_ok) == (verdict, refused_by, True)


@pytest.mark.parametrize(
    ("first", "retest", "verdict", "refused_by"),
    [
        ((True, 3), (True, 1), Verdict.BLOCKED, (DROPPED_REQUESTS,)),
        ((True, 3), (True, 0), Verdict.INCONCLUSIVE, (DROPPED_REQUESTS,)),
        ((True, 0), (True, 2), Verdict.INCONCLUSIVE, (DROPPED_REQUESTS,)),
        ((False, 0), (True, 2), Verdict.INCONCLUSIVE, (DROPPED_REQUESTS,)),
        ((False, 2), (False, 0), Verdict.BLOCKED, ()),
    ],
    ids=["dropped-twice", "dropped-first", "dropped-retest", "two-defects", "accuracy-twice"],
)
def test_a_gain_is_blocked_only_by_a_defect_both_measurements_show(first, retest, verdict, refused_by):
    provisional = compare(InferenceXReading(1030.0, SCORES[first[0]], dropped=first[1]), BASELINE, PARENT, "r1")
    reading = InferenceXReading(1025.0, SCORES[retest[0]], dropped=retest[1])
    judgement = confirm(provisional, reading, BASELINE, PARENT, "r2")
    assert (judgement.verdict, judgement.refused_by) == (verdict, refused_by)


def test_a_retest_that_loses_the_gain_does_not_confirm_it_whatever_the_mean():
    first = compare(_reading(1050.0), BASELINE, PARENT, "r1")
    judgement = confirm(first, _reading(995.0), BASELINE, PARENT, "r2")
    assert judgement.gain_pct == pytest.approx(2.25)
    assert (judgement.verdict, judgement.refused_by) == (Verdict.FLAT, (GAIN_NOT_CONFIRMED,))


@pytest.mark.parametrize(
    ("parent", "first", "retest"),
    [(7861.0, 8037.0, 7908.0), (8070.0, 8145.0, 8175.0)],
    ids=["b07-686cd806", "b09-cuda-graph-backend-prefill"],
)
def test_a_gain_near_one_percent_on_a_quiet_machine_is_confirmed(parent, first, retest):
    quiet = InferenceXBaseline(DispersionThreshold((8000.0, 8008.0, 7992.0)).threshold_pct, 0.90)
    provisional = compare(_reading(first), quiet, {GRADED_OUTPUT: parent}, "r1")
    judgement = confirm(provisional, _reading(retest), quiet, {GRADED_OUTPUT: parent}, "r2")
    assert (provisional.verdict, judgement.verdict) == (Verdict.NEEDS_CONFIRM, Verdict.IMPROVED)


def test_only_a_provisional_judgement_is_retested():
    first = compare(_reading(1010.0), BASELINE, PARENT, "r1")
    with pytest.raises(ValueError, match="provisional"):
        confirm(first, _reading(1030.0), BASELINE, PARENT, "r2")


@pytest.mark.parametrize(
    ("throughput", "accuracy", "dropped"), [(0.0, 0.9, 0), (math.inf, 0.9, 0), (1000.0, math.nan, 0), (1000.0, 0.9, -1)]
)
def test_a_reading_needs_a_positive_throughput_a_finite_score_and_no_negative_drops(throughput, accuracy, dropped):
    with pytest.raises(ValueError):
        InferenceXReading(throughput, accuracy, dropped)


@pytest.mark.parametrize(("bar", "accuracy"), [(0.0, 0.9), (math.nan, 0.9), (2.0, math.nan)])
def test_a_baseline_needs_a_positive_bar_and_a_finite_score(bar, accuracy):
    with pytest.raises(ValueError):
        InferenceXBaseline(bar, accuracy)


def test_a_parent_without_a_usable_record_is_not_compared():
    with pytest.raises(ValueError, match="cannot compare"):
        compare(_reading(1000.0), BASELINE, {GRADED_OUTPUT: 0.0}, "r1")
