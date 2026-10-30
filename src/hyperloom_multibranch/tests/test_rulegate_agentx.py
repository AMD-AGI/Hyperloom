# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from hyperloom.common.perf_metric import (
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
)
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Verdict
from hyperloom_multibranch.rulegate.agentx import P50_BELOW_THRESHOLD, compare, confirm
from hyperloom_multibranch.rulegate.readings import (
    GRADED_REQUESTS_TOTAL,
    GRADED_SHORT_REQUESTS,
    GRADED_UNSERVED_REQUESTS,
)

PARENT = {
    GRADED_INTVTY_P50: 100.0,
    GRADED_INTVTY: 80.0,
    GRADED_OUTPUT: 1000.0,
    GRADED_DURATION: 600.0,
    GRADED_ERROR_RATE: 0.2,
}
P90_BROKEN = {GRADED_INTVTY_P50: 106.0, GRADED_INTVTY: 75.2}


@pytest.fixture(autouse=True)
def _default_noise_band(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PERF_NOISE_PCT", raising=False)


@pytest.mark.parametrize(
    ("changes", "verdict", "refused_by"),
    [
        ({GRADED_INTVTY_P50: 104.0}, Verdict.IMPROVED, ()),
        ({GRADED_INTVTY_P50: 102.0}, Verdict.FLAT, (P50_BELOW_THRESHOLD,)),
        ({GRADED_INTVTY_P50: 98.0}, Verdict.FLAT, (P50_BELOW_THRESHOLD,)),
        ({GRADED_INTVTY_P50: 96.0}, Verdict.REGRESSED, (P50_BELOW_THRESHOLD,)),
        (P90_BROKEN, Verdict.NEEDS_CONFIRM, ("p90_regressed",)),
        ({GRADED_INTVTY_P50: 104.0, GRADED_OUTPUT: 940.0}, Verdict.NEEDS_CONFIRM, ("output_regressed",)),
        ({GRADED_INTVTY_P50: 104.0, GRADED_ERROR_RATE: 1.2}, Verdict.NEEDS_CONFIRM, ("extra_failed_requests",)),
        ({GRADED_INTVTY_P50: 102.0, GRADED_INTVTY: 75.2}, Verdict.REGRESSED, ("p90_regressed", P50_BELOW_THRESHOLD)),
        ({GRADED_INTVTY_P50: 104.0, GRADED_DURATION: 660.0}, Verdict.REGRESSED, ("duration_drift",)),
    ],
)
def test_a_measurement_against_its_parent(changes, verdict, refused_by):
    candidate = {**PARENT, **changes}
    judgement = compare(candidate, PARENT, "r1")
    assert judgement.verdict is verdict
    assert judgement.refused_by == refused_by
    assert judgement.threshold_pct == pytest.approx(3.0)
    assert judgement.recorded_value == candidate
    assert judgement.request_ids == ("r1",)


def test_the_error_rate_bar_blocks_a_gain_and_fails_the_accuracy_gate():
    judgement = compare({**PARENT, GRADED_INTVTY_P50: 104.0, GRADED_ERROR_RATE: 12.0}, PARENT, "r1")
    assert judgement.verdict is Verdict.NEEDS_CONFIRM
    assert judgement.accuracy is AccuracyVerdict.FAILED


@pytest.mark.parametrize(
    ("retest", "verdict", "refused_by"),
    [
        ({GRADED_INTVTY_P50: 105.0, GRADED_INTVTY: 74.0}, Verdict.BLOCKED, ("p90_regressed",)),
        ({GRADED_INTVTY_P50: 105.0, GRADED_OUTPUT: 940.0}, Verdict.BLOCKED, ("p90_regressed", "output_regressed")),
        ({GRADED_INTVTY_P50: 105.0}, Verdict.INCONCLUSIVE, ("p90_regressed",)),
        (
            {GRADED_INTVTY_P50: 101.0, GRADED_INTVTY: 74.0},
            Verdict.INCONCLUSIVE,
            ("p90_regressed", P50_BELOW_THRESHOLD),
        ),
    ],
)
def test_a_retest_settles_a_blocked_gain(retest, verdict, refused_by):
    first = compare({**PARENT, **P90_BROKEN}, PARENT, "r1")
    judgement = confirm(first, {**PARENT, **retest}, PARENT, "r2")
    assert judgement.verdict is verdict
    assert judgement.refused_by == refused_by
    assert judgement.request_ids == ("r1", "r2")
    assert judgement.recorded_value[GRADED_INTVTY_P50] == pytest.approx((106.0 + retest[GRADED_INTVTY_P50]) / 2)


def test_only_a_provisional_judgement_is_retested():
    with pytest.raises(ValueError, match="provisional"):
        confirm(compare({**PARENT, GRADED_INTVTY_P50: 104.0}, PARENT, "r1"), PARENT, PARENT, "r2")


def test_a_pair_missing_a_comparability_input_counts_as_regressed():
    candidate = {key: value for key, value in PARENT.items() if key != GRADED_ERROR_RATE}
    judgement = compare({**candidate, GRADED_INTVTY_P50: 104.0}, PARENT, "r1")
    assert judgement.verdict is Verdict.REGRESSED
    assert judgement.refused_by == ("comparability_inputs_missing",)


def test_a_run_without_a_p50_is_not_compared():
    candidate = {key: value for key, value in PARENT.items() if key != GRADED_INTVTY_P50}
    with pytest.raises(ValueError, match="cannot compare"):
        compare(candidate, PARENT, "r1")


COUNTED = {**PARENT, GRADED_UNSERVED_REQUESTS: 1.0, GRADED_SHORT_REQUESTS: 0.0, GRADED_REQUESTS_TOTAL: 200.0}


def test_the_request_classes_are_recorded_on_the_judgement():
    candidate = {**COUNTED, GRADED_INTVTY_P50: 104.0, GRADED_SHORT_REQUESTS: 2.0}
    judgement = compare(candidate, COUNTED, "r1")
    assert judgement.recorded_value[GRADED_SHORT_REQUESTS] == 2.0
    assert judgement.recorded_value[GRADED_UNSERVED_REQUESTS] == 1.0
    assert judgement.recorded_value[GRADED_REQUESTS_TOTAL] == 200.0


def test_more_short_answers_alone_never_refuse_a_candidate():
    candidate = {**COUNTED, GRADED_INTVTY_P50: 104.0, GRADED_ERROR_RATE: 5.0, GRADED_SHORT_REQUESTS: 30.0}
    judgement = compare(candidate, COUNTED, "r1")
    assert judgement.verdict is Verdict.IMPROVED
    assert judgement.refused_by == ()


@pytest.mark.parametrize(
    ("candidate", "parent"),
    [
        ({**COUNTED, GRADED_UNSERVED_REQUESTS: 3.0}, COUNTED),
        ({key: value for key, value in COUNTED.items() if key != GRADED_UNSERVED_REQUESTS}, COUNTED),
        (COUNTED, PARENT),
    ],
    ids=["more-unserved", "candidate-uncounted", "parent-uncounted"],
)
def test_more_failed_requests_refuse_unless_the_unserved_count_shows_none_more(candidate, parent):
    judgement = compare({**candidate, GRADED_INTVTY_P50: 104.0, GRADED_ERROR_RATE: 5.0}, parent, "r1")
    assert judgement.verdict is Verdict.NEEDS_CONFIRM
    assert judgement.refused_by == ("extra_failed_requests",)
