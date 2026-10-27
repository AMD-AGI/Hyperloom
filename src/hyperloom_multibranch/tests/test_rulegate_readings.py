# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from hyperloom.common.gain_math import ACCURACY_METRICS
from hyperloom.common.perf_metric import GRADED_DURATION, GRADED_ERROR_RATE, GRADED_INTVTY_P50, GRADED_OUTPUT
from hyperloom.contracts.measurement import AccuracyValue, RequestClasses, RoundValue, TestResult, Validity
from hyperloom_multibranch.rulegate.inferencex import InferenceXReading
from hyperloom_multibranch.rulegate.readings import (
    UnreadableResult,
    agentx_reading,
    dropped_requests,
    inferencex_reading,
)
from hyperloom_multibranch.rulegate.readings import (
    GRADED_REQUESTS_TOTAL,
    GRADED_SHORT_REQUESTS,
    GRADED_UNSERVED_REQUESTS,
)

STRICT, FLEXIBLE, _, ACC = ACCURACY_METRICS
GSM8K = "gsm8k"
CONC = 64
MEASURE = "measure"
SCORES = (AccuracyValue(GSM8K, STRICT, 0.61), AccuracyValue(GSM8K, FLEXIBLE, 0.64))


def _result(*rounds: RoundValue, accuracy=SCORES, validity=Validity.SUCCEEDED, classes=()) -> TestResult:
    return TestResult("r1", validity, rounds=rounds, accuracy=tuple(accuracy), request_classes=tuple(classes))


def _classes(point: int, transport: int, empty: int | None, short: int | None) -> RequestClasses:
    no_content = None if empty is None else empty + short
    counts = (100, transport + no_content, transport, no_content) if no_content is not None else (None,) * 4
    reason = "" if empty is not None else "the export carries no output token count"
    return RequestClasses(point, "", None, *counts, empty, short, reason)


def _agentx_rounds(*classes: RequestClasses) -> TestResult:
    rounds = [RoundValue(MEASURE, None, {GRADED_INTVTY_P50: 40.0}) for _ in range(2)]
    return _result(RoundValue("warmup", None, {GRADED_INTVTY_P50: 1.0}), *rounds, accuracy=(), classes=classes)


def test_inferencex_reads_the_measure_rounds_at_the_graded_concurrency_and_the_strict_score():
    result = _result(
        RoundValue("warmup", CONC, {GRADED_OUTPUT: 500.0}),
        RoundValue(MEASURE, CONC, {GRADED_OUTPUT: 1000.0}),
        RoundValue(MEASURE, CONC, {GRADED_OUTPUT: 1020.0}),
        RoundValue(MEASURE, 128, {GRADED_OUTPUT: 1800.0}),
    )
    assert inferencex_reading(result, CONC) == InferenceXReading(1010.0, 0.61)


@pytest.mark.parametrize(
    ("accuracy", "score"),
    [
        ((AccuracyValue(GSM8K, FLEXIBLE, 0.64),), 0.64),
        ((AccuracyValue("mmlu", ACC, 0.70), AccuracyValue(GSM8K, STRICT, 0.61)), 0.70),
        ((AccuracyValue(GSM8K, "other,none", 0.99), AccuracyValue(GSM8K, FLEXIBLE, 0.64)), 0.64),
    ],
)
def test_the_eval_score_is_picked_as_the_old_flow_picks_it(accuracy, score):
    result = _result(RoundValue(MEASURE, CONC, {GRADED_OUTPUT: 1000.0}), accuracy=accuracy)
    assert inferencex_reading(result, CONC).accuracy == score


@pytest.mark.parametrize(
    "result",
    [
        _result(validity=Validity.MEASUREMENT_FAILED),
        _result(validity=Validity.CANDIDATE_FAILED),
        _result(RoundValue(MEASURE, 128, {GRADED_OUTPUT: 1000.0})),
        _result(RoundValue(MEASURE, CONC, {GRADED_INTVTY_P50: 40.0})),
        _result(RoundValue(MEASURE, CONC, {GRADED_OUTPUT: 1000.0}), accuracy=()),
        _result(RoundValue(MEASURE, CONC, {GRADED_OUTPUT: 1000.0}), accuracy=(AccuracyValue(GSM8K, "bleu", 0.3),)),
        _result(RoundValue(MEASURE, CONC, {GRADED_OUTPUT: 0.0})),
    ],
    ids=[
        "measurement-failed",
        "candidate-failed",
        "other-concurrency",
        "no-throughput",
        "no-score",
        "unknown-metric",
        "zero-throughput",
    ],
)
def test_inferencex_refuses_what_it_cannot_judge(result):
    with pytest.raises(UnreadableResult):
        inferencex_reading(result, CONC)


def test_the_dropped_requests_are_those_of_the_measure_rounds_at_the_graded_concurrency():
    result = _result(
        RoundValue("warmup", CONC, {GRADED_OUTPUT: 500.0}, completed_requests=90, failed_requests=10),
        RoundValue(MEASURE, CONC, {GRADED_OUTPUT: 1000.0}, completed_requests=99, failed_requests=1),
        RoundValue(MEASURE, CONC, {GRADED_OUTPUT: 1020.0}, completed_requests=98, failed_requests=2),
        RoundValue(MEASURE, 128, {GRADED_OUTPUT: 1800.0}, completed_requests=50, failed_requests=50),
    )
    assert dropped_requests(result, CONC) == 3


def test_a_graded_round_that_does_not_count_its_requests_leaves_the_drops_unread():
    result = _result(
        RoundValue(MEASURE, CONC, {GRADED_OUTPUT: 1000.0}, completed_requests=100, failed_requests=0),
        RoundValue(MEASURE, CONC, {GRADED_OUTPUT: 1020.0}),
    )
    with pytest.raises(UnreadableResult, match="does not count the requests it dropped in 1 of its rounds"):
        dropped_requests(result, CONC)
    assert inferencex_reading(result, CONC) == InferenceXReading(1010.0, 0.61)


def test_agentx_averages_the_measure_rounds_and_leaves_out_a_metric_one_round_lacks():
    result = _result(
        RoundValue("warmup", None, {GRADED_INTVTY_P50: 1.0}),
        RoundValue(MEASURE, None, {GRADED_INTVTY_P50: 40.0, GRADED_DURATION: 600.0, GRADED_ERROR_RATE: 0.0}),
        RoundValue(MEASURE, None, {GRADED_INTVTY_P50: 42.0, GRADED_DURATION: 610.0}),
        accuracy=(),
    )
    assert agentx_reading(result) == {GRADED_DURATION: 605.0, GRADED_INTVTY_P50: 41.0}


@pytest.mark.parametrize(
    "result",
    [
        _result(validity=Validity.MEASUREMENT_FAILED),
        _result(RoundValue("warmup", None, {GRADED_INTVTY_P50: 1.0})),
        _result(RoundValue(MEASURE, None, {GRADED_DURATION: 600.0, GRADED_ERROR_RATE: 0.0})),
    ],
    ids=["measurement-failed", "no-measure-round", "no-p50"],
)
def test_agentx_refuses_what_it_cannot_judge(result):
    with pytest.raises(UnreadableResult):
        agentx_reading(result)


def test_agentx_sums_the_unserved_and_the_short_requests_over_the_measured_rounds():
    reading = agentx_reading(_agentx_rounds(_classes(0, 1, 2, 3), _classes(1, 4, 0, 5)))
    assert reading[GRADED_UNSERVED_REQUESTS] == 7.0
    assert reading[GRADED_SHORT_REQUESTS] == 8.0
    assert reading[GRADED_REQUESTS_TOTAL] == 200.0


@pytest.mark.parametrize(
    "classes",
    [(), (_classes(0, 1, 2, 3),), (_classes(0, 1, 2, 3), _classes(1, 0, None, None))],
    ids=["no-classes", "a-round-without-classes", "an-uncounted-round"],
)
def test_agentx_leaves_the_request_classes_out_when_a_round_does_not_count_them(classes):
    reading = agentx_reading(_agentx_rounds(*classes))
    assert GRADED_UNSERVED_REQUESTS not in reading
    assert GRADED_SHORT_REQUESTS not in reading


def test_agentx_sums_the_requests_of_rounds_that_counted_them_but_did_not_classify_the_failures():
    unsplit = RequestClasses(1, "", None, 100, 3, 1, 2, None, None, "the export carries no output token count")
    reading = agentx_reading(_agentx_rounds(_classes(0, 1, 2, 3), unsplit))
    assert GRADED_UNSERVED_REQUESTS not in reading
    assert reading[GRADED_REQUESTS_TOTAL] == 200.0


@pytest.mark.parametrize(
    "classes",
    [
        (),
        (_classes(0, 1, 2, 3),),
        (_classes(0, 1, 2, 3), RequestClasses(1, "", None, reason="the round left no export")),
    ],
    ids=["no-classes", "a-round-without-classes", "an-uncounted-round"],
)
def test_agentx_leaves_the_requests_total_out_when_a_round_does_not_count_its_requests(classes):
    assert GRADED_REQUESTS_TOTAL not in agentx_reading(_agentx_rounds(*classes))
