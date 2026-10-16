# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from hyperloom.common.accuracy import ACCURACY_METRICS
from hyperloom.common.perf_metric import GRADED_DURATION, GRADED_ERROR_RATE, GRADED_INTVTY_P50, GRADED_OUTPUT
from hyperloom.contracts.measurement import AccuracyValue, RoundValue, TestResult, Validity
from hyperloom_multibranch.rulegate.inferencex import InferenceXReading
from hyperloom_multibranch.rulegate.readings import UnreadableResult, agentx_reading, inferencex_reading

STRICT, FLEXIBLE, _, ACC = ACCURACY_METRICS
GSM8K = "gsm8k"
CONC = 64
MEASURE = "measure"
SCORES = (AccuracyValue(GSM8K, STRICT, 0.61), AccuracyValue(GSM8K, FLEXIBLE, 0.64))


def _result(*rounds: RoundValue, accuracy=SCORES, validity=Validity.SUCCEEDED) -> TestResult:
    return TestResult("r1", validity, rounds=rounds, accuracy=tuple(accuracy))


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
