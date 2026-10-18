# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the rule gate compares, read from a measured test: its measure-phase rounds and its eval score."""

from __future__ import annotations

import statistics

from hyperloom.common.accuracy import ACCURACY_METRICS
from hyperloom.common.perf_metric import GRADED_INTVTY_P50, GRADED_OUTPUT
from hyperloom.contracts.measurement import RoundValue, TestResult, Validity
from hyperloom_multibranch.rulegate.inferencex import InferenceXReading

_MEASURE = "measure"


class UnreadableResult(ValueError):
    """A result the rule gate does not judge: no successful measurement, or without what its rule compares."""


def inferencex_reading(result: TestResult, concurrency: int) -> InferenceXReading:
    """The mean graded throughput of the measure-phase rounds at ``concurrency``, and the test's eval score."""
    rounds = [r for r in _measured(result) if r.concurrency == concurrency]
    if not rounds:
        raise UnreadableResult(f"{result.request_id} has no measure-phase round at concurrency {concurrency}")
    missing = [r for r in rounds if GRADED_OUTPUT not in r.metrics]
    if missing:
        raise UnreadableResult(f"{result.request_id} reports no {GRADED_OUTPUT} in {len(missing)} of its rounds")
    score = eval_score(result)
    try:
        return InferenceXReading(statistics.fmean(r.metrics[GRADED_OUTPUT] for r in rounds), score)
    except ValueError as error:
        raise UnreadableResult(f"{result.request_id}: {error}") from error


def eval_score(result: TestResult) -> float:
    """Read as the old flow reads it: the first task reporting one of ``ACCURACY_METRICS``, by that order."""
    _require_measurement(result)
    by_task: dict[str, dict[str, float]] = {}
    for value in result.accuracy:
        by_task.setdefault(value.task, {})[value.metric] = value.score
    for scores in by_task.values():
        for metric in ACCURACY_METRICS:
            if metric in scores:
                return scores[metric]
    raise UnreadableResult(f"{result.request_id} has no eval score of {', '.join(ACCURACY_METRICS)}")


def agentx_reading(result: TestResult) -> dict[str, float]:
    """Each metric every measure-phase round reports, averaged over them.

    A guard or comparability metric some round lacks stays out, so the comparison fails closed on it rather than
    trusting a partial reading; without the graded P50 there is nothing to compare.
    """
    rounds = _measured(result)
    if not rounds:
        raise UnreadableResult(f"{result.request_id} has no measure-phase round")
    shared = set(rounds[0].metrics).intersection(*(r.metrics for r in rounds[1:]))
    if GRADED_INTVTY_P50 not in shared:
        raise UnreadableResult(f"{result.request_id} does not report {GRADED_INTVTY_P50} in every measure-phase round")
    return {key: statistics.fmean(r.metrics[key] for r in rounds) for key in sorted(shared)}


def _measured(result: TestResult) -> list[RoundValue]:
    _require_measurement(result)
    return [r for r in result.rounds if r.phase == _MEASURE]


def _require_measurement(result: TestResult) -> None:
    if result.validity is not Validity.SUCCEEDED:
        raise UnreadableResult(f"{result.request_id} is {result.validity.value}, not a measurement")


__all__ = ["UnreadableResult", "agentx_reading", "eval_score", "inferencex_reading"]
