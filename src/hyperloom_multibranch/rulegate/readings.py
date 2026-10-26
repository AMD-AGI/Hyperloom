# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the rule gate compares, read from a measured test: its measure-phase rounds and its eval score."""

from __future__ import annotations

import statistics

from hyperloom.common.accuracy import ACCURACY_METRICS
from hyperloom.common.perf_metric import GRADED_INTVTY_P50, GRADED_OUTPUT
from hyperloom.contracts.measurement import RequestClasses, RoundValue, TestResult, Validity
from hyperloom_multibranch.rulegate.inferencex import InferenceXReading

_MEASURE = "measure"
#: Requests the server left unserved (a transport failure or zero output tokens), summed over the measured rounds.
GRADED_UNSERVED_REQUESTS = "graded_unserved_requests"
#: Requests answered with output tokens the client still counted failed, summed over the measured rounds.
GRADED_SHORT_REQUESTS = "graded_short_requests"
#: Requests profiled, summed over the measured rounds: the denominator of the two counts above.
GRADED_REQUESTS_TOTAL = "graded_requests_total"


class UnreadableResult(ValueError):
    """A result the rule gate does not judge: no successful measurement, or without what its rule compares."""


def inferencex_reading(result: TestResult, concurrency: int) -> InferenceXReading:
    """The mean graded throughput of the measure-phase rounds at ``concurrency``, and the test's eval score."""
    return _inferencex(result, _graded_throughput(result, concurrency), eval_score(result))


def unscored_reading(result: TestResult, concurrency: int) -> InferenceXReading:
    """The mean graded throughput of a test that ran no eval, with no score."""
    return _inferencex(result, _graded_throughput(result, concurrency), None)


def dropped_requests(result: TestResult, concurrency: int) -> int:
    """The requests the measure-phase rounds at ``concurrency`` sent but did not complete; a candidate whose round does
    not count them is not judged, since its throughput may be over fewer requests than were sent."""
    rounds = _graded(result, concurrency)
    counts = [r.failed_requests for r in rounds if r.failed_requests is not None]
    if len(counts) < len(rounds):
        uncounted = len(rounds) - len(counts)
        raise UnreadableResult(
            f"{result.request_id} does not count the requests it dropped in {uncounted} of its rounds"
        )
    return sum(counts)


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
    """Each metric every measure-phase round reports, averaged over them, and the request classes summed over them.

    A guard or comparability metric some round lacks stays out, so the comparison fails closed on it rather than
    trusting a partial reading; without the graded P50 there is nothing to compare.
    """
    rounds = _measured(result)
    if not rounds:
        raise UnreadableResult(f"{result.request_id} has no measure-phase round")
    shared = set(rounds[0].metrics).intersection(*(r.metrics for r in rounds[1:]))
    if GRADED_INTVTY_P50 not in shared:
        raise UnreadableResult(f"{result.request_id} does not report {GRADED_INTVTY_P50} in every measure-phase round")
    reading = {key: statistics.fmean(r.metrics[key] for r in rounds) for key in sorted(shared)}
    return {**reading, **_request_classes(result.request_classes, len(rounds))}


def _request_classes(classes: tuple[RequestClasses, ...], measured: int) -> dict[str, float]:
    """Each count only when every measured round counted it: an uncounted round is no "none"."""
    if len(classes) != measured:
        return {}
    counts: dict[str, float] = {}
    if all(c.requests_total is not None for c in classes):
        counts[GRADED_REQUESTS_TOTAL] = float(sum(c.requests_total for c in classes))
    if all(c.failed_empty is not None and c.failed_transport is not None for c in classes):
        counts[GRADED_UNSERVED_REQUESTS] = float(sum(c.failed_transport + c.failed_empty for c in classes))
        counts[GRADED_SHORT_REQUESTS] = float(sum(c.failed_short for c in classes))
    return counts


def _graded_throughput(result: TestResult, concurrency: int) -> float:
    rounds = _graded(result, concurrency)
    missing = [r for r in rounds if GRADED_OUTPUT not in r.metrics]
    if missing:
        raise UnreadableResult(f"{result.request_id} reports no {GRADED_OUTPUT} in {len(missing)} of its rounds")
    return statistics.fmean(r.metrics[GRADED_OUTPUT] for r in rounds)


def _inferencex(result: TestResult, throughput: float, score: float | None) -> InferenceXReading:
    try:
        return InferenceXReading(throughput, score)
    except ValueError as error:
        raise UnreadableResult(f"{result.request_id}: {error}") from error


def _measured(result: TestResult) -> list[RoundValue]:
    _require_measurement(result)
    return [r for r in result.rounds if r.phase == _MEASURE]


def _graded(result: TestResult, concurrency: int) -> list[RoundValue]:
    rounds = [r for r in _measured(result) if r.concurrency == concurrency]
    if not rounds:
        raise UnreadableResult(f"{result.request_id} has no measure-phase round at concurrency {concurrency}")
    return rounds


def _require_measurement(result: TestResult) -> None:
    if result.validity is not Validity.SUCCEEDED:
        raise UnreadableResult(f"{result.request_id} is {result.validity.value}, not a measurement")


__all__ = [
    "GRADED_REQUESTS_TOTAL",
    "GRADED_SHORT_REQUESTS",
    "GRADED_UNSERVED_REQUESTS",
    "UnreadableResult",
    "agentx_reading",
    "dropped_requests",
    "eval_score",
    "inferencex_reading",
    "unscored_reading",
]
