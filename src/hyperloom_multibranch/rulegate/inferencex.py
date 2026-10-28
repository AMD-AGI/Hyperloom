# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The InferenceX rule: a bar from the sandbox-to-sandbox spread, and a retest on another sandbox before a gain counts."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass

from hyperloom.common.gain_math import finite_score
from hyperloom.common.gain_math import gain_pct
from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom_multibranch.contracts.judgement import ACCURACY_SCORE, AccuracyVerdict, Judgement, Verdict
from hyperloom_multibranch.rulegate.accuracy import inferencex_accuracy_ok
from hyperloom_multibranch.rulegate.short_prompts import SHORT_PROMPTS_INCONSISTENT

# With the retest, at a 1% spread and three baselines (simulated): about 6% of no-op candidates get committed and about
# 75% of real +3% gains are confirmed.
NOISE_MULTIPLE = 2.0

# Three near-identical baselines would otherwise let every flicker through. Repeated tests of one recipe in session
# c7eddf22 spread 0.34%; at that spread this floor commits about 2% of no-op candidates and confirms about 67% of real
# +1% gains (simulated), where 1.0 confirmed 29%.
FLOOR_PCT = 0.6

MIN_BASELINES = 2
# PRELUDE measures the root this many times, so one failed measurement still leaves the bar its two.
BASELINE_TESTS = 3

GAIN_BELOW_THRESHOLD = "gain_below_threshold"
GAIN_NOT_CONFIRMED = "gain_not_confirmed"
# A synthetic benchmark a healthy server completes in full: any request it did not complete is a defect, whatever the
# parent did, since throughput over the requests that did complete is not the work that was asked for.
DROPPED_REQUESTS = "dropped_requests"
#: The long-context check dropped beyond its own sampling noise below the root's reading (#142 c3219, bar (b)).
LONG_CONTEXT_DROPPED = "long_context_dropped"
_ACCURACY = "accuracy"


def _checked(value: float) -> float:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"a throughput must be positive and finite, got {value!r}")
    return value


def _gain(throughput: float, parent: Mapping[str, float]) -> float:
    gain = gain_pct(throughput, parent[GRADED_OUTPUT])
    if gain is None:
        raise ValueError(f"cannot compare {throughput!r} against a parent record of {parent[GRADED_OUTPUT]!r}")
    return gain


@dataclass(frozen=True)
class DispersionThreshold:
    """The bar, from the spread of the PRELUDE baselines; it holds for the whole session."""

    baselines: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.baselines) < MIN_BASELINES:
            raise ValueError(f"the spread needs at least {MIN_BASELINES} baselines, got {len(self.baselines)}")
        for value in self.baselines:
            _checked(value)

    @property
    def spread_pct(self) -> float:
        """The relative standard deviation of one measurement, in percent."""
        return statistics.stdev(self.baselines) / statistics.fmean(self.baselines) * 100.0

    @property
    def threshold_pct(self) -> float:
        return max(FLOOR_PCT, NOISE_MULTIPLE * self.spread_pct)


@dataclass(frozen=True)
class InferenceXReading:
    """One bench-eval-bench test: its graded throughput, the eval score measured in the same sandbox (``None`` for an
    explore that skipped the eval), the requests its graded rounds sent but did not complete, whether its
    short-prompt check was more inconsistent than the baseline's, and whether its long-context check dropped below the
    root's."""

    throughput: float
    accuracy: float | None
    dropped: int = 0
    inconsistent: bool = False
    long_context_dropped: bool = False

    def __post_init__(self) -> None:
        _checked(self.throughput)
        if self.accuracy is not None and finite_score(self.accuracy) is None:
            raise ValueError(f"an eval score must be a finite number, got {self.accuracy!r}")
        if self.dropped < 0:
            raise ValueError(f"a count of dropped requests is at least 0, got {self.dropped!r}")

    @property
    def defects(self) -> tuple[str, ...]:
        return (
            *((DROPPED_REQUESTS,) if self.dropped else ()),
            *((SHORT_PROMPTS_INCONSISTENT,) if self.inconsistent else ()),
            *((LONG_CONTEXT_DROPPED,) if self.long_context_dropped else ()),
        )


@dataclass(frozen=True)
class InferenceXBaseline:
    """What every candidate of a session is held to: the bar on its throughput gain, the baselines' eval score, and the
    most short-prompt inconsistency they measured (``None`` when none did)."""

    bar_pct: float
    accuracy: float
    short_prompt_floor: int | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.bar_pct) and self.bar_pct > 0) or finite_score(self.accuracy) is None:
            raise ValueError(f"a baseline needs a positive bar and a finite eval score, got {self!r}")


def _recorded(throughput: float, accuracy: float | None) -> dict[str, float]:
    return {GRADED_OUTPUT: throughput} if accuracy is None else {GRADED_OUTPUT: throughput, ACCURACY_SCORE: accuracy}


def compare(
    reading: InferenceXReading, baseline: InferenceXBaseline, parent: Mapping[str, float], request_id: str
) -> Judgement:
    """The first measurement against the parent's record: past the bar it waits for a retest whatever its accuracy and
    defects, since the retest settles both whether the gain holds and whether a defect is real. Below it, a
    measurement with a defect is regressed, so a correctness fix measured on it forks rather than landing. A
    measurement that skipped its eval is judged on throughput alone, its accuracy ``NOT_MEASURED``."""
    gain = _gain(reading.throughput, parent)
    bar = baseline.bar_pct
    if reading.accuracy is None:
        accuracy = AccuracyVerdict.NOT_MEASURED
    else:
        accuracy = AccuracyVerdict.measured(inferencex_accuracy_ok(baseline.accuracy, reading.accuracy))
    recorded = _recorded(reading.throughput, reading.accuracy)
    if gain > bar:
        return Judgement(Verdict.NEEDS_CONFIRM, gain, bar, (request_id,), recorded, accuracy, reading.defects)
    verdict = Verdict.REGRESSED if gain < -bar or reading.defects else Verdict.FLAT
    refused_by = (*reading.defects, GAIN_BELOW_THRESHOLD)
    return Judgement(verdict, gain, bar, (request_id,), recorded, accuracy, refused_by)


def confirm(
    first: Judgement,
    reading: InferenceXReading,
    baseline: InferenceXBaseline,
    parent: Mapping[str, float],
    request_id: str,
) -> Judgement:
    """Settles a provisional gain with a retest on another sandbox; the record is the mean of both measurements, and its
    eval score the mean of those that ran their eval.

    A gain holds when the mean of both measurements clears the bar and the retest is still a gain; one that holds is
    improved when neither showed a defect (a failed accuracy gate, dropped requests, short prompts answered more
    inconsistently than the baselines, a long-context drop), blocked when both showed the same one, and inconclusive otherwise. After a first
    measurement that skipped its eval, the retest's eval is the only one, so its failure blocks the gain. A blocked judgement names the performance defects both showed; the accuracy gate is
    in ``accuracy``.
    """
    if first.verdict is not Verdict.NEEDS_CONFIRM:
        raise ValueError(f"only a provisional judgement is retested, not a {first.verdict.value} one")
    if reading.accuracy is None:
        raise ValueError(f"the retest {request_id} ran no eval, which settling a gain needs")
    bar = baseline.bar_pct
    unmeasured = first.accuracy is AccuracyVerdict.NOT_MEASURED
    scores = (reading.accuracy,)
    if not unmeasured:
        scores = (first.recorded_value[ACCURACY_SCORE], *scores)
    recorded = _recorded((first.recorded_value[GRADED_OUTPUT] + reading.throughput) / 2.0, sum(scores) / len(scores))
    gain = _gain(recorded[GRADED_OUTPUT], parent)
    request_ids = (*first.request_ids, request_id)
    accuracy_ok = inferencex_accuracy_ok(baseline.accuracy, reading.accuracy)
    first_ok = accuracy_ok if unmeasured else first.accuracy is AccuracyVerdict.PASSED
    accuracy = AccuracyVerdict.measured(accuracy_ok and first_ok)
    seen = tuple(dict.fromkeys((*first.refused_by, *reading.defects)))
    if gain > bar and _gain(reading.throughput, parent) > 0:
        before = {*first.refused_by, *(() if first_ok else (_ACCURACY,))}
        after = {*reading.defects, *(() if accuracy_ok else (_ACCURACY,))}
        if before & after:
            shared = tuple(defect for defect in seen if defect in after)
            return Judgement(Verdict.BLOCKED, gain, bar, request_ids, recorded, accuracy, shared)
        verdict = Verdict.INCONCLUSIVE if before or after else Verdict.IMPROVED
        return Judgement(verdict, gain, bar, request_ids, recorded, accuracy, seen)
    verdict = Verdict.REGRESSED if gain < -bar or seen else Verdict.FLAT
    return Judgement(verdict, gain, bar, request_ids, recorded, accuracy, (*seen, GAIN_NOT_CONFIRMED))


__all__ = [
    "BASELINE_TESTS",
    "DROPPED_REQUESTS",
    "FLOOR_PCT",
    "GAIN_BELOW_THRESHOLD",
    "GAIN_NOT_CONFIRMED",
    "MIN_BASELINES",
    "NOISE_MULTIPLE",
    "DispersionThreshold",
    "InferenceXBaseline",
    "InferenceXReading",
    "compare",
    "confirm",
]
