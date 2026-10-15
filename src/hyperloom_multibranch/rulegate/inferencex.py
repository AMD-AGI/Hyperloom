# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The InferenceX rule: a bar from the sandbox-to-sandbox spread, and a retest on another sandbox before a gain counts."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass

from hyperloom.common.accuracy import finite_score
from hyperloom.common.gain_math import gain_pct
from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom_multibranch.contracts.judgement import ACCURACY_SCORE, Judgement, Verdict
from hyperloom_multibranch.rulegate.accuracy import inferencex_accuracy_ok

# With the retest, at a 1% spread and three baselines (simulated): about 2% of no-op candidates get committed and about
# 70% of real +3% gains are confirmed.
NOISE_MULTIPLE = 2.0

# Three near-identical baselines would otherwise let every flicker through.
FLOOR_PCT = 1.0

MIN_BASELINES = 2

GAIN_BELOW_THRESHOLD = "gain_below_threshold"
GAIN_NOT_CONFIRMED = "gain_not_confirmed"


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
    """One bench-eval-bench test: its graded throughput and the eval score measured in the same sandbox."""

    throughput: float
    accuracy: float

    def __post_init__(self) -> None:
        _checked(self.throughput)
        if finite_score(self.accuracy) is None:
            raise ValueError(f"an eval score must be a finite number, got {self.accuracy!r}")


@dataclass(frozen=True)
class InferenceXBaseline:
    """What every candidate of a session is held to: the bar on its throughput gain, and the baselines' eval score."""

    bar_pct: float
    accuracy: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.bar_pct) and self.bar_pct > 0) or finite_score(self.accuracy) is None:
            raise ValueError(f"a baseline needs a positive bar and a finite eval score, got {self!r}")


def _recorded(throughput: float, accuracy: float) -> dict[str, float]:
    return {GRADED_OUTPUT: throughput, ACCURACY_SCORE: accuracy}


def compare(
    reading: InferenceXReading, baseline: InferenceXBaseline, parent: Mapping[str, float], request_id: str
) -> Judgement:
    """The first measurement against the parent's record: past the bar it waits for a retest whatever its accuracy,
    since the retest settles both whether the gain holds and whether a failed accuracy gate is real."""
    gain = _gain(reading.throughput, parent)
    bar = baseline.bar_pct
    accuracy_ok = inferencex_accuracy_ok(baseline.accuracy, reading.accuracy)
    recorded = _recorded(reading.throughput, reading.accuracy)
    if gain > bar:
        return Judgement(Verdict.NEEDS_CONFIRM, gain, bar, (request_id,), recorded, accuracy_ok)
    verdict = Verdict.REGRESSED if gain < -bar else Verdict.FLAT
    return Judgement(verdict, gain, bar, (request_id,), recorded, accuracy_ok, (GAIN_BELOW_THRESHOLD,))


def confirm(
    first: Judgement,
    reading: InferenceXReading,
    baseline: InferenceXBaseline,
    parent: Mapping[str, float],
    request_id: str,
) -> Judgement:
    """Settles a provisional gain with a retest on another sandbox; the record is the mean of both measurements, their
    eval scores included.

    A gain both measurements cleared is improved when both passed the accuracy gate, blocked when both failed it and
    inconclusive when they split.
    """
    if first.verdict is not Verdict.NEEDS_CONFIRM:
        raise ValueError(f"only a provisional judgement is retested, not a {first.verdict.value} one")
    bar = baseline.bar_pct
    recorded = _recorded(
        (first.recorded_value[GRADED_OUTPUT] + reading.throughput) / 2.0,
        (first.recorded_value[ACCURACY_SCORE] + reading.accuracy) / 2.0,
    )
    gain = _gain(recorded[GRADED_OUTPUT], parent)
    request_ids = (*first.request_ids, request_id)
    accuracy_ok = inferencex_accuracy_ok(baseline.accuracy, reading.accuracy)
    both_ok = first.accuracy_ok and accuracy_ok
    if first.gain_pct > bar and _gain(reading.throughput, parent) > bar:
        if both_ok:
            verdict = Verdict.IMPROVED
        elif first.accuracy_ok or accuracy_ok:
            verdict = Verdict.INCONCLUSIVE
        else:
            verdict = Verdict.BLOCKED
        return Judgement(verdict, gain, bar, request_ids, recorded, both_ok)
    verdict = Verdict.REGRESSED if gain < -bar else Verdict.FLAT
    return Judgement(verdict, gain, bar, request_ids, recorded, both_ok, (GAIN_NOT_CONFIRMED,))


__all__ = [
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
