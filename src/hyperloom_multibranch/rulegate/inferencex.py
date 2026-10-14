# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The InferenceX rule: a bar from the sandbox-to-sandbox spread, and a retest on another sandbox before a gain counts."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, replace

from hyperloom.common.gain_math import gain_pct
from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict

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
    """The bar, from the PRELUDE baselines and the baselines spot-checked since.

    Confirm-retest pairs stay out: only a first measurement that cleared the bar is retested, so a pair overstates the
    spread and would push the bar up as the session runs.
    """

    baselines: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.baselines) < MIN_BASELINES:
            raise ValueError(f"the spread needs at least {MIN_BASELINES} baselines, got {len(self.baselines)}")
        for value in self.baselines:
            _checked(value)

    def with_baseline(self, value: float) -> DispersionThreshold:
        return replace(self, baselines=(*self.baselines, value))

    @property
    def spread_pct(self) -> float:
        """The relative standard deviation of one measurement, in percent."""
        return statistics.stdev(self.baselines) / statistics.fmean(self.baselines) * 100.0

    @property
    def threshold_pct(self) -> float:
        return max(FLOOR_PCT, NOISE_MULTIPLE * self.spread_pct)


def compare(
    throughput: float, accuracy_ok: bool, parent: Mapping[str, float], threshold: DispersionThreshold, request_id: str
) -> Judgement:
    """The first measurement against the parent's record: past the bar it waits for a retest whatever its accuracy,
    since the retest settles both whether the gain holds and whether a failed accuracy gate is real."""
    gain = _gain(throughput, parent)
    bar = threshold.threshold_pct
    recorded = {GRADED_OUTPUT: throughput}
    if gain > bar:
        return Judgement(Verdict.NEEDS_CONFIRM, gain, bar, (request_id,), recorded, accuracy_ok)
    verdict = Verdict.REGRESSED if gain < -bar else Verdict.FLAT
    return Judgement(verdict, gain, bar, (request_id,), recorded, accuracy_ok, (GAIN_BELOW_THRESHOLD,))


def confirm(
    first: Judgement,
    throughput: float,
    accuracy_ok: bool,
    parent: Mapping[str, float],
    threshold: DispersionThreshold,
    request_id: str,
) -> Judgement:
    """Settles a provisional gain with a retest on another sandbox; the record is the mean of both measurements.

    A gain both measurements cleared is improved when both passed the accuracy gate, blocked when both failed it and
    inconclusive when they split. Both are held to the bar as it stands now, which spot-checked baselines may have moved
    since the first measurement.
    """
    if first.verdict is not Verdict.NEEDS_CONFIRM:
        raise ValueError(f"only a provisional judgement is retested, not a {first.verdict.value} one")
    bar = threshold.threshold_pct
    mean = (first.recorded_value[GRADED_OUTPUT] + throughput) / 2.0
    gain = _gain(mean, parent)
    request_ids = (*first.request_ids, request_id)
    both_ok = first.accuracy_ok and accuracy_ok
    if first.gain_pct > bar and _gain(throughput, parent) > bar:
        if both_ok:
            verdict = Verdict.IMPROVED
        elif first.accuracy_ok or accuracy_ok:
            verdict = Verdict.INCONCLUSIVE
        else:
            verdict = Verdict.BLOCKED
        return Judgement(verdict, gain, bar, request_ids, {GRADED_OUTPUT: mean}, both_ok)
    verdict = Verdict.REGRESSED if gain < -bar else Verdict.FLAT
    return Judgement(verdict, gain, bar, request_ids, {GRADED_OUTPUT: mean}, both_ok, (GAIN_NOT_CONFIRMED,))


__all__ = [
    "FLOOR_PCT",
    "GAIN_BELOW_THRESHOLD",
    "GAIN_NOT_CONFIRMED",
    "MIN_BASELINES",
    "NOISE_MULTIPLE",
    "DispersionThreshold",
    "compare",
    "confirm",
]
