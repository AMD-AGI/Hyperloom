# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decide whether a throughput measurement has converged.

A real run measured the same configuration three times: 14,202.70 -> 19,373.98
-> 22,424.80 tok/s. A 58% spread, monotonically rising -- the measurement window
sat on the warm-up climb, not on steady state. At that noise level a 3% KEEP
threshold cannot separate anything, so every number downstream of it, including
the ones used to argue about it, is unusable.

A controlled repeat pinned the cause. One resident vLLM server, five *identical*
benchmark passes:

    round 1: 63.90 req/s   (TTFT 1906.52 ms)
    round 2: 133.85        (415.12)
    round 3: 139.04        (363.99)
    round 4: 137.29        (368.15)
    round 5: 117.13        (621.12)

Full spread 117.6%. Drop round 1 -- whose TTFT is 5x the rest, i.e. plainly cold
start -- and rounds 2-4 span 3.9%. So the fix is not a looser threshold, which
would let the genuine climb through as well; it is to discard the warm-up round
and then require consecutive rounds to agree.

Round 5 falling back to 117.13 is the other lesson: that box was shared, and
another workload arrived. Convergence on one side is necessary but not
sufficient -- paired alternating measurement is what removes drift between the
A and B legs, and it lives in its own module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Same order of magnitude as the KEEP threshold: a measurement that cannot
# resolve the decision it feeds is not converged.
DEFAULT_TOLERANCE_PCT = 3.0

# Rounds discarded before judging. The observed cold-start round on its own took
# the spread from 3.9% to 117.6%.
DEFAULT_WARMUP_ROUNDS = 1

# Below this many usable rounds there is nothing to compare against.
MIN_ROUNDS_FOR_VERDICT = 2

# A rising pair is not a trend: with two noisy samples, half of all steady
# measurements rise. Claiming "still warming up" needs three points; with two,
# spread alone decides.
MIN_ROUNDS_FOR_TREND = 3


@dataclass(frozen=True)
class ConvergenceVerdict:
    """Why a series was accepted or rejected, with the numbers behind it."""

    converged: bool
    reason: str
    value: float | None
    used: list[float] = field(default_factory=list)
    discarded: list[float] = field(default_factory=list)
    spread_pct: float | None = None
    monotonic: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "reason": self.reason,
            "value": self.value,
            "rounds_used": list(self.used),
            "rounds_discarded": list(self.discarded),
            "spread_pct": self.spread_pct,
            "monotonic_increasing": self.monotonic,
        }


def _spread_pct(values: list[float]) -> float | None:
    """Max-to-min spread as a percentage of the minimum."""
    usable = [v for v in values if v > 0]
    if len(usable) < 2:
        return None
    lo, hi = min(usable), max(usable)
    return (hi - lo) / lo * 100.0


def _is_monotonic_increasing(values: list[float]) -> bool:
    """True only for a series long enough for a rise to mean something."""
    return len(values) >= MIN_ROUNDS_FOR_TREND and all(b > a for a, b in zip(values, values[1:], strict=False))


def assess_convergence(
    rounds: list[float],
    *,
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
    warmup_rounds: int = DEFAULT_WARMUP_ROUNDS,
) -> ConvergenceVerdict:
    """Judge a throughput series measured under one unchanged configuration.

    Args:
        rounds: Throughputs in chronological order.
        tolerance_pct: Allowed spread across the retained rounds.
        warmup_rounds: Leading rounds discarded before judging.

    Returns:
        A verdict carrying every round, so a reader can audit the call rather
        than trusting a single surviving number.
    """
    series = [float(r) for r in rounds if isinstance(r, (int, float))]
    positive = [r for r in series if r > 0]
    if not positive:
        return ConvergenceVerdict(False, "no_measurements", None, [], series)

    discarded = positive[:warmup_rounds]
    used = positive[warmup_rounds:]
    if len(used) < MIN_ROUNDS_FOR_VERDICT:
        # One usable round cannot be shown to be steady. Saying so beats
        # reporting it as if it were.
        return ConvergenceVerdict(
            False,
            "insufficient_rounds",
            None,
            used,
            discarded,
            spread_pct=_spread_pct(used),
        )

    spread = _spread_pct(used)
    monotonic = _is_monotonic_increasing(used)

    if monotonic:
        # Still climbing: the last round is the least settled, so taking it
        # would systematically overstate the result.
        return ConvergenceVerdict(
            False,
            "monotonic_increasing",
            None,
            used,
            discarded,
            spread_pct=spread,
            monotonic=True,
        )
    if spread is not None and spread > tolerance_pct:
        return ConvergenceVerdict(
            False,
            "spread_exceeds_tolerance",
            None,
            used,
            discarded,
            spread_pct=spread,
        )

    value = sum(used) / len(used)
    return ConvergenceVerdict(
        True,
        "converged",
        value,
        used,
        discarded,
        spread_pct=spread,
    )
