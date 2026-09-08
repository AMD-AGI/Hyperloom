# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Re-time a generated tuner's candidates with our own clock.

This is what makes a generated tuner safe to run at all: it may propose
configurations, and nothing it reports about their speed is used.

Each rule below replaced something that gave a wrong answer on this fleet first.
Warm the clocks, because the GPU idles at 94MHz and whatever is measured first
pays the ramp. Interleave baseline and candidate rather than timing them in
blocks, which put one default at 1269us against 517us the day before -- a 2.5x
swing owed to a neighbour. Take the minimum across repeats, not the median:
interference only ever adds time, and median-based runs rejected 9 of 16
measurements as unstable on spreads of 40-170%. Refuse a result whose best case
and typical case disagree about which side is faster, since the two sides were
then not measured under one machine state. And require a win to clear the noise
rather than merely be positive.

Dispatch is the caller's business: a candidate is only meaningful against the
backend it names, so this takes callables and never interprets a config.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

WARMUP_CALLS = 20
CALLS_PER_SAMPLE = 30
REPEATS = 9

#: How much faster a candidate has to read before we call it faster.
#:
#: Measured, not chosen. On an MI355X, ``time_paired`` was run 25 times per
#: shape with *the same* callable on both sides, over four shapes demanded by
#: the fleet (8192x3456x1152, 4096x4096x4096, 1024x1536x7168, 128x2048x2048).
#: The pairing rules above did most of the work -- 43 of the 100 trials were
#: refused outright as unstable -- but of the 57 that produced a number, 28
#: came out above 1.0. The unmodified path was judged an improvement over
#: itself roughly half the time it was judged at all, the largest such reading
#: being 1.00925x. Not one trial reached 1.01x.
#:
#: So this sits just above the whole observed null distribution rather than at
#: its edge, and the rule it implements is "beat the noise", not "be positive".
#: Without it a generated tuner that proposes the baseline backend verbatim is
#: verified as an improvement and deployed, which is the one outcome the
#: referee exists to make impossible.
MIN_SPEEDUP = 1.01


@dataclass(frozen=True)
class PairedTiming:
    """One baseline-versus-candidate comparison, with why it is trustworthy."""

    baseline_us: float
    candidate_us: float
    speedup: float | None
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.speedup is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_us": self.baseline_us,
            "candidate_us": self.candidate_us,
            "speedup": self.speedup,
            "usable": self.usable,
            "reason": self.reason,
        }


@dataclass
class Judgement:
    """What the referee concluded about one shape's candidates."""

    shape: str
    best: dict[str, Any] | None = None
    best_timing: PairedTiming | None = None
    timings: list[tuple[dict[str, Any], PairedTiming]] = field(default_factory=list)
    rejected_incorrect: int = 0

    @property
    def improved(self) -> bool:
        """Faster than the baseline by more than the baseline beats itself."""
        return bool(self.best_timing and self.best_timing.usable and (self.best_timing.speedup or 0) >= MIN_SPEEDUP)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "best": self.best,
            "best_timing": self.best_timing.to_dict() if self.best_timing else None,
            "improved": self.improved,
            "rejected_incorrect": self.rejected_incorrect,
            "candidates_timed": len(self.timings),
        }


def _sample(call: Callable[[], Any], sync: Callable[[], Any]) -> float:
    sync()
    t0 = time.perf_counter()
    for _ in range(CALLS_PER_SAMPLE):
        call()
    sync()
    return (time.perf_counter() - t0) / CALLS_PER_SAMPLE * 1e6


def time_paired(
    baseline: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    sync: Callable[[], Any] | None = None,
    repeats: int = REPEATS,
) -> PairedTiming:
    """Interleave the two and report the paired result, or why there is none."""
    sync = sync or (lambda: None)
    try:
        for _ in range(WARMUP_CALLS):
            baseline()
            candidate()
        sync()
    except Exception as exc:  # noqa: BLE001 - a candidate that cannot run is data
        return PairedTiming(0.0, 0.0, None, f"{type(exc).__name__}: {exc}")

    base_s: list[float] = []
    cand_s: list[float] = []
    try:
        for _ in range(max(repeats, 1)):
            base_s.append(_sample(baseline, sync))
            cand_s.append(_sample(candidate, sync))
    except Exception as exc:  # noqa: BLE001
        return PairedTiming(0.0, 0.0, None, f"{type(exc).__name__}: {exc}")

    mb, mc = min(base_s), min(cand_s)
    if mc <= 0 or mb <= 0:
        return PairedTiming(mb, mc, None, "a side measured no time at all")

    best_ratio = mb / mc
    typical_ratio = statistics.median(base_s) / statistics.median(cand_s)
    if (best_ratio - 1.0) * (typical_ratio - 1.0) < 0:
        return PairedTiming(
            mb,
            mc,
            None,
            f"unstable: best-case {best_ratio:.4f}x contradicts typical-case {typical_ratio:.4f}x",
        )
    return PairedTiming(mb, mc, best_ratio)


def judge_candidates(
    shape: str,
    candidates: list[dict[str, Any]],
    *,
    baseline: Callable[[], Any],
    dispatch: Callable[[dict[str, Any]], Callable[[], Any] | None],
    is_correct: Callable[[Callable[[], Any]], bool] | None = None,
    sync: Callable[[], Any] | None = None,
) -> Judgement:
    """Re-time one shape's candidates and pick the best that stands up."""
    result = Judgement(shape=shape)
    for cand in candidates:
        call = dispatch(cand)
        if call is None:
            result.timings.append((cand, PairedTiming(0.0, 0.0, None, "not dispatchable")))
            continue
        if is_correct is not None and not is_correct(call):
            result.rejected_incorrect += 1
            result.timings.append((cand, PairedTiming(0.0, 0.0, None, "failed the correctness check")))
            continue
        timing = time_paired(baseline, call, sync=sync)
        result.timings.append((cand, timing))
        if timing.usable and (
            result.best_timing is None
            or not result.best_timing.usable
            or timing.candidate_us < result.best_timing.candidate_us
        ):
            result.best, result.best_timing = cand, timing

    log.info(
        "referee %s: %d candidate(s), %d rejected as incorrect, best %s",
        shape,
        len(candidates),
        result.rejected_incorrect,
        f"{result.best_timing.speedup:.4f}x" if result.improved else "none",
    )
    return result
