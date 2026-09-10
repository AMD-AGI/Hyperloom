# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Re-time a generated tuner's candidates with our own clock."""

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
        return bool(self.best_timing and self.best_timing.usable and (self.best_timing.speedup or 0) > 1.0)

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
