# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Time two kernels side by side: each pair runs both arms back to back, in an order a coin decides.

A score compared with one measured hours earlier carries every drift in between -- clocks, temperature, a
neighbour on the node. Timing the incumbent in the same window as the candidate cancels what both arms share, so
the ratio of the pair is the candidate's own effect.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass

from kernelforge.evaluation.benchmark.runner import BenchRun, bench_wallclock

CANDIDATE, INCUMBENT = "candidate", "incumbent"


@dataclass(frozen=True)
class PairRecord:
    """One pair: which arm ran first, both arms' results, and the equal-weight mean of incumbent/candidate."""

    order: tuple[str, str]
    candidate: dict
    incumbent: dict
    case_ratios: dict[str, float]
    score: float


@dataclass(frozen=True)
class PairedMeasurement:
    """Every pair timed, or why the measurement stopped."""

    pairs: tuple[PairRecord, ...] = ()
    failure: str = ""

    @property
    def scores(self) -> list[float]:
        return [pair.score for pair in self.pairs]

    def extended(self, other: PairedMeasurement) -> PairedMeasurement:
        return PairedMeasurement(self.pairs + other.pairs, other.failure)

    def to_dict(self) -> dict:
        return {
            "pairs": [asdict(pair) for pair in self.pairs],
            "scores": self.scores,
            "mean": statistics.fmean(self.scores) if self.pairs else None,
            "failure": self.failure,
        }


def _pair_ratios(candidate: dict, incumbent: dict, scored: frozenset[str] | None) -> dict[str, float] | str:
    """Per-case incumbent/candidate, or what kept the pair from being compared."""
    candidate_times, incumbent_times = candidate.get("case_times") or {}, incumbent.get("case_times") or {}
    if set(candidate_times) != set(incumbent_times):
        return f"the arms timed different cases: {sorted(set(candidate_times) ^ set(incumbent_times))}"
    cases = sorted(case for case in candidate_times if scored is None or case in scored)
    if not cases:
        return "the arms timed no scored case"
    if any(float(candidate_times[case]) <= 0 or float(incumbent_times[case]) <= 0 for case in cases):
        return "an arm reported a non-positive case time"
    return {case: float(incumbent_times[case]) / float(candidate_times[case]) for case in cases}


async def measure_pairs(
    candidate: BenchRun,
    incumbent: BenchRun,
    *,
    pairs: int,
    rng: random.Random,
    timeout_sec: int,
    scored: frozenset[str] | None = None,
) -> PairedMeasurement:
    """Time ``pairs`` pairs; stop at the first arm that fails, and say which."""
    records: list[PairRecord] = []
    arms = {CANDIDATE: candidate, INCUMBENT: incumbent}
    for _ in range(pairs):
        order = (CANDIDATE, INCUMBENT) if rng.random() < 0.5 else (INCUMBENT, CANDIDATE)
        results = {}
        for arm in order:
            results[arm] = await bench_wallclock(arms[arm], timeout_sec=timeout_sec)
            if not results[arm].get("success"):
                detail = results[arm].get("message") or "failed"
                return PairedMeasurement(tuple(records), f"{arm} arm: {detail}")
        ratios = _pair_ratios(results[CANDIDATE], results[INCUMBENT], scored)
        if isinstance(ratios, str):
            return PairedMeasurement(tuple(records), ratios)
        records.append(
            PairRecord(order, results[CANDIDATE], results[INCUMBENT], ratios, statistics.fmean(ratios.values()))
        )
    return PairedMeasurement(tuple(records))
