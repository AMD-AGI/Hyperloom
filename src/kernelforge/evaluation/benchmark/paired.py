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
_TAIL_LINES = 8
# Further runs of an arm that failed, each its own process: the profiler under SIKL's timer fails a process now and
# then (an empty Kineto trace, a hang) and keeps failing within it, while a fresh process measures.
ARM_RETRIES = 2


@dataclass(frozen=True)
class PairRecord:
    """One pair: which arm ran first, both arms' results, the equal-weight mean of incumbent/candidate, and every
    failed run of an arm that a fresh process then measured."""

    order: tuple[str, str]
    candidate: dict
    incumbent: dict
    case_ratios: dict[str, float]
    score: float
    retries: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairedMeasurement:
    """Every pair timed, or why the measurement stopped."""

    pairs: tuple[PairRecord, ...] = ()
    failure: str = ""

    @property
    def scores(self) -> list[float]:
        return [pair.score for pair in self.pairs]

    @property
    def retries(self) -> list[str]:
        return [retry for pair in self.pairs for retry in pair.retries]

    def extended(self, other: PairedMeasurement) -> PairedMeasurement:
        return PairedMeasurement(self.pairs + other.pairs, other.failure)

    def to_dict(self) -> dict:
        return {
            "pairs": [asdict(pair) for pair in self.pairs],
            "scores": self.scores,
            "mean": statistics.fmean(self.scores) if self.pairs else None,
            "failure": self.failure,
        }


def _arm_failure(arm: str, result: dict) -> str:
    """What an arm's failed run says: its message, then the last lines of its output."""
    detail = result.get("message") or "failed"
    tail = "\n".join(str(result.get("output") or "").strip().splitlines()[-_TAIL_LINES:])
    return f"{arm} arm: {detail}" + (f"\n{tail}" if tail else "")


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
    arm_retries: int = ARM_RETRIES,
) -> PairedMeasurement:
    """Time ``pairs`` pairs; an arm that fails runs again in a fresh process up to ``arm_retries`` times, and the
    measurement stops at the first arm that never succeeds, saying which and why."""
    records: list[PairRecord] = []
    arms = {CANDIDATE: candidate, INCUMBENT: incumbent}
    for _ in range(pairs):
        order = (CANDIDATE, INCUMBENT) if rng.random() < 0.5 else (INCUMBENT, CANDIDATE)
        results: dict[str, dict] = {}
        retries: list[str] = []
        for arm in order:
            for _attempt in range(1 + arm_retries):
                results[arm] = await bench_wallclock(arms[arm], timeout_sec=timeout_sec)
                if results[arm].get("success"):
                    break
                retries.append(_arm_failure(arm, results[arm]))
            else:
                runs = 1 + arm_retries
                return PairedMeasurement(
                    tuple(records),
                    f"{retries[-1]}\n(failed in each of {runs} fresh processes)" if runs > 1 else retries[-1],
                )
        ratios = _pair_ratios(results[CANDIDATE], results[INCUMBENT], scored)
        if isinstance(ratios, str):
            return PairedMeasurement(tuple(records), ratios)
        records.append(
            PairRecord(
                order,
                results[CANDIDATE],
                results[INCUMBENT],
                ratios,
                statistics.fmean(ratios.values()),
                tuple(retries),
            )
        )
    return PairedMeasurement(tuple(records))
