# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pick two rejected candidates whose wins do not overlap, to measure stacked."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import statistics


MERGE_PLAN_PREFIX = "stacked candidates"

# Consecutive iterations without a KEEP -- the stall only a real result clears, not the supervisor cooldown an
# intervention resets -- before stacking is worth trying.
MERGE_ATTEMPT_STALL_THRESHOLD = 2

# Consecutive iterations a stack may hold before one must go to the queue.
MERGE_PRECEDENCE_STREAK_LIMIT = 2


@dataclass(frozen=True)
class MergeCandidate:
    """One archived candidate that took a case from the incumbent, unkept."""

    iteration: int
    plan: str
    mean_case_speedup: float
    winning_cases: frozenset[str]


def merge_plan(first: MergeCandidate, second: MergeCandidate) -> str:
    """The plan text a stacked attempt is recorded under."""
    low, high = sorted((first.iteration, second.iteration))
    return f"{MERGE_PLAN_PREFIX} {low}+{high}: {first.plan} | {second.plan}"


def attempted_pairs(plans: list[str]) -> frozenset[frozenset[int]]:
    """The iteration pairs already measured stacked, read back from plan text."""
    found: set[frozenset[int]] = set()
    for plan in plans:
        text = str(plan or "").strip()
        if not text.startswith(MERGE_PLAN_PREFIX):
            continue
        head = text[len(MERGE_PLAN_PREFIX) :].split(":", 1)[0].strip()
        parts = head.split("+")
        if len(parts) != 2:
            continue
        try:
            found.add(frozenset({int(parts[0]), int(parts[1])}))
        except ValueError:
            continue
    return frozenset(found)


def case_spreads(
    measurements: Sequence[dict] | None,
) -> dict[str, float]:
    """Each case's run-to-run spread across one candidate's own measurements."""
    per_case: dict[str, list[float]] = {}
    for measurement in measurements or ():
        if not isinstance(measurement, dict):
            continue
        for case_id, value in (measurement.get("case_times") or {}).items():
            if not isinstance(value, (int, float)) or float(value) <= 0.0:
                continue
            per_case.setdefault(str(case_id), []).append(float(value))
    return {case_id: statistics.stdev(times) for case_id, times in per_case.items() if len(times) >= 2}


def cases_beating_reference(
    case_times: Mapping[str, float],
    reference_case_times: Mapping[str, float],
    spreads: Mapping[str, float],
) -> frozenset[str]:
    """The cases this candidate ran faster than *reference* by more than noise."""
    owned: set[str] = set()
    for case_id, reference in reference_case_times.items():
        measured = case_times.get(case_id)
        spread = spreads.get(str(case_id))
        if not isinstance(reference, (int, float)) or float(reference) <= 0.0:
            continue
        if not isinstance(measured, (int, float)) or float(measured) <= 0.0:
            continue
        if spread is None or float(spread) <= 0.0:
            continue
        if float(measured) < float(reference) - float(spread):
            owned.add(str(case_id))
    return frozenset(owned)


def eligible_candidates(
    metas: list[dict],
    incumbent_case_times: Mapping[str, float],
) -> list[MergeCandidate]:
    """Archived candidates worth stacking, newest last."""
    found: list[MergeCandidate] = []
    for meta in metas:
        if str(meta.get("decision") or "") != "REVERT_PERF":
            continue
        if str(meta.get("plan") or "").strip().startswith(MERGE_PLAN_PREFIX):
            # A stack that reverted is archived like any other candidate, so without this a later pair selects it and
            # measures three diffs under a record that names two -- which is the one thing the staged/kept counts are
            # for.
            continue
        score = meta.get("mean_case_speedup")
        if not isinstance(score, (int, float)):
            continue
        bench = meta.get("bench") if isinstance(meta.get("bench"), dict) else {}
        owned = cases_beating_reference(
            dict(bench.get("case_times") or {}),
            incumbent_case_times,
            case_spreads(bench.get("measurements")),
        )
        if not owned:
            continue
        found.append(
            MergeCandidate(
                iteration=int(meta.get("iteration") or 0),
                plan=str(meta.get("plan") or ""),
                mean_case_speedup=float(score),
                winning_cases=owned,
            )
        )
    return sorted(found, key=lambda item: item.iteration)


def select_merge_pair(
    candidates: list[MergeCandidate],
    *,
    already_attempted: frozenset[frozenset[int]] = frozenset(),
) -> tuple[MergeCandidate, MergeCandidate] | None:
    """The pair covering the most cases where each owns ground the other loses."""
    ordered = sorted(candidates, key=lambda item: item.iteration)
    chosen: tuple[tuple, tuple[MergeCandidate, MergeCandidate]] | None = None
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if frozenset({first.iteration, second.iteration}) in already_attempted:
                continue
            if not (first.winning_cases - second.winning_cases):
                continue
            if not (second.winning_cases - first.winning_cases):
                continue
            key = (
                len(first.winning_cases | second.winning_cases),
                first.mean_case_speedup + second.mean_case_speedup,
                -first.iteration,
                -second.iteration,
            )
            if chosen is None or key > chosen[0]:
                chosen = (key, (first, second))
    return chosen[1] if chosen else None
