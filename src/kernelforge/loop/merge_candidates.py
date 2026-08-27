# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pick two rejected candidates whose wins do not overlap, to measure stacked.

A REVERT that still beat the incumbent is a measured gain the gate turned down.
Forge produces one candidate per iteration and drops the rest, so those gains
are never combined even when they improve different cases and would clear the
gate together. Stacking is only ever a hypothesis: additivity has no predictable
sign, so a selected pair is measured under the ordinary KEEP protocol and is
worth nothing until it is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import statistics


MERGE_PLAN_PREFIX = "stacked candidates"

# Consecutive iterations without a KEEP -- the stall only a real result clears,
# not the supervisor cooldown an intervention resets -- before stacking is worth
# trying. A stacked attempt spends no Implementer session, so it can be reached
# before the supervisor's own stall threshold, but it needs a field of rejected
# gains to choose from first.
MERGE_ATTEMPT_STALL_THRESHOLD = 2

# Consecutive iterations a stack may hold before one must go to the queue.
#
# A stacked attempt neither drains the queue nor resolves the stall that
# admitted it -- it reverts, which leaves ``unresolved_stall_iters`` higher than
# it found it -- so nothing about running one makes the next one less likely. A
# streak runs for as long as the archive still offers a distinct, mutually
# complementary, not-yet-attempted pair.
#
# That field does not grow while a streak runs. The only candidate a stacked
# iteration archives is the stack, and :func:`eligible_candidates` skips any
# candidate whose plan carries ``MERGE_PLAN_PREFIX``, so the pool is frozen at
# whatever the streak started with and each iteration spends one pair out of it.
# A streak is therefore already finite without this constant -- but only at the
# size of the pool's pair set, which goes as the square of the pool: across the
# thirty archived forge runs of 2026-08-22 and 08-23 the eligible pool reaches 8
# candidates, and 8 candidates admit 28 pairs.
#
# What the archive exhibits is far short of that. Replayed with both the
# held-plan guard and this precedence rule in place, 9 stacks fire over the 549
# iterations, in 9 streaks of one iteration each; the deepest streak the pool
# could have sustained at any of the 121 iterations reaching the stall gate is
# 3. So the limit is set at 2: it refuses none of the 9 firings the archive
# exhibits, and it replaces a square-law tail with a constant.
#
# The reachability the limit restores does not rest on its value -- that
# argument is at ``IterationLoop._merge_attempt_refusal`` and holds for any
# finite limit. It is not a cap on firings either: a refused pair is not staged,
# is not remembered against, and is selected again an iteration later.
MERGE_PRECEDENCE_STREAK_LIMIT = 2


@dataclass(frozen=True)
class MergeCandidate:
    """One archived candidate that took a case from the incumbent, unkept."""

    iteration: int
    plan: str
    mean_case_speedup: float
    winning_cases: frozenset[str]


def merge_plan(first: MergeCandidate, second: MergeCandidate) -> str:
    """The plan text a stacked attempt is recorded under.

    The pair is named in the plan because that archived line is what tells a
    later iteration the combination was already measured. Keeping the record in
    the archive rather than in control state means a resumed run sees the same
    history without a state field to migrate.
    """
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
    """Each case's run-to-run spread across one candidate's own measurements.

    The sample standard deviation, matching
    :func:`~kernelforge.loop.scoring.measurement_sigma`: it is the same
    quantity asked of a different level of the measurement. A case seen fewer
    than twice has no spread and is omitted, which makes it unwinnable below --
    an unmeasured case is not evidence of a gain.
    """
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
    """The cases this candidate ran faster than *reference* by more than noise.

    This is both the admission test and the ownership measure: whether a
    rejected candidate demonstrated a real per-case gain over the incumbent, and
    which cases it therefore brings to a stack.

    "By more than noise" is the case's own spread across the candidate's
    measurements, from :func:`case_spreads`. A case whose spread is unknown or
    exactly zero cannot be won. Unknown, because one timing of a case is not a
    measurement of it; zero, because three byte-identical timings mean the
    driver's resolution swallowed the case rather than that it is noiseless, and
    admitting a win against zero noise is the one-lucky-draw failure this test
    exists to prevent.
    """
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
    """Archived candidates worth stacking, newest last.

    A rejected candidate is worth stacking when it took at least one scored case
    from the *incumbent* by more than that case's own measured spread. The
    equal-weight mean is what the arena scores, so it stays the KEEP gate's
    business; but a candidate that won 2% on one of the two cases carrying a
    campaign's whole deficit has measured something real, and dropping it
    because a third case moved the other way throws away the only evidence the
    run produced.

    The same measurement is the candidate's ownership -- the ``winning_cases``
    field the pair selector reasons over. Taking ownership against the pristine
    baseline instead is why this mechanism never ran: across the thirty archived
    forge runs of 2026-08-22 and 08-23, 549 iterations, the marker ``stacked
    candidates`` appears in no log. Once a campaign has banked a few KEEPs almost
    every candidate beats pristine on almost every case, so the
    pristine-relative sets come out nested or identical, and the selector's
    mutual-complementarity test -- the thing that separates a stack worth
    measuring from a re-run of the better half -- can never be satisfied.
    Replaying those same archives against the incumbent yields a complementary
    pair on 45 iterations where pristine yields one on 23.

    Measuring against the incumbent is not a relaxation of what a case has to
    show. The pristine comparison asked only that the number be smaller;
    :func:`cases_beating_reference` requires the gain to clear that case's own
    run-to-run spread, which is strictly harder, and it is asked against a
    moving reference, so a gain the campaign has since banked stops counting
    without a separate staleness test. What it gives up is the candidate whose
    only distinguishable ground is ground the incumbent already holds, and such
    a candidate has nothing a stack can add: on the archives that costs one pair
    against twenty-three gained.
    """
    found: list[MergeCandidate] = []
    for meta in metas:
        if str(meta.get("decision") or "") != "REVERT_PERF":
            continue
        if str(meta.get("plan") or "").strip().startswith(MERGE_PLAN_PREFIX):
            # A stack that reverted is archived like any other candidate, so
            # without this a later pair selects it and measures three diffs
            # under a record that names two -- which is the one thing the
            # staged/kept counts are for. Capping at two costs nothing the
            # archives show: across the thirty runs of 2026-08-22 and 08-23 a
            # mutually-complementary triple exists at 2 of the 121 consulted
            # iterations, and at neither does it cover more cases than the best
            # available pair.
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
    """The pair covering the most cases where each owns ground the other loses.

    Mutual complementarity is the requirement rather than a preference: if one
    side's wins are a subset of the other's, the stack re-measures what the
    better candidate already showed. Ties break on combined speedup and then on
    the earlier iterations, and the ordering is total, so two runs reading one
    archive choose the same pair in the same order however it reaches them.
    """
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
