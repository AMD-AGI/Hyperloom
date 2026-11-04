# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which rounds' plans go to the plan critic.

A campaign that builds a critic reviews its first round, and a later round only when the previous round kept no
candidate of its own, the search is in DIVERSIFY mode, or the last critic asked for a challenger lane. A round that
continues, in EXPLOIT mode, a route its predecessor kept a candidate on publishes its drafts unreviewed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from kernelforge.workflows.forge_loop.search.search_policy import SEARCH_MODE_EXPLOIT

# The critic verdict that makes the next round give one lane to the route it names.
CHALLENGE_VERDICT = "REPLACE"


@dataclass(frozen=True)
class PlanReview:
    """Whether one round's plans are reviewed, and the rule that decided it."""

    review: bool
    reason: str


def decide_plan_review(
    *,
    previous_round_kept: bool | None,
    search_mode: str,
    last_critic_verdict: str,
) -> PlanReview:
    """Decide whether the round about to plan sends its plans to the critic.

    ``previous_round_kept`` is whether the last round that planned kept a candidate of its own, or ``None`` before any
    round planned.
    """
    if previous_round_kept is None:
        return PlanReview(review=True, reason="first_round")
    if last_critic_verdict == CHALLENGE_VERDICT:
        return PlanReview(review=True, reason="challenge_requested")
    if search_mode != SEARCH_MODE_EXPLOIT:
        return PlanReview(review=True, reason="search_mode_diversify")
    if not previous_round_kept:
        return PlanReview(review=True, reason="previous_round_blocked")
    return PlanReview(review=False, reason="previous_round_kept")


def review_seconds(phase_durations_sec: Mapping[str, float]) -> float:
    """What a round spent on the critic and the revision it asked for."""
    return sum(float(phase_durations_sec.get(phase, 0.0)) for phase in ("plan_critic", "plan_revision"))
