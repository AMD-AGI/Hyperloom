# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which rounds' plans go to the plan critic.

``FORGE_PLAN_CRITIC`` chooses, for a campaign long enough to run the critic at all:

- ``always`` (the default): every round a synthesis produced is reviewed;
- ``after-blocked-round``: a round is reviewed unless it continues, in EXPLOIT mode, a route the previous round kept
  a candidate on and carries no challenge the last critic asked for;
- ``off``: no round is reviewed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from kernelforge.loop.search_policy import SEARCH_MODE_EXPLOIT

PLAN_CRITIC_ENV = "FORGE_PLAN_CRITIC"
# The critic verdict that makes the next round give one lane to the route it names.
CHALLENGE_VERDICT = "REPLACE"


class PlanCriticMode(str, Enum):
    ALWAYS = "always"
    AFTER_BLOCKED_ROUND = "after-blocked-round"
    OFF = "off"


@dataclass(frozen=True)
class PlanCriticSetting:
    """The campaign's plan-critic mode, and whether the campaign is long enough to run a critic."""

    mode: PlanCriticMode
    long_horizon: bool

    @classmethod
    def from_env(cls, *, long_horizon: bool, environ: Mapping[str, str] = os.environ) -> PlanCriticSetting:
        raw = environ.get(PLAN_CRITIC_ENV, "").strip()
        try:
            mode = PlanCriticMode(raw or PlanCriticMode.ALWAYS.value)
        except ValueError:
            choices = ", ".join(choice.value for choice in PlanCriticMode)
            raise ValueError(f"{PLAN_CRITIC_ENV}={raw!r} is not one of: {choices}") from None
        return cls(mode=mode, long_horizon=bool(long_horizon))

    @property
    def enabled(self) -> bool:
        """Whether the campaign builds a plan critic at all."""
        return self.long_horizon and self.mode is not PlanCriticMode.OFF

    @property
    def loop_mode(self) -> PlanCriticMode:
        """The mode the loop decides each round by: ``off`` for a campaign that builds no critic."""
        return self.mode if self.enabled else PlanCriticMode.OFF

    def describe(self) -> str:
        if not self.long_horizon:
            return "disabled (requires --max-hours > 2)"
        if self.mode is PlanCriticMode.OFF:
            return f"disabled ({PLAN_CRITIC_ENV}={self.mode.value})"
        if self.mode is PlanCriticMode.ALWAYS:
            return "enabled (long-horizon, same backend/model)"
        return f"enabled for rounds that do not continue a kept route ({PLAN_CRITIC_ENV}={self.mode.value})"


@dataclass(frozen=True)
class PlanReview:
    """Whether one round's plans are reviewed, and the rule that decided it."""

    review: bool
    reason: str


def decide_plan_review(
    mode: PlanCriticMode,
    *,
    previous_round_iteration: int | None,
    best_iteration: int,
    search_mode: str,
    last_critic_verdict: str,
) -> PlanReview:
    """Decide whether the round about to plan sends its plans to the critic.

    ``previous_round_iteration`` is the first iteration of the last round that planned, or ``None`` before any did;
    ``best_iteration`` is the iteration of the current best, which a KEEP at or after that round's first iteration
    moved there.
    """
    if mode is PlanCriticMode.ALWAYS:
        return PlanReview(review=True, reason="always")
    if mode is PlanCriticMode.OFF:
        return PlanReview(review=False, reason="off")
    if previous_round_iteration is None:
        return PlanReview(review=True, reason="first_round")
    if last_critic_verdict == CHALLENGE_VERDICT:
        return PlanReview(review=True, reason="challenge_requested")
    if search_mode != SEARCH_MODE_EXPLOIT:
        return PlanReview(review=True, reason="search_mode_diversify")
    if best_iteration < previous_round_iteration:
        return PlanReview(review=True, reason="previous_round_blocked")
    return PlanReview(review=False, reason="previous_round_kept")


def review_seconds(phase_durations_sec: Mapping[str, float]) -> float:
    """What a round spent on the critic and the revision it asked for."""
    return sum(float(phase_durations_sec.get(phase, 0.0)) for phase in ("plan_critic", "plan_revision"))
