# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Divide the KERNEL phase's remaining time between the lanes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

LANE_FUSION = "fusion"
LANE_GEMM = "gemm"

#: gemm: a tuner's own estimate is the cost, and the router supplies it per
#: tuner. This is only the fallback for a tuner that reports none.
GEMM_DEFAULT_TARGET_SEC = 20 * 60

#: fusion: no admission floor and no per-recipe estimate, so its ceiling is a
#: count rather than a division. Travels to forge-fuse as ``--max-recipes``.
FUSION_MAX_TARGETS = 3

#: Held back from the phase share so the lanes cannot consume the time the phase
#: itself needs to close out.
PHASE_RESERVE_SEC = 300

#: Why KERNEL work did not start: what is left of a bounded phase cannot fund one run of it.
#: A skip, not a verdict on the work: a later KERNEL entry may still fund it.
INSUFFICIENT_BUDGET = "insufficient_budget"

#: Rewrite is not a lane: the route budgets itself against the wall clock (see
#: ``_kernel_rewrite_controller_timeouts``), so its half comes off the top and
#: never becomes an allocation the caller has to discard.
REWRITE_RESERVE_SHARE = 0.5

#: Default split of what is left for the lanes once rewrite has taken its share.
DEFAULT_LANE_WEIGHTS: Mapping[str, float] = {
    LANE_FUSION: 0.6,
    LANE_GEMM: 0.4,
}


class LaneBudgetError(ValueError):
    """A budget that cannot be derived is a programming error, not a skip."""


@dataclass(frozen=True)
class LaneAllocation:
    """One lane's share, and how many targets that share can actually pay for."""

    budget_sec: int
    max_targets: int

    @property
    def is_fundable(self) -> bool:
        """Whether the share pays for at least one target."""
        return self.max_targets > 0


def phase_budget_sec(remaining_minutes: object) -> int:
    """Convert time left in the phase into the seconds the lanes may divide."""
    if remaining_minutes is None:
        return 0
    if isinstance(remaining_minutes, bool) or not isinstance(remaining_minutes, (int, float)):
        raise LaneBudgetError(f"remaining minutes must be numeric, got {remaining_minutes!r}")
    minutes = float(remaining_minutes)
    if not math.isfinite(minutes) or minutes < 0:
        raise LaneBudgetError(f"remaining minutes must be finite and non-negative, got {minutes}")
    return max(0, int(minutes * 60) - PHASE_RESERVE_SEC)


def split_lanes(phase_sec: int) -> dict[str, int]:
    """Divide a phase budget between lanes by :data:`DEFAULT_LANE_WEIGHTS`."""
    total = sum(DEFAULT_LANE_WEIGHTS.values())
    return {lane: int(phase_sec * (weight / total)) for lane, weight in DEFAULT_LANE_WEIGHTS.items()}


def max_targets(
    lane: str,
    budget_sec: int,
    *,
    target_costs_sec: tuple[int, ...] = (),
) -> int:
    """How many targets a lane may pick with the budget it was given."""
    if lane == LANE_GEMM:
        return _greedy_fit(budget_sec, target_costs_sec)
    return FUSION_MAX_TARGETS if budget_sec > 0 else 0


def allocate(
    remaining_minutes: object,
    *,
    gemm_target_costs_sec: tuple[int, ...] = (),
) -> dict[str, LaneAllocation]:
    """Derive every lane's share and target ceiling in one pass."""
    phase_sec = phase_budget_sec(remaining_minutes)
    lane_sec = phase_sec - int(phase_sec * REWRITE_RESERVE_SHARE)
    shares = split_lanes(lane_sec)
    return {
        lane: LaneAllocation(
            budget_sec=budget,
            max_targets=max_targets(lane, budget, target_costs_sec=gemm_target_costs_sec),
        )
        for lane, budget in shares.items()
    }


#: What one tuner may take, as a fraction of the whole gemm session. Strictly
#: below 1 so the producer's own ``min(per_tuner, remaining)`` actually bounds
#: something: at 1 the first tuner could consume the session and every later one
#: was skipped for lack of time.
GEMM_PER_TUNER_SHARE = 0.5


def gemm_per_tuner_timeout_sec(global_timeout_sec: object) -> int:
    """Cap one gemm tuner below the session budget it is drawn from."""
    if isinstance(global_timeout_sec, bool) or not isinstance(global_timeout_sec, (int, float)):
        return 0
    total = int(global_timeout_sec)
    if total <= 0:
        return 0
    # Strictly below the global cap for any session of two seconds or more.
    return max(1, min(total - 1, int(total * GEMM_PER_TUNER_SHARE)))


def _greedy_fit(budget_sec: int, costs_sec: tuple[int, ...]) -> int:
    """Count how many estimates fit, in order, without exceeding the budget."""
    if not costs_sec:
        return budget_sec // GEMM_DEFAULT_TARGET_SEC
    remaining = budget_sec
    fitted = 0
    for cost in costs_sec:
        price = int(cost) if not isinstance(cost, bool) and isinstance(cost, (int, float)) else 0
        price = price if price > 0 else GEMM_DEFAULT_TARGET_SEC
        if price > remaining:
            break
        remaining -= price
        fitted += 1
    return fitted
