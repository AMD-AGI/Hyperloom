# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Divide the KERNEL phase's remaining time between the lanes.

The phase currently has no allocation at all: each lane carries its own timeout,
those timeouts sum to far more than the phase ever gets, and whoever runs last is
starved by whoever ran first. Once forge decides how many kernels to work on, it
needs to be told a real number instead of inheriting a per-kernel default that
was never divided by anything.

Two decisions live here:

* **How much each lane gets.** Derived from time actually left, following the
  one lane that already does this correctly rather than a fixed per-kernel
  figure.
* **How many targets a lane may pick.** Its share divided by what one target
  costs at minimum. Overshooting is not a small loss: below its admission floor
  a target produces nothing at all, so picking one too many can cost the entire
  lane's budget rather than a fraction of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

LANE_REWRITE = "rewrite"
LANE_FUSION = "fusion"
LANE_GEMM = "gemm"

#: Rewrite: the producer's own floor plus the reserve the caller keeps back for
#: apply-back. Below this sum the route declines outright, so it is the divisor
#: for how many kernels fit.
REWRITE_MIN_TARGET_SEC = 4500

#: gemm: a tuner's own estimate is the cost, and the router supplies it per
#: tuner. This is only the fallback for a tuner that reports none.
GEMM_DEFAULT_TARGET_SEC = 20 * 60

#: fusion: no admission floor and no per-recipe estimate exist, so its ceiling is
#: a count rather than a division. Mirrors ``LoopConfig.max_recipes``' own default,
#: which the forge-fuse path overrides with the full discovered recipe count.
FUSION_MAX_TARGETS = 3

#: Held back from the phase share so the lanes cannot consume the time the phase
#: itself needs to close out.
PHASE_RESERVE_SEC = 300

#: Default split. Rewrite is weighted highest because its targets are the most
#: expensive and the only ones with a hard admission floor.
DEFAULT_LANE_WEIGHTS: Mapping[str, float] = {
    LANE_REWRITE: 0.5,
    LANE_FUSION: 0.3,
    LANE_GEMM: 0.2,
}


class LaneBudgetError(ValueError):
    """A budget that cannot be derived is a programming error, not a skip."""


@dataclass(frozen=True)
class LaneAllocation:
    """One lane's share, and how many targets that share can actually pay for."""

    lane: str
    budget_sec: int
    max_targets: int

    @property
    def is_fundable(self) -> bool:
        """Whether the share pays for at least one target."""
        return self.max_targets > 0


def phase_budget_sec(remaining_minutes: object, *, reserve_sec: int = PHASE_RESERVE_SEC) -> int:
    """Convert time left in the phase into the seconds the lanes may divide.

    Args:
        remaining_minutes: Minutes left, as reported by session state. ``None``
            means an unbounded session.
        reserve_sec: Held back for the phase to close out.

    Returns:
        Seconds available to the lanes, never negative. An unbounded session
        yields ``0``, which callers read as "no allocation to make".

    Raises:
        LaneBudgetError: If the reported remaining time is not a finite,
            non-negative number.
    """
    if remaining_minutes is None:
        return 0
    if isinstance(remaining_minutes, bool) or not isinstance(remaining_minutes, (int, float)):
        raise LaneBudgetError(f"remaining minutes must be numeric, got {remaining_minutes!r}")
    minutes = float(remaining_minutes)
    if not math.isfinite(minutes) or minutes < 0:
        raise LaneBudgetError(f"remaining minutes must be finite and non-negative, got {minutes}")
    return max(0, int(minutes * 60) - max(0, reserve_sec))


def split_lanes(
    phase_sec: int,
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, int]:
    """Divide a phase budget between lanes by weight.

    Args:
        phase_sec: Seconds available to the lanes.
        weights: Lane weights; need not sum to one, they are normalized.

    Returns:
        Seconds per lane, floored so the parts never exceed the whole.

    Raises:
        LaneBudgetError: On a negative budget, an unknown lane, or weights that
            are not positive and finite.
    """
    if isinstance(phase_sec, bool) or not isinstance(phase_sec, int) or phase_sec < 0:
        raise LaneBudgetError(f"phase budget must be a non-negative int, got {phase_sec!r}")
    resolved = dict(DEFAULT_LANE_WEIGHTS if weights is None else weights)
    if not resolved:
        raise LaneBudgetError("at least one lane weight is required")
    total = 0.0
    for lane, weight in resolved.items():
        if lane not in {LANE_REWRITE, LANE_FUSION, LANE_GEMM}:
            raise LaneBudgetError(f"unknown lane {lane!r}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise LaneBudgetError(f"weight for {lane!r} must be numeric, got {weight!r}")
        value = float(weight)
        if not math.isfinite(value) or value <= 0:
            raise LaneBudgetError(f"weight for {lane!r} must be finite and positive, got {value}")
        total += value
    return {lane: int(phase_sec * (float(weight) / total)) for lane, weight in resolved.items()}


def max_targets(
    lane: str,
    budget_sec: int,
    *,
    target_costs_sec: tuple[int, ...] = (),
) -> int:
    """How many targets a lane may pick with the budget it was given.

    Each lane measures a target differently, so each gets its own rule rather
    than a shared division:

    * rewrite divides by a hard admission floor;
    * gemm consumes per-tuner estimates in the order the router gave them,
      because a tuner that does not fit is skipped rather than shortened;
    * fusion has neither a floor nor an estimate, so it is capped by count.

    Args:
        lane: One of the module's lane constants.
        budget_sec: Seconds allotted to this lane.
        target_costs_sec: Per-target estimates, used by the gemm rule.

    Returns:
        The ceiling, ``0`` when the budget cannot fund a single target.

    Raises:
        LaneBudgetError: On an unknown lane or a negative budget.
    """
    if isinstance(budget_sec, bool) or not isinstance(budget_sec, int) or budget_sec < 0:
        raise LaneBudgetError(f"lane budget must be a non-negative int, got {budget_sec!r}")
    if lane == LANE_REWRITE:
        return budget_sec // REWRITE_MIN_TARGET_SEC
    if lane == LANE_GEMM:
        return _greedy_fit(budget_sec, target_costs_sec)
    if lane == LANE_FUSION:
        return FUSION_MAX_TARGETS if budget_sec > 0 else 0
    raise LaneBudgetError(f"unknown lane {lane!r}")


def allocate(
    remaining_minutes: object,
    *,
    weights: Mapping[str, float] | None = None,
    gemm_target_costs_sec: tuple[int, ...] = (),
    reserve_sec: int = PHASE_RESERVE_SEC,
) -> dict[str, LaneAllocation]:
    """Derive every lane's share and target ceiling in one pass.

    Args:
        remaining_minutes: Minutes left in the phase.
        weights: Lane weights; defaults to :data:`DEFAULT_LANE_WEIGHTS`.
        gemm_target_costs_sec: Per-tuner estimates from the router.
        reserve_sec: Held back for the phase to close out.

    Returns:
        One allocation per lane in ``weights``.
    """
    phase_sec = phase_budget_sec(remaining_minutes, reserve_sec=reserve_sec)
    shares = split_lanes(phase_sec, weights=weights)
    return {
        lane: LaneAllocation(
            lane=lane,
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
    """Cap one gemm tuner below the session budget it is drawn from.

    Args:
        global_timeout_sec: The whole-session ceiling. ``0`` or a non-positive
            value means unbounded, which is passed through unchanged.

    Returns:
        The per-tuner ceiling; ``0`` when the session is unbounded.
    """
    if isinstance(global_timeout_sec, bool) or not isinstance(global_timeout_sec, (int, float)):
        return 0
    total = int(global_timeout_sec)
    if total <= 0:
        return 0
    # Strictly below the global cap for any session of two seconds or more. A
    # one-second session is degenerate: there is no integer both below it and
    # usable, so it yields one second rather than zero.
    return max(1, min(total - 1, int(total * GEMM_PER_TUNER_SHARE)))


def _greedy_fit(budget_sec: int, costs_sec: tuple[int, ...]) -> int:
    """Count how many estimates fit, in order, without exceeding the budget.

    Order is the router's priority order, and a target that does not fit is
    skipped rather than shortened, so this stops at the first one that would
    overrun rather than searching for a better packing.
    """
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
