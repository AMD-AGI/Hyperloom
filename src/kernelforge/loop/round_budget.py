# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the remaining campaign budget can pay for another round."""

from __future__ import annotations

from dataclasses import dataclass

from kernelforge.loop.run_state import RoundCost
from kernelforge.orchestrator.plan_critic import PLAN_CRITIC_TIMEOUT_SEC

# The fastest of the 75 measured rounds.
PLANNING_FLOOR_SEC = 750.0

# What an Implementer session is priced at BEFORE planning, where the only question is whether a round could run at
# all.
ADMISSION_SESSION_SEC = 480.0

# What an Implementer session is priced at AFTER planning, where the loop is about to start something it cannot
# interrupt.
DISPATCH_SESSION_SEC = 738.0

# The floor under the dispatch requirement, and the one number here that this campaign's own speed cannot lower.
DISPATCH_FLOOR_SEC = 1176.0

# What the canonical validation and benchmark cost a round that has observed none of its own.
FIRST_ROUND_MEASUREMENT_SEC = 600.0


@dataclass(frozen=True)
class RoundAdmission:
    """The decision on planning one round, and every number it was made from."""

    admitted: bool
    lanes: int
    remaining_sec: float
    required_sec: float
    planning_sec: float
    execution_sec: float
    # True when the round was admitted at less than the width it asked for.
    narrowed: bool

    def summary(self) -> str:
        """One line naming the decision's cost breakdown, in minutes."""
        return (
            f"{self.required_sec / 60:.0f} min needed at {self.lanes} lane(s) "
            f"at least (planning {self.planning_sec / 60:.0f}, session and "
            f"measurement {self.execution_sec / 60:.0f}); "
            f"{self.remaining_sec / 60:.0f} min remain"
        )


@dataclass(frozen=True)
class DispatchAdmission:
    """The decision on dispatching a round whose plans are already bought."""

    admitted: bool
    remaining_sec: float
    required_sec: float
    session_sec: float
    measurement_sec: float
    # True when what this round is estimated to spend came in under :data:`DISPATCH_FLOOR_SEC` and the floor is what
    # is being required.
    floored: bool

    def summary(self) -> str:
        """One line naming the decision's cost breakdown, in minutes."""
        priced = f"session {self.session_sec / 60:.0f}, measurement {self.measurement_sec / 60:.0f}"
        if self.floored:
            priced = f"external-timeout floor over {priced}"
        return (
            f"{self.required_sec / 60:.0f} min needed after planning "
            f"({priced}); {self.remaining_sec / 60:.0f} min remain"
        )


def estimate_measurement_sec(history: list[RoundCost]) -> float:
    """Seconds the canonical validation and benchmark are expected to take."""
    observed = [cost.measurement_sec for cost in history if cost.measurement_sec > 0]
    if not observed:
        return FIRST_ROUND_MEASUREMENT_SEC
    return max(observed)


def estimate_planning_sec(history: list[RoundCost], *, lanes: int) -> float:
    """A LOWER bound on what a round of ``lanes`` lanes will spend planning."""
    observed = [cost.planning_sec for cost in history if cost.lanes == lanes]
    if observed:
        return min(observed)
    wider = [cost for cost in history if cost.lanes > lanes]
    if wider:
        cheapest = min(wider, key=lambda cost: cost.planning_sec)
        unread_plans = cheapest.lanes - lanes
        # Held at the floor -- or at this campaign's own cheapest round, on the campaign that plans faster than
        # production ever did.
        return max(
            min(PLANNING_FLOOR_SEC, cheapest.planning_sec),
            cheapest.planning_sec - unread_plans * PLAN_CRITIC_TIMEOUT_SEC,
        )
    narrower = [cost.planning_sec for cost in history if cost.lanes < lanes]
    if narrower:
        return min(narrower)
    return PLANNING_FLOOR_SEC


def admit_round(
    *,
    remaining_sec: float,
    requested_lanes: int,
    history: list[RoundCost],
    measurement_sec: float,
) -> RoundAdmission:
    """Decide whether -- and how wide -- the next round may be PLANNED."""
    requested = max(1, int(requested_lanes))
    # What a round costs once its plans exist: the least a session can be given and still return something, and the
    # canonical measurement that judges it.
    execution_sec = ADMISSION_SESSION_SEC + max(0.0, measurement_sec)

    def priced(lanes: int) -> RoundAdmission:
        planning_sec = estimate_planning_sec(history, lanes=lanes)
        required_sec = planning_sec + execution_sec
        return RoundAdmission(
            admitted=remaining_sec >= required_sec,
            lanes=lanes,
            remaining_sec=remaining_sec,
            required_sec=required_sec,
            planning_sec=planning_sec,
            execution_sec=execution_sec,
            narrowed=lanes < requested,
        )

    for lanes in range(requested, 1, -1):
        decision = priced(lanes)
        if decision.admitted:
            return decision
    # The single-lane round is both the last width tried and the one a refusal reports, so it is priced outside the
    # loop and its verdict IS the answer. "Some decision is always returned" is then a property of the code rather
    # than of an assertion, which ``python -O`` would strip.
    return priced(1)


def admit_dispatch(
    *,
    remaining_sec: float,
    measurement_sec: float,
) -> DispatchAdmission:
    """Decide whether a round whose plans are bought may be dispatched."""
    measurement = max(0.0, measurement_sec)
    estimated_sec = DISPATCH_SESSION_SEC + measurement
    required_sec = max(DISPATCH_FLOOR_SEC, estimated_sec)
    return DispatchAdmission(
        admitted=remaining_sec >= required_sec,
        remaining_sec=remaining_sec,
        required_sec=required_sec,
        session_sec=DISPATCH_SESSION_SEC,
        measurement_sec=measurement,
        floored=estimated_sec < DISPATCH_FLOOR_SEC,
    )
