# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Campaign clock, round admission and stop conditions for the forge-loop."""

from __future__ import annotations

import logging
import time
from pathlib import Path


from kernelforge.loop.round_budget import (
    admit_dispatch,
    admit_round,
    estimate_measurement_sec,
)
from kernelforge.loop.run_state import (
    ORCHESTRATION_CIRCUIT_OPEN,
    RoundCost,
    apply_round_cost,
    make_event,
)
from kernelforge.loop.device_hazard import DeviceHazard

log = logging.getLogger(__name__)


class BudgetRuntimeMixin:
    """Own the campaign clock, the pricing of each round, and when the loop stops."""

    def _time_remaining(self) -> float:
        """Seconds remaining in the budget."""
        elapsed = time.time() - self.start_time
        return max(0, self.ic.max_time_hours * 3600 - elapsed)

    def _analysis_deadline_unix(self) -> float:
        """Absolute Analysis deadline preserving iteration/finalization reserve."""
        now = time.time()
        started_at = self.start_time or now
        deadlines = [started_at + self.ic.max_time_hours * 3600 - self.ic.budget_reserve_sec]
        if self.ic.deadline_unix is not None:
            deadlines.append(self.ic.deadline_unix - self.ic.budget_reserve_sec)
        return max(now, min(deadlines))

    def _is_budget_exhausted(self) -> bool:
        """Whether remaining campaign time cannot admit another Agent session."""
        return self._time_remaining() < self.ic.budget_reserve_sec

    def _advance_campaign_clock(self) -> float:
        """Bring the campaign's cumulative wall-clock up to now, and return it."""
        costs = self.run_state.round_costs
        if self._campaign_started_at <= 0:
            return costs.campaign_sec
        costs.campaign_sec = max(
            costs.campaign_sec,
            max(0.0, time.time() - self._campaign_started_at),
        )
        return costs.campaign_sec

    def _open_round(self, iteration: int, *, lanes: int) -> None:
        """Start timing the round the budget has just admitted."""
        self._round_started_at = time.time()
        self._round_iteration = iteration
        self._round_lanes = max(1, int(lanes))
        self._round_planning_sec = 0.0
        self._round_measurement_sec = 0.0

    def _close_round(self) -> None:
        """Record what the open round cost, if it bought any planning."""
        started_at = self._round_started_at
        planning_sec = self._round_planning_sec
        measurement_sec = self._round_measurement_sec
        self._round_started_at = None
        self._round_planning_sec = 0.0
        self._round_measurement_sec = 0.0
        if started_at is None or planning_sec <= 0:
            return
        total_sec = max(planning_sec, time.time() - started_at)
        apply_round_cost(
            self.run_state,
            RoundCost(
                iteration=self._round_iteration,
                lanes=self._round_lanes,
                planning_sec=planning_sec,
                total_sec=total_sec,
                measurement_sec=measurement_sec,
            ),
            campaign_sec=self._advance_campaign_clock(),
        )
        self.state_store.append_event(
            make_event(
                "round_cost",
                self._round_iteration,
                lanes=self._round_lanes,
                planning_sec=round(planning_sec, 3),
                total_sec=round(total_sec, 3),
                measurement_sec=round(measurement_sec, 3),
            )
        )
        self.state_store.save(self.run_state)

    def _round_budget_summary(self) -> dict:
        """What the campaign's rounds have cost, for the published report."""
        costs = self.run_state.round_costs
        if not costs.rounds and not self._refused_round:
            return {}
        summary = {
            "rounds": costs.rounds,
            "planning_total_sec": round(costs.planning_total_sec, 3),
            "total_sec": round(costs.total_sec, 3),
            "campaign_sec": round(self._advance_campaign_clock(), 3),
        }
        share = costs.planning_share_pct()
        if share is not None:
            summary["planning_share_pct"] = round(share, 1)
        if self._refused_round:
            summary["refused"] = self._refused_round
        return summary

    def _observe_measurement(self, started_at: float) -> None:
        """Charge the open round for one canonical validate-and-benchmark cycle."""
        if self._round_started_at is None:
            return
        self._round_measurement_sec += max(0.0, time.time() - started_at)

    def _measurement_estimate_sec(self) -> float:
        """Wall-clock the canonical validation and benchmark may still take."""
        return estimate_measurement_sec(list(self.run_state.round_costs.recent))

    def _admit_next_round(self, iteration: int) -> int | None:
        """How many lanes the next round may PLAN, or ``None`` if none fit."""
        decision = admit_round(
            remaining_sec=self._time_remaining(),
            requested_lanes=self.ic.lanes,
            history=list(self.run_state.round_costs.recent),
            measurement_sec=self._measurement_estimate_sec(),
        )
        if decision.admitted and not decision.narrowed:
            return decision.lanes
        event_fields = {
            "lanes": decision.lanes,
            "requested_lanes": max(1, int(self.ic.lanes)),
            "remaining_sec": round(decision.remaining_sec, 3),
            "required_sec": round(decision.required_sec, 3),
            "planning_sec": round(decision.planning_sec, 3),
            "execution_sec": round(decision.execution_sec, 3),
        }
        if decision.admitted:
            print(f"  [budget] round narrowed to {decision.lanes} lane(s): {decision.summary()}")
        else:
            self._refuse_round(iteration, decision.summary())
        self.state_store.append_event(
            make_event(
                "round_admission",
                iteration,
                admitted=decision.admitted,
                **event_fields,
            )
        )
        return decision.lanes if decision.admitted else None

    def _admit_dispatch(self, iteration: int) -> bool:
        """Whether the round now holding plans may start its session."""
        decision = admit_dispatch(
            remaining_sec=self._time_remaining(),
            measurement_sec=self._measurement_estimate_sec(),
        )
        self.state_store.append_event(
            make_event(
                "round_dispatch",
                iteration,
                admitted=decision.admitted,
                remaining_sec=round(decision.remaining_sec, 3),
                required_sec=round(decision.required_sec, 3),
                session_sec=round(decision.session_sec, 3),
                measurement_sec=round(decision.measurement_sec, 3),
                # Recorded because it is the one case where the parts do not add up to the requirement: this
                # campaign estimated less than the external-timeout floor and was held at it.
                floored=decision.floored,
            )
        )
        if decision.admitted:
            return True
        self._refuse_round(iteration, decision.summary())
        return False

    def _refuse_round(self, iteration: int, summary: str) -> None:
        """End the campaign on a round the remaining budget cannot pay for."""
        self._refused_round = summary
        self.termination_reason = "round_budget_exhausted"
        print(f"\nROUND REFUSED FOR BUDGET at iteration {iteration}: {summary}")

    def _is_force_stopped(self) -> bool:
        """Whether the operator requested an early stop via <workspace>/.stop."""
        return (Path(self.ic.workspace_dir) / ".stop").exists()

    def _is_gate_met(self) -> bool:
        """Check if performance target is met."""
        if self.ic.target_wall_ms is None or self.best_wall_ms is None:
            return False
        return self.best_wall_ms <= self.ic.target_wall_ms

    def _stop_condition_met(self, iteration) -> tuple[bool, DeviceHazard | None]:
        """Whether this campaign stops, and the device standing when it does not.

        Every answer here is terminal for the run rather than for one iteration,
        so each sets ``termination_reason`` before saying so.

        The device is probed last, after every answer that ends the run without
        touching it. ``recheck`` is a write -- it clears the hazard file, or
        counts one more blocked iteration into it -- so an iteration that was
        going to stop anyway must not leave that mark for the next resume.
        """
        if self._is_gate_met():
            self.termination_reason = "gate_met"
            print(f"\nGATE MET at iteration {iteration}: raw wall target reached at {self.best_wall_ms:.6f} ms")
            return True, None
        if self._is_roofline_target_met():
            self.termination_reason = "roofline_target_met"
            standing = self._roofline_attainment()
            print(
                f"\nROOFLINE TARGET MET at iteration {iteration}: mean attainment "
                f"{standing.mean * 100:.1f}% of the estimated ceiling, at or above the "
                f"{float(self.ic.roofline_target) * 100:.0f}% target. The ceiling is an estimate: if this "
                "looks early, the derivation in performance_ceiling_analysis.md is where it would be wrong."
            )
            for entry in sorted(standing.cases, key=lambda c: c.attainment):
                print(
                    f"  [roofline] {entry.case_id}: {entry.attainment * 100:.1f}% "
                    f"({entry.t_current_ms:.6g} ms vs ceiling {entry.t_ideal_ms:.6g} ms)"
                )
            return True, None
        if self.run_state.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_OPEN:
            self.termination_reason = "orchestration_failed"
            print(
                "\nORCHESTRATION FAILED repeatedly; stopping after "
                f"{self.run_state.orchestration_error_streak} "
                "consecutive infrastructure errors"
            )
            return True, None
        if self._is_budget_exhausted():
            self.termination_reason = "budget_exhausted"
            print(f"\nBUDGET EXHAUSTED after {len(self.results)} iterations in this session")
            return True, None
        if self._is_force_stopped():
            self.termination_reason = "force_stop"
            print("\nFORCE STOP: .stop file detected — remove it and --resume to continue")
            return True, None
        hazard: DeviceHazard | None = self.device_hazard.recheck(iteration)
        if hazard is not None and hazard.exhausted:
            self.termination_reason = "device_contended"
            print(
                "\nDEVICE CONTENDED: nothing this campaign may clear has "
                f"released the device in {hazard.blocked_iterations} "
                "iterations, so no measurement can be trusted; stopping "
                f"rather than spending the budget on unmeasurable "
                f"iterations. {hazard.describe()}"
            )
            return True, hazard
        return False, hazard
