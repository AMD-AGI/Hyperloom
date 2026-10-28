# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The campaign clock: what a round costs, and whether the budget admits the next one."""

from __future__ import annotations

import time
from pathlib import Path

from kernelforge.loop.round_budget import admit_dispatch, admit_round, estimate_measurement_sec
from kernelforge.loop.run_state import RoundCost, StallPolicy, apply_round_cost, make_event
from kernelforge.loop.search_policy import (
    MARGINAL_GAIN_SCAN_WINDOW,
    MARGINAL_GAIN_WINDOW,
    NO_CHANGES_STREAK_WINDOW,
    RecentProgress,
    SearchPolicyDecision,
    SearchState,
)


class RoundAdmissionMixin:
    """Charge the campaign for the wall-clock a round spends, and decide how many lanes the next one may plan and dispatch."""

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

    def _stall_policy(self) -> StallPolicy:
        """The stall and orchestration-error thresholds this campaign was configured with."""
        return StallPolicy(self.ic.supervise_after, self.ic.max_consecutive_orchestration_errors)

    def _price_round(self, iteration: int) -> int | None:
        """Price the round before anything is spent on it: the lanes it may plan, or ``None`` if none fit."""
        self._close_round()
        round_lanes: int | None = self.ic.lanes
        if not self._lane_queue:
            round_lanes = self._admit_next_round(iteration)
            if round_lanes is not None:
                self._open_round(iteration, lanes=round_lanes)
        return round_lanes

    def _update_search_policy(self, iteration: int) -> SearchPolicyDecision:
        """Derive and persist the search mode before planning an iteration."""
        window_gain = self._exploit_window_gain(
            self.state_store.recent_results(MARGINAL_GAIN_SCAN_WINDOW),
            window=MARGINAL_GAIN_WINDOW,
            since_iteration=self.run_state.stall.last_supervisor_iter,
        )
        decision = self.search_policy_engine.decide(
            SearchState(
                best_source=self.run_state.best.source,
                no_improvement_iters=self.run_state.stall.unresolved_stall_iters,
                current_mode=self.run_state.search_mode,
                residence_iterations_remaining=(self.run_state.search_mode_residence_remaining),
                diversification_cycle_completed=(self.run_state.diversification_cycle_completed),
            ),
            stall_threshold=self.ic.supervise_after,
            recent=RecentProgress(
                consecutive_no_changes=self._consecutive_no_changes(
                    self.state_store.recent_results(NO_CHANGES_STREAK_WINDOW)
                ),
                window_gain_ratio=window_gain.ratio,
            ),
        )
        decision = self._relay_search_policy(iteration, decision)
        previous_mode = self.run_state.search_mode
        previous_reasons = tuple(self.run_state.search_reason_codes)
        self.run_state.search_mode = decision.mode
        self.run_state.search_reason_codes = list(decision.reason_codes)
        self.run_state.search_objective = decision.objective_kind
        self.run_state.search_mode_residence_remaining = decision.residence_iterations_remaining
        self.run_state.diversification_cycle_completed = False
        self._search_policy_decision = decision
        self.state_store.append_event(
            make_event(
                "search_policy_decision",
                iteration,
                mode=decision.mode,
                reason_codes=list(decision.reason_codes),
                objective_kind=decision.objective_kind,
                residence_iterations_remaining=(decision.residence_iterations_remaining),
                # ``make_event`` drops empty fields, so a ratio of None on its own would leave the event silent
                # about a trigger that could not be evaluated at all -- indistinguishable from a young campaign.
                window_gain_ratio=window_gain.ratio,
                window_gain_unavailable=window_gain.unavailable,
                mode_changed=(decision.mode != previous_mode),
            )
        )
        self.state_store.save(self.run_state)
        # A window that has not filled yet is the ordinary state of a young campaign.
        fault = window_gain.unavailable
        if fault is not None and fault != "short_window" and fault not in self._reported_window_gain_faults:
            self._reported_window_gain_faults.add(fault)
            print(f"  [search-policy] diminishing-returns trigger unavailable: {fault}")
        if decision.mode != previous_mode or decision.reason_codes != previous_reasons:
            print(f"  [search-policy] {decision.mode}: " + ", ".join(decision.reason_codes))
        return decision

    async def _plan_round(
        self,
        *,
        iteration: int,
        orchestration_service,
        lanes: int = 1,
    ) -> tuple[Path | None, str]:
        """Buy the round's plans and charge the round for the wall-clock."""
        started_at = time.time()
        try:
            return await self._run_orchestration(
                iteration=iteration,
                orchestration_service=orchestration_service,
                lanes=lanes,
            )
        finally:
            self._round_planning_sec += max(0.0, time.time() - started_at)
