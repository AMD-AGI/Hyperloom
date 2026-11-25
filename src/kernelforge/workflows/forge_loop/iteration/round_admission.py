# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The campaign clock, whether the budget admits the next round, and the search mode the round runs under."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from kernelforge.workflows.forge_loop.policy.round_budget import admit_dispatch
from kernelforge.workflows.forge_loop.campaign.run_state import make_event
from kernelforge.workflows.forge_loop.search.search_policy import (
    MARGINAL_GAIN_SCAN_WINDOW,
    MARGINAL_GAIN_WINDOW,
    NO_CHANGES_STREAK_WINDOW,
    RecentProgress,
    SearchPolicyDecision,
    SearchState,
)
from kernelforge.workflows.forge_loop.roles.contracts import EvidenceRef, EvidenceContext
from kernelforge.workflows.forge_loop.roles.supervisor import latest_supervisor_ruling_path


class RoundAdmissionMixin:
    """Keep the campaign's clock, admit a round against what is left of it, and decide the round's search mode."""

    def _time_remaining(self) -> float:
        """Seconds remaining in the budget."""
        deadline = self.start_time + self.ic.max_time_hours * 3600
        if self.ic.deadline_unix is not None:
            deadline = min(deadline, self.ic.deadline_unix)
        return max(0, deadline - time.time())

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
        state = self.run_state
        if self._campaign_started_at > 0:
            state.campaign_sec = max(state.campaign_sec, max(0.0, time.time() - self._campaign_started_at))
        return state.campaign_sec

    def _round_budget_summary(self) -> dict:
        """Why the campaign stopped for budget, for the published report; empty when it did not."""
        if not self._refused_round:
            return {}
        return {"campaign_sec": round(self._advance_campaign_clock(), 3), "refused": self._refused_round}

    def _admit_dispatch(self, iteration: int) -> bool:
        """Whether the round may start its sessions."""
        decision = admit_dispatch(remaining_sec=self._time_remaining())
        self.state_store.append_event(
            make_event(
                "round_dispatch",
                iteration,
                admitted=decision.admitted,
                remaining_sec=round(decision.remaining_sec, 3),
                required_sec=round(decision.required_sec, 3),
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

    def _update_search_policy(self, iteration: int) -> SearchPolicyDecision:
        """Derive and persist the search mode before an iteration's round."""
        window_gain = self._exploit_window_gain(
            self.state_store.recent_results(MARGINAL_GAIN_SCAN_WINDOW),
            window=MARGINAL_GAIN_WINDOW,
            since_iteration=self.run_state.stall.last_supervisor_iter,
        )
        decision = self.search_policy_engine.decide(
            SearchState(
                best_source=self.run_state.best.source,
                no_improvement_iters=self.run_state.stall.unresolved_stall_iters,
            ),
            diversify_after=self.ic.diversify_after,
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
        self._search_policy_decision = decision
        self.state_store.append_event(
            make_event(
                "search_policy_decision",
                iteration,
                mode=decision.mode,
                reason_codes=list(decision.reason_codes),
                objective_kind=decision.objective_kind,
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

    def _with_loop_state(self, context: EvidenceContext) -> EvidenceContext:
        """``context`` carrying the active Supervisor ruling and the search policy.

        Each is read from its owner as it stands now. The Analysis context is resolved before the Supervisor runs and
        the policy is decided after it, so a context taken from Analysis carries none of them.
        """
        ruling_path = latest_supervisor_ruling_path(self.ic.workspace_dir)
        ruling_refs = (
            (
                EvidenceRef(
                    kind="supervisor_guidance",
                    path=str(ruling_path.resolve()),
                    summary=(
                        "Latest free-form Supervisor Ruling. It overrides subjective conclusions in historical "
                        "lesson records but not objective measurements."
                    ),
                ),
            )
            if self._supervisor_ruling and ruling_path.is_file()
            else ()
        )
        return replace(
            context,
            supervisor_guidance=self._supervisor_ruling,
            search_mode=self.run_state.search_mode,
            search_reason_codes=tuple(self.run_state.search_reason_codes),
            search_objective=self.run_state.search_objective,
            evidence_refs=(*context.evidence_refs, *ruling_refs),
        )
