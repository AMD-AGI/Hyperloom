# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Search policy, orchestrated planning, Plan Critic rulings and Supervisor review for the forge-loop."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path


from kernelforge.loop.search_policy import (
    MARGINAL_GAIN_SCAN_WINDOW,
    MARGINAL_GAIN_WINDOW,
    NO_CHANGES_STREAK_WINDOW,
    SEARCH_MODE_EXPLOIT,
    RecentProgress,
    SearchPolicyDecision,
    SearchState,
)
from kernelforge.loop.run_state import (
    CriticRuling,
    StallPolicy,
    apply_supervisor_attempt,
    apply_supervisor_intervention,
    make_event,
    measured_nothing,
)
from kernelforge.orchestrator.orchestration import (
    OrchestrationInfrastructureError,
)
from kernelforge.orchestrator.supervisor import (
    clear_latest_supervisor_ruling,
    persist_supervisor_ruling,
)
from kernelforge.tools.durable_io import atomic_write_text
from kernelforge.loop.iteration_types import WindowGain

log = logging.getLogger(__name__)


class PlanningRuntimeMixin:
    """Own how each round is planned: search policy, orchestration, critic and supervisor rulings."""

    def _expire_supervisor_ruling(self) -> None:
        """Stop injecting a ruling after its stall episode ends."""
        self._supervisor_ruling = ""
        clear_latest_supervisor_ruling(self.ic.workspace_dir)

    def _search_control_snapshot(self) -> dict:
        """Capture decision-critical planning state for pending KEEP recovery."""
        return {
            "diversification_cycle_completed": (self.run_state.diversification_cycle_completed),
        }

    def _apply_iteration_planning_state(
        self,
        *,
        optimization_plan_created: bool,
    ) -> None:
        """Reduce one completed iteration's planning outcome into run_state."""
        self.run_state.diversification_cycle_completed = (
            self.run_state.search_mode == "DIVERSIFY" and optimization_plan_created
        )

    @staticmethod
    def _consecutive_no_changes(events: list[dict]) -> int:
        """Count the trailing run of empty diffs under the latest search mode."""
        streak = 0
        mode: str | None = None
        for event in reversed(events):
            if event.get("type") != "iteration_result":
                continue
            decision = str(event.get("decision") or "").strip().upper()
            if measured_nothing(decision):
                continue
            if decision != "NO_CHANGES":
                break
            recorded_mode = str(event.get("search_mode") or "")
            if mode is None:
                mode = recorded_mode
            elif recorded_mode != mode:
                break
            if not mode:
                break
            streak += 1
        return streak

    @staticmethod
    def _exploit_window_gain(
        events: list[dict],
        *,
        window: int,
        since_iteration: int,
    ) -> WindowGain:
        """Relative incumbent gain over the last full window of EXPLOIT outcomes."""
        scores: list[float] = []
        unavailable = "short_window"
        for event in reversed(events):
            if event.get("type") != "iteration_result":
                continue
            if int(event.get("iter", 0) or 0) <= since_iteration:
                break
            if str(event.get("search_mode") or "") != SEARCH_MODE_EXPLOIT:
                break
            decision = str(event.get("decision") or "").strip().upper()
            if measured_nothing(decision):
                continue
            score = event.get("best_after_mean_case_speedup")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                unavailable = "non_numeric_score"
                break
            score = float(score)
            if not math.isfinite(score):
                unavailable = "non_finite_score"
                break
            if score <= 0:
                unavailable = "non_positive_score"
                break
            scores.append(score)
            if len(scores) > window:
                break
        if len(scores) <= window:
            return WindowGain(ratio=None, unavailable=unavailable)
        anchor = scores[-1]
        return WindowGain(ratio=(scores[0] - anchor) / anchor, unavailable=None)

    def _stall_policy(self) -> StallPolicy:
        """The stall and orchestration-error thresholds this campaign was configured with."""
        return StallPolicy(self.ic.supervise_after, self.ic.max_consecutive_orchestration_errors)

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

    async def _run_orchestration(
        self,
        *,
        iteration: int,
        orchestration_service,
        lanes: int = 1,
    ) -> tuple[Path | None, str]:
        """Run planning and durably publish every lane's plan for the round."""
        context = self._with_ceiling_standing(
            self._with_case_config_coverage(
                self._active_analysis_context
                if self._active_analysis_context is not None
                else self._build_orchestration_context()
            )
        )
        try:
            result = await orchestration_service.run(
                context,
                usage=self._usage,
                lanes=lanes,
            )
        except OrchestrationInfrastructureError as error:
            detail = f"{type(error).__name__}: {error}"
            print(f"  [orchestration] failed ({detail})")
            return None, detail
        finally:
            self._checkpoint_llm_usage()

        self._record_probe_hazard(iteration, result)
        self._last_lane_plans = [plan for plan in result.optimization_plans if str(plan).strip()]
        self._persist_orchestration_result(iteration, context, result)
        plan_path = self._persist_lane_plans(
            iteration,
            self._last_lane_plans,
            analysis_commit=context.analysis_commit,
        )
        self._record_orchestration_final_plan(iteration, plan_path)
        self._last_orchestration_plan_executable = bool(getattr(result, "optimization_plan_executable", True))
        critic = result.plan_critic
        self._last_critic_verdict = critic.verdict if critic is not None else ""
        self._last_critic_review = critic.review if critic is not None else ""
        self._record_critic_ruling(iteration, critic)
        self._latest_optimization_plan_path = str(plan_path)
        print(f"  [orchestration] optimization plan: {plan_path}")
        if len(self._last_lane_plans) > 1:
            print(f"  [orchestration] {len(self._last_lane_plans)} lane plans published under {plan_path.parent}")
        return plan_path, ""

    def _record_probe_hazard(self, iteration: int, result) -> None:
        """Turn a probe the analysis phase could not clear into a live hazard."""
        diagnostics = getattr(result, "structured_output_diagnostics", None)
        finding = (diagnostics or {}).get("probe_device_hazard")
        if not isinstance(finding, dict):
            return
        hazard = self.device_hazard.record(
            iteration=iteration,
            detail=f"probe round: {finding.get('describe', '')}",
            pids=finding.get("pids") or (),
        )
        print(
            "  [probe] the round's probe scratch tree left the device "
            f"contended; this round measures nothing. {hazard.describe()}"
        )

    def _record_critic_ruling(self, iteration: int, critic) -> None:
        """Put this round's verdict where the next process can still find it."""
        ruling = CriticRuling()
        if critic is not None and not critic.fail_open:
            ruling = CriticRuling(
                verdict=critic.verdict,
                review_path=str((self._orchestration_root(iteration) / "critic_review.md").resolve()),
            )
        self.run_state.last_critic = ruling

    def _restore_critic_ruling(self) -> None:
        """Resume the ruling whose round ended with the process that bought it."""
        ruling = self.run_state.last_critic
        if not ruling.verdict or not ruling.review_path:
            return
        try:
            review = Path(ruling.review_path).read_text(encoding="utf-8").strip()
        except OSError as error:
            log.warning(
                "critic review at %s is unreadable (%s); resuming without the %s verdict it carried",
                ruling.review_path,
                error,
                ruling.verdict,
            )
            self.run_state.last_critic = CriticRuling()
            return
        if not review:
            self.run_state.last_critic = CriticRuling()
            return
        self._last_critic_verdict = ruling.verdict
        self._last_critic_review = review
        print(f"  [critic] resuming the {ruling.verdict} verdict an earlier process recorded")

    def _persist_orchestration_result(self, iteration, context, result) -> None:
        """Persist planning diagnostics before publishing the executable plan."""
        from kernelforge.loop.recovery import atomic_write_json

        root = self._orchestration_root(iteration)
        atomic_write_json(
            root / "context.json",
            context.to_prompt_dict(),
        )
        if result.dispatch_plan is not None:
            atomic_write_json(
                root / "dispatch.json",
                result.dispatch_plan.to_dict(),
            )
        atomic_write_json(
            root / "specialists.json",
            {
                "analysis_commit": context.analysis_commit,
                "outcomes": [outcome.to_dict() for outcome in result.specialist_outcomes],
            },
        )
        diagnostics = dict(result.structured_output_diagnostics or {})
        artifact_paths = {}
        draft = str(result.optimization_plan_draft or "").strip()
        if draft:
            draft_path = root / "draft_plan.md"
            atomic_write_text(draft_path, draft + "\n")
            artifact_paths["draft_plan"] = str(draft_path.resolve())
        critic = result.plan_critic
        if critic is not None:
            critic_path = root / "critic_review.md"
            atomic_write_text(
                critic_path,
                critic.render_artifact().rstrip() + "\n",
            )
            artifact_paths["critic_review"] = str(critic_path.resolve())
        if artifact_paths:
            diagnostics["artifact_paths"] = artifact_paths
            diagnostics["plan_revised"] = bool(result.plan_revised)
        if diagnostics:
            atomic_write_json(
                root / "structured_output.json",
                diagnostics,
            )

    def _record_orchestration_final_plan(
        self,
        iteration: int,
        plan_path: Path,
    ) -> None:
        """Publish the final-plan pointer only after the plan exists."""
        from kernelforge.loop.recovery import atomic_write_json

        diagnostics_path = self._orchestration_root(iteration) / "structured_output.json"
        if not diagnostics_path.is_file():
            return
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if not isinstance(diagnostics, dict):
            raise ValueError(f"invalid orchestration diagnostics: {diagnostics_path}")
        artifact_paths = diagnostics.get("artifact_paths")
        if not isinstance(artifact_paths, dict):
            return
        artifact_paths["final_plan"] = str(plan_path.resolve())
        atomic_write_json(diagnostics_path, diagnostics)

    async def _run_supervisor_review(
        self, supervisor_fn, *, iteration, digest, supervisor_due, supervisor_reason
    ) -> None:
        """Ask the supervisor for a ruling when the run has stalled, and record whatever comes back.

        The memo is optional: a supervisor that fails, or returns nothing,
        leaves the run to continue without an active ruling.
        """
        if self.monitor is not None and supervisor_fn is not None:
            if supervisor_due:
                print(f"\n[supervisor] intervening at iteration {iteration}: {supervisor_reason}")
                memo = ""
                try:
                    evidence_context = self._build_supervisor_evidence_context(iteration)
                    # A new review attempt supersedes the prior stall episode's ruling even when the backend
                    # returns empty.
                    self._expire_supervisor_ruling()
                    self.monitor.mark_attempted(iteration)
                    apply_supervisor_attempt(
                        self.run_state,
                        iteration=iteration,
                    )
                    self.state_store.append_event(
                        make_event(
                            "supervisor_attempt",
                            iteration,
                            reason=supervisor_reason,
                        )
                    )
                    self.state_store.save(self.run_state)
                    memo = await supervisor_fn(
                        digest=digest,
                        reason=supervisor_reason,
                        workspace=self.ic.workspace_dir,
                        iteration=iteration,
                        evidence_context=evidence_context,
                    )
                except Exception as e:  # noqa: BLE001 - supervisor memo is optional
                    print(f"  [supervisor] failed ({e}); continuing without a memo")
                finally:
                    self._checkpoint_llm_usage()
                memo = memo or ""
                if memo.strip():
                    interaction_path, ruling_path = persist_supervisor_ruling(
                        self.ic.workspace_dir,
                        iteration,
                        supervisor_reason,
                        memo,
                    )
                    self._supervisor_ruling = memo
                    print(f"  [supervisor] injected free-form ruling: {len(self._supervisor_ruling)} chars")
                    self.state_store.append_event(
                        make_event(
                            "supervisor_ruling",
                            iteration,
                            reason=supervisor_reason,
                            ruling_len=len(self._supervisor_ruling),
                            interaction_path=(
                                str(interaction_path.relative_to(Path(self.ic.workspace_dir)))
                                if interaction_path is not None
                                else None
                            ),
                            ruling_path=(
                                str(ruling_path.relative_to(Path(self.ic.workspace_dir)))
                                if ruling_path is not None
                                else None
                            ),
                        )
                    )
                else:
                    print("  [supervisor] no new ruling returned; continuing without an active ruling")
                if memo.strip():
                    self.monitor.mark_intervened(iteration)
                    apply_supervisor_intervention(
                        self.run_state,
                        iteration=iteration,
                        stall_threshold=self.ic.supervise_after,
                    )
                    self.state_store.save(self.run_state)
