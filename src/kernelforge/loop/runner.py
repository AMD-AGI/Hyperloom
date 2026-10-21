# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Durable, evidence-driven autonomous kernel optimization loop."""

from __future__ import annotations

import logging
import os
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from kernelforge.config import Config
from kernelforge.loop.experience import ExperienceLedger
from kernelforge.loop.lessons import LessonStore
from kernelforge.loop.archive import CandidateArchive
from kernelforge.loop.handoffs import HandoffStore
from kernelforge.loop.jit_rebuild import force_jit_rebuild
from kernelforge.loop.analysis_runtime import AnalysisRuntimeMixin
from kernelforge.loop.attempt_verdict import AttemptVerdictMixin
from kernelforge.loop.best_publication import BestPublicationMixin
from kernelforge.loop.campaign_report import CampaignReportMixin
from kernelforge.loop.campaign_runtime import CampaignMixin
from kernelforge.loop.ceiling_runtime import CeilingMixin
from kernelforge.loop.iteration_phases import IterationPhasesMixin
from kernelforge.loop.iteration_types import (
    HeldRound,
    IterationConfig,
    IterationResult,
    LoopAgents,
    LoopCallbacks,
    _IterationFrame,
    _IterationSupply,
    _with_search_actor,
)
from kernelforge.loop.keep_checkpoint import KeepCheckpointMixin
from kernelforge.loop.lane_runtime import LaneRuntimeMixin
from kernelforge.loop.outcome_runtime import OutcomeMixin
from kernelforge.loop.relay_runtime import RelayRuntimeMixin
from kernelforge.loop.round_admission import RoundAdmissionMixin
from kernelforge.loop.scoring_runtime import ScoringMixin
from kernelforge.loop.workspace_git import WorkspaceGitMixin
from kernelforge.loop.search_policy import (
    MARGINAL_GAIN_SCAN_WINDOW,
    MARGINAL_GAIN_WINDOW,
    NO_CHANGES_STREAK_WINDOW,
    RecentProgress,
    SearchPolicyDecision,
    SearchPolicyEngine,
    SearchState,
)
from kernelforge.loop.run_state import (
    ORCHESTRATION_CIRCUIT_OPEN,
    CriticRuling,
    LoopStateStore,
    StallPolicy,
    apply_supervisor_attempt,
    apply_supervisor_intervention,
    make_event,
    reconcile_stale_running_session,
    start_session,
)
from kernelforge.orchestrator.orchestration import OrchestrationInfrastructureError
from kernelforge.orchestrator.supervisor import (
    clear_latest_supervisor_ruling,
    latest_supervisor_ruling_path,
    load_latest_supervisor_ruling,
    persist_supervisor_ruling,
)
from kernelforge.loop.reporting import BestResultPublisher
from kernelforge.loop.device_hazard import DeviceHazard, DeviceHazardLog
from kernelforge.loop.fanout import LaneResult
from kernelforge.tracker import CampaignSegment, ExperimentTracker, Experiment, ExperimentSubject

log = logging.getLogger(__name__)


class IterationLoop(
    AnalysisRuntimeMixin,
    AttemptVerdictMixin,
    BestPublicationMixin,
    CampaignMixin,
    CampaignReportMixin,
    CeilingMixin,
    IterationPhasesMixin,
    KeepCheckpointMixin,
    LaneRuntimeMixin,
    OutcomeMixin,
    RelayRuntimeMixin,
    RoundAdmissionMixin,
    ScoringMixin,
    WorkspaceGitMixin,
):
    """Autonomous kernel optimization loop."""

    def __init__(
        self,
        iter_config: IterationConfig,
        tracker: ExperimentTracker,
        config: Config | None = None,
        resume: bool = False,
        ceiling_estimator: Callable[..., Awaitable[Any]] | None = None,
    ):
        self.ic = iter_config
        # Produces a ceiling report for this kernel, given the case set and per-case latencies the baseline
        # established; ``None`` unless the operator turned the roofline ceiling on. Injected rather than built here so
        # the loop needs no view of the agent-backend registry: the analyst is a different role, and resolving it is
        # the caller's job.
        self._ceiling_estimator = ceiling_estimator
        # Where this campaign's ceiling was published, once it has one: set by ``_establish_ceiling``, never by the
        # caller. It decides when to stop, never which candidate is better: see ``_is_roofline_target_met``.
        self._ceiling_report_path = ""
        # Declared here so persistence works before the methods that populate them have run.
        self._best_case_times: dict[str, float] = {}
        # Pairs this process selected and could not stage.
        self._declined_merge_pairs: set[frozenset[int]] = set()
        # Last reported per-case bandwidth.
        self.last_case_bandwidth: dict[str, dict[str, float | int]] = {}
        self._scoring_state_restored = False
        self._unscored_cases: set[str] = set()
        # New files the last commit or discard could act on neither way, because no ``commit_new_paths`` entry admits
        # them.
        self._refused_new_paths: list[str] = []
        # Allowlisted new files a discard left on the tree because they were already there when this loop took the
        # workspace over.
        self._retained_new_paths: list[str] = []
        # Why the last new-file enumeration could not be read, "" when it could.
        self._new_paths_unreadable: str = ""
        # Untracked paths present when the current iteration began -- captured again before resume recovery, which
        # also discards and runs before any iteration.
        self._pre_untracked: set[str] | None = None
        # Validation and benchmarking invoke the driver with no arguments, so a multi-rank task has no other way to
        # tell it how many ranks to launch.
        if self.ic.nproc_per_node > 1:
            os.environ["FORGE_NPROC_PER_NODE"] = str(self.ic.nproc_per_node)
        else:
            os.environ.pop("FORGE_NPROC_PER_NODE", None)
        self.tracker = tracker
        self.config = config or Config.from_env()
        self.resume = resume
        self.experiment: Experiment | None = None
        self.results: list[IterationResult] = []
        # FIXED per-case baseline wall times (case_id -> ms), captured once on the pristine kernel and never
        # overwritten.
        self._baseline_case_times: dict = dict(self.ic.baseline_case_times)
        # UsageAccumulator for the run (set in run()); lets the analyst fold its token spend into the run total.
        self._usage = None
        self.best_wall_ms: float | None = None
        self.best_mean_case_speedup: float | None = None
        # What the kernel the search starts from scores against the anchor. 1.0 whenever that kernel IS the anchor,
        # and the port's own speedup when a caller supplied the anchor it was ported from.
        self.search_start_mean_case_speedup: float | None = None
        self.start_time: float = 0
        # Total LLM token spend for the run, populated from the UsageAccumulator passed to run() (empty when no agent
        # / no accumulator).
        self.llm_usage: dict = {}
        self.persistence_degraded = False
        self.persistence_errors: list[str] = []
        self._analysis_bundle = None
        self._last_published_analysis_commit = ""
        self._active_analysis_context = None
        self._analysis_diff_results = {}
        self.search_policy_engine = SearchPolicyEngine()
        self._search_policy_decision: SearchPolicyDecision | None = None
        self._reported_window_gain_faults: set[str] = set()
        self.handoff_store: HandoffStore | None = None
        # A committed KEEP recovered during synchronous resume preflight waits here until the async run can restore
        # its post-KEEP profile/archive.
        self._recovered_pending_keep: tuple[dict, IterationResult] | None = None
        # Why the loop stopped: "gate_met" (target reached), "budget_exhausted" (not enough time to admit another
        # session), or "round_budget_exhausted" (time is left, but not enough to finish even the narrowest round).
        self.termination_reason: str = ""
        # The round currently being timed, opened when an iteration is admitted and closed once the next one starts.
        self._round_started_at: float | None = None
        self._round_iteration = 0
        self._round_lanes = 1
        self._round_planning_sec = 0.0
        # What the round spent in the canonical validation and benchmark, which is what prices the next round's
        # dispatch.
        self._round_measurement_sec = 0.0
        # The instant the CAMPAIGN began, which on a resumed session is before this process did: ``start_time`` less
        # the wall-clock earlier sessions already banked.
        self._campaign_started_at: float = 0.0
        # The refusal that ended the campaign, as the line the operator sees, kept so the run summary and the
        # published report can say a round was priced out rather than let it read as a round that found nothing.
        self._refused_round: str = ""
        # Supervisor self-supervision (AVO): the latest free-form ruling to pass verbatim through planning, and the
        # factual progress monitor that decides when to request a new review.
        self._supervisor_ruling: str = ""
        self._latest_optimization_plan_path = ""
        self._last_orchestration_plan_executable: bool | None = None
        # The previous round's Plan Critic ruling, kept for the next round's partition.
        self._last_critic_verdict = ""
        self._last_critic_review = ""
        self.monitor = None

    def _expire_supervisor_ruling(self) -> None:
        """Stop injecting a ruling after its stall episode ends."""
        self._supervisor_ruling = ""
        clear_latest_supervisor_ruling(self.ic.workspace_dir)

    def _checkpoint_llm_usage(self) -> None:
        """Best-effort checkpoint of the latest cumulative LLM usage."""
        if self._usage is None:
            return
        try:
            self.llm_usage = dict(self._usage.totals())
            has_usage = bool(self.llm_usage.get("calls")) or any(
                self.llm_usage.get(key)
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            )
            if self.experiment is not None and has_usage:
                self.tracker.set_llm_usage(
                    self.experiment.experiment_id,
                    self.llm_usage,
                )
        except (OSError, TypeError, ValueError):
            log.debug("failed to checkpoint LLM usage", exc_info=True)

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

    async def run(
        self,
        agents: LoopAgents = LoopAgents(),
        *,
        callbacks: LoopCallbacks = LoopCallbacks(),
        usage=None,
        workspace_lock_held: bool = False,
    ) -> list[IterationResult]:
        """Run the loop and clean attempt-owned processes on every exit path."""
        try:
            return await self._run_impl(
                agents,
                callbacks=callbacks,
                usage=usage,
                workspace_lock_held=workspace_lock_held,
            )
        finally:
            try:
                from kernelforge.tools.aiter_cache import (
                    cleanup_current_owned_aiter_locks,
                )

                cleanup_current_owned_aiter_locks()
            except (OSError, ValueError):
                log.debug("failed to clean AITER locks on loop exit", exc_info=True)

    async def _run_impl(
        self,
        agents: LoopAgents,
        *,
        callbacks: LoopCallbacks,
        usage=None,
        workspace_lock_held: bool = False,
    ) -> list[IterationResult]:
        """Run while exclusively owning this campaign workspace."""
        if workspace_lock_held:
            return await self._run_locked(agents, callbacks=callbacks, usage=usage)
        store = LoopStateStore(self.ic.workspace_dir)
        with store.workspace_lock():
            return await self._run_locked(agents, callbacks=callbacks, usage=usage)

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

    def _initialize_campaign_runtime(self, usage) -> None:
        """Initialize process-local stores and candidate queues for one session."""
        self.start_time = time.time()
        self.results = []
        self._usage = usage

        # Safety net: if the kernel is an aiter HIP kernel, force it to recompile from the current source
        # (AITER_REBUILD) so the agent's edits are never silently ignored via aiter's prebuilt in-tree .so.
        force_jit_rebuild(self._jit_source_files())

        # Cross-iteration objective ledger: records each iteration's net diff, measured outcome, and real error
        # signatures, then feeds concise toolchain observations and recent entries into the next prompt.
        self.ledger = ExperienceLedger(self.ic.workspace_dir)

        # Per-iteration lesson documents.
        self.lessons = LessonStore(self.ic.workspace_dir)
        self.handoff_store = HandoffStore(self.ic.workspace_dir)

        # Full-fidelity candidate archive: persists each iteration's WHOLE solution (kernel snapshot + diff + full
        # profile + measurements + decision) so a later iteration can read back any prior attempt's real code.
        self.archive = CandidateArchive(self.ic.workspace_dir, self.ic.kernel_file)
        self.best_publisher = BestResultPublisher(self.ic.workspace_dir)
        # Candidates from one concurrent fan-out, spent one per iteration so each is measured and judged on its own by
        # the ordinary decision path.
        self._lane_queue: list[LaneResult] = []
        self._last_lane_plans: list[str] = []
        # Stacked iterations run back to back so far.
        self._merge_precedence_streak = 0
        # A device the campaign may not measure on, recorded by whichever iteration found it and re-checked by every
        # iteration after it.
        self.device_hazard = DeviceHazardLog(self.ic.workspace_dir)

    async def _run_locked(
        self,
        agents: LoopAgents,
        *,
        callbacks: LoopCallbacks,
        usage=None,
    ) -> list[IterationResult]:
        """Run the autonomous iteration loop."""
        import functools

        self._search_actors = agents.search_actors

        global print
        print = functools.partial(print, flush=True)  # noqa: A001 - deliberate flush-on-print rebind

        self._initialize_campaign_runtime(usage)

        # Durable, file-backed run state + append-only event log.
        self.state_store = LoopStateStore(self.ic.workspace_dir)
        state_exists = self.state_store.state_path.exists()
        self.run_state = self.state_store.load()
        actors = _with_search_actor(agents, self._session_start_bundle())
        agent_fn, agent_factory, supervisor_fn = actors.agent_fn, actors.agent_factory, actors.supervisor_fn
        analysis_service, orchestration_service = actors.analysis_service, actors.orchestration_service
        current_ruling_path = latest_supervisor_ruling_path(self.ic.workspace_dir)
        self._supervisor_ruling = load_latest_supervisor_ruling(self.ic.workspace_dir) if self.resume else ""

        # Ownership boundary for anything a candidate creates, taken before this loop touches the workspace.
        self._pre_untracked = self._untracked_snapshot()

        self._restore_or_begin_campaign(
            state_exists=state_exists,
            current_ruling_path=current_ruling_path,
            agent_fn=agent_fn,
            orchestration_service=orchestration_service,
            on_best_committed=callbacks.on_best_committed,
        )

        # Anchor the campaign clock now that the state carrying what earlier sessions spent is loaded.
        self._campaign_started_at = self.start_time - max(0.0, float(self.run_state.round_costs.campaign_sec))

        if state_exists and reconcile_stale_running_session(self.run_state):
            self.state_store.append_event(
                make_event(
                    "session_interrupted",
                    self.run_state.iteration,
                    reason="stale_running_session_reconciled",
                )
            )
            self.state_store.save(self.run_state)

        parent_experiment_id = self.run_state.last_experiment_id
        next_segment_index = self.run_state.session_index + 1
        self._set_state_identity(self.run_state)
        self._stage_validated_warm_start_state()
        start_session(self.run_state)
        self._record_agent_availability()
        self.experiment = self.tracker.create_segment(
            CampaignSegment(self.run_state.campaign_id, next_segment_index, parent_experiment_id),
            ExperimentSubject(
                task_id=Path(self.ic.kernel_file).stem,
                backend=self.ic.backend,
                kernel_backend=self.ic.kernel_backend,
                description=f"Autonomous optimization of {self.ic.kernel_file}",
                target_wall_ms=self.ic.target_wall_ms,
                baseline_wall_ms=self.ic.baseline_wall_ms,
            ),
        )
        self.run_state.last_experiment_id = self.experiment.experiment_id
        self.state_store.append_event(
            make_event(
                "session_started",
                self.run_state.iteration,
            )
        )
        self.state_store.save(self.run_state)

        self._persist_pr_knowledge()
        self._announce_run(supervisor_fn)

        await self._anchor_speedup_reporting()

        # Seed the run state's baseline and, guardedly, resume a prior best from a reused workspace (only when the
        # recorded best commit is still HEAD).
        self._seed_and_hydrate_run_state()
        self._adopt_validated_warm_start()

        self._verify_baseline_anchor()

        # Once the anchor is settled and verified, so the estimate is handed the case set the objective scores and
        # latencies from the campaign's own clock.
        await self._establish_ceiling()

        # A crash immediately after a verified commit can leave the KEEP's archive unfinished.
        await self._finish_recovered_pending_keep()

        # Candidates an earlier round bought and never measured.
        self._restore_lane_queue()

        # The previous round's verdict, for the same reason: a REPLACE is spent on the round after the one it judged,
        # and the budget often ends between the two.
        self._restore_critic_ruling()

        # Analyze the baseline canonical commit once before any specialist or Implementer session.
        if analysis_service is not None:
            await self._resolve_analysis_context(analysis_service)

        iteration = self.run_state.next_iteration - 1
        while True:
            iteration += 1

            # Build the lineage digest once per iteration — reused by BOTH the supervisor (trajectory to review) and
            # the implementer (prompt history).
            digest = ""
            if getattr(self, "archive", None) is not None:
                digest = self.archive.render_digest()

            # Ruled on once per iteration, with the other conditions that decide whether this iteration may run at
            # all -- and after them, so a run that was stopping anyway never touches the device.
            stop, hazard = self._stop_condition_met(iteration)
            if stop:
                break

            actors = _with_search_actor(actors, self._iteration_bundle(iteration))
            agent_fn, agent_factory, supervisor_fn = actors.agent_fn, actors.agent_factory, actors.supervisor_fn
            orchestration_service = actors.orchestration_service

            # Price the round before anything is spent on it.
            self._close_round()
            round_lanes = self.ic.lanes
            if not self._lane_queue:
                admitted_lanes = self._admit_next_round(iteration)
                if admitted_lanes is None:
                    break
                round_lanes = admitted_lanes
                self._open_round(iteration, lanes=round_lanes)

            await self._open_iteration(
                iteration=iteration,
                digest=digest,
                supervisor_fn=supervisor_fn,
                analysis_service=analysis_service,
            )

            # Agent proposes modification
            session_sink: dict = {}
            # What a fan-out round leaves this iteration holding, so the single-session path below spends it instead
            # of buying it again.
            fan_out_plan: HeldRound | None = None
            if hazard is None and (
                not self._lane_queue
                and round_lanes > 1
                and agent_factory is not None
                and agent_fn is not None
                and orchestration_service is not None
            ):
                fan_out_plan = await self._fan_out_round(
                    iteration=iteration,
                    orchestration_service=orchestration_service,
                    agent_factory=agent_factory,
                    lanes=round_lanes,
                )
                # The round's own lanes share one device with the canonical measurement, so a lane whose teardown
                # could not clear it has just refused this iteration -- whatever its siblings produced.
                hazard = self.device_hazard.live
                if self._refused_round:
                    # Planning cost more than the round had left.
                    break
            merge = self._stage_merge(
                iteration=iteration,
                hazard=hazard,
                fan_out_plan=fan_out_plan,
            )
            # A fan-out round already paid for these candidates, so they are measured before anything new is planned.
            queued_lane = None if hazard or merge.diff else self._take_lane_candidate()
            frame = _IterationFrame(iteration, session_sink, merge)
            attempt = await self._source_attempt(
                frame,
                _IterationSupply(hazard, queued_lane, fan_out_plan),
                agents=agents,
                digest=digest,
                callbacks=callbacks,
            )
            if attempt is None:
                if self._refused_round:
                    break
                continue
            (
                unmeasured_result,
                commit_hash,
                rationale,
                attempt_source,
                attempt_diff,
                reusable_benchmark,
                optimization_plan_path,
                optimization_plan_executable,
            ) = attempt

            # Snapshot the best-so-far BEFORE this iteration is measured, so the archive can record the true delta vs
            # the standard it had to beat (run_one_iteration mutates self.best_wall_ms on an improvement).
            best_before = self.best_wall_ms
            best_mean_case_speedup_before = self.best_mean_case_speedup
            frame = replace(
                frame,
                best_before=best_before,
                best_mean_case_speedup_before=best_mean_case_speedup_before,
            )

            # Run validation + bench.
            if unmeasured_result is not None:
                result = unmeasured_result
            else:
                try:
                    run_kwargs = {}
                    if reusable_benchmark is not None:
                        run_kwargs["benchmark_measurement"] = reusable_benchmark
                    result = await self.run_one_iteration(
                        iteration,
                        plan=session_sink.get("plan", ""),
                        **run_kwargs,
                    )
                except Exception as e:  # noqa: BLE001
                    # Turn the crash into a FAILED result (crashed=True) and let it flow through the same
                    # verdict/ledger/archive path.
                    print(f"  [CRASH] iteration {iteration} crashed during run: {e}")
                    result = IterationResult(
                        iteration=iteration,
                        duration_sec=0.0,
                        validation_passed=False,
                        validation_summary=f"iteration crashed: {e}",
                        error_output=traceback.format_exc()[-4000:],
                        kept=False,
                        crashed=True,
                    )
            result.commit_hash = commit_hash
            result.agent_rationale = rationale
            self._stamp_result_profile(result, session_sink)
            # Session end reason + turns spent (from the in-session gate / SDK via session_sink) — persisted so a
            # run's end-reason distribution (edit cap / turn cap / converged / …) is analyzable.
            result.session_end_reason = session_sink.get("end_reason", "")
            result.turns = session_sink.get("turns")

            if agent_fn is not None:
                self._apply_iteration_planning_state(
                    optimization_plan_created=optimization_plan_executable,
                )

            ruling = self._rule_on_attempt(result, attempt, frame, on_best_committed=callbacks.on_best_committed)

            await self._close_iteration(result, attempt, frame, ruling=ruling, callbacks=callbacks)

        self._persist_terminal_state()

        self._print_final_report()

        return self.results
