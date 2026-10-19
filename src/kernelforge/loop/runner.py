# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Durable, evidence-driven autonomous kernel optimization loop."""

from __future__ import annotations

import logging
import os
from .iteration.attempt_verdict import AttemptVerdictMixin
from .iteration.best_publication import BestPublicationMixin
from .iteration.campaign_report import CampaignReportMixin
from .iteration.campaign_runtime import CampaignMixin
from .iteration.case_coverage import CaseCoverageMixin
from .iteration.ceiling_runtime import CeilingMixin
from .iteration.iteration_phases import IterationPhasesMixin
from .iteration.iteration_types import IterationConfig, IterationResult, LoopAgents, LoopCallbacks
from .iteration.keep_checkpoint import KeepCheckpointMixin
from .iteration.resume_recovery import ResumeRecoveryMixin
from .iteration.lane_runtime import LaneRuntimeMixin
from .iteration.locked_run import LockedRunMixin
from .iteration.outcome_runtime import OutcomeMixin
from .iteration.round_admission import RoundAdmissionMixin
from .iteration.scoring_runtime import ScoringMixin
from .iteration.supervisor_review import SupervisorReviewMixin
from .iteration.wiki_guidance_hooks import WikiGuidanceMixin
from .iteration.workspace_git import WorkspaceGitMixin
from collections.abc import Awaitable, Callable
from kernelforge.config import Config
from kernelforge.loop.analysis_runtime import AnalysisRuntimeMixin
from kernelforge.loop.handoffs import HandoffStore
from kernelforge.loop.relay_runtime import RelayRuntimeMixin
from kernelforge.loop.run_state import LoopStateStore
from kernelforge.loop.search_policy import SearchPolicyDecision, SearchPolicyEngine
from kernelforge.loop.wiki_guided_rounds import WikiGuidedRounds
from kernelforge.tracker import Experiment, ExperimentTracker
from typing import Any

log = logging.getLogger(__name__)


class IterationLoop(
    AnalysisRuntimeMixin,
    AttemptVerdictMixin,
    BestPublicationMixin,
    CampaignMixin,
    CampaignReportMixin,
    CaseCoverageMixin,
    CeilingMixin,
    IterationPhasesMixin,
    KeepCheckpointMixin,
    LaneRuntimeMixin,
    LockedRunMixin,
    OutcomeMixin,
    ResumeRecoveryMixin,
    RelayRuntimeMixin,
    RoundAdmissionMixin,
    ScoringMixin,
    SupervisorReviewMixin,
    WikiGuidanceMixin,
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
        # Knowledge-base-guided planning; None leaves every planning path as it was.
        self.wiki_rounds: WikiGuidedRounds | None = self._open_wiki_rounds(resume) if self.ic.wiki_guided else None
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

    # The one obstacle a later iteration clears on its own, and so the one the caller must not hold against the pair:
    # a tree carrying work is this iteration's accident, not a fact about two archived diffs.
    TREE_ALREADY_DIRTY_OBSTACLE = "the working tree already carried uncommitted work"

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
