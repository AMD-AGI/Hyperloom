# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One iteration run while this process holds the device lock: taking it, settling the attempt, releasing it."""

from __future__ import annotations

import sys
import logging
import traceback
from dataclasses import replace
from pathlib import Path

from kernelforge.workflows.forge_loop.iteration.iteration_types import (
    HeldRound,
    IterationResult,
    LoopAgents,
    LoopCallbacks,
    _IterationFrame,
    _IterationSupply,
    _with_search_actor,
)
from kernelforge.workflows.forge_loop.campaign.run_state import (
    LoopStateStore,
    make_event,
    reconcile_stale_running_session,
    start_session,
)
from kernelforge.workflows.forge_loop.roles.supervisor import (
    latest_supervisor_ruling_path,
    load_latest_supervisor_ruling,
)
from kernelforge.workflows.forge_loop.campaign.schema import CampaignSegment, ExperimentSubject

log = logging.getLogger(__name__)


class LockedRunMixin:
    """Running an iteration under the exclusive device lock."""

    async def _begin_locked_run(
        self,
        agents: LoopAgents,
        *,
        callbacks: LoopCallbacks,
        usage=None,
    ) -> LoopAgents:
        """Load durable state, open the session segment, and seed the campaign clock."""
        self._search_actors = agents.search_actors
        # The loop's progress is read off a pipe (workflows.flydsl_rewrite.optimize captures stdout
        # and may hard-kill the run), so every print has to leave the buffer as it is made.
        # One line-buffered stream covers all of them; a ``print`` rebind would only reach
        # the module it is made in.
        sys.stdout.reconfigure(line_buffering=True)
        self._initialize_campaign_runtime(usage)
        self.state_store = LoopStateStore(self.ic.workspace_dir)
        state_exists = self.state_store.state_path.exists()
        self.run_state = self.state_store.load()
        actors = _with_search_actor(agents, self._session_start_bundle())
        self._supervisor_ruling = load_latest_supervisor_ruling(self.ic.workspace_dir) if self.resume else ""
        # Ownership boundary for anything a candidate creates, taken before this loop touches the workspace.
        self._pre_untracked = self._untracked_snapshot()
        self._restore_or_begin_campaign(
            state_exists=state_exists,
            current_ruling_path=latest_supervisor_ruling_path(self.ic.workspace_dir),
            agent_fn=actors.agent_fn,
            orchestration_service=actors.orchestration_service,
            on_best_committed=callbacks.on_best_committed,
        )
        # Anchor the campaign clock now that the state carrying what earlier sessions spent is loaded.
        self._campaign_started_at = self.start_time - max(0.0, float(self.run_state.round_costs.campaign_sec))
        if reconcile_stale_running_session(self.run_state):
            self.state_store.append_event(
                make_event(
                    "session_interrupted",
                    self.run_state.iteration,
                    reason="stale_running_session_reconciled",
                )
            )
            self.state_store.save(self.run_state)
        # Captured before start_session, which increments session_index and clears last_experiment_id.
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
        self.state_store.append_event(make_event("session_started", self.run_state.iteration))
        self.state_store.save(self.run_state)
        self._persist_pr_knowledge()
        self._announce_run(actors.supervisor_fn)
        await self._anchor_speedup_reporting()
        self._seed_and_hydrate_run_state()
        self._adopt_validated_warm_start()
        self._verify_baseline_anchor()
        # Once the anchor is settled and verified, so the estimate is handed the case set the objective scores and
        # latencies from the campaign's own clock.
        await self._establish_ceiling()
        # A crash immediately after a verified commit can leave the KEEP's archive unfinished.
        await self._finish_recovered_pending_keep()
        self._restore_lane_queue()
        # The previous round's verdict: a REPLACE is spent on the round after the one it judged, and the budget often
        # ends between the two.
        self._restore_critic_ruling()
        # Analyze the baseline canonical commit once before any specialist or Implementer session.
        if actors.analysis_service is not None:
            await self._resolve_analysis_context(actors.analysis_service)
        return actors

    async def _run_locked(
        self,
        agents: LoopAgents,
        *,
        callbacks: LoopCallbacks,
        usage=None,
    ) -> list[IterationResult]:
        """Run the autonomous iteration loop."""
        actors = await self._begin_locked_run(agents, callbacks=callbacks, usage=usage)
        analysis_service = actors.analysis_service
        iteration = self.run_state.next_iteration - 1
        while True:
            iteration += 1
            # Build the lineage digest once per iteration — reused by BOTH the supervisor (trajectory to review) and
            # the implementer (prompt history).
            digest = self.archive.render_digest()
            stop, hazard = self._stop_condition_met(iteration)
            if stop:
                break
            actors = _with_search_actor(actors, self._iteration_bundle(iteration))
            agent_fn, agent_factory, supervisor_fn = actors.agent_fn, actors.agent_factory, actors.supervisor_fn
            orchestration_service = actors.orchestration_service
            round_lanes = self._price_round(iteration)
            if round_lanes is None:
                break
            await self._open_iteration(
                iteration=iteration,
                digest=digest,
                supervisor_fn=supervisor_fn,
                analysis_service=analysis_service,
            )
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
            await self._settle_locked_attempt(attempt, frame, callbacks=callbacks)
        self._persist_terminal_state()
        self._print_final_report()
        return self.results

    async def _settle_locked_attempt(
        self,
        attempt,
        frame: _IterationFrame,
        *,
        callbacks: LoopCallbacks,
    ) -> None:
        """Measure one sourced attempt, stamp it, rule on it, and close the iteration."""
        (
            unmeasured_result,
            commit_hash,
            rationale,
            _attempt_source,
            _attempt_diff,
            reusable_benchmark,
            _optimization_plan_path,
            optimization_plan_executable,
            _lane_record,
        ) = attempt
        # Snapshot the best-so-far BEFORE this iteration is measured, so the archive can record the true delta vs
        # the standard it had to beat (run_one_iteration mutates self.best_wall_ms on an improvement).
        frame = replace(
            frame,
            best_before=self.best_wall_ms,
            best_mean_case_speedup_before=self.best_mean_case_speedup,
        )
        if unmeasured_result is not None:
            result = unmeasured_result
        else:
            try:
                result = await self.run_one_iteration(
                    frame.iteration,
                    plan=frame.session_sink.get("plan", ""),
                    **({"benchmark_measurement": reusable_benchmark} if reusable_benchmark is not None else {}),
                )
            except Exception as e:  # isolation boundary: one iteration's measurement
                # Turn the crash into a FAILED result (crashed=True) and let it flow through the same
                # verdict/ledger/archive path.
                log.warning("iteration %s crashed during run", frame.iteration, exc_info=True)
                print(f"  [CRASH] iteration {frame.iteration} crashed during run: {e}")
                result = IterationResult(
                    iteration=frame.iteration,
                    duration_sec=0.0,
                    validation_passed=False,
                    validation_summary=f"iteration crashed: {e}",
                    error_output=traceback.format_exc()[-4000:],
                    kept=False,
                    crashed=True,
                )
        result.commit_hash = commit_hash
        result.agent_rationale = rationale
        self._stamp_result_profile(result, frame.session_sink)
        # Session end reason + turns spent (from the in-session gate / SDK via session_sink) — persisted so a
        # run's end-reason distribution (edit cap / turn cap / converged / …) is analyzable.
        result.session_end_reason = frame.session_sink.get("end_reason", "")
        result.turns = frame.session_sink.get("turns")
        self._apply_iteration_planning_state(
            optimization_plan_created=optimization_plan_executable,
        )
        ruling = self._rule_on_attempt(result, attempt, frame, on_best_committed=callbacks.on_best_committed)
        await self._close_iteration(result, attempt, frame, ruling=ruling, callbacks=callbacks)
