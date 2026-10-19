# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The front half of one iteration: what it is scoped to own, and how the candidate it will measure is produced."""

from __future__ import annotations

import inspect
import logging
import subprocess
from pathlib import Path

from kernelforge.agent_backends.session_resume import EXHAUSTED_END_REASON
from kernelforge.loop.iteration.iteration_types import (
    IterationResult,
    LoopAgents,
    LoopCallbacks,
    _Attempt,
    _IterationFrame,
    _IterationSupply,
    _RoundPlan,
    _append_finding,
)
from kernelforge.loop.fanout import LaneResult
from kernelforge.loop.merge_candidates import merge_plan
from kernelforge.loop.run_state import complete_orchestration_probe, make_event
from kernelforge.orchestrator.orchestration import OrchestrationInfrastructureError
from kernelforge.tools.process_reaping import processes_under

log = logging.getLogger(__name__)


def _lane_rationale(lane: LaneResult) -> str:
    """The rationale of the iteration that measures a queued lane, with the lane Implementer's own when it gave one."""
    head = f"lane {lane.lane_id} of a fan-out round"
    return f"{head}: {lane.rationale}" if lane.rationale else head


class IterationPhasesMixin:
    """Open an iteration, source the attempt it measures from a session, a lane or a merge, and abandon it when nothing can be."""

    async def _open_iteration(self, *, iteration, digest, supervisor_fn, analysis_service) -> None:
        """Resolve evidence, review the trajectory and re-scope what this iteration owns.

        Analysis resolves before any Supervisor or planning call, so both read
        the same evidence. A stall triggers the reviewer and the loop ALWAYS
        CONTINUES -- it never self-terminates on one. Untracked files already
        on the tree belong to the operator or to an earlier round, so the
        boundary is re-taken here and this iteration's REVERT cannot delete them.
        """
        supervisor_due = False
        supervisor_reason = ""
        if self.monitor is not None and supervisor_fn is not None:
            supervisor_due, supervisor_reason = self._supervisor_due(iteration)

        # Resolve Analysis before any Supervisor or planning call.
        if analysis_service is not None:
            await self._resolve_analysis_context(
                analysis_service,
                supervisor_due=supervisor_due,
                iteration=iteration,
            )

        # Self-supervision (AVO): when supervised, a stall triggers a reviewer that injects fresh directions and
        # the loop ALWAYS CONTINUES — it never self-terminates on stall.
        await self._run_supervisor_review(
            supervisor_fn,
            iteration=iteration,
            digest=digest,
            supervisor_due=supervisor_due,
            supervisor_reason=supervisor_reason,
        )

        self._update_search_policy(iteration)

        print(
            f"--- Iteration {iteration} "
            f"(best mean case speedup: {self.best_mean_case_speedup:.6f}x, "
            f"remaining: {self._time_remaining() / 60:.0f} min) ---"
            if self.best_mean_case_speedup is not None
            else f"--- Iteration {iteration} ---"
        )
        roofline_progress = self._render_roofline_progress()
        if roofline_progress:
            print(roofline_progress)

        # Re-scope the ownership boundary to this iteration: untracked files already here are the operator's or an
        # earlier round's, and this iteration's REVERT must not delete them.
        snapshot = self._untracked_snapshot()
        if snapshot is not None:
            self._pre_untracked = snapshot

        # Durable per-iteration marker (facts only; detail lives in files).
        self.run_state.iteration = iteration
        self.state_store.append_event(
            make_event(
                "iteration_started",
                iteration,
                best_before_ms=self.best_wall_ms,
                best_before_mean_case_speedup=self.best_mean_case_speedup,
                phase=self.run_state.phase,
            )
        )

    async def _source_attempt(
        self,
        frame: _IterationFrame,
        supply: _IterationSupply,
        *,
        agents: LoopAgents,
        digest,
        callbacks: LoopCallbacks,
    ) -> _Attempt | None:
        """Decide what this iteration measures, and produce it.

        A held device, a queued lane candidate and a staged merge each cost
        nothing to source, so they are answered before an Implementer
        session is bought. Returns None when the iteration has nothing to
        measure: either it was closed out here, or the round was refused for
        budget and ``_refused_round`` now says the campaign is over.
        """
        iteration, session_sink = frame.iteration, frame.session_sink
        merge_pair, merge_diff = frame.merge
        hazard, queued_lane, fan_out_plan = supply.hazard, supply.queued_lane, supply.fan_out_plan
        agent_fn, orchestration_service, on_iteration = (
            agents.agent_fn,
            agents.orchestration_service,
            callbacks.on_iteration,
        )
        # Only an orchestrated round synthesizes a plan; every other way of
        # sourcing an attempt leaves the iteration without one.
        optimization_plan_path = ""
        optimization_plan_executable = False
        if hazard is not None:
            unmeasured_result = self._unmeasurable_on_a_held_device(
                iteration=iteration,
                detail=hazard.describe(),
                session_sink=session_sink,
            )
            commit_hash = ""
            rationale = "device held; nothing was planned, run or measured"
            attempt_source = ""
            attempt_diff = ""
            reusable_benchmark = None
        elif queued_lane is not None:
            unmeasured_result = None
            commit_hash = ""
            rationale = _lane_rationale(queued_lane)
            attempt_source = self._read_kernel_source()
            attempt_diff = self._working_tree_diff()
            reusable_benchmark = None
            session_sink["plan"] = queued_lane.plan
            self._bind_session_producer(session_sink, queued_lane.agent_profile_id)
            print(f"  [lane {queued_lane.lane_id}] measuring queued candidate")
        elif merge_diff and merge_pair is not None:
            unmeasured_result = None
            commit_hash = ""
            rationale = (
                f"stacked iterations {merge_pair[0].iteration} and {merge_pair[1].iteration}; no Implementer session"
            )
            attempt_source = self._read_kernel_source()
            attempt_diff = merge_diff
            reusable_benchmark = None
            session_sink["plan"] = merge_plan(*merge_pair)
            print(f"  [merge] {session_sink['plan']}")
            # How often the mechanism engaged.
            self.state_store.append_event(
                make_event(
                    "merge_attempt_staged",
                    iteration,
                    first_iteration=merge_pair[0].iteration,
                    second_iteration=merge_pair[1].iteration,
                    cases=sorted(merge_pair[0].winning_cases | merge_pair[1].winning_cases),
                    unresolved_stall_iters=(self.run_state.stall.unresolved_stall_iters),
                )
            )
        elif agent_fn is not None:
            print("  [agent] Querying agent for kernel modification...")
            self._bind_session_producer(session_sink, self.run_state.agent_relay.active_profile_id)
            if orchestration_service is not None:
                if fan_out_plan is not None:
                    plan_path, orchestration_error = fan_out_plan
                else:
                    print("  [orchestration] analyzing and dispatching specialists...")
                    self._last_orchestration_plan_executable = None
                    plan_path, orchestration_error = await self._plan_round(
                        iteration=iteration,
                        orchestration_service=orchestration_service,
                    )
                if plan_path is None:
                    await self._abandon_iteration(
                        IterationResult(
                            iteration=iteration,
                            duration_sec=0.0,
                            validation_passed=False,
                            validation_summary=(f"ORCHESTRATION ERROR: {orchestration_error}"),
                            session_end_reason="orchestration_error",
                        ),
                        frame,
                        decision_label="ORCHESTRATION_ERROR",
                        on_iteration=on_iteration,
                    )
                    return None
                optimization_plan_path = str(plan_path)
                optimization_plan_executable = (
                    self._last_orchestration_plan_executable
                    if self._last_orchestration_plan_executable is not None
                    else True
                )
                complete_orchestration_probe(self.run_state)
                self.state_store.save(self.run_state)
            # The last point before the round buys its session, and the first at which what planning cost is a
            # measurement rather than an estimate.
            if not self._admit_dispatch(iteration):
                return None
            session_sink["session_started"] = True
            history = self._implementer_history(
                digest=digest,
                optimization_plan_path=optimization_plan_path,
            )
            # Pass extras only to agent_fns that declare them: the in-session gate uses the current best mean case
            # speedup and immutable pristine per-case timings; session_sink hands back findings and the
            # resumed-session factual-record callback.
            extra_kwargs = {}
            try:
                params = inspect.signature(agent_fn).parameters
                if "baseline_case_times" in params:
                    extra_kwargs["baseline_case_times"] = dict(self._baseline_case_times)
                if "best_mean_case_speedup" in params:
                    extra_kwargs["best_mean_case_speedup"] = self.best_mean_case_speedup
                if "session_sink" in params:
                    extra_kwargs["session_sink"] = session_sink
            except (ValueError, TypeError):
                log.debug("could not introspect agent_fn signature", exc_info=True)
            agent_error = None
            try:
                rationale = await agent_fn(
                    self.ic.kernel_file,
                    history,
                    **extra_kwargs,
                )
                print(f"  [agent] Rationale: {rationale[:200]}")
            except Exception as e:  # isolation boundary: one implementer agent session
                log.warning("implementer agent session failed at iteration %s", iteration, exc_info=True)
                agent_error = e
                print(f"  [agent] ERROR: {e}")
                rationale = f"agent session ended with error after edits: {e}"
                session_sink.setdefault("end_reason", "sdk_error")
                session_sink.setdefault(
                    "findings",
                    f"Agent session error before outer validation: {e}",
                )
            finally:
                # The SDK result stream has completed (or unwound).
                self._checkpoint_llm_usage()

            return await self._collect_session_candidate(
                frame,
                agent_error=agent_error,
                rationale=rationale,
                plan=_RoundPlan(optimization_plan_path, optimization_plan_executable),
                on_iteration=on_iteration,
            )
        else:
            unmeasured_result = None
            commit_hash = ""
            rationale = "no-agent (baseline measurement)"
            attempt_source = ""
            attempt_diff = ""
            reusable_benchmark = None
        return _Attempt(
            unmeasured_result=unmeasured_result,
            commit_hash=commit_hash,
            rationale=rationale,
            attempt_source=attempt_source,
            attempt_diff=attempt_diff,
            reusable_benchmark=reusable_benchmark,
            optimization_plan_path=optimization_plan_path,
            optimization_plan_executable=optimization_plan_executable,
        )

    async def _collect_session_candidate(
        self,
        frame: _IterationFrame,
        *,
        agent_error,
        rationale,
        plan: _RoundPlan,
        on_iteration,
    ) -> _Attempt | None:
        """Collect what the Implementer session left behind, or None if nothing can be.

        A session that tainted protected state or left the workspace
        contended produced a candidate that exists but must not be measured,
        so it comes back already judged. A session that changed nothing
        tracked is closed out here and the iteration is abandoned.
        """
        iteration, session_sink = frame.iteration, frame.session_sink
        optimization_plan_path, optimization_plan_executable = plan.path, plan.executable
        # Either kind of "this candidate exists but must not be measured": protected state was tainted, or the
        # workspace is still busy.
        unmeasured_result: IterationResult | None = None
        if session_sink.get("integrity_violation") is True:
            # Capture evidence before restoration.
            attempt_diff = self._working_tree_diff()
            attempt_source = self._read_kernel_source()
            integrity_reason = str(session_sink.get("integrity_reason") or "protected workspace state changed")
            restore_errors: list[str] = []
            restore = session_sink.get("integrity_restore")
            if callable(restore):
                try:
                    restore()
                except (RuntimeError, OSError) as error:
                    restore_errors.append(f"protected snapshot restore failed: {type(error).__name__}: {error}")
            else:
                restore_errors.append("protected snapshot restore callback unavailable")
            try:
                self._git_discard_worktree()
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                restore_errors.append(f"tracked candidate restore failed: {type(error).__name__}: {error}")
            if not restore_errors:
                try:
                    self._validate_driver_integrity(self.run_state)
                except (ValueError, OSError) as error:
                    restore_errors.append(str(error))
            summary = (
                "REVERT (protected integrity violation): canonical "
                "correctness and benchmark were skipped before executing "
                f"the measurement surface. {integrity_reason}"
            )
            if restore_errors:
                summary += " Restoration errors: " + "; ".join(restore_errors)
            _append_finding(session_sink, summary)
            unmeasured_result = IterationResult(
                iteration=iteration,
                duration_sec=0.0,
                validation_passed=False,
                validation_summary=summary,
                kept=False,
                integrity_violation=True,
            )
            reusable_benchmark = None
            print("  [REVERT] Protected integrity violation; canonical validation skipped")
        elif str(session_sink.get("workspace_contention") or ""):
            # The session's own processes are still running in the workspace, or someone else's are and they
            # are not ours to kill.
            contention = str(session_sink["workspace_contention"])
            # Nothing about the end of this iteration makes those processes leave, so the refusal is recorded
            # and re-checked rather than forgotten here.
            self.device_hazard.record(
                iteration=iteration,
                detail=contention,
                pids=processes_under(self.ic.workspace_dir),
            )
            attempt_diff = self._working_tree_diff()
            attempt_source = self._read_kernel_source()
            summary = (
                "REVERT (workspace contention): canonical correctness "
                "and benchmark were skipped because the session's "
                f"workspace could not be cleared. {contention}"
            )
            try:
                # The candidate itself may be sound, but nothing here can establish that, and HEAD has to stay
                # at the last measured best rather than carry an unmeasured diff into the next iteration.
                self._git_discard_worktree()
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                summary += f" Candidate restore failed: {type(error).__name__}: {error}"
            _append_finding(session_sink, summary)
            unmeasured_result = IterationResult(
                iteration=iteration,
                duration_sec=0.0,
                validation_passed=False,
                validation_summary=summary,
                kept=False,
                workspace_contention=contention,
            )
            reusable_benchmark = None
            print("  [REVERT] Workspace still contended; canonical measurement skipped")
        else:
            # The driver is the measurement boundary and must remain byte-for-byte canonical.
            self._validate_driver_integrity(self.run_state)

            # Keep HEAD at the last validated best state while this candidate remains unverified.
            attempt_diff = self._working_tree_diff()
            if not attempt_diff.strip():
                # An outage leaves the same empty diff as a deliberate no-op.
                api_failed = session_sink.get("end_reason") == EXHAUSTED_END_REASON
                # A file the agent created is not in the tracked diff.
                new_files_only = self._new_paths_need_discard()
                if new_files_only:
                    self._git_discard_worktree()
                if agent_error:
                    decision_label = "AGENT_ERROR"
                    summary = f"agent_fn error: {agent_error}"
                elif api_failed:
                    decision_label = "API_ERROR"
                    summary = (
                        "LLM API never answered this Session; no candidate was attempted (not an optimization result)"
                    )
                elif new_files_only:
                    decision_label = "NO_CHANGES"
                    summary = (
                        "NO TRACKED CHANGES: the whole candidate was in "
                        "new file(s) matching "
                        f"{', '.join(self.ic.commit_new_paths)}. A KEEP "
                        "commit is built from the tracked diff, so an "
                        "allowlisted new file can only ship alongside a "
                        "tracked edit. The file was taken off the tree "
                        "rather than measured with the next candidate."
                    )
                else:
                    decision_label = "NO_CHANGES"
                    summary = "NO TRACKED CHANGES: agent produced no candidate diff"
                print("  [agent] No tracked source changes; skipping candidate")
                result = IterationResult(
                    iteration=iteration,
                    duration_sec=0.0,
                    validation_passed=False,
                    validation_summary=summary,
                    kept=False,
                )
                self._stamp_result_profile(result, session_sink)
                result.agent_rationale = rationale
                result.session_end_reason = session_sink.get("end_reason", "")
                result.turns = session_sink.get("turns")
                await self._abandon_iteration(
                    result,
                    frame,
                    decision_label=decision_label,
                    on_iteration=on_iteration,
                    round_plan=_RoundPlan(optimization_plan_path, optimization_plan_executable),
                )
                return None
            reusable_benchmark = None
            gate_measurement = session_sink.get("benchmark_measurement")
            if session_sink.get("gate_passed") is True and self._can_reuse_insession_benchmark(
                gate_measurement,
                attempt_diff=attempt_diff,
            ):
                reusable_benchmark = gate_measurement
            # Capture the attempt before any discard or keep commit.
            attempt_source = self._read_kernel_source()
        return _Attempt(
            unmeasured_result=unmeasured_result,
            commit_hash="",
            rationale=rationale,
            attempt_source=attempt_source,
            attempt_diff=attempt_diff,
            reusable_benchmark=reusable_benchmark,
            optimization_plan_path=optimization_plan_path,
            optimization_plan_executable=optimization_plan_executable,
        )

    async def _abandon_iteration(
        self,
        result: IterationResult,
        frame: _IterationFrame,
        *,
        decision_label: str,
        on_iteration,
        round_plan: _RoundPlan = _RoundPlan(),
    ) -> None:
        """Close out an iteration that ended before anything could be measured.

        Nothing reached the measurement surface, but the iteration still owes
        the campaign its outcome, its lesson and its handoff: without them a
        resume would re-plan the round this one already spent.
        """
        iteration, session_sink = frame.iteration, frame.session_sink
        optimization_plan_path, optimization_plan_created = round_plan.path, round_plan.executable
        self.results.append(result)
        self._apply_iteration_planning_state(
            optimization_plan_created=optimization_plan_created,
        )
        self._record_iteration_outcome(
            result,
            plan=session_sink.get("plan", ""),
            decision_label=decision_label,
        )
        await self._record_lesson(
            iteration=iteration,
            result=result,
            decision=decision_label,
            session_sink=session_sink,
        )
        self._record_iteration_handoff(
            iteration=iteration,
            decision=decision_label,
            optimization_plan_path=optimization_plan_path,
            session_sink=session_sink,
        )
        self._publish_optimization_history()
        if on_iteration:
            on_iteration(result)

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
