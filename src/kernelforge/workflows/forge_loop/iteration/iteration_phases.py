# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The front half of one iteration: what it is scoped to own, and how the candidate it will measure is produced."""

from __future__ import annotations

import logging
import subprocess

from kernelforge.agent_backends.session_resume import EXHAUSTED_END_REASON
from kernelforge.evaluation.scoring import GateScoring
from kernelforge.workflows.forge_loop.campaign.decision_labels import DecisionLabel
from kernelforge.workflows.forge_loop.roles.agent_extras import declared_extras
from kernelforge.workflows.forge_loop.iteration.iteration_types import (
    IterationResult,
    LoopAgents,
    LoopCallbacks,
    RoundSession,
    _Attempt,
    _IterationFrame,
    _append_finding,
)
from kernelforge.workflows.forge_loop.campaign.loop_log import iteration_header
from kernelforge.workflows.forge_loop.campaign.run_state import ITERATION_STARTED, make_event
from kernelforge.utils.process_reaping import processes_under

log = logging.getLogger(__name__)


class IterationPhasesMixin:
    """Open an iteration, source the attempt it measures from the round's lanes and their merge, and abandon it when nothing can be."""

    async def _open_iteration(self, *, iteration, digest, supervisor_fn, analysis_service) -> None:
        """Resolve evidence, review the trajectory and re-scope what this iteration owns.

        Analysis resolves before any Supervisor call, so the Supervisor and
        the round's sessions read the same evidence. A stall triggers the reviewer and the loop ALWAYS
        CONTINUES -- it never self-terminates on one. Untracked files already
        on the tree belong to the operator or to an earlier round, so the
        boundary is re-taken here and this iteration's REVERT cannot delete them.
        """
        supervisor_due = False
        supervisor_reason = ""
        if self.monitor is not None and supervisor_fn is not None:
            supervisor_due, supervisor_reason = self._supervisor_due(iteration)

        # Resolve Analysis before any Supervisor call.
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

        print(iteration_header(iteration, self.best_mean_case_speedup, self._time_remaining()))
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
                ITERATION_STARTED,
                iteration,
                best_before_ms=self.best_wall_ms,
                best_before_mean_case_speedup=self.best_mean_case_speedup,
                phase=self.run_state.phase,
            )
        )

    async def _source_attempt(
        self,
        frame: _IterationFrame,
        hazard,
        *,
        agents: LoopAgents,
        digest,
        callbacks: LoopCallbacks,
    ) -> _Attempt | None:
        """Decide what this iteration measures, and produce it.

        A held device costs nothing to answer, so it is answered before a round is bought. Returns None when the
        iteration has nothing to measure: either it was closed out here, or the round was refused for budget and
        ``_refused_round`` now says the campaign is over.
        """
        iteration, session_sink = frame.iteration, frame.session_sink
        if hazard is not None:
            unmeasured = self._unmeasurable_on_a_held_device(
                iteration=iteration, detail=hazard.describe(), session_sink=session_sink
            )
            return _Attempt(unmeasured, "", "device held; nothing was run or measured", "", "", None)
        if agents.agent_fn is None or agents.agent_factory is None:
            return _Attempt(None, "", "no-agent (baseline measurement)", "", "", None)
        return await self._round_attempt(frame, agents=agents, digest=digest, on_iteration=callbacks.on_iteration)

    async def _round_attempt(
        self, frame: _IterationFrame, *, agents: LoopAgents, digest, on_iteration
    ) -> _Attempt | None:
        """Run the round's lanes, merge what they left on the canonical tree, and collect the candidate.

        One changed lane is the candidate as it stands. Several are merged by a session in the canonical workspace,
        which starts from the best of them and is judged against it.
        """
        iteration, session_sink = frame.iteration, frame.session_sink
        # The last point before the round buys its sessions, which the loop cannot interrupt.
        if not self._admit_dispatch(iteration):
            return None
        self._bind_session_producer(session_sink, self.run_state.agent_relay.active_profile_id)
        results = await self._run_lanes(frame, agent_factory=agents.agent_factory, digest=digest)
        sessions = self._lane_sessions(results)
        hazard = self.device_hazard.live
        if hazard is not None:
            session_sink["session_started"] = bool(sessions)
            unmeasured = self._unmeasurable_on_a_held_device(
                iteration=iteration, detail=hazard.describe(), session_sink=session_sink
            )
            return _Attempt(unmeasured, "", "a lane left the device held", "", "", None, tuple(sessions))
        changed = [lane for lane in results if lane.changed]
        start = self._merge_start(changed)
        # A lane whose patch does not apply drops out of the round, and the start is chosen again without it.
        while start is not None and not self._apply_lane(start):
            changed = [lane for lane in changed if lane.changed]
            start = self._merge_start(changed)
        rest = [lane for lane in changed if lane is not start]
        agent_error, rationale = None, ""
        if rest:
            agent_error, rationale = await self._merge_session(
                frame, start, rest, agent_fn=agents.agent_fn, digest=digest
            )
            sessions.append(RoundSession("merge", session_sink))
        else:
            # The round's only lane to leave a change, or its only lane: what that session handed back is the
            # iteration's.
            lead = start if start is not None else results[0]
            session_sink.update(lead.sink)
            rationale = lead.rationale
            agent_error = lead.error or None
            # The lane's own workspace was cleared, or the round refused above; the canonical one was never busy.
            session_sink.pop("workspace_contention", None)
            # A measurement taken while other lanes compiled is not reused as the canonical one.
            if len(results) > 1:
                session_sink.pop("benchmark_measurement", None)
        return await self._collect_session_candidate(
            frame, agent_error=agent_error, rationale=rationale, sessions=tuple(sessions), on_iteration=on_iteration
        )

    async def _merge_session(
        self, frame: _IterationFrame, start, rest, *, agent_fn, digest
    ) -> tuple[Exception | None, str]:
        """Run the session that merges the round's remaining lanes into the canonical workspace."""
        best = self.best_mean_case_speedup
        bar = start.score if start is not None and start.score is not None else best
        print(
            f"  [merge] merging {len(rest)} lane(s) onto "
            f"{'lane ' + start.lane_id if start is not None else 'the round base'} (bar {bar})"
        )
        history = self._implementer_history(
            digest=digest, direction=self._merge_direction(frame.iteration, start, rest)
        )
        return await self._run_session(
            agent_fn,
            self.ic.kernel_file,
            history,
            sink=frame.session_sink,
            gate_scoring=GateScoring(dict(self._baseline_case_times), bar),
        )

    async def _run_session(
        self, agent_fn, kernel_path: str, history: str, *, sink: dict, gate_scoring: GateScoring
    ) -> tuple[Exception | None, str]:
        """Run one Implementer session on ``kernel_path``; return its error, if any, and its rationale."""
        # Pass extras only to agent_fns that declare them: the in-session gate uses the score to beat and the
        # immutable pristine per-case timings; the sink hands back findings and the resumed-session factual-record
        # callback. The lesson turn, not this session, writes the record.
        extras = declared_extras(agent_fn, {"gate_scoring": gate_scoring, "session_sink": sink})
        agent_error = None
        try:
            rationale = await agent_fn(kernel_path, history, **extras)
            print(f"  [agent] Rationale: {rationale[:200]}")
        except Exception as e:  # isolation boundary: one implementer agent session
            log.warning("implementer agent session failed", exc_info=True)
            agent_error = e
            print(f"  [agent] ERROR: {e}")
            rationale = f"agent session ended with error after edits: {e}"
            sink.setdefault("end_reason", "sdk_error")
            sink.setdefault("findings", f"Agent session error before outer validation: {e}")
        finally:
            # The SDK result stream has completed (or unwound).
            self._checkpoint_llm_usage()
        return agent_error, rationale

    async def _collect_session_candidate(
        self,
        frame: _IterationFrame,
        *,
        agent_error,
        rationale,
        sessions: tuple[RoundSession, ...],
        on_iteration,
    ) -> _Attempt | None:
        """Collect what the round left on the canonical tree, or None if nothing can be.

        A merge session that tainted protected state or left the workspace
        contended produced a candidate that exists but must not be measured,
        so it comes back already judged. A round that changed nothing
        tracked is closed out here and the iteration is abandoned.
        """
        iteration, session_sink = frame.iteration, frame.session_sink
        # Either kind of "this candidate exists but must not be measured": protected state was tainted, or the
        # workspace is still busy.
        unmeasured_result: IterationResult | None = None
        if session_sink.get("integrity_violation") is True:
            # Capture evidence before restoration.
            attempt_diff = self._working_tree_diff()
            attempt_source = self._read_kernel_source()
            unmeasured_result = self._refuse_integrity_violation(frame)
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
            unmeasured_result = self._refuse_contended_workspace(frame, contention)
            reusable_benchmark = None
            print("  [REVERT] Workspace still contended; canonical measurement skipped")
        else:
            # The driver is the measurement boundary and must remain byte-for-byte canonical.
            self._validate_driver_integrity(self.run_state)

            # Keep HEAD at the last validated best state while this candidate remains unverified.
            attempt_diff = self._working_tree_diff()
            if not attempt_diff.strip():
                await self._abandon_unchanged_session(
                    frame,
                    agent_error=agent_error,
                    rationale=rationale,
                    sessions=sessions,
                    on_iteration=on_iteration,
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
            sessions=sessions,
        )

    def _refuse_integrity_violation(self, frame: _IterationFrame) -> IterationResult:
        """Restore the protected state the session tainted and judge its candidate unmeasurable."""
        iteration, session_sink = frame.iteration, frame.session_sink
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
        return IterationResult(
            iteration=iteration,
            duration_sec=0.0,
            validation_passed=False,
            validation_summary=summary,
            kept=False,
            integrity_violation=True,
        )

    def _refuse_contended_workspace(self, frame: _IterationFrame, contention: str) -> IterationResult:
        """Take the candidate off a workspace that is still busy and judge it unmeasurable."""
        iteration, session_sink = frame.iteration, frame.session_sink
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
        return IterationResult(
            iteration=iteration,
            duration_sec=0.0,
            validation_passed=False,
            validation_summary=summary,
            kept=False,
            workspace_contention=contention,
        )

    async def _abandon_unchanged_session(
        self,
        frame: _IterationFrame,
        *,
        agent_error,
        rationale,
        sessions: tuple[RoundSession, ...],
        on_iteration,
    ) -> None:
        """Close out a round that left no tracked change, saying why there was none."""
        iteration, session_sink = frame.iteration, frame.session_sink
        # An outage leaves the same empty diff as a deliberate no-op.
        api_failed = session_sink.get("end_reason") == EXHAUSTED_END_REASON
        # A file the agent created is not in the tracked diff.
        new_files_only = self._new_paths_need_discard()
        if new_files_only:
            self._git_discard_worktree()
        if agent_error:
            decision_label = DecisionLabel.AGENT_ERROR.value
            summary = f"agent_fn error: {agent_error}"
        elif api_failed:
            decision_label = DecisionLabel.API_ERROR.value
            summary = "LLM API never answered this Session; no candidate was attempted (not an optimization result)"
        elif new_files_only:
            decision_label = DecisionLabel.NO_CHANGES.value
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
            decision_label = DecisionLabel.NO_CHANGES.value
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
            sessions=sessions,
        )

    async def _abandon_iteration(
        self,
        result: IterationResult,
        frame: _IterationFrame,
        *,
        decision_label: str,
        on_iteration,
        sessions: tuple[RoundSession, ...] = (),
    ) -> None:
        """Close out an iteration that ended before anything could be measured.

        Nothing reached the measurement surface, but the iteration still owes
        the campaign its outcome, its lesson and its handoff: without them a
        resume would run again the round this one already spent.
        """
        iteration, session_sink = frame.iteration, frame.session_sink
        self.results.append(result)
        self._record_iteration_outcome(
            result,
            plan=session_sink.get("plan", ""),
            decision_label=decision_label,
        )
        await self._record_lesson(
            result=result,
            decision=decision_label,
            session_sink=session_sink,
            sessions=sessions,
        )
        self._record_iteration_handoff(iteration=iteration, decision=decision_label, session_sink=session_sink)
        self._publish_optimization_history()
        if on_iteration:
            on_iteration(result)
