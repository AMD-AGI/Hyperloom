# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a finished iteration leaves on the record: the state it advances, the events it writes and the lesson it teaches the next session."""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path

from kernelforge.evaluation.scoring import beats_current_best
from kernelforge.workflows.forge_loop.campaign.decision_labels import DecisionLabel
from kernelforge.workflows.forge_loop.campaign.archive import CandidateRecord
from kernelforge.workflows.forge_loop.iteration.best_publication import _require_best_is_keep
from kernelforge.workflows.forge_loop.iteration.handoffs import IterationHandoff
from kernelforge.workflows.forge_loop.iteration.iteration_types import (
    DirectionEvidence,
    IterationResult,
    WindowGain,
    _Attempt,
    _IterationFrame,
    _Ruling,
    _decision_label,
    _verdict_of,
)
from kernelforge.workflows.forge_loop.memory.lessons import (
    LessonScope,
    OutcomeMeasurement,
    PrReferenceNote,
    SUMMARY_MIN_SECONDS,
    SessionEnd,
    SummaryOutcome,
    SessionObservations,
    UNDISPROVEN_CLAIM,
    build_fallback_document,
    cases_named_in,
    format_outcome_line,
    format_scope_line,
    is_claim_disproved,
    parse_disproof_marker,
    parse_held_fixed,
    parse_negatives_marker,
    summarize_iteration,
)
from kernelforge.workflows.forge_loop.iteration.merge_candidates import case_spreads, cases_beating_reference
from kernelforge.workflows.forge_loop.campaign.run_state import (
    ITERATION_RESULT,
    LoopStanding,
    RunState,
    apply_iteration,
    is_infrastructure_decision,
    make_event,
    measured_nothing,
    pin_iteration,
)
from kernelforge.workflows.forge_loop.search.search_policy import SEARCH_MODE_EXPLOIT
from kernelforge.workflows.forge_loop.roles.supervisor import latest_supervisor_ruling_path

log = logging.getLogger(__name__)


# Decision labels that can possibly mean nothing came out worse this iteration: a kept candidate, and the ones that
# mean no candidate was ever measured.
_LABELS_WITHOUT_A_MEASURED_NEGATIVE = frozenset(
    {
        DecisionLabel.KEEP.value,
        DecisionLabel.NO_CHANGES.value,
        DecisionLabel.API_ERROR.value,
        DecisionLabel.AGENT_ERROR.value,
        DecisionLabel.ORCHESTRATION_ERROR.value,
    }
)


class OutcomeMixin:
    """Reduce one completed iteration into campaign state, the durable event stream, the experience ledger and the session lesson."""

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

    @staticmethod
    def _record_direction_verdict(
        state: RunState,
        *,
        iteration: int,
        decision_label: str,
        evidence: DirectionEvidence,
    ) -> None:
        """Pin a rejected candidate that still measured faster than the incumbent."""
        mean_case_speedup, best_mean_case_speedup = evidence.mean_case_speedup, evidence.best_mean_case_speedup
        bench_detail, incumbent_case_times = evidence.bench_detail, evidence.incumbent_case_times
        if decision_label != "REVERT_PERF":
            return
        if beats_current_best(
            mean_case_speedup,
            best_mean_case_speedup=best_mean_case_speedup,
        ):
            pin_iteration(state, iteration)
            return
        detail = bench_detail if isinstance(bench_detail, dict) else {}
        if not incumbent_case_times or not detail:
            return
        if cases_beating_reference(
            dict(detail.get("case_times") or {}),
            incumbent_case_times,
            case_spreads(detail.get("measurements")),
        ):
            pin_iteration(state, iteration)

    def _apply_new_iteration_outcome(
        self,
        result: IterationResult,
        decision_label: str,
        plan: str,
    ) -> None:
        """Reduce one previously unseen outcome into campaign and relay state."""
        apply_iteration(
            self.run_state,
            _verdict_of(result, decision_label, plan),
            LoopStanding(self.ic.baseline_wall_ms, self.best_wall_ms, self.best_mean_case_speedup),
            self._stall_policy(),
        )
        self._apply_relay_outcome(result, decision_label)

    def _record_iteration_outcome(
        self,
        result: IterationResult,
        *,
        plan: str = "",
        decision_label: str | None = None,
        require_durable: bool = False,
        checkpoint_metadata: dict | None = None,
    ) -> bool:
        """Synchronize one completed attempt into live and durable control state."""
        plan = (plan or "").strip()

        try:
            if decision_label is None:
                decision_label = _decision_label(result)

            existing_events = self._journaled_results_of(result.iteration)
            if len(existing_events) > 1:
                raise ValueError(f"duplicate iteration_result events for iteration {result.iteration}")
            if existing_events:
                existing = existing_events[0]
                if existing.get("decision") != decision_label or (
                    result.kept and existing.get("commit_hash") != result.commit_hash
                ):
                    raise ValueError(f"iteration_result conflicts with iteration {result.iteration}")

            newly_applied = self.run_state.next_iteration <= result.iteration
            if newly_applied:
                self._apply_new_iteration_outcome(result, decision_label, plan)
            elif result.kept:
                _require_best_is_keep(self.run_state.best, result)

            # The monitor remains operational even if durable state I/O fails, but an idempotent recovery must not
            # count the outcome twice.
            if newly_applied and self.monitor is not None and not is_infrastructure_decision(decision_label):
                self.monitor.record(kept=result.kept)
            if result.kept:
                self._expire_supervisor_ruling()
            head_out = self._git("rev-parse", "HEAD").strip()
            if head_out:
                self.run_state.head_commit = head_out.splitlines()[0]
            if not existing_events:
                self.state_store.append_event(
                    self._iteration_result_event(
                        result,
                        plan=plan,
                        decision_label=decision_label,
                        checkpoint_metadata=checkpoint_metadata,
                    )
                )
            if newly_applied:
                self._record_direction_verdict(
                    self.run_state,
                    iteration=result.iteration,
                    decision_label=decision_label,
                    evidence=DirectionEvidence(
                        mean_case_speedup=result.mean_case_speedup,
                        best_mean_case_speedup=self.best_mean_case_speedup,
                        bench_detail=result.bench_detail,
                        incumbent_case_times=self._scored_incumbent_case_times(),
                    ),
                )
            self.state_store.save(self.run_state)

            if require_durable:
                self._require_durable_outcome(result, decision_label)
            return True
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            if require_durable:
                raise RuntimeError(f"failed to finalize iteration {result.iteration} checkpoint") from error
            log.debug("run_state: iteration reduce/save failed", exc_info=True)
            return False

    def _iteration_result_event(
        self,
        result: IterationResult,
        *,
        plan: str,
        decision_label: str | None = None,
        checkpoint_metadata: dict | None = None,
    ) -> dict:
        """Build the canonical durable event for one completed iteration."""
        resolved_decision = decision_label or _decision_label(result)
        return make_event(
            ITERATION_RESULT,
            result.iteration,
            decision=resolved_decision,
            plan=(plan or "").strip()[:120] or None,
            # The mode this iteration actually ran under, which is the direction identity the empty-diff streak is
            # counted against.
            search_mode=self.run_state.search_mode,
            wall_ms=result.wall_ms,
            mean_case_speedup=result.mean_case_speedup,
            snr_db=result.snr_db,
            agent_profile_id=result.agent_profile_id or None,
            agent_provider=result.agent_provider or None,
            agent_model=result.agent_model or None,
            blockage_epoch=result.blockage_epoch,
            error_sig=result.error_signature() or None,
            session_end_reason=result.session_end_reason or None,
            session_index=int(
                (checkpoint_metadata or {}).get(
                    "session_index",
                    self.run_state.session_index,
                )
            ),
            experiment_id=(
                str((checkpoint_metadata or {}).get("experiment_id") or "")
                or (self.experiment.experiment_id if self.experiment else None)
            ),
            turns=result.turns,
            validation_passed=result.validation_passed,
            commit_hash=result.commit_hash or None,
            best_after_ms=(result.wall_ms if result.kept else self.best_wall_ms),
            best_after_mean_case_speedup=(result.mean_case_speedup if result.kept else self.best_mean_case_speedup),
            is_new_best=result.kept,
            diversification_cycle_completed=(self.run_state.diversification_cycle_completed),
        )

    def _record_iteration_handoff(
        self,
        *,
        iteration: int,
        decision: str,
        optimization_plan_path: str,
        session_sink: dict,
        archived_path: Path | None = None,
    ) -> Path | None:
        """Persist one lightweight handoff without duplicating full artifacts."""
        if self.handoff_store is None:
            return None
        try:
            head_lines = self._git("rev-parse", "HEAD").splitlines()
            analysis_commit = (
                self.run_state.best.commit_hash
                or self.run_state.head_commit
                or (head_lines[0] if head_lines else "")
                or self.ic.campaign_base_commit
                or "uncommitted"
            )
            lesson_path = ""
            handoff_plan = str(session_sink.get("plan") or "")
            if getattr(self, "lessons", None) is not None:
                candidate = self.lessons.path(iteration)
                if candidate.is_file():
                    lesson_path = str(candidate.resolve().relative_to(Path(self.ic.workspace_dir).resolve()))
                    # The planner reads handoffs, not lesson documents, and a refutation quoted out of a handoff is
                    # how one sweep at one M became a ban on every case.
                    scope = self.lessons.scope_of(iteration)
                    if scope is not None:
                        line = format_scope_line(scope)
                        handoff_plan = f"{handoff_plan}\n{line}" if handoff_plan else line
            workspace = Path(self.ic.workspace_dir).resolve()
            relative_plan_path = ""
            orchestration_artifacts = ""
            if optimization_plan_path:
                plan_path = Path(optimization_plan_path).resolve()
                if plan_path.is_file():
                    relative_plan_path = str(plan_path.relative_to(workspace))
                    orchestration_artifacts = str(plan_path.parent.relative_to(workspace))
            supervisor_ruling_path = ""
            current_ruling = latest_supervisor_ruling_path(self.ic.workspace_dir)
            if current_ruling.is_file():
                supervisor_ruling_path = str(current_ruling.resolve().relative_to(workspace))
            handoff = IterationHandoff(
                iteration=iteration,
                analysis_commit=analysis_commit,
                canonical_verdict=decision,
                search_mode=self.run_state.search_mode,
                search_reason_codes=tuple(self.run_state.search_reason_codes),
                search_objective=self.run_state.search_objective,
                search_mode_residence_remaining=(self.run_state.search_mode_residence_remaining),
                diversification_cycle_complete=(self.run_state.diversification_cycle_completed),
                optimization_plan_path=relative_plan_path,
                supervisor_ruling_path=supervisor_ruling_path,
                plan=handoff_plan,
                lesson_path=lesson_path,
                orchestration_artifacts=orchestration_artifacts,
                candidate_archive=(
                    str(archived_path.resolve().relative_to(Path(self.ic.workspace_dir).resolve()))
                    if archived_path is not None
                    else ""
                ),
                **self._handoff_agent_provenance(session_sink),
            )
            return self.handoff_store.write(handoff)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.persistence_degraded = True
            self.persistence_errors.append(f"persist handoff iteration {iteration}: {error}")
            self.persistence_errors = self.persistence_errors[-10:]
            log.debug(
                "failed to persist iteration handoff %s",
                iteration,
                exc_info=True,
            )
            return None

    def _loop_measured_a_negative(self, decision: str, result: IterationResult, session_sink: dict) -> bool:
        """Whether the LOOP itself saw something come out worse this iteration."""
        label = (decision or "").strip().upper()
        if label not in _LABELS_WITHOUT_A_MEASURED_NEGATIVE:
            return True
        if label != "KEEP" and result.mean_case_speedup is not None:
            return True
        return bool(str(session_sink.get("findings") or "").strip())

    def _carries_measured_negative(
        self,
        decision: str,
        result: IterationResult,
        session_sink: dict,
        *,
        document: str,
        agent_narrative: bool,
    ) -> bool | None:
        """Whether this DOCUMENT records anything that measured worse."""
        if self._loop_measured_a_negative(decision, result, session_sink):
            return True
        if not agent_narrative:
            return False
        return parse_negatives_marker(document)

    def _lesson_scope(
        self,
        iteration: int,
        session_sink: dict,
        *,
        decision: str,
        result: IterationResult,
        agent_narrative: bool,
    ) -> LessonScope:
        """The conditions this iteration's observations were taken under."""
        scored = tuple(self._scored_case_ids())
        named = cases_named_in(str(session_sink.get("plan") or ""), scored)
        restricted = bool(named) and len(named) < len(scored)
        document = self.lessons.read(iteration)
        return LessonScope(
            cases=named if restricted else scored,
            held_fixed=parse_held_fixed(document),
            lane_restricted=restricted,
            carries_negative=self._carries_measured_negative(
                decision,
                result,
                session_sink,
                document=document,
                agent_narrative=agent_narrative,
            ),
            disproof=parse_disproof_marker(document),
        )

    async def _request_lesson(self, iteration: int, end_reason: str, summarizer) -> SummaryOutcome:
        """Ask the session for its record while the campaign still has time for the turn."""
        if self._time_remaining() < SUMMARY_MIN_SECONDS:
            # Time to produce the summary, not the loop's session-admission reserve: a campaign that can no
            # longer admit an implementer session is resumed later, and that session reads this document.
            return SummaryOutcome(reason="insufficient campaign time to run summarizer")
        try:
            return await summarize_iteration(
                iteration=iteration,
                end_reason=end_reason,
                summarizer=summarizer,
                references=PrReferenceNote(self.ic.pr_reference_labels, self.ic.pr_reference_context),
            )
        except Exception as error:  # isolation boundary: one iteration's lesson summarizer step
            log.debug("lessons: summarizer step failed", exc_info=True)
            return SummaryOutcome(reason=f"{type(error).__name__}: {str(error)[:200]}")
        finally:
            self._checkpoint_llm_usage()

    async def _record_lesson(
        self,
        *,
        result: IterationResult,
        decision: str,
        session_sink: dict,
        diff_summary: str = "",
        lane_record: SummaryOutcome | None = None,
    ) -> None:
        """Write this iteration's free-form factual session record."""
        store = getattr(self, "lessons", None)
        if store is None or session_sink.get("session_started") is not True:
            # No agent ran this iteration (e.g. the baseline measurement path): there is no exploration to record.
            return

        iteration = result.iteration
        # A lane's record was taken in its round. None means this iteration ran the session itself.
        record = lane_record
        if record is None:
            record = await self._request_lesson(iteration, result.session_end_reason, session_sink.get("summarize"))
        if record and store.write(iteration, record.text) is None:
            record = SummaryOutcome(reason="failed to persist lesson document")
        has_narrative = bool(record)
        # Narrated by the resumed session itself, as opposed to machine-written by the loop below.
        agent_narrative = has_narrative
        summary_failure = "" if has_narrative else record.reason
        if has_narrative:
            print(f"  [lesson] recorded iter {iteration}: {len(record.text)} chars")
        else:
            print(f"  [lesson] no summary ({record.reason}) — falling back to machine-observed session progress")

        if not has_narrative:
            # No session could describe what was explored, but the gate's block reasons are a real record of what the
            # agent ran into.
            fallback = build_fallback_document(
                SessionObservations(
                    diff_summary=diff_summary,
                    findings=session_sink.get("findings", ""),
                    plan=session_sink.get("plan", ""),
                    progress_log=session_sink.get("progress_log"),
                ),
                SessionEnd(result.session_end_reason, turns=result.turns, summary_failure=summary_failure),
            )
            if fallback and store.write(iteration, fallback) is not None:
                has_narrative = True
                print(f"  [lesson] machine-recorded iter {iteration} from gate findings: {len(fallback)} chars")

        scope = self._lesson_scope(
            iteration,
            session_sink,
            decision=decision,
            result=result,
            agent_narrative=agent_narrative,
        )
        if store.append_scope(iteration, scope):
            print(f"  [lesson] {format_scope_line(scope)}")
            if scope.carries_negative is not False and not scope.held_fixed:
                print(
                    "  [lesson] no held-fixed constants recorded — "
                    "negatives from this iteration re-open on the next "
                    "change"
                )
            if is_claim_disproved(scope.disproof):
                print(
                    "  [lesson] a direction reported unreachable was "
                    "shown reachable by the experiment run against it — "
                    "later iterations are told to re-enter it, not to "
                    "treat this record as closing it"
                )
            elif scope.disproof == UNDISPROVEN_CLAIM:
                print(
                    "  [lesson] a direction was reported unreachable "
                    "without running the experiment that would falsify "
                    "that — it stays open for later iterations"
                )
            elif scope.disproof is None and agent_narrative:
                print(
                    "  [lesson] the record answered nothing about "
                    "unreachable directions — any 'cannot' in it is "
                    "recorded as unchecked, and closes nothing"
                )
        else:
            print(
                f"  [lesson] scope not recorded for iter {iteration}: the document renders unscoped and closes nothing"
            )

        turns, failure = (None, "") if has_narrative else (result.turns, summary_failure)
        store.append_outcome(
            iteration,
            format_outcome_line(
                decision=decision,
                measurement=OutcomeMeasurement(
                    result.wall_ms,
                    self.best_wall_ms,
                    result.snr_db,
                    result.mean_case_speedup,
                    self.best_mean_case_speedup,
                ),
                session=SessionEnd(result.session_end_reason, turns=turns, summary_failure=failure),
            ),
        )

    def _record_ledger_iteration(
        self, result, frame: _IterationFrame, *, iteration_diff_summary, commit_hash, attempt_diff
    ) -> None:
        """Record what this iteration decided, and the evidence behind it, in the experience ledger."""
        iteration, session_sink = frame.iteration, frame.session_sink
        if getattr(self, "ledger", None) is not None and (commit_hash or attempt_diff):
            if not result.validation_passed:
                last = ""
                if result.validation_summary:
                    lines = [l for l in result.validation_summary.splitlines() if l.strip()]
                    last = lines[-1][:120] if lines else ""
                outcome = f"CRASH: {last}" if result.crashed else f"REVERT (validation failed): {last}"
            elif result.kept:
                outcome = f"KEPT — new best mean case speedup={result.mean_case_speedup:.6f}x"
            else:
                best_txt = f"{self.best_mean_case_speedup:.6f}x" if self.best_mean_case_speedup is not None else "?"
                speedup_txt = f"{result.mean_case_speedup:.6f}x" if result.mean_case_speedup is not None else "?"
                outcome = f"REVERT (correct but not faster): mean case speedup={speedup_txt} vs best={best_txt}"
            error_text = (
                session_sink.get("findings", "")
                or getattr(result, "error_output", "")
                or (result.validation_summary if not result.validation_passed else "")
            )
            self.ledger.record_iteration(
                iteration=iteration,
                outcome=outcome,
                diff_summary=iteration_diff_summary,
                error_text=error_text,
            )

    def _archive_attempt(
        self,
        result,
        attempt: _Attempt,
        frame: _IterationFrame,
        *,
        ruling: _Ruling,
        decision_label,
    ) -> Path | None:
        """Archive this attempt's candidate, and say where it landed.

        Archiving is best-effort: a candidate that cannot be written must
        not take the iteration down with it, so a failure reads as "nowhere".
        """
        iteration, session_sink = frame.iteration, frame.session_sink
        best_before, best_mean_case_speedup_before = frame.best_before, frame.best_mean_case_speedup_before
        attempt_diff, attempt_source, rationale = attempt.attempt_diff, attempt.attempt_source, attempt.rationale
        commit_hash, keep_checkpoint_finalized = ruling.commit_hash, ruling.keep_checkpoint_finalized
        archived_path: Path | None = None
        if getattr(self, "archive", None) is not None and (commit_hash or attempt_diff):
            try:
                decision = decision_label
                validation_text = result.validation_summary or ""
                if getattr(result, "error_output", ""):
                    validation_text = f"{validation_text}\n\n{result.error_output}".strip()
                archived_path = self.archive.record(
                    CandidateRecord(
                        iteration=iteration,
                        commit_hash=commit_hash,
                        decision=decision,
                        kept=result.kept,
                        validation_passed=result.validation_passed,
                        wall_ms=result.wall_ms,
                        mean_case_speedup=result.mean_case_speedup,
                        bench_detail=result.bench_detail,
                        snr_db=result.snr_db,
                        vgpr=result.vgpr,
                        baseline_wall_ms=self.ic.baseline_wall_ms,
                        best_wall_ms_before=best_before,
                        best_mean_case_speedup_before=best_mean_case_speedup_before,
                        plan=session_sink.get("plan", ""),
                        rationale=rationale,
                        agent_profile_id=result.agent_profile_id,
                        agent_provider=result.agent_provider,
                        agent_model=result.agent_model,
                        blockage_epoch=result.blockage_epoch,
                        kernel_file=self.ic.kernel_file,
                        shape={},
                        kernel_source=attempt_source,
                        change_diff=self._full_diff(commit_hash) if commit_hash else attempt_diff,
                        validation_text=validation_text,
                        session_end_reason=result.session_end_reason,
                        turns=result.turns,
                    )
                )
                if keep_checkpoint_finalized and archived_path is None:
                    raise RuntimeError("candidate archive returned no published path")
            except (OSError, RuntimeError, ValueError) as e:
                if keep_checkpoint_finalized:
                    self.persistence_degraded = True
                    self.persistence_errors.append(f"archive derived KEEP view iteration {iteration}: {e}")
                    self.persistence_errors = self.persistence_errors[-10:]
                log.debug("could not archive iteration %s: %s", iteration, e)
                archived_path = None
        return archived_path

    def _journaled_results_of(self, iteration: int) -> list[dict]:
        """Every journaled iteration_result for ``iteration``, duplicates included."""
        return [
            event
            for event in self.state_store.read_events()
            if event.get("type") == "iteration_result" and int(event.get("iter", 0) or 0) == iteration
        ]

    def _require_durable_outcome(self, result: IterationResult, decision_label: str) -> None:
        """Reload the run state and journal and refuse unless both carry this outcome exactly once."""
        persisted = self.state_store.load()
        durable_events = self._journaled_results_of(result.iteration)
        if (
            persisted.next_iteration <= result.iteration
            or len(durable_events) != 1
            or durable_events[0].get("decision") != decision_label
            or (
                result.kept
                and (
                    persisted.best.iteration != result.iteration
                    or persisted.best.commit_hash != result.commit_hash
                    or persisted.best.wall_ms != result.wall_ms
                )
            )
        ):
            raise RuntimeError(f"iteration {result.iteration} checkpoint was not durable")
