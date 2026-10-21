# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Iteration outcome events, handoffs, ledger entries and candidate archiving for the forge-loop."""

from __future__ import annotations

import logging
from pathlib import Path


from kernelforge.loop.lessons import (
    LessonScope,
    cases_named_in,
    format_scope_line,
    parse_disproof_marker,
    parse_held_fixed,
    parse_negatives_marker,
)
from kernelforge.loop.archive import CandidateRecord
from kernelforge.loop.handoffs import IterationHandoff
from kernelforge.loop.run_state import (
    IterationVerdict,
    LoopStanding,
    RunState,
    pin_iteration,
    apply_iteration,
    make_event,
)
from kernelforge.orchestrator.supervisor import (
    latest_supervisor_ruling_path,
)
from kernelforge.evaluation.scoring import (
    beats_current_best,
)
from kernelforge.loop.merge_candidates import (
    case_spreads,
    cases_beating_reference,
)
from kernelforge.loop.iteration_types import (
    DirectionEvidence,
    IterationResult,
    _Attempt,
    _IterationFrame,
    _Ruling,
)

log = logging.getLogger(__name__)


def _verdict_of(result: "IterationResult", decision: str, plan: str) -> IterationVerdict:
    """The run-state verdict of one decided iteration."""
    return IterationVerdict(
        result.iteration, decision, result.kept, result.wall_ms, result.commit_hash, plan, result.mean_case_speedup
    )


# Decision labels that can possibly mean nothing came out worse this iteration: a kept candidate, and the ones that
# mean no candidate was ever measured.
_LABELS_WITHOUT_A_MEASURED_NEGATIVE = frozenset(
    {
        "KEEP",
        "NO_CHANGES",
        "API_ERROR",
        "AGENT_ERROR",
        "ORCHESTRATION_ERROR",
    }
)


def _decision_label(result: IterationResult) -> str:
    """The canonical keep/revert label for one finished attempt."""
    if result.integrity_violation:
        return "REVERT_INTEGRITY"
    if result.workspace_contention:
        return "REVERT_CONTENDED"
    if result.crashed:
        return "CRASH"
    if not result.validation_passed:
        if (result.validation_summary or "").startswith("BUILD FAILED"):
            return "BUILD_FAILED"
        if result.validation_outcome == "timeout":
            return "REVERT_VALIDATION_TIMEOUT"
        if result.validation_outcome in {"driver_error", "invalid_result"}:
            return "REVERT_VALIDATION_ERROR"
        return "REVERT_VALIDATION"
    return "KEEP" if result.kept else "REVERT_PERF"


class OutcomeRuntimeMixin:
    """Own what a decided iteration leaves behind: events, handoffs, ledger rows and archives."""

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

    def _iteration_result_event(
        self,
        result: IterationResult,
        *,
        plan: str,
        decision_label: str | None = None,
        error_sig: str = "",
        checkpoint_metadata: dict | None = None,
    ) -> dict:
        """Build the canonical durable event for one completed iteration."""
        resolved_decision = decision_label or _decision_label(result)
        return make_event(
            "iteration_result",
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
            error_sig=error_sig or None,
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
        except Exception as error:
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
                        pmc_diagnosis=result.pmc_diagnosis,
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
                        pmc_full=result.pmc_full,
                        profile_meta=result.profile_meta,
                        validation_text=validation_text,
                        session_end_reason=result.session_end_reason,
                        turns=result.turns,
                    )
                )
                if keep_checkpoint_finalized and archived_path is None:
                    raise RuntimeError("candidate archive returned no published path")
            except Exception as e:  # noqa: BLE001
                if keep_checkpoint_finalized:
                    self.persistence_degraded = True
                    self.persistence_errors.append(f"archive derived KEEP view iteration {iteration}: {e}")
                    self.persistence_errors = self.persistence_errors[-10:]
                log.debug("could not archive iteration %s: %s", iteration, e)
                archived_path = None
        return archived_path
