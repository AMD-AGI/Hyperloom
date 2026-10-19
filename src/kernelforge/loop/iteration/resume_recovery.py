# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resuming from a pending-keep checkpoint: what it validates, replays and settles."""

from __future__ import annotations

import copy
import logging

from kernelforge.loop.agent_relay import apply_relay_keep, apply_relay_search_outcome
from kernelforge.loop.iteration.best_publication import _require_best_is_keep
from kernelforge.loop.iteration.iteration_types import (
    DirectionEvidence,
    IterationResult,
)
from kernelforge.loop.run_state import (
    IterationVerdict,
    LoopStanding,
    LoopStateStore,
    RunState,
    SESSION_COMPLETED,
    apply_iteration,
)

log = logging.getLogger(__name__)


class ResumeRecoveryMixin:
    """Recovering a campaign that stopped with a keep in flight."""

    def _validate_resume_scoring_state(self, state: RunState) -> None:
        """Reject checkpoints that cannot restore the original scoring rules."""
        if state.best.commit_hash and not state.best_case_times:
            raise ValueError("resume state has no incumbent per-case timings; start a fresh campaign")

    def _validate_resume_state(
        self,
        state: RunState,
        *,
        expected_head: str | None = None,
        allow_dirty: bool = False,
    ) -> None:
        """Fail closed before a resumed invocation mutates persistent state."""
        self._validate_resume_scoring_state(state)
        if state.session_status == SESSION_COMPLETED:
            raise ValueError("completed campaign cannot be resumed")
        if state.best.commit_hash and state.best.mean_case_speedup is None:
            raise ValueError("resume state has a best commit without its mean case speedup; start a fresh campaign")
        if not state.baseline_case_times:
            raise ValueError(
                "resume state has no pristine per-case timings; start a fresh "
                "campaign so mean case speedup can be computed"
            )

        self._validate_driver_integrity(state)

        current_kernel = self._workspace_path(self.ic.kernel_file)
        if state.kernel_path and state.kernel_path != current_kernel:
            raise ValueError(f"kernel path mismatch: expected {state.kernel_path}, got {current_kernel}")

        if state.task_fingerprint and state.task_fingerprint != self._task_fingerprint():
            raise ValueError("task fingerprint mismatch")

        current_branch = self._git("branch", "--show-current").splitlines()[0]
        if state.git_branch and state.git_branch != self.ic.git_branch:
            raise ValueError(f"branch mismatch: state uses {state.git_branch}, configuration uses {self.ic.git_branch}")
        if current_branch != self.ic.git_branch:
            raise ValueError(
                f"branch mismatch: workspace is on {current_branch or 'detached HEAD'}, expected {self.ic.git_branch}"
            )

        current_head = self._git("rev-parse", "HEAD").splitlines()[0]
        resume_head = expected_head or state.best.commit_hash or state.head_commit
        if not resume_head:
            raise ValueError("resume state has no HEAD anchor")
        if current_head != resume_head:
            raise ValueError(f"HEAD mismatch: expected {resume_head}, got {current_head}")

        dirty = self._git("status", "--porcelain", "--untracked-files=no")
        if dirty and not allow_dirty:
            raise ValueError("workspace has uncommitted tracked changes")

    def validate_resume_preflight(self) -> RunState:
        """Validate a persisted resume checkpoint without mutating campaign files."""
        store = LoopStateStore(self.ic.workspace_dir)
        if not store.state_path.is_file():
            raise ValueError(f"resume state not found: {store.state_path}")
        state = store.load()
        self.state_store = store
        pending = self._load_pending_keep()
        planned, status, _result, _append_keep = self._plan_resume_recovery(
            state,
            pending,
        )
        self._validate_resume_state(
            planned,
            allow_dirty=status == "uncommitted",
        )
        return state

    def _restore_resume_baseline_case_times(self, state: RunState) -> None:
        """Restore the immutable scoring baseline for a validated resume."""
        if not self.resume:
            return
        state_cases = dict(state.baseline_case_times)
        if not state_cases:
            raise ValueError(
                "resume state has no pristine per-case timings; start a fresh "
                "campaign so mean case speedup can be computed"
            )
        if self._baseline_case_times and self._baseline_case_times != state_cases:
            raise ValueError("resume baseline case timings conflict with the persisted campaign")
        self._baseline_case_times = state_cases
        self.ic.baseline_case_times = dict(state_cases)

    @staticmethod
    def _require_matching_keep_event(
        event: dict,
        pending: dict,
        commit_hash: str,
    ) -> None:
        """Reject a KEEP event that does not describe the pending journal."""
        expected = {
            "decision": "KEEP",
            "commit_hash": commit_hash,
            "plan": str(pending.get("plan") or "").strip()[:120],
            "wall_ms": pending.get("wall_ms"),
            "mean_case_speedup": pending.get("mean_case_speedup"),
            "snr_db": pending.get("snr_db"),
            "session_end_reason": (str(pending.get("session_end_reason") or "") or None),
            "session_index": int(pending.get("session_index", 0) or 0),
            "experiment_id": str(pending.get("experiment_id") or "") or None,
            "turns": pending.get("turns"),
            "validation_passed": True,
            "is_new_best": True,
        }
        conflicts = [key for key, value in expected.items() if event.get(key) != value]
        if conflicts:
            raise ValueError("pending KEEP event payload mismatch: " + ", ".join(sorted(conflicts)))

    def _apply_replayed_non_keep(self, state: RunState, event: dict) -> None:
        """Reduce one validated non-KEEP event without persisting state."""
        iteration = int(event["iter"])
        decision = str(event.get("decision") or "")
        if not decision or decision == "KEEP":
            raise ValueError(f"iteration {iteration} is not a replayable non-KEEP event")
        apply_iteration(
            state,
            IterationVerdict(
                iteration=iteration,
                decision=decision,
                kept=False,
                wall_ms=event.get("wall_ms"),
                commit_hash=str(event.get("commit_hash") or ""),
                plan=str(event.get("plan") or ""),
                mean_case_speedup=event.get("mean_case_speedup"),
            ),
            LoopStanding(
                baseline_wall_ms=state.baseline_wall_ms,
                best_wall_ms=event.get("best_after_ms"),
                best_mean_case_speedup=event.get("best_after_mean_case_speedup"),
            ),
            self._stall_policy(),
        )
        state.diversification_cycle_completed = event.get("diversification_cycle_completed") is True
        apply_relay_search_outcome(
            state.agent_relay,
            decision=decision,
            producer_profile_id=str(event.get("agent_profile_id") or ""),
        )
        self._record_direction_verdict(
            state,
            iteration=iteration,
            decision_label=decision,
            evidence=DirectionEvidence(
                mean_case_speedup=event.get("mean_case_speedup"),
                # A non-KEEP leaves the incumbent untouched, so the recorded post-decision score is the bar this
                # candidate had to clear.
                best_mean_case_speedup=event.get("best_after_mean_case_speedup"),
            ),
        )

    def _iteration_result_events(self) -> dict[int, dict]:
        """Every recorded iteration result, keyed by iteration.

        A duplicate is refused rather than taken twice: the journal is what a
        resume replays from, so two records of one iteration would apply it
        twice.
        """
        events: dict[int, dict] = {}
        for event in self.state_store.read_events():
            if event.get("type") != "iteration_result":
                continue
            iteration = int(event["iter"])
            if iteration in events:
                raise ValueError(f"duplicate iteration_result events for iteration {iteration}")
            events[iteration] = event
        return events

    def _replay_non_keep_events(
        self,
        planned: RunState,
        events_by_iteration: dict[int, dict],
        iterations: list[int],
        cursor: int,
        *,
        boundary: int | None,
    ) -> int:
        """Replay the contiguous non-KEEP iterations below ``boundary``; return the cursor after them.

        The window has to be contiguous and KEEP-free: a gap means the journal
        is missing a record this resume would skip over, and a KEEP with no
        pending journal means one landed that was never checkpointed.
        """
        for iteration in iterations:
            if boundary is not None and iteration >= boundary:
                break
            if iteration != cursor:
                raise ValueError(f"iteration_result recovery gap: expected {cursor}, got {iteration}")
            event = events_by_iteration[iteration]
            if event.get("decision") == "KEEP":
                raise ValueError(f"uncheckpointed KEEP iteration {iteration} has no matching pending journal")
            self._apply_replayed_non_keep(planned, event)
            cursor = planned.next_iteration
        return cursor

    def _adopt_pending_keep(
        self,
        planned: RunState,
        pending: dict,
        keep_event: dict | None,
        *,
        pending_iteration: int,
        already_applied: bool,
    ) -> tuple[str, IterationResult | None, bool]:
        """Fold the pending KEEP into the planned state; return its status, result and whether it still needs an event."""
        status = self._inspect_pending_keep(planned, pending)
        if keep_event is not None and keep_event.get("decision") != "KEEP":
            raise ValueError(f"pending KEEP conflicts with iteration {pending_iteration} event")
        if status == "uncommitted":
            if keep_event is not None:
                raise ValueError(f"uncommitted pending KEEP iteration {pending_iteration} already has a KEEP event")
            return status, None, False

        current_head = self._git("rev-parse", "HEAD").splitlines()[0]
        result = self._pending_keep_result(pending, current_head)
        if keep_event is not None:
            self._require_matching_keep_event(
                keep_event,
                pending,
                current_head,
            )
        if not already_applied:
            apply_iteration(
                planned,
                IterationVerdict(
                    iteration=result.iteration,
                    decision="KEEP",
                    kept=True,
                    wall_ms=result.wall_ms,
                    commit_hash=result.commit_hash,
                    plan=str(pending.get("plan") or ""),
                    mean_case_speedup=result.mean_case_speedup,
                ),
                LoopStanding(
                    baseline_wall_ms=planned.baseline_wall_ms,
                    best_wall_ms=result.wall_ms,
                    best_mean_case_speedup=result.mean_case_speedup,
                ),
                self._stall_policy(),
            )
            control = pending.get("search_control")
            if isinstance(control, dict):
                planned.diversification_cycle_completed = control.get("diversification_cycle_completed") is True
            apply_relay_keep(planned.agent_relay)
        else:
            _require_best_is_keep(planned.best, result)
        return status, result, keep_event is None

    def _plan_resume_recovery(
        self,
        state: RunState,
        pending: dict | None,
    ) -> tuple[RunState, str, IterationResult | None, bool]:
        """Validate and reduce the complete contiguous recovery window."""
        planned = copy.deepcopy(state)
        events_by_iteration = self._iteration_result_events()

        cursor = planned.next_iteration
        pending_iteration = int(pending.get("iteration", 0) or 0) if pending is not None else None
        if pending is not None and pending_iteration <= 0:
            raise ValueError("pending KEEP metadata is incomplete")

        forward_iterations = sorted(iteration for iteration in events_by_iteration if iteration >= cursor)
        cursor = self._replay_non_keep_events(
            planned,
            events_by_iteration,
            forward_iterations,
            cursor,
            boundary=pending_iteration,
        )
        if pending is None:
            return planned, "", None, False

        assert pending_iteration is not None
        already_applied = pending_iteration < cursor
        if not already_applied and pending_iteration != cursor:
            raise ValueError(
                f"iteration_result recovery gap before pending KEEP: expected {cursor}, got {pending_iteration}"
            )
        later_events = [iteration for iteration in forward_iterations if iteration > pending_iteration]
        if later_events:
            raise ValueError(
                f"iteration_result exists after pending KEEP iteration {pending_iteration}: {later_events[0]}"
            )

        status, result, append_keep = self._adopt_pending_keep(
            planned,
            pending,
            events_by_iteration.get(pending_iteration),
            pending_iteration=pending_iteration,
            already_applied=already_applied,
        )
        return planned, status, result, append_keep

    def _coordinate_resume_recovery(self, on_best_committed=None) -> None:
        """Replay, reconcile, and checkpoint one ordered recovery transaction."""
        pending = self._load_pending_keep()
        planned, pending_status, result, append_keep = self._plan_resume_recovery(self.run_state, pending)
        if pending_status == "uncommitted":
            if self._tracked_diff_from_head():
                self._git_discard_all_tracked_changes()
            if self._tracked_diff_from_head():
                raise RuntimeError("pending KEEP workspace remained dirty after restore")
            self._clear_pending_keep()

        head_out = self._git("rev-parse", "HEAD").strip()
        if head_out:
            planned.head_commit = head_out.splitlines()[0]
        self.run_state = planned
        if append_keep:
            assert pending is not None and result is not None
            self.state_store.append_event(
                self._iteration_result_event(
                    result,
                    plan=str(pending.get("plan") or ""),
                    checkpoint_metadata=pending,
                )
            )
            if self.wiki_rounds is not None:
                self.wiki_rounds.recover_keep(result.iteration)
        self.state_store.save(self.run_state)
        persisted = self.state_store.load()
        if persisted.to_dict() != self.run_state.to_dict():
            raise RuntimeError("resume recovery state was not persisted")

        if result is not None and pending is not None:
            self._promote_best(result)
            self.best_mean_case_speedup = result.mean_case_speedup
            if on_best_committed is not None:
                on_best_committed(result)
            self._recovered_pending_keep = (pending, result)

    async def _finish_recovered_pending_keep(self) -> None:
        """Rebuild optional post-KEEP views after critical recovery is safe."""
        recovered = self._recovered_pending_keep
        if recovered is None:
            return
        pending, result = recovered
        try:
            self._archive_pending_keep(pending, result)
        except (OSError, ValueError, RuntimeError) as error:
            self.persistence_degraded = True
            self.persistence_errors.append(f"rebuild candidate archive iteration {result.iteration}: {error}")
            self.persistence_errors = self.persistence_errors[-10:]
            log.debug("failed to rebuild recovered candidate archive", exc_info=True)
        self._publish_best_result(
            result,
            plan=str(pending.get("plan") or ""),
            best_before=pending.get("best_wall_ms_before"),
            pending=pending,
        )
        self._clear_pending_keep()
        self._recovered_pending_keep = None
        self._publish_optimization_history()
        self._checkpoint_llm_usage()
