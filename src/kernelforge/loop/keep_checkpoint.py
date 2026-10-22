# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pending-keep checkpoint: what an interrupted session left behind, and how the next one finishes it."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path

from kernelforge.loop.agent_relay import apply_relay_keep, apply_relay_search_outcome
from kernelforge.loop.archive import CandidateRecord
from kernelforge.loop.iteration_types import DirectionEvidence, IterationResult
from kernelforge.loop.run_state import (
    IterationVerdict,
    LoopStanding,
    LoopStateStore,
    RunState,
    SESSION_COMPLETED,
    apply_iteration,
)
from kernelforge.tools.durable_io import atomic_write_text

log = logging.getLogger(__name__)


# The exact key set ``_build_pending_keep`` writes to ``pending_keep.json``; recovery refuses any other shape.
PENDING_KEEP_FIELDS = frozenset(
    {
        "campaign_id",
        "session_index",
        "experiment_id",
        "base_head",
        "iteration",
        "wall_ms",
        "mean_case_speedup",
        "snr_db",
        "vgpr",
        "plan",
        "rationale",
        "validation_text",
        "benchmark",
        "changed_files",
        "patch",
        "patch_sha256",
        "publication_base_commit",
        "publication_changed_files",
        "publication_patch",
        "kernel_source",
        "kernel_file",
        "shape",
        "baseline_wall_ms",
        "pristine_baseline_wall_ms",
        "best_wall_ms_before",
        "best_mean_case_speedup_before",
        "session_end_reason",
        "turns",
        "agent_profile_id",
        "agent_provider",
        "agent_model",
        "blockage_epoch",
        "search_control",
        "commit_message",
        "commit_subject",
        "task_fingerprint",
        "git_branch",
    }
)


class KeepCheckpointMixin:
    """Carry a verified KEEP across a restart: the journal it writes, the identity a resume checks, and the outcomes it replays."""

    def _task_fingerprint(self) -> str:
        """Stable identity for the task inputs that define a resume campaign."""
        payload = {
            "kernel_path": self._workspace_path(self.ic.kernel_file),
            "driver_path": self._workspace_path(self.ic.driver_script),
            "task_type": self.ic.task_type,
            "source_files": sorted(self._workspace_path(path) for path in self.ic.source_files),
            "target_functions": sorted(self.ic.target_functions),
            "operator_name": self.ic.operator_name,
            "implementation_signature": self.ic.implementation_signature,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _driver_sha256(self) -> str:
        """Hash the exact driver bytes that validation and benchmarking will run."""
        path = Path(self.ic.driver_script)
        if not path.is_file():
            raise ValueError(f"driver integrity check failed: file is missing: {path}")
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f"driver integrity check failed: could not read {path}: {error}") from error

    def _validate_driver_integrity(self, state: RunState) -> str:
        """Accept only the canonical campaign driver."""
        canonical = (self.ic.canonical_driver_sha256 or "").strip().lower()
        if not canonical:
            return ""
        current = self._driver_sha256()
        if current == canonical:
            return current
        raise ValueError("driver integrity check failed: workspace driver does not match the campaign canonical digest")

    def _set_state_identity(self, state: RunState) -> None:
        """Stamp the current workspace/task/git identity onto campaign state."""
        state.kernel_path = self._workspace_path(self.ic.kernel_file)
        state.task_fingerprint = self._task_fingerprint()
        state.git_branch = self.ic.git_branch
        state.head_commit = self._git("rev-parse", "HEAD").splitlines()[0]

    @property
    def _pending_keep_path(self) -> Path:
        return Path(self.ic.workspace_dir) / "forge_experiments" / "pending_keep.json"

    def _persist_pending_keep(self, pending: dict) -> None:
        """Atomically persist a verified candidate before creating its commit."""
        atomic_write_text(
            self._pending_keep_path,
            json.dumps(pending, indent=2, sort_keys=True, default=str) + "\n",
        )

    def _load_pending_keep(self) -> dict | None:
        path = self._pending_keep_path
        if not path.exists():
            return None
        try:
            pending = json.loads(path.read_text())
        except Exception as error:
            raise ValueError(f"invalid pending KEEP metadata: {path}") from error
        relay_fields = {
            "agent_profile_id": "",
            "agent_provider": "",
            "agent_model": "",
            "blockage_epoch": 0,
        }
        if isinstance(pending, dict) and not set(relay_fields) & set(pending):
            pending.update(relay_fields)
        if not isinstance(pending, dict) or set(pending) != PENDING_KEEP_FIELDS:
            raise ValueError(f"invalid pending KEEP metadata: {path}")
        return pending

    def _clear_pending_keep(self) -> None:
        try:
            self._pending_keep_path.unlink()
        except FileNotFoundError:
            return

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

    def _build_pending_keep(
        self,
        result: IterationResult,
        *,
        plan: str,
        best_before: float | None,
        rationale: str,
        kernel_source: str,
    ) -> dict:
        """Capture every fact needed to finish a verified KEEP after restart."""
        base_head = self._git("rev-parse", "HEAD").splitlines()[0]
        patch, changed_files = self._candidate_changes(base_head)
        # Keep the journal's existing fingerprint convention; export the raw diff below.
        patch = patch.strip()
        if not patch:
            raise ValueError("verified KEEP has no candidate diff")
        validation_text = result.validation_summary or "canonical validation passed"
        if result.error_output:
            validation_text = f"{validation_text}\n\n{result.error_output}".strip()
        benchmark = dict(result.bench_detail or {})
        benchmark.setdefault("median_ms", result.wall_ms)
        publication_base = self.ic.campaign_base_commit or base_head
        publication_patch, publication_changed_files = self._candidate_changes(publication_base)
        commit_message = f"iter-{result.iteration}: {rationale[:72]}"
        return {
            "campaign_id": self.run_state.campaign_id,
            "session_index": self.run_state.session_index,
            "experiment_id": (self.experiment.experiment_id if self.experiment else ""),
            "base_head": base_head,
            "iteration": result.iteration,
            "wall_ms": result.wall_ms,
            "mean_case_speedup": result.mean_case_speedup,
            "snr_db": result.snr_db,
            "vgpr": result.vgpr,
            "plan": (plan or "").strip(),
            "rationale": rationale,
            "validation_text": validation_text,
            "benchmark": benchmark,
            "changed_files": changed_files,
            "patch": patch,
            "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "publication_base_commit": publication_base,
            "publication_changed_files": publication_changed_files,
            "publication_patch": publication_patch,
            "kernel_source": kernel_source,
            "kernel_file": self.ic.kernel_file,
            "shape": {},
            "baseline_wall_ms": (
                self.ic.publication_baseline_wall_ms or self.ic.baseline_wall_ms or self.run_state.baseline_wall_ms
            ),
            "pristine_baseline_wall_ms": (
                self.ic.pristine_baseline_wall_ms
                if self.ic.pristine_baseline_wall_ms is not None
                else self.ic.baseline_wall_ms
            ),
            "best_wall_ms_before": best_before,
            "best_mean_case_speedup_before": self.best_mean_case_speedup,
            "session_end_reason": result.session_end_reason,
            "turns": result.turns,
            "agent_profile_id": result.agent_profile_id,
            "agent_provider": result.agent_provider,
            "agent_model": result.agent_model,
            "blockage_epoch": result.blockage_epoch,
            "search_control": self._search_control_snapshot(),
            "commit_message": commit_message,
            "commit_subject": commit_message.splitlines()[0],
            "task_fingerprint": self._task_fingerprint(),
            "git_branch": self.ic.git_branch,
        }

    def _inspect_pending_keep(self, state: RunState, pending: dict) -> str:
        """Classify a pending KEEP as uncommitted or the exact expected child."""
        if state.session_status == SESSION_COMPLETED:
            raise ValueError("completed campaign cannot be resumed")
        base_head = str(pending.get("base_head") or "")
        patch = str(pending.get("patch") or "")
        iteration = int(pending.get("iteration", 0) or 0)
        expected_hash = str(pending.get("patch_sha256") or "")
        if not base_head or not patch or iteration <= 0:
            raise ValueError("pending KEEP metadata is incomplete")
        if hashlib.sha256(patch.encode()).hexdigest() != expected_hash:
            raise ValueError("pending KEEP metadata patch checksum mismatch")
        if pending.get("campaign_id") != state.campaign_id:
            raise ValueError("pending KEEP campaign mismatch")
        current_kernel = self._workspace_path(self.ic.kernel_file)
        if state.kernel_path and state.kernel_path != current_kernel:
            raise ValueError(f"kernel path mismatch: expected {state.kernel_path}, got {current_kernel}")
        if state.task_fingerprint and state.task_fingerprint != pending.get("task_fingerprint"):
            raise ValueError("pending KEEP state task fingerprint mismatch")
        if pending.get("task_fingerprint") != self._task_fingerprint():
            raise ValueError("pending KEEP task fingerprint mismatch")
        if state.git_branch and state.git_branch != pending.get("git_branch"):
            raise ValueError("pending KEEP state branch mismatch")
        if pending.get("git_branch") != self.ic.git_branch:
            raise ValueError("pending KEEP branch mismatch")

        current_branch = self._git("branch", "--show-current").splitlines()[0]
        if current_branch != self.ic.git_branch:
            raise ValueError(
                f"branch mismatch: workspace is on {current_branch or 'detached HEAD'}, expected {self.ic.git_branch}"
            )
        current_head = self._git("rev-parse", "HEAD").splitlines()[0]
        state_anchor = state.best.commit_hash or state.head_commit
        already_finalized = state.best.iteration == iteration and state.best.commit_hash == current_head
        if not already_finalized and state.next_iteration != iteration:
            raise ValueError(f"pending KEEP iteration mismatch: expected {state.next_iteration}, got {iteration}")
        if not already_finalized and base_head != state_anchor:
            raise ValueError(f"pending KEEP base mismatch: expected {state_anchor}, got {base_head}")

        tracked_diff = self._tracked_diff_from_head()
        if current_head == base_head:
            candidate_diff, _ = self._candidate_changes(base_head)
            if candidate_diff and hashlib.sha256(candidate_diff.strip().encode()).hexdigest() != expected_hash:
                raise ValueError("pending KEEP working tree mismatch")
            return "uncommitted"

        parents = self._git("rev-list", "--parents", "-n", "1", current_head).split()
        if len(parents) != 2 or parents[1] != base_head:
            raise ValueError(f"pending KEEP HEAD mismatch: {current_head} is not the expected child")
        if tracked_diff:
            raise ValueError("pending KEEP committed child has tracked workspace changes")
        committed_patch = self._git("diff", base_head, current_head, "--", ".")
        if hashlib.sha256(committed_patch.encode()).hexdigest() != expected_hash:
            raise ValueError("pending KEEP committed patch mismatch")
        subject = self._git("show", "-s", "--format=%s", current_head)
        if subject != pending["commit_subject"]:
            raise ValueError("pending KEEP commit message mismatch")
        return "committed"

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

    def _archive_pending_keep(
        self,
        pending: dict,
        commit_hash: str,
        *,
        result: IterationResult | None = None,
    ) -> None:
        """Recover the candidate archive when a KEEP was interrupted post-commit."""
        iteration = int(pending["iteration"])
        existing = self.archive.load_meta(iteration)
        if existing:
            if existing.get("decision") != "KEEP" or existing.get("commit_hash") != commit_hash:
                raise ValueError(f"candidate archive conflicts with pending KEEP iteration {iteration}")
            return
        archived = self.archive.record(
            CandidateRecord(
                iteration=iteration,
                commit_hash=commit_hash,
                decision="KEEP",
                kept=True,
                validation_passed=True,
                wall_ms=pending.get("wall_ms"),
                mean_case_speedup=pending.get("mean_case_speedup"),
                bench_detail=pending.get("benchmark") or {},
                snr_db=pending.get("snr_db"),
                vgpr=pending.get("vgpr"),
                pmc_diagnosis=result.pmc_diagnosis if result else "",
                profile_meta=result.profile_meta if result else {},
                baseline_wall_ms=pending.get("baseline_wall_ms"),
                best_wall_ms_before=pending.get("best_wall_ms_before"),
                best_mean_case_speedup_before=pending.get("best_mean_case_speedup_before"),
                plan=str(pending.get("plan") or ""),
                rationale=str(pending.get("rationale") or ""),
                session_end_reason=str(pending.get("session_end_reason") or ""),
                turns=pending.get("turns"),
                agent_profile_id=str(pending.get("agent_profile_id") or ""),
                agent_provider=str(pending.get("agent_provider") or ""),
                agent_model=str(pending.get("agent_model") or ""),
                blockage_epoch=int(pending.get("blockage_epoch") or 0),
                kernel_file=str(pending.get("kernel_file") or self.ic.kernel_file),
                shape=pending.get("shape") or {},
                kernel_source=str(pending.get("kernel_source") or ""),
                change_diff=str(pending.get("patch") or ""),
                pmc_full=result.pmc_full if result else "",
                validation_text=str(pending.get("validation_text") or ""),
            )
        )
        if archived != self.archive._iter_dir(iteration):
            raise RuntimeError(f"failed to recover candidate archive for iteration {iteration}")

    def _pending_keep_result(
        self,
        pending: dict,
        commit_hash: str,
    ) -> IterationResult:
        """Rebuild the compact KEEP result represented by its journal."""
        return IterationResult(
            iteration=int(pending["iteration"]),
            duration_sec=0.0,
            validation_passed=True,
            validation_summary=str(pending.get("validation_text") or ""),
            wall_ms=pending.get("wall_ms"),
            mean_case_speedup=pending.get("mean_case_speedup"),
            snr_db=pending.get("snr_db"),
            vgpr=pending.get("vgpr"),
            kept=True,
            commit_hash=commit_hash,
            agent_rationale=str(pending.get("rationale") or ""),
            bench_detail=dict(pending.get("benchmark") or {}),
            session_end_reason=str(pending.get("session_end_reason") or ""),
            turns=pending.get("turns"),
            agent_profile_id=str(pending.get("agent_profile_id") or ""),
            agent_provider=str(pending.get("agent_provider") or ""),
            agent_model=str(pending.get("agent_model") or ""),
            blockage_epoch=int(pending.get("blockage_epoch") or 0),
        )

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
        elif (
            planned.best.iteration != result.iteration
            or planned.best.commit_hash != result.commit_hash
            or planned.best.wall_ms != result.wall_ms
            or planned.best.mean_case_speedup != result.mean_case_speedup
        ):
            raise ValueError(f"run state conflicts with KEEP iteration {result.iteration}")
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
            self._archive_pending_keep(
                pending,
                result.commit_hash,
                result=result,
            )
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
