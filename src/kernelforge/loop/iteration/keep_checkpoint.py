# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pending-keep checkpoint: what an interrupted session left behind, and how the next one validates and finishes it."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

from kernelforge.loop.archive import CandidateRecord
from kernelforge.loop.iteration.attempt_verdict import _defer_termination_signals
from kernelforge.loop.iteration.iteration_types import (
    IterationResult,
    _Attempt,
    _IterationFrame,
    _Ruling,
)
from kernelforge.loop.run_state import (
    RunState,
    SESSION_COMPLETED,
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
            "validation_text": result.keep_validation_text(),
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
        base_head = pending["base_head"]
        patch = str(pending.get("patch") or "")
        iteration = int(pending["iteration"])
        expected_hash = pending["patch_sha256"]
        if not base_head or not patch:
            raise ValueError("pending KEEP metadata is incomplete")
        if hashlib.sha256(patch.encode()).hexdigest() != expected_hash:
            raise ValueError("pending KEEP metadata patch checksum mismatch")
        self._require_pending_keep_identity(state, pending)
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

    def _archive_pending_keep(self, pending: dict, result: IterationResult) -> None:
        """Recover the candidate archive when a KEEP was interrupted post-commit."""
        iteration = int(pending["iteration"])
        existing = self.archive.load_meta(iteration)
        if existing:
            if existing.get("decision") != "KEEP" or existing.get("commit_hash") != result.commit_hash:
                raise ValueError(f"candidate archive conflicts with pending KEEP iteration {iteration}")
            return
        archived = self.archive.record(
            CandidateRecord(
                iteration=iteration,
                commit_hash=result.commit_hash,
                decision="KEEP",
                kept=True,
                validation_passed=True,
                wall_ms=pending.get("wall_ms"),
                mean_case_speedup=pending.get("mean_case_speedup"),
                bench_detail=pending.get("benchmark") or {},
                snr_db=pending.get("snr_db"),
                vgpr=pending.get("vgpr"),
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

    def _require_pending_keep_identity(self, state: RunState, pending: dict) -> None:
        """Refuse a pending KEEP written for another campaign, kernel, task or branch."""
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

    def _commit_keep(self, result, attempt: _Attempt, frame: _IterationFrame, *, on_best_committed) -> _Ruling:
        """Commit a verified KEEP behind a durable pending journal, or demote it when the commit fails."""
        commit_hash, attempt_diff = attempt.commit_hash, attempt.attempt_diff
        pending_keep: dict | None = None
        keep_checkpoint_finalized = False
        # Defer SIGTERM/SIGINT across the durable best-commit publication (main #hardening) so a kill
        # mid-checkpoint cannot leave the pending-keep/run-state half-written.
        with _defer_termination_signals(bool(attempt_diff)):
            if attempt_diff:
                try:
                    pending_keep = self._build_pending_keep(
                        result,
                        plan=frame.session_sink.get("plan", ""),
                        best_before=frame.best_before,
                        rationale=attempt.rationale,
                        kernel_source=attempt.attempt_source,
                    )
                    self._persist_pending_keep(pending_keep)
                    commit_hash = self._git_commit(str(pending_keep["commit_message"]))
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as e:
                    result.kept = False
                    result.validation_passed = False
                    result.crashed = True
                    result.validation_summary = f"COMMIT FAILED: {e}"
                    result.error_output = str(e)
                    self._git_discard_all_tracked_changes()
                    self._clear_pending_keep()
                    print(f"  [REVERT] Commit failed after validation ({result.duration_sec:.0f}s)")
                    print(f"           {str(e)[-300:]}")
                else:
                    result.commit_hash = commit_hash
                    self._promote_best(result)
                    self.best_mean_case_speedup = result.mean_case_speedup
                    self._finalize_keep_checkpoint(
                        result,
                        plan=frame.session_sink.get("plan", ""),
                        best_before=frame.best_before,
                        pending=pending_keep,
                    )
                    keep_checkpoint_finalized = True
                    print(f"  [agent] Committed verified best: {commit_hash[:8]}")
            else:
                # No-agent measurement path: there is no candidate diff to commit, but the measurement can
                # still establish a best.
                self._promote_best(result)
                self.best_mean_case_speedup = result.mean_case_speedup
            # Bridge to the caller's checkpoint sink (main): lets the CLI persist a Hyperloom-recovery
            # checkpoint JSON alongside our run-state durability.
            if result.kept and on_best_committed:
                on_best_committed(result)
            if keep_checkpoint_finalized:
                self._clear_pending_keep()
        return _Ruling(pending_keep, keep_checkpoint_finalized, commit_hash)
