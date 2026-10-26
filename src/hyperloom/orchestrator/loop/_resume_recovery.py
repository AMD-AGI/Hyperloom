# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Settling the windows a crash left open: integrations, builds, revalidations, orphaned KEEPs."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.breakdown.recorder import enablement_event
from hyperloom.inference_optimizer.session.resume_guard import confirmed_stopped_scope

from ..state.task_registry import Task
from ._current_best_lift import _StackKey, _orphan_stack_key, _stacked_keep_keys
from ._integrate_promotion import _IntegrateMarker, _KeptResultScan, _integrate_marker_of, _task_recorded_a_result

log = logging.getLogger(__name__)


class ResumeRecoveryMixin:
    """Recovering the work in flight when the coordinator died."""

    def _resume_rollback_pending_integrate(self, pending: dict[str, Any], gone_scope: str | None) -> dict[str, Any]:
        """Restore a pending integration from its durable file and stash evidence.

        Incomplete evidence or an unverifiable restore remains pending; the
        resume entry point refuses to continue to measurement in that state.

        Args:
            pending: The ``pending_integrate`` sentinel dict.
            gone_scope: What an operator confirmed stopped with the task (``confirmed_stopped_scope``), if anything.

        Returns:
            A summary ``{"reversed": [...], "failed": [...]}``.
        """
        from ..actions.executors._patch_git import restore_pending_integrate
        from ..actions.executors.source_tree_identity import settle_replaced_tree, tree_replaced

        if tree_replaced(pending, gone_scope):
            return settle_replaced_tree(pending)
        return restore_pending_integrate(pending)

    @staticmethod
    def _gc_attempt_runtime(attempt_dir: str) -> bool:
        """Remove an attempt-runtime dir (best-effort).

        Returns True when a directory was present and removal was attempted.
        """
        import shutil

        path = Path(str(attempt_dir or "").strip())
        if not attempt_dir or not path.exists():
            return False
        shutil.rmtree(path, ignore_errors=True)
        return True

    async def _resume_recover_interrupted_stack(self, report: dict[str, Any]) -> None:
        """Settle a stack attempt a crash or halt left behind, before anything here can benchmark.

        The recovery halts the session if the tree cannot be settled, which is
        the point: the alternative is measuring a tree whose contents no resume
        can account for.
        """
        if await self._coord.phase_kernel_stack.recover_interrupted_stack_validation():
            report["fixes"].append({"kind": "interrupted_stack_validation_recovered"})

    def _clear_pending_integrate(self, pending: dict[str, Any], *, gc_runtime: bool) -> None:
        """Drop a discharged sentinel, and its attempt runtime once nothing runs from it."""
        self.shared_state.pending_integrate = {}
        self.shared_state.save(self.session_dir)
        if gc_runtime and pending.get("attempt_venv_root"):
            self._gc_attempt_runtime(str(Path(pending["attempt_venv_root"]).parent))

    def _clear_settled_integrate_window(
        self,
        pending: dict[str, Any],
        report: dict[str, Any],
        marker: _IntegrateMarker,
    ) -> bool:
        task_id = str(pending.get("task_id") or "")
        if marker.verdict == "settled":
            fix = {
                "kind": "settled_pending_integrate",
                "task_id": task_id,
                "status": str((marker.outcome or {}).get("status") or ""),
            }
        elif marker.phase == "restored" and marker.outcome is not None:
            fix = {"kind": "settled_pending_integrate", "task_id": task_id, "phase": marker.phase}
        else:
            return False
        report["fixes"].append(fix)
        self._clear_pending_integrate(pending, gc_runtime=True)
        return True

    async def _scan_for_published_integrate_keep(self, task_id: str) -> _KeptResultScan:
        try:
            for msg in await self.bus.tail(topic="delegated_result", n=-1):
                payload = msg.payload or {}
                if str(payload.get("task_id") or "") != task_id:
                    continue
                res = payload.get("result") or {}
                # Require an explicit integrate_patch kind: an empty-kind wildcard
                # could misclassify a non-integrate event that happens to share
                # this task_id as a kept integrate result, skipping rollback of a
                # half-applied patch.
                if (
                    isinstance(res, dict)
                    and str(res.get("kind") or payload.get("kind") or "") == "integrate_patch"
                    and str(res.get("status") or "").lower() == "kept"
                ):
                    return _KeptResultScan(scanned=True, kept=res)
        except Exception:
            log.exception("Coordinator: pending_integrate kept-result scan failed")
            return _KeptResultScan(scanned=False, kept=None)
        return _KeptResultScan(scanned=True, kept=None)

    async def _recover_pending_integrate_from_task(
        self,
        pending: dict[str, Any],
        task: Task | None,
        report: dict[str, Any],
    ) -> bool:
        """Settle an integrate window from durable records rather than the bus.

        Args:
            pending: The ``pending_integrate`` sentinel dict.
            task: The sentinel's ``integrate_patch`` task row, if it still exists.
            report: The resume report dict to append fixes/warnings to.

        Returns:
            ``True`` when the sentinel was discharged or deliberately retained,
            ``False`` when the caller should roll the window back (a no-op when it changed nothing in the tree).
        """
        state = self.shared_state
        task_id = str(pending.get("task_id") or "")
        # Event retention can erase a KEEP. A promoted stack row is
        # independent positive evidence; a terminal/missing task without
        # its result is not evidence that a candidate was rejected.
        stack = state.optimization_stack or []
        if any(isinstance(row, dict) and row.get("task_id") == task_id for row in stack):
            state.pending_integrate = {}
            state.save(self.session_dir)
            report["fixes"].append({"kind": "retained_promoted_pending_integrate", "task_id": task_id})
            return True
        marker = _integrate_marker_of(pending, task)
        if marker.verdict == "retained":
            appended = self._replay_keep_from_result("integrate_patch", marker.outcome or {})
            report["fixes"].append(
                {"kind": "replayed_pending_integrate", "task_id": task_id, "appended": bool(appended)}
            )
            self._clear_pending_integrate(pending, gc_runtime=False)
            return True
        if not (pending.get("patches") or pending.get("artifacts")):
            return False  # env/args only: short of a recorded KEEP there is nothing to hold, only to clear
        if self._clear_settled_integrate_window(pending, report, marker):
            return True
        if marker.phase == "accepted":
            # A KEEP verdict left the candidate in the tree on purpose and popped the user's stash, so neither rollback
            # nor clear is correct here. Hold the obligation: dropping it would strand unrecorded candidate code, and
            # standing in a baseline number for the lost measurement would poison current_best and misattribute the
            # candidate's gain on the next stack rebench.
            report["warnings"].append(
                {"kind": "pending_integrate_outcome_unknown", "task_id": task_id, "phase": marker.phase}
            )
            return True
        if marker.verdict == "incomplete":
            # A confirmed outcome that discharged nothing: the teardown the
            # attempt owed is still owed, so retry it rather than hold the
            # sentinel until someone edits state.json. An online restore that
            # failed lands here on a task the runner recorded as succeeded.
            return False
        if confirmed_stopped_scope(task.history if task is not None else None) is not None:
            return False  # confirmed stopped with no verdict: nothing will ever finish this window
        if task is None or task.state not in ("queued", "running", "failed"):
            report["warnings"].append({"kind": "pending_integrate_outcome_unknown", "task_id": task_id})
            return True
        if _task_recorded_a_result(task):
            report["warnings"].append({"kind": "pending_integrate_result_missing", "task_id": task_id})
            return True
        return False

    async def _resume_recover_pending_integrate(self, report: dict[str, Any]) -> None:
        """Recover a crashed integrate_patch window from the sentinel.

        The verdict is taken from the first source that still carries one: a
        published KEEP on the bus, then the task row's own completion evidence,
        which the runner writes before it publishes. Whatever is found is
        classified by the same ``(status, phase)`` table the live promotion path
        reads, then replayed, cleared, or rolled back accordingly. A scan that
        could not read the event log establishes nothing, so it must neither
        roll back nor clear the sentinel.

        Args:
            report: The resume report dict to append fixes/warnings to.
        """
        state = self.shared_state
        pending = state.pending_integrate or {}
        if not (isinstance(pending, dict) and pending):
            return
        task_id = str(pending.get("task_id") or "")
        if not task_id:
            report["warnings"].append({"kind": "pending_integrate_task_missing"})
            return
        scan = await self._scan_for_published_integrate_keep(task_id)
        if not scan.scanned:
            report["warnings"].append({"kind": "pending_integrate_scan_failed", "task_id": task_id})
            return
        if scan.kept is not None:
            appended = self._replay_keep_from_result("integrate_patch", scan.kept)
            report["fixes"].append(
                {"kind": "replayed_pending_integrate", "task_id": task_id, "appended": bool(appended)}
            )
            self._clear_pending_integrate(pending, gc_runtime=False)
            return
        task = await self.tasks.find(task_id, kind="integrate_patch")
        if await self._recover_pending_integrate_from_task(pending, task, report):
            return
        gone_scope = confirmed_stopped_scope(task.history if task is not None else None)
        summary = self._resume_rollback_pending_integrate(pending, gone_scope)
        if summary.get("failed"):
            report["warnings"].append({"kind": "pending_integrate_rollback_failed", "task_id": task_id, **summary})
            state.save(self.session_dir)
            return
        if summary.get("reversed") or summary.get("artifacts_reverted") or summary.get("tree_replaced"):
            report["fixes"].append({"kind": "rolled_back_pending_integrate", "task_id": task_id, **summary})
        else:
            report["fixes"].append({"kind": "cleared_stale_pending_integrate", "task_id": task_id})
        self._clear_pending_integrate(pending, gc_runtime=True)

    async def _resume_recover_pending_warm_replay(
        self,
        report: dict[str, Any],
    ) -> None:
        """Rollback a combined PRELUDE set whose verdict was lost to a crash."""
        state = self.shared_state
        pending = state.warm_replay_pending or {}
        if not isinstance(pending, dict) or not pending:
            return
        rollback = self._coord.phase_prelude.rollback_combined_warm({}, None)
        errors = list(rollback.get("errors") or [])
        if errors:
            report["warnings"].append(
                {
                    "kind": "resume_warm_rollback_failed",
                    "task_id": pending.get("task_id"),
                    "errors": errors,
                }
            )
            state.set_stop_reason("warm_replay_rollback_failed")
            state.save(self.session_dir)
            return
        task_id = str(pending.get("task_id") or "").strip()
        task_state = ""
        if task_id:
            try:
                from ..state.task_registry import TaskNotFound

                try:
                    task = await self.tasks.get(task_id)
                except TaskNotFound:
                    task = None
                if task is not None:
                    task_state = str(task.state or "")
                    if task_state == "queued":
                        await self.tasks.transition(
                            task_id,
                            "cancelled",
                            evidence={"reason": "resume_interrupted_warm_replay"},
                        )
                        task_state = "cancelled"
                    elif task_state == "running":
                        await self.tasks.transition(
                            task_id,
                            "failed",
                            evidence={"failure_class": "resume_interrupted"},
                        )
                        task_state = "failed"
            except (sqlite3.Error, RuntimeError, ValueError) as exc:
                state.warm_replay_pending = {
                    **dict(pending),
                    "status": "task_invalidation_failed",
                }
                report["warnings"].append(
                    {
                        "kind": "resume_warm_task_invalidation_failed",
                        "task_id": task_id,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
                state.set_stop_reason("warm_replay_rollback_failed")
                state.save(self.session_dir)
                return
        state.warm_replay_outcome = {
            **dict(state.warm_replay_outcome or {}),
            "status": "failed",
            "reason": "interrupted_combined_validation_rolled_back",
            # This terminal branch never runs ``promote_warm_replay``, which is
            # what normally stamps ``settled_at``; stamp it here so the SBD
            # warm_replay event reports the real span instead of collapsing its
            # end_time back onto ``enqueued_at`` (a zero-duration replay).
            "settled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kernel": {
                "status": "reverted",
                "reason": "interrupted_combined_validation",
            },
        }
        report["fixes"].append(
            {
                "kind": "recovered_pending_warm_replay",
                "task_id": pending.get("task_id"),
                "task_state": task_state,
            }
        )
        state.save(self.session_dir)

    async def _resume_recover_pending_targeted_build(self, report: dict[str, Any]) -> None:
        """Reclaim an off-loop build that was in flight when the coordinator died.

        A detached compile cannot be re-adopted across a restart: kill its
        recorded process group, rmtree the attempt dir, sweep its per-attempt
        aiter JIT locks (a killed compile leaves a pid-less lock that wedges
        every later build of that module), mark the row failed with a
        ``timeout`` failure_class for the framework channel, and clear the
        sentinel. Best-effort throughout.
        """
        import shutil
        import signal

        from ..enablement.runtime.targeted_build import kill_build_pgroup

        state = self.shared_state
        pending = state.pending_targeted_build or {}
        if not (isinstance(pending, dict) and pending):
            return
        task_id = str(pending.get("task_id") or "")
        summary: dict[str, Any] = {"kind": "reclaimed_pending_targeted_build", "task_id": task_id}

        try:
            pgid = int(pending.get("pgid") or 0)
        except (TypeError, ValueError):
            pgid = 0
        if pgid > 0:
            kill_build_pgroup(pgid, sig=signal.SIGKILL)
            summary["killed_pgid"] = pgid

        attempt_root = str(pending.get("attempt_root") or "").strip()
        if attempt_root and Path(attempt_root).exists():
            shutil.rmtree(attempt_root, ignore_errors=True)
            summary["removed_attempt_root"] = attempt_root

        jit_dir = str(pending.get("aiter_jit_dir") or "").strip()
        if jit_dir:
            try:
                from ..actions.executors._aiter_jit import sweep_stale_aiter_locks_if_dead

                sweep_stale_aiter_locks_if_dead(aiter_jit_dir=Path(jit_dir))
                summary["swept_jit_dir"] = jit_dir
            except Exception:
                log.debug("resume: targeted-build jit sweep failed for %s", jit_dir, exc_info=True)

        state.enablement.last_build_failure = {
            "failure_class": "timeout",
            "failure_summary": "targeted build interrupted by coordinator restart",
        }
        if task_id:
            try:
                task = await self.tasks.get(task_id)
                if getattr(task, "state", "") == "running":
                    await self.tasks.transition(task_id, "failed", evidence={"failure_class": "resume_interrupted"})
                    summary["failed_row"] = True
            except Exception:
                log.debug("resume: targeted-build row fail raced for %s", task_id, exc_info=True)

        state.pending_targeted_build = {}
        report["fixes"].append(summary)

    async def _resume_recover_pending_revalidation(self, report: dict[str, Any]) -> None:
        """Unstick enablement_validation_pending when the tracked revalidation task is terminal.

        If the coordinator died while a revalidation baseline was running, the
        task row may already be in a terminal state on resume.  Without this
        recovery the pending flag stays set indefinitely and the next
        revalidation cannot be enqueued (tracked_tid is still the old row).

        Which recovery depends on how the row ended, not on this being the resume
        path. A row the run cancelled -- which is what the queue scan does to a
        revalidation the wall-clock budget can no longer fit -- measured nothing
        and produced no result to route, so it is no evidence about the baseline
        and gets :meth:`_reopen_revalidation_window`, the same verdict the
        writeback reaches for the same round. Anything else ended having had its
        chance: the window closes and the round charges an observation carrying
        no evidence, which spends an evidence-stall credit.
        """
        state = self.shared_state
        if not bool(state.enablement.validation_pending):
            return
        tracked_tid = str(state.enablement.revalidation_task_id or "").strip()
        if not tracked_tid:
            return
        try:
            from ..state.task_registry import TERMINAL_STATES, TaskNotFound

            row_state = ""
            try:
                row = await self.tasks.get(tracked_tid)
                row_state = str(getattr(row, "state", "") or "")
                is_terminal = row_state in TERMINAL_STATES
            except TaskNotFound:
                is_terminal = True
            if not is_terminal:
                return
            if row_state == "cancelled":
                await self._reopen_revalidation_window()
                report["fixes"].append({"kind": "reopened_revalidation_the_run_cancelled", "task_id": tracked_tid})
                log.info(
                    "resume: revalidation task %s was cancelled by the run; window left open at generation %d",
                    tracked_tid,
                    int(state.enablement.revalidation_generation or 0),
                )
                return
            enablement_event.record_revalidation_cleared(
                generation=int(state.enablement.revalidation_generation or 0), task_id=tracked_tid
            )
            state.enablement.validation_pending = False
            state.enablement.revalidation_task_id = ""
            report["fixes"].append({"kind": "cleared_orphaned_revalidation_pending", "task_id": tracked_tid})
            log.info(
                "resume: cleared stale enablement_validation_pending for terminal revalidation task %s",
                tracked_tid,
            )
        except Exception:
            log.debug("resume: revalidation pending recovery check failed", exc_info=True)

    async def _surface_orphaned_keep(self, report: dict[str, Any], kind: str, variant: str, task_id: Any) -> None:
        report["warnings"].append(
            {"kind": "orphaned_keep", "orphan_kind": kind, "variant": variant, "task_id": task_id}
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {"kind": "orphaned_keep", "severity": "medium", "orphan_kind": kind, "variant": variant},
        )

    async def _recover_orphaned_integrate_keep(
        self,
        result: dict[str, Any],
        report: dict[str, Any],
        variant: str,
        task_id: Any,
    ) -> bool:
        kind = "integrate_patch"
        workspace = str(result.get("workspace") or "").strip()
        if workspace and not Path(workspace).exists():
            report["warnings"].append(
                {
                    "kind": "orphaned_keep_discarded",
                    "orphan_kind": kind,
                    "variant": variant,
                    "task_id": task_id,
                    "reason": "workspace_missing",
                }
            )
            await self.bus.record_observation(
                "coordinator",
                "observation",
                {"kind": "orphaned_keep_discarded", "severity": "medium", "orphan_kind": kind, "variant": variant},
            )
            return False
        if self._replay_keep_from_result(kind, result):
            report["fixes"].append({"kind": "replayed_orphaned_keep", "orphan_kind": kind, "variant": variant})
            return True
        report["warnings"].append({"kind": "orphaned_keep_replay_noop", "orphan_kind": kind, "variant": variant})
        return False

    async def _resume_recover_orphaned_keeps(self, report: dict[str, Any]) -> None:
        """Recover / surface KEEPs present in the event log but absent from the stack (Gap B).

        ``integrate_patch`` KEEPs are well-defined (a ``kept`` status means the
        single-variant bench + accuracy gate passed and the patch was committed),
        so a kept-but-absent one is a crash before the append landed → replay it
        (idempotent), unless its run workspace is gone → discard + alert. ``explore``
        / ``framework`` KEEPs are ambiguous (the stack they landed on is only
        known to the round that ran), so they are surfaced as a ``medium``
        alert rather than resurrected. Whatever the stack ends up as
        is re-validated by the Gap A full-stack rebench.

        Args:
            report: The resume report dict to append fixes/warnings to.
        """
        try:
            stack_keys = _stacked_keep_keys(self.shared_state.optimization_stack)
            seen: set[_StackKey] = set()
            for msg in await self.bus.tail(topic="delegated_result", n=10_000):
                payload = msg.payload or {}
                res = payload.get("result") or {}
                if not isinstance(res, dict) or str(res.get("status") or "").lower() != "kept":
                    continue
                kind = str(payload.get("kind") or "")
                key = _orphan_stack_key(kind, res)
                if key is None or not key.variant or key in stack_keys or key in seen:
                    continue
                seen.add(key)
                if kind != "integrate_patch":
                    await self._surface_orphaned_keep(report, kind, key.variant, payload.get("task_id"))
                elif await self._recover_orphaned_integrate_keep(res, report, key.variant, payload.get("task_id")):
                    stack_keys.add(key)
        except Exception:
            log.exception("Coordinator: orphaned KEEP resume recovery failed")
