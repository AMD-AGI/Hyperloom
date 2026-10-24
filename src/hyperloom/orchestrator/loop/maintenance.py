# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Periodic Coordinator maintenance: lease reaping, DB retention, disk trim."""

from __future__ import annotations
import shutil
import sqlite3
from pathlib import Path
from typing import Any
from ..state.shared_state import SharedState
from ..collaborator import CoordinatorCollaborator

import logging as _logging

log = _logging.getLogger(__name__)


async def run_lease_and_db_reclaim(
    host: Any,
    summary: dict[str, Any],
    *,
    reason: str,
) -> None:
    """Report confirmed-dead cleanup and prune retained database history.

    Shared by periodic maintenance and cycle soft-restart. Resource ownership
    is resolved by the reconciler: owners it proved dead, and lanes whose holder
    both ended and proved nothing is still using them -- never inferred from
    elapsed lease budgets.

    ``leases_unverifiable`` rides the same summary because it is the other half
    of that answer: lanes still held by a holder that ended without confirming
    its cleanup. Nothing decides those -- no identity available to this process
    survives a served process that setsid's away from it -- so they are retained
    on purpose. A number that stays put while the queue does not drain is where
    an operator starts; the remedy for each one is logged once by the diagnostic.

    Args:
        host: Coordinator exposing ``reconciler``, ``locks`` and ``db``.
        summary: Mutated in place with the per-step counts.
        reason: Log prefix identifying the maintenance caller.
    """
    try:
        report = host.reconciler.last_report
        summary["leases_reaped"] = report.leases_reaped
        summary["leases_unverifiable"] = report.leases_unverifiable
        # Whether retained lanes are an accident or the ordinary outcome. Every
        # portable way to release them automatically was refuted (see
        # docs/task-containment.md), and the one candidate left is
        # safety-critical, so this is the number that decides whether anyone
        # should build it.
        unconfirmed, ended = await host.locks.cleanup_confirmation_rate()
        if ended:
            summary["cleanup_unconfirmed"] = f"{unconfirmed}/{ended}"
        summary["running_tasks_reclaimed"] = len(report.failed_tasks)
    except sqlite3.Error:
        log.exception("%s: reading the reconciler's cleanup report failed", reason)
    try:
        from ..bus import db_maintenance as _db_maint

        res = await _db_maint.run_db_retention(host.db)
        summary["events_pruned"] = res.events_deleted
        summary["tasks_pruned"] = res.tasks_deleted
    except sqlite3.Error:
        log.exception("%s: DB retention failed", reason)


class MaintenanceCollaborator(CoordinatorCollaborator):
    """Handles session maintenance: disk cleanup, task reclaim, and health checks."""

    # Advisory disk guard: when free space runs low, LRU-trim per-task runs/ workspaces; durable state is never
    # touched, and neither is a workspace a live task still reads or one the enablement stack still replays. Free
    # space, not the used fraction: a shared volume's used fraction is mostly other tenants'.
    _DISK_FREE_MIN_GB: float = 20.0
    _DISK_RUNS_KEEP_PER_ACTION: int = 50
    _STATE_JSON_WARN_BYTES: int = 50 * 1024 * 1024

    async def run(
        self,
        *,
        tick: int,
    ) -> dict[str, Any] | None:
        """Report ownership cleanup, prune the DB, and trim ``runs/`` when disk is low."""
        summary: dict[str, Any] = {"tick": tick}
        await run_lease_and_db_reclaim(self._coord, summary, reason="maintenance_watchdog")
        disk = self._maybe_prune_runs_for_disk(await self._workspaces_in_use())
        if disk is not None:
            summary["disk"] = disk
        log.info("maintenance tick %d: %s", tick, summary)
        return summary

    async def _workspaces_in_use(self) -> frozenset[str]:
        """Task ids whose ``runs/`` workspace a queued or running task still reads: its own, and its specialist's."""
        names: set[str] = set()
        for task in (*await self.tasks.queued(), *await self.tasks.running()):
            names.add(task.task_id)
            if task.params.get("specialist_task_id"):
                names.add(task.params["specialist_task_id"])
        return frozenset(names)

    def _maybe_prune_runs_for_disk(self, in_use: frozenset[str]) -> dict[str, Any] | None:
        """LRU-trim per-task ``runs/`` workspaces outside ``in_use`` when disk is low."""
        from hyperloom.inference_optimizer.session.session_paths import runs_root as _runs_root

        try:
            usage = shutil.disk_usage(str(self.session_dir))
        except OSError:
            return None
        free_gb = usage.free / (1024.0**3)
        used_frac = usage.used / usage.total if usage.total else 0.0
        summary: dict[str, Any] = {
            "free_gb": round(free_gb, 2),
            "used_frac": round(used_frac, 4),
        }
        try:
            state_path = SharedState.state_path(self.session_dir)
            if state_path.is_file() and state_path.stat().st_size > self._STATE_JSON_WARN_BYTES:
                log.warning(
                    "maintenance: state.json is %.1f MB (soft cap %.0f MB)",
                    state_path.stat().st_size / (1024.0**2),
                    self._STATE_JSON_WARN_BYTES / (1024.0**2),
                )
        except OSError:
            # Best-effort disk/size warning only; never block on a stat() that races a concurrent prune or a transient
            # filesystem error.
            pass

        if free_gb >= self._DISK_FREE_MIN_GB:
            return summary

        runs_root = _runs_root(self.session_dir)
        if not runs_root.is_dir():
            return summary
        removed = self._prune_stale_runs(runs_root, in_use)
        summary["runs_pruned"] = removed
        if removed:
            log.info(
                "maintenance: low disk (free=%.1fGB used=%.0f%%) pruned %d run dirs",
                free_gb,
                used_frac * 100.0,
                removed,
            )
        return summary

    def _prune_stale_runs(self, runs_root: Path, in_use: frozenset[str]) -> int:
        """Delete each action's task dirs past the newest ``_DISK_RUNS_KEEP_PER_ACTION``; returns how many went.

        A dir in ``in_use`` or holding an enablement stack file is kept.
        """
        stack_files = {Path(f).resolve() for f in self.shared_state.enablement.stack_files()}
        removed = 0
        for action_dir in (d for d in runs_root.iterdir() if d.is_dir()):
            task_dirs = sorted((p for p in action_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
            for stale in task_dirs[: -self._DISK_RUNS_KEEP_PER_ACTION]:
                if stale.name in in_use or any(f.is_relative_to(stale.resolve()) for f in stack_files):
                    continue
                try:
                    shutil.rmtree(stale)
                except OSError:
                    log.warning("maintenance: failed to prune %s", stale)
                    continue
                removed += 1
        return removed
