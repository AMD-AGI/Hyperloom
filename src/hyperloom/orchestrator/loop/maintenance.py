# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Periodic Coordinator maintenance: lease reaping, DB retention, disk trim."""

from __future__ import annotations
from typing import Any
from ..state.shared_state import SharedState

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
        host: Coordinator exposing ``reconciler`` and ``db``.
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
        unconfirmed, ended = await host.reconciler.cleanup_confirmation_rate()
        if ended:
            summary["cleanup_unconfirmed"] = f"{unconfirmed}/{ended}"
        summary["running_tasks_reclaimed"] = len(report.failed_tasks)
    except Exception:
        log.exception("%s: reading the reconciler's cleanup report failed", reason)
    try:
        from ..bus import db_maintenance as _db_maint

        res = await _db_maint.run_db_retention(host.db)
        summary["events_pruned"] = res.events_deleted
        summary["tasks_pruned"] = res.tasks_deleted
    except Exception:
        log.exception("%s: DB retention failed", reason)


class MaintenanceCollaborator:
    """Extracted collaborator; delegates unknown attrs to its Coordinator."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_coord"), name)

    async def _run_maintenance(
        self,
        *,
        tick: int,
    ) -> dict[str, Any] | None:
        """Report ownership cleanup, prune the DB, and trim ``runs/`` when disk is low."""
        summary: dict[str, Any] = {"tick": tick}
        await run_lease_and_db_reclaim(self, summary, reason="maintenance_watchdog")
        disk = self._maybe_prune_runs_for_disk()
        if disk is not None:
            summary["disk"] = disk
        log.info("maintenance tick %d: %s", tick, summary)
        return summary

    def _maybe_prune_runs_for_disk(self) -> dict[str, Any] | None:
        """LRU-trim per-task ``runs/`` workspaces when disk is low."""
        import shutil

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

        if free_gb >= self._DISK_FREE_MIN_GB and used_frac <= self._DISK_USED_MAX_FRAC:
            return summary

        runs_root = _runs_root(self.session_dir)
        if not runs_root.is_dir():
            return summary
        removed = 0
        for action_dir in runs_root.iterdir():
            if not action_dir.is_dir():
                continue
            task_dirs = [p for p in action_dir.iterdir() if p.is_dir()]
            if len(task_dirs) <= self._DISK_RUNS_KEEP_PER_ACTION:
                continue
            task_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for stale in task_dirs[self._DISK_RUNS_KEEP_PER_ACTION :]:
                try:
                    shutil.rmtree(stale, ignore_errors=True)
                    removed += 1
                except OSError:
                    log.warning("maintenance: failed to prune %s", stale)
        summary["runs_pruned"] = removed
        if removed:
            log.info(
                "maintenance: low disk (free=%.1fGB used=%.0f%%) pruned %d run dirs",
                free_gb,
                used_frac * 100.0,
                removed,
            )
        return summary
