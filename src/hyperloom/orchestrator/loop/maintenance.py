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
    """Reap expired serving/GPU leases, reclaim orphaned running tasks, prune the DB.

    Shared by the periodic maintenance tick and the cycle soft-restart. The
    task reclaim is the R6 watchdog: a running task whose execution lease
    expired is failed so a dead worker never wedges a lane indefinitely. Every
    step is individually best-effort — maintenance never aborts the run loop.

    Args:
        host: Anything exposing the Coordinator's ``locks``,
            ``gpu_specialist_pool``, ``tasks`` and ``db``.
        summary: Mutated in place with the per-step counts.
        reason: Reclaim reason recorded on the tasks and used as the log prefix.
    """
    try:
        reaped = await host.locks.reap_expired()
        summary["leases_reaped"] = len(reaped or [])
    except Exception:  # noqa: BLE001
        log.exception("%s: serving-lease reap failed", reason)
    try:
        summary["gpu_leases_reaped"] = await host.gpu_specialist_pool.reap_expired()
    except Exception:  # noqa: BLE001
        log.exception("%s: gpu-lease reap failed", reason)
    try:
        reclaimed = await host.tasks.reclaim_expired_running(reason=reason)
        summary["running_tasks_reclaimed"] = len(reclaimed)
    except Exception:  # noqa: BLE001
        log.exception("%s: running-task reclaim failed", reason)
    try:
        from ..bus import db_maintenance as _db_maint

        res = await _db_maint.run_db_retention(host.db)
        summary["events_pruned"] = res.events_deleted
        summary["tasks_pruned"] = res.tasks_deleted
    except Exception:  # noqa: BLE001
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
        """Reap expired leases, prune the DB, and trim ``runs/`` when disk is low.

        Every step is independently guarded so one failure never aborts the
        run loop. The Coordinator's wall-clock gate owns the cadence.

        Args:
            tick: The current coordinator tick; recorded in the summary.

        Returns:
            A summary dict of work performed (leases reaped, tasks reclaimed,
            rows pruned, disk status).
        """
        summary: dict[str, Any] = {"tick": tick}
        await run_lease_and_db_reclaim(self, summary, reason="maintenance_watchdog")
        try:
            disk = self._maybe_prune_runs_for_disk()
            if disk is not None:
                summary["disk"] = disk
        except Exception:  # noqa: BLE001
            log.exception("maintenance: disk monitor failed")
        log.info("maintenance tick %d: %s", tick, summary)
        return summary

    def _maybe_prune_runs_for_disk(self) -> dict[str, Any] | None:
        """LRU-trim per-task ``runs/`` workspaces when disk is low.

        No-op unless the session partition is below the free-space floor or
        above the used-fraction ceiling. When triggered, keeps only the most
        recently modified N task dirs per action and deletes the rest. Also
        warns (only) when ``state.json`` grows past a soft size cap.

        Returns:
            dict | None: A summary when the check ran, else ``None``.
        """
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
            # Best-effort disk/size warning only; never block on a stat() that
            # races a concurrent prune or a transient filesystem error.
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
