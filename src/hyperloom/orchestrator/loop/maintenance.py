# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
import time
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
            ``gpu_specialist_pool``, ``tasks``, ``db`` and ``cursors``.
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

        res = await _db_maint.run_db_retention(host.db, host.cursors)
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

    async def _maybe_run_maintenance_tick(
        self,
        *,
        tick: int,
    ) -> dict[str, Any] | None:
        """Periodic in-process maintenance (R5 reaper + R4 DB retention).

        On a fixed tick cadence: actively reap TTL-expired serving + GPU leases
        and prune the events/tasks DB (strictly below the resume anchor) so a
        multi-day single-session run never leaks capacity or grows the DB
        unbounded. Best-effort — every step is independently guarded so one
        failure never aborts the run loop. Returns a summary dict when it ran,
        else ``None``.

        Args:
            tick: The current coordinator tick; maintenance only runs when it
                is positive and a multiple of the configured cadence.

        Returns:
            A summary dict of work performed (leases reaped, tasks reclaimed,
            rows pruned, disk status) when the cadence fired, else ``None``.
        """
        every = int(getattr(self, "_maintenance_every_ticks", 0) or 0)
        if every <= 0 or tick <= 0 or (tick % every) != 0:
            return None
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

    async def _maybe_checkpoint_orchestration(
        self,
        *,
        tick: int,
        phase_changed: bool = False,
        force: bool = False,
    ) -> bool:
        """Compact the orchestration conversation into durable memory.

        Returns True when a checkpoint was taken. Best-effort. ``force`` bypasses
        the throttle policy (used by the R6 cycle-boundary soft restart) but
        still requires a seeded conversational backend.

        Args:
            tick: The current coordinator tick, recorded on the checkpoint
                tracker and emitted in the observation.
            phase_changed: Whether the phase changed this tick; influences the
                throttle policy's decision to checkpoint.
            force: Bypass the throttle policy and checkpoint regardless of
                cadence (still requires a seeded conversational backend).

        Returns:
            ``True`` if a checkpoint was taken, else ``False``.
        """
        if not self._checkpoint_enabled:
            return False
        if not self._orchestration_conversational():
            return False
        backend = self.backends.get("orchestration")
        if backend is None or not getattr(backend, "conversational", False):
            return False
        # Nothing to compact before the first real turn seeded the session.
        if not self._orchestration_seeded:
            return False

        from ..state import orchestration_memory as _orch_mem

        now_min = 0.0
        if self._run_started_monotonic is not None:
            now_min = (time.monotonic() - self._run_started_monotonic) / 60.0
        tracker = self._checkpoint_tracker
        ticks_since = max(0, tick - tracker.last_tick)
        minutes_since = max(0.0, now_min - tracker.last_minute_mark)
        # Growth signal is the context-token water level; char count is the
        # fallback for backends that don't report token usage.
        if (
            not force
            and not self._checkpoint_policy.should_checkpoint(
                ticks_since_last=ticks_since,
                minutes_since_last=minutes_since,
                chars_since_last=tracker.chars_since_last,
                phase_changed=phase_changed,
                context_tokens_now=tracker.context_tokens_now,
            )
        ):
            return False

        try:
            sys_prompt = await self._load_system_prompt("orchestration")
            result = await backend.run(
                prompt=_orch_mem.CHECKPOINT_REQUEST_PROMPT,
                system_prompt=sys_prompt,
                tools=[],
                max_turns=0,
                # Checkpoint summary is plain-text, not emit_intent; relax no-intent guard.
                allow_no_intent=True,
            )
            raw_text = getattr(result, "raw_text", "") or ""
            parsed = _orch_mem.parse_checkpoint_reply(raw_text)
            degenerate = _orch_mem.is_degenerate_checkpoint(parsed)
            cur_phase = str(getattr(self.shared_state, "phase", "") or "")
            # Degenerate reply: skip compaction, preserve the live conversation +
            # prior memory, but reset the tracker to avoid a checkpoint storm.
            if degenerate:
                self._coord._consec_degenerate_ckpt += 1
                tracker.reset(tick=tick, minute_mark=now_min, phase=cur_phase)
                await self._record_observation(
                    "coordinator",
                    "observation",
                    {
                        "kind": "orchestration_checkpoint_degraded",
                        "tick": tick,
                        "consecutive": self._consec_degenerate_ckpt,
                        "parse_error": str(parsed.get("parse_error", "") or ""),
                    },
                )
                # Repeated degeneracy: raise the observation's severity
                # (advisory only).
                if self._consec_degenerate_ckpt >= 3:
                    await self._record_observation(
                        "coordinator",
                        "observation",
                        {
                            "kind": "orchestration_checkpoint_degraded",
                            "severity": "medium",
                            "tick": tick,
                            "consecutive": self._consec_degenerate_ckpt,
                            "detail": "orchestration checkpoint summaries repeatedly degenerate",
                        },
                    )
                return False
            # Usable summary — compact for real.
            self._coord._consec_degenerate_ckpt = 0
            seq = 0
            try:
                row = self.bus.db.fetchone_sync("SELECT COALESCE(MAX(seq), 0) AS s FROM events")
                seq = int(row["s"]) if row else 0
            except Exception:  # noqa: BLE001
                seq = 0
            record = _orch_mem.build_memory_record(
                parsed,
                seq=seq,
                tick=tick,
                previous=dict(getattr(self.shared_state, "orchestration_memory", {}) or {}),
            )
            self.shared_state.orchestration_memory = record
            # Append to the bounded rollback ring so a later bad compaction can
            # be recovered from a prior good snapshot.
            try:
                hist = list(getattr(self.shared_state, "orchestration_memory_history", []) or [])
                hist.append(record)
                self.shared_state.orchestration_memory_history = hist[-10:]
            except Exception:  # noqa: BLE001 — history is best-effort
                log.exception("Coordinator: failed to append orchestration_memory_history")
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001
                log.exception("Coordinator: failed to persist orchestration_memory")
            # Reset so the next turn re-seeds from the compacted memory.
            self._coord._orchestration_seed_memory = _orch_mem.render_memory_for_seed(record)
            self._reset_orchestration_conversation()
            # The level that decided this compaction; the reset clears it.
            level_at_trigger = int(tracker.context_tokens_now)
            tracker.reset(
                tick=tick,
                minute_mark=now_min,
                phase=cur_phase,
            )
            await self._record_observation(
                "coordinator",
                "observation",
                {
                    "kind": "orchestration_checkpoint",
                    "tick": tick,
                    "seq": seq,
                    "checkpoint_count": record.get("checkpoint_count", 0),
                    "phase_changed": bool(phase_changed),
                    "context_tokens": level_at_trigger,
                },
            )
            return True
        except Exception:  # noqa: BLE001 — never let a checkpoint kill the loop
            log.exception("Coordinator: orchestration checkpoint failed")
            # Reset the tracker even on failure so a transient backend error
            # (e.g. gateway 401) can't trigger a checkpoint storm next tick.
            try:
                tracker.reset(
                    tick=tick,
                    minute_mark=now_min,
                    phase=str(getattr(self.shared_state, "phase", "") or ""),
                )
            except Exception:  # noqa: BLE001
                pass
            return False
