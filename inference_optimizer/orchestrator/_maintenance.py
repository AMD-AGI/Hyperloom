# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
import time
from typing import Any
from .shared_state import SharedState
from .coordinator_helpers import (  # noqa: F401 - re-exported for callers/tests
    _BASELINE_FINGERPRINT_KEYS,
    _baseline_params_fingerprint,
    _dedupe_extra_server_args,
    _infer_model_class_from_config,
    _merge_cumulative_extra_sglang_args,
    _parse_baseline_workload_extra,
    _parse_iso_unix,
    _resolve_roofline_watermark_ratio,
    effective_closing_grace_sec,
    format_exc_brief,
    serialize_verdict_advisory,
)

import logging as _logging
log = _logging.getLogger(__name__)


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
        try:
            reaped = await self.locks.reap_expired()
            summary["leases_reaped"] = len(reaped or [])
        except Exception:  # noqa: BLE001 — maintenance never aborts the run loop
            log.exception("maintenance: serving-lease reap failed")
        try:
            summary["gpu_leases_reaped"] = await self.gpu_specialist_pool.reap_expired()
        except Exception:  # noqa: BLE001
            log.exception("maintenance: gpu-lease reap failed")
        # R6 watchdog/self-heal: reclaim orphaned running tasks whose execution
        # lease has expired so a dead worker never wedges a lane indefinitely.
        try:
            reclaimed = await self.tasks.reclaim_expired_running(
                reason="maintenance_watchdog",
            )
            summary["running_tasks_reclaimed"] = len(reclaimed)
        except Exception:  # noqa: BLE001
            log.exception("maintenance: running-task reclaim failed")
        try:
            from . import db_maintenance as _db_maint

            res = await _db_maint.run_db_retention(self.db, self.cursors)
            summary["events_pruned"] = res.events_deleted
            summary["tasks_pruned"] = res.tasks_deleted
        except Exception:  # noqa: BLE001
            log.exception("maintenance: DB retention failed")
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

        from ..session_paths import runs_root as _runs_root

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
        """Compact the orchestration conversation into durable memory (plan Step 4).

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

        from . import orchestration_memory as _orch_mem

        now_min = 0.0
        if self._run_started_monotonic is not None:
            now_min = (time.monotonic() - self._run_started_monotonic) / 60.0
        tracker = self._checkpoint_tracker
        ticks_since = max(0, tick - tracker.last_tick)
        minutes_since = max(0.0, now_min - tracker.last_minute_mark)
        # Hard context-token guardrail: near the window we MUST compact even when
        # the LLM summary is degenerate (deterministic fallback) to avoid overflow.
        hard = self._checkpoint_policy.is_hard_compaction(tracker.context_tokens_now)
        # Anti-thrash floor for the TOKEN-budget triggers only. A compaction
        # RESETS the persistent conversation, so the next tick re-sends the full
        # SEED whose own token cost re-trips the soft/hard budget — looping
        # forever (checkpoint every tick → conversation never persists →
        # orchestration loses cross-tick memory and re-does discovery instead of
        # progressing). When the token budget is what's firing, require a minimum
        # tick gap between compactions; genuine cadence triggers (every_ticks /
        # minutes / chars / phase boundary) are unaffected. A true near-window
        # emergency (>= 98% of the window — unreachable by a single SEED) always
        # bypasses the floor so we never overflow the context window.
        suppress_token_trigger = False
        if not force and ticks_since < max(1, int(getattr(self, "_checkpoint_min_tick_gap", 2) or 2)):
            ctx_token_hard = int(getattr(self._checkpoint_policy, "context_token_hard", 0) or 0)
            ctx_token_soft = int(getattr(self._checkpoint_policy, "context_token_soft", 0) or 0)
            ctx_now = int(tracker.context_tokens_now)
            token_due = (
                (ctx_token_hard > 0 and ctx_now >= ctx_token_hard)
                or (ctx_token_soft > 0 and ctx_now >= ctx_token_soft)
            )
            emergency_ceiling = (
                int(ctx_token_hard / max(0.01, _orch_mem.DEFAULT_CONTEXT_TOKEN_HARD_FRACTION) * 0.98)
                if ctx_token_hard > 0
                else 0
            )
            in_emergency = emergency_ceiling > 0 and ctx_now >= emergency_ceiling
            suppress_token_trigger = token_due and not in_emergency
            if suppress_token_trigger:
                # Suppress the token-driven hard flag so the freshly-seeded
                # conversation can persist; fall through to the cadence check,
                # which will return False unless a non-token trigger fired.
                hard = False
        # Authoritative growth signal is the context-token water level; the char
        # count is a fallback for backends that don't report token usage.
        if (
            not force
            and not hard
            and not self._checkpoint_policy.should_checkpoint(
                ticks_since_last=ticks_since,
                minutes_since_last=minutes_since,
                chars_since_last=tracker.chars_since_last,
                phase_changed=phase_changed,
                # During the anti-thrash window, zero out the token level so the
                # soft-token trigger can't re-fire the compaction we just
                # suppressed; cadence triggers still evaluate normally.
                context_tokens_now=0 if suppress_token_trigger else tracker.context_tokens_now,
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
            # Path 1 — degenerate reply, NOT near the window: skip compaction.
            # Preserve the live conversation + prior memory (a forgetful summary
            # must never blank the plan or drop the history). Still reset the
            # tracker so we don't immediately retry next tick (checkpoint storm).
            if degenerate and not hard:
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
                # Repeated degeneracy: raise the observation's severity so the
                # operator/robustness sees it (advisory only — never
                # auto-changes strategy, and does not hijack the alert intent).
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
            # Path 2 — degenerate but near the window: compact anyway using a
            # deterministic fallback synthesised from authoritative state, so the
            # conversation is reset and never overflows.
            if degenerate and hard:
                parsed = _orch_mem.deterministic_memory_fallback(self.shared_state)
            # Path 3 (and post-fallback): a usable summary — compact for real.
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
            # Append to the rollback ring (bounded) so a later bad compaction can
            # be recovered from a prior good snapshot (long-run #1).
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
                    "hard_compaction": bool(hard),
                    "context_tokens": int(tracker.context_tokens_now),
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
