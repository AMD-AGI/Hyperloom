# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Off-loop targeted-build lifecycle (Rung 5, S2).

Drives compiled-component builds as resume-safe task rows that run OFF the
coordinator tick loop: one build at a time (``build_lane`` capacity 1), spawned
detached, and reaped across ticks against a wall-clock deadline. The build never
runs inside ``_pump_dispatcher_once``'s inflight drain (the dispatcher skips
``kind == "targeted_build"``); this collaborator pumps and reaps it instead.

The in-flight handle is held in memory keyed by task_id; the durable copy is the
``pending_targeted_build`` sentinel, so a crash/resume can reclaim the orphan
(see ``writeback._resume_recover_pending_targeted_build``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from ..bus.resource_lock import Lease
from ..framework.build_actions import TargetedBuildAction, build_novelty_key
from ..framework.targeted_build import BuildHandle, poll_build, spawn_build

log = logging.getLogger(__name__)

_BUILD_KIND = "targeted_build"
# Grace added to the wall-clock budget for the lease TTL reclaim backstop.
_LEASE_GRACE_SEC = 300


def _novelty_idempotency_key(action: TargetedBuildAction) -> str:
    key = build_novelty_key(action)
    digest = hashlib.sha1(
        json.dumps(list(key), sort_keys=True, default=str).encode(),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return f"targeted_build:{action.component}:{digest}"


class BuildLifecycleCollaborator:
    """Extracted collaborator; delegates unknown attrs to its Coordinator."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator
        # In-memory handles + leases for in-flight builds, keyed by task_id.
        self._build_handles: dict[str, BuildHandle] = {}
        self._build_leases: dict[str, Lease] = {}

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_coord"), name)

    def _attempt_root(self, task_id: str) -> str:
        return str(self.session_dir / "enablement" / "builds" / task_id)

    async def enqueue_targeted_build(self, action: TargetedBuildAction) -> str:
        """Enqueue a resume-safe ``targeted_build`` row (idempotent by novelty).

        Args:
            action: The build to run.

        Returns:
            str: The task id (existing row's id on a repeat novelty tuple).
        """
        from ..framework.targeted_build import _resolve_budget_sec

        ttl = int(_resolve_budget_sec(action)) + _LEASE_GRACE_SEC
        task, _existing = await self.tasks.create_or_return_existing(
            kind=_BUILD_KIND,
            params=action.to_state(),
            idempotency_key=_novelty_idempotency_key(action),
            requires_lanes=["build_lane"],
            lease_ttl_sec=ttl,
        )
        return str(getattr(task, "task_id", "") or "")

    async def _maybe_pump_targeted_build(self, *, tick: int) -> dict[str, Any] | None:
        """Start the oldest queued build when ``build_lane`` is free.

        One build at a time (derived single-in-flight gate: a running row
        exists). Best-effort; never aborts the tick.
        """
        try:
            running = [t for t in await self.tasks.by_state("running") if t.kind == _BUILD_KIND]
            if running:
                return None
            queued = [t for t in await self.tasks.queued() if t.kind == _BUILD_KIND]
            if not queued:
                return None
            task = queued[0]
            lease = await self.locks.try_acquire_many(
                ["build_lane"],
                holder_id=task.task_id,
                task_id=task.task_id,
                action=_BUILD_KIND,
                ttl_sec=task.lease_ttl_sec or 60,
            )
            if lease is None:
                return None
            try:
                action = TargetedBuildAction.from_state(task.params)
                attempt_root = self._attempt_root(task.task_id)
                handle = spawn_build(action, attempt_root=attempt_root)
            except Exception as exc:  # noqa: BLE001 — spawn failure is a clean fail
                await self.locks.release(lease)
                log.exception("targeted_build: spawn failed for %s", task.task_id)
                await self._fail_build_row(task.task_id, failure_class="compile_error", summary=repr(exc))
                return {"tick": tick, "spawn_failed": task.task_id}
            self._build_handles[task.task_id] = handle
            self._build_leases[task.task_id] = lease
            self.shared_state.pending_targeted_build = handle.to_sentinel(task.task_id)
            self.shared_state.save(self.session_dir)
            await self.tasks.transition(task.task_id, "running", evidence={"build": action.component})
            log.info("targeted_build: started %s (%s) pid=%d", task.task_id, action.component, handle.pid)
            return {"tick": tick, "started": task.task_id}
        except Exception:  # noqa: BLE001 — lifecycle never aborts the run loop
            log.exception("targeted_build: pump failed")
            return None

    async def _maybe_reap_targeted_build(self, *, tick: int) -> dict[str, Any] | None:
        """Poll the in-flight build; finalize + release on a terminal result."""
        try:
            running = [t for t in await self.tasks.by_state("running") if t.kind == _BUILD_KIND]
            if not running:
                return None
            task = running[0]
            handle = self._build_handles.get(task.task_id)
            if handle is None:
                # No in-memory handle (crash/resume): the reclaim path owns it.
                return None
            result = poll_build(handle)
            if result is None:
                return None
            self._record_build_result(result)
            await self._release_build(task.task_id)
            new_state = "succeeded" if result.ok else "failed"
            await self.tasks.transition(
                task.task_id,
                new_state,
                evidence={"failure_class": result.failure_class},
            )
            self.shared_state.pending_targeted_build = {}
            self.shared_state.save(self.session_dir)
            log.info(
                "targeted_build: %s -> %s (failure_class=%s)",
                task.task_id,
                new_state,
                result.failure_class,
            )
            return {"tick": tick, "finished": task.task_id, "ok": result.ok}
        except Exception:  # noqa: BLE001
            log.exception("targeted_build: reap failed")
            return None

    def _record_build_result(self, result: Any) -> None:
        """Append the manifest entry and record the failure carrier (framework channel §9)."""
        state = self.shared_state
        manifest = list(getattr(state, "enablement_build_manifest", []) or [])
        manifest.append(result.to_state())
        state.enablement_build_manifest = manifest
        if not result.ok:
            state.enablement_last_build_failure = {
                "failure_class": result.failure_class,
                "failure_summary": result.failure_summary or result.error,
            }
            # A killed build can wedge every later compile of a module (L4); the
            # per-attempt jit dir is swept so a resumed/next attempt is clean.
            self._sweep_build_jit(result.attempt_root)

    @staticmethod
    def _sweep_build_jit(attempt_root: str) -> None:
        from ..actions.executors._aiter_jit import sweep_stale_aiter_locks_if_dead

        jit_dir = Path(str(attempt_root or "")) / "aiter_jit"
        try:
            sweep_stale_aiter_locks_if_dead(aiter_jit_dir=jit_dir)
        except Exception:  # noqa: BLE001 — sweep is best-effort
            log.debug("targeted_build: jit sweep failed for %s", jit_dir, exc_info=True)

    async def _release_build(self, task_id: str) -> None:
        self._build_handles.pop(task_id, None)
        lease = self._build_leases.pop(task_id, None)
        if lease is not None:
            try:
                await self.locks.release(lease)
            except Exception:  # noqa: BLE001 — reclaim/reap is the backstop
                log.debug("targeted_build: lease release raced for %s", task_id, exc_info=True)

    async def _fail_build_row(self, task_id: str, *, failure_class: str, summary: str) -> None:
        self.shared_state.enablement_last_build_failure = {
            "failure_class": failure_class,
            "failure_summary": summary,
        }
        self.shared_state.pending_targeted_build = {}
        self.shared_state.save(self.session_dir)
        try:
            await self.tasks.transition(task_id, "failed", evidence={"failure_class": failure_class})
        except Exception:  # noqa: BLE001
            log.debug("targeted_build: fail transition raced for %s", task_id, exc_info=True)


__all__ = ["BuildLifecycleCollaborator"]
