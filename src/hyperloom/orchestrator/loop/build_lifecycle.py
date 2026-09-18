# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enqueues ``targeted_build`` rows; execution is handled by TargetedBuildExecutor."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys as _sys
import uuid

from hyperloom.inference_optimizer.session.session_paths import enablement_builds_dir

from ..enablement.runtime.build_actions import TargetedBuildAction, build_novelty_key

_BUILD_KIND = "targeted_build"
_LEASE_GRACE_SEC = 300  # added to build budget for the lease TTL reclaim backstop


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

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_coord"), name)

    async def enqueue_targeted_build(self, action: TargetedBuildAction) -> str:
        """Enqueue a ``targeted_build`` row (idempotent by novelty key)."""
        from ..enablement.runtime.targeted_build import _resolve_budget_sec

        # The default attempt_root derives from the task_id, so the id is minted
        # here rather than by the insert: the params must carry the path they name.
        task_id = uuid.uuid4().hex
        if not action.attempt_root:
            action = dataclasses.replace(action, attempt_root=str(enablement_builds_dir(self.session_dir, task_id)))
        ttl = int(_resolve_budget_sec(action)) + _LEASE_GRACE_SEC
        task, _existing = await self.tasks.create_or_return_existing(
            kind=_BUILD_KIND,
            params=action.to_state(),
            idempotency_key=_novelty_idempotency_key(action),
            requires_lanes=["build_lane"],
            lease_ttl_sec=ttl,
            task_id=task_id,
        )
        return str(getattr(task, "task_id", "") or "")


def _driver_command(action: TargetedBuildAction, attempt_root: str) -> list[str]:
    """Return the spawn argv for this action."""
    if action.build_command:
        return list(action.build_command)
    import hyperloom.orchestrator.enablement.runtime.targeted_build as _tb_mod
    from pathlib import Path as _Path

    root = _Path(str(attempt_root))
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(json.dumps(action.to_state()), encoding="utf-8")
    return [
        _sys.executable,
        "-m",
        _tb_mod.__name__,
        "--attempt-root",
        str(root),
    ]


__all__ = ["BuildLifecycleCollaborator", "_driver_command", "_novelty_idempotency_key"]
