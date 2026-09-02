# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Durable per-task state for the kernel rewrite controller."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from kernelforge.durable_io import atomic_write_text
from kernelforge.kernel_rewrite_controller.contracts import (
    TASK_STATUS_FAILED,
    TASK_STATUS_READY,
    TASK_STATUS_RUNNING,
    TASK_STATUS_SKIPPED,
    TASK_STATUS_SUCCEEDED,
    TERMINAL_TASK_STATUSES,
    TaskState,
    TaskStateError,
)
from kernelforge.kernel_rewrite_controller.paths import TaskLayout

_ALLOWED_TRANSITIONS = {
    TASK_STATUS_READY: frozenset({TASK_STATUS_RUNNING, TASK_STATUS_SKIPPED}),
    TASK_STATUS_RUNNING: frozenset(
        {
            TASK_STATUS_SUCCEEDED,
            TASK_STATUS_FAILED,
            TASK_STATUS_SKIPPED,
        }
    ),
    TASK_STATUS_SUCCEEDED: frozenset(),
    TASK_STATUS_FAILED: frozenset(),
    TASK_STATUS_SKIPPED: frozenset(),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStateStore:
    """Single-writer store for one task's ``state.json``."""

    def __init__(self, task_dir: str | Path) -> None:
        self.layout = TaskLayout(Path(task_dir))

    @property
    def path(self) -> Path:
        return self.layout.state_json

    def load(self) -> TaskState | None:
        """Read the current state, returning ``None`` before the first write."""
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TaskStateError(f"could not read task state {self.path}: {error}") from error
        return TaskState.from_dict(payload)

    def save(self, state: TaskState) -> None:
        """Atomically persist a validated task state."""
        self.layout.root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.path,
            json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n",
        )

    def initialize_ready(self) -> TaskState:
        """Create the initial ready state for a valid task."""
        current = self.load()
        if current is not None:
            raise TaskStateError(f"task state already exists with status {current.status!r}")
        state = TaskState(status=TASK_STATUS_READY)
        self.save(state)
        return state

    def mark_skipped(self, reason: str) -> TaskState:
        """Persist a terminal skipped state for an invalid or declined task."""
        current = self.load()
        if current is not None and current.status in TERMINAL_TASK_STATUSES:
            return current
        if current is not None and TASK_STATUS_SKIPPED not in _ALLOWED_TRANSITIONS[current.status]:
            raise TaskStateError(f"cannot transition task from {current.status!r} to {TASK_STATUS_SKIPPED!r}")
        state = TaskState(
            status=TASK_STATUS_SKIPPED,
            reason=str(reason or ""),
            started_at=current.started_at if current is not None else "",
            finished_at=_now_iso(),
            workspace_dir=current.workspace_dir if current is not None else "",
            result_patch_dir=current.result_patch_dir if current is not None else "",
        )
        self.save(state)
        return state

    def transition(
        self,
        status: str,
        *,
        reason: str = "",
        workspace_dir: str | None = None,
        result_patch_dir: str | None = None,
    ) -> TaskState:
        """Apply one valid lifecycle transition and persist it atomically."""
        current = self.load()
        if current is None:
            raise TaskStateError("task state must be initialized before transition")
        if status not in _ALLOWED_TRANSITIONS[current.status]:
            raise TaskStateError(f"cannot transition task from {current.status!r} to {status!r}")

        started_at = current.started_at
        if status == TASK_STATUS_RUNNING and not started_at:
            started_at = _now_iso()
        finished_at = _now_iso() if status in TERMINAL_TASK_STATUSES else ""
        state = TaskState(
            status=status,
            reason=str(reason or ""),
            started_at=started_at,
            finished_at=finished_at,
            workspace_dir=current.workspace_dir if workspace_dir is None else str(workspace_dir),
            result_patch_dir=(current.result_patch_dir if result_patch_dir is None else str(result_patch_dir)),
        )
        self.save(state)
        return state

    def mark_recovered_success(
        self,
        *,
        result_patch_dir: str,
        reason: str,
    ) -> TaskState:
        """Publish a recovered success even when the interrupted state was terminal."""
        current = self.load()
        state = TaskState(
            status=TASK_STATUS_SUCCEEDED,
            reason=str(reason or ""),
            started_at=current.started_at if current is not None else "",
            finished_at=_now_iso(),
            workspace_dir=current.workspace_dir if current is not None else "",
            result_patch_dir=str(result_patch_dir),
        )
        self.save(state)
        return state


__all__ = ["TaskStateStore"]
