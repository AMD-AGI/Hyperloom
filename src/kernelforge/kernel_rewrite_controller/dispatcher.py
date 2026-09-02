# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dispatch one validated controller task to a named-kernel forge-loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kernelforge.durable_io import atomic_write_text
from kernelforge.kernel_rewrite_controller.contracts import (
    TASK_STATUS_FAILED,
    TASK_STATUS_RUNNING,
    TASK_STATUS_SKIPPED,
    TASK_STATUS_SUCCEEDED,
    KernelRewriteTask,
)
from kernelforge.kernel_rewrite_controller.forge_runner import (
    ForgeLoopOutcome,
    build_forge_loop_invocation,
    run_forge_loop,
)
from kernelforge.kernel_rewrite_controller.paths import ControllerLayout
from kernelforge.kernel_rewrite_controller.state import TaskStateStore
from kernelforge.kernel_rewrite_controller.task import load_task
from kernelforge.kernel_rewrite_controller.worktree import (
    OperatorWorktree,
    create_operator_worktree,
    export_patch_from_base,
)

CONTROLLER_PATCH_FILENAME = "controller-result.patch"


@dataclass(frozen=True)
class SingleTaskResult:
    """Outcome of one controller task dispatch."""

    task: KernelRewriteTask | None
    worktree: OperatorWorktree | None
    forge_outcome: ForgeLoopOutcome | None
    patch_path: Path | None
    status: str
    reason: str = ""


def _failure_detail(outcome: ForgeLoopOutcome) -> str:
    if outcome.timed_out:
        return "forge-loop timed out"
    if outcome.returncode != 0:
        detail = (outcome.stderr or outcome.stdout).strip()
        return f"forge-loop exited {outcome.returncode}: {detail[-1000:]}"
    if outcome.result is None:
        return "forge-loop emitted no structured result"
    if not outcome.best_commit:
        return "forge-loop produced no best commit"
    return "forge-loop produced no validated improvement"


def dispatch_single_task(
    task_dir: str | Path,
    *,
    layout: ControllerLayout,
    deadline_unix: float,
    expected_base_commit: str | None = None,
) -> SingleTaskResult:
    """Validate and run one task without affecting sibling task failures."""
    task_path = Path(task_dir).expanduser().resolve()
    parsed = load_task(
        task_path,
        expected_base_commit=expected_base_commit,
        record_state=True,
    )
    if parsed.task is None:
        return SingleTaskResult(
            task=None,
            worktree=None,
            forge_outcome=None,
            patch_path=None,
            status=TASK_STATUS_SKIPPED,
            reason=parsed.reason,
        )

    task = parsed.task
    state_store = TaskStateStore(task_path)
    state_store.transition(
        TASK_STATUS_RUNNING,
        workspace_dir=str(layout.workspace_dir(task.operator_id)),
    )
    worktree: OperatorWorktree | None = None
    outcome: ForgeLoopOutcome | None = None
    try:
        worktree = create_operator_worktree(task, layout)
        invocation = build_forge_loop_invocation(
            task,
            task_dir=task_path,
            worktree=worktree,
            deadline_unix=deadline_unix,
        )
        outcome = run_forge_loop(invocation)
        if not outcome.improved:
            reason = _failure_detail(outcome)
            state_store.transition(TASK_STATUS_FAILED, reason=reason)
            return SingleTaskResult(
                task=task,
                worktree=worktree,
                forge_outcome=outcome,
                patch_path=None,
                status=TASK_STATUS_FAILED,
                reason=reason,
            )
        patch = export_patch_from_base(
            worktree,
            best_commit=outcome.best_commit,
        )
        if not patch.strip():
            reason = "forge-loop best commit has no changes from the controller base"
            state_store.transition(TASK_STATUS_FAILED, reason=reason)
            return SingleTaskResult(
                task=task,
                worktree=worktree,
                forge_outcome=outcome,
                patch_path=None,
                status=TASK_STATUS_FAILED,
                reason=reason,
            )
        patch_path = task_path / CONTROLLER_PATCH_FILENAME
        atomic_write_text(patch_path, patch)
        state_store.transition(
            TASK_STATUS_SUCCEEDED,
            result_patch_dir=str(task_path),
        )
        return SingleTaskResult(
            task=task,
            worktree=worktree,
            forge_outcome=outcome,
            patch_path=patch_path,
            status=TASK_STATUS_SUCCEEDED,
        )
    except Exception as error:
        reason = f"single-task dispatch failed: {error}"
        state_store.transition(TASK_STATUS_FAILED, reason=reason)
        return SingleTaskResult(
            task=task,
            worktree=worktree,
            forge_outcome=outcome,
            patch_path=None,
            status=TASK_STATUS_FAILED,
            reason=reason,
        )


__all__ = [
    "CONTROLLER_PATCH_FILENAME",
    "SingleTaskResult",
    "dispatch_single_task",
]
