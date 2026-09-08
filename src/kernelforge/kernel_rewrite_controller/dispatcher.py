# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dispatch one validated controller task to a named-kernel forge-loop."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

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
from kernelforge.kernel_rewrite_controller.publisher import PATCH_FILENAME
from kernelforge.kernel_rewrite_controller.recovery import recover_task_result
from kernelforge.kernel_rewrite_controller.state import TaskStateStore
from kernelforge.kernel_rewrite_controller.task import load_task
from kernelforge.kernel_rewrite_controller.worktree import (
    FORGE_LOOP_OUTPUT_DIRNAME,
    OperatorWorktree,
    create_operator_worktree,
    operator_workspace,
    release_operator_worktree,
    stage_operator_driver,
)

log = logging.getLogger(__name__)

#: Where forge-loop's task preparer writes its per-attempt record, relative to
#: the workspace it prepares.
_PREPARATION_AUDIT_DIRNAME = "task_preparation"


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


def _keep_preparation_audit(
    layout: ControllerLayout,
    task: KernelRewriteTask,
    worktree: OperatorWorktree | None,
) -> None:
    """Save the driver-preparation record before the workspace goes away.

    Preparation is where a task whose driver does not yet conform either
    becomes runnable or stops. Its record is the only account of which of those
    happened and why, and it is written inside the workspace, which the release
    below deletes. Best-effort: losing the copy must not turn a dispatch that
    otherwise succeeded into a failure.
    """
    if worktree is None:
        return
    source = worktree.workspace / FORGE_LOOP_OUTPUT_DIRNAME / _PREPARATION_AUDIT_DIRNAME
    if not source.is_dir():
        return
    destination = layout.preparation_audit_dir(task.operator_id)
    try:
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    except OSError as error:
        log.warning("could not keep the preparation audit for %s: %s", task.operator_id, error)


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
        workspace_dir=str(operator_workspace(task, layout)),
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
            driver=stage_operator_driver(task, task_path, worktree),
        )
        outcome = run_forge_loop(
            invocation,
            on_checkpoint=lambda: recover_task_result(
                layout,
                task_path,
                update_state=False,
            ),
        )
        recovered = recover_task_result(
            layout,
            task_path,
            update_state=False,
        )
        if recovered.patch_dir is not None:
            patch_path = recovered.patch_dir / PATCH_FILENAME
            state_store.transition(
                TASK_STATUS_SUCCEEDED,
                reason=(
                    "published from forge-loop checkpoint"
                    if outcome.timed_out
                    else "published from completed forge-loop"
                ),
                result_patch_dir=str(recovered.patch_dir),
            )
            return SingleTaskResult(
                task=task,
                worktree=worktree,
                forge_outcome=outcome,
                patch_path=patch_path,
                status=TASK_STATUS_SUCCEEDED,
            )
        forge_failure = _failure_detail(outcome)
        reason = (
            forge_failure
            if outcome.timed_out or outcome.returncode != 0 or outcome.result is None
            else recovered.reason or forge_failure
        )
        state_store.transition(TASK_STATUS_FAILED, reason=reason)
        return SingleTaskResult(
            task=task,
            worktree=worktree,
            forge_outcome=outcome,
            patch_path=None,
            status=TASK_STATUS_FAILED,
            reason=reason,
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
    finally:
        # After the recovery above, never before it: the patch is what the
        # campaign was for, and this returns the tree the patch was built in.
        # A private checkout is left standing instead, because the controller's
        # closing sweep still reads results out of it.
        if task is not None:
            _keep_preparation_audit(layout, task, worktree)
        release_operator_worktree(worktree)


__all__ = [
    "SingleTaskResult",
    "dispatch_single_task",
]
