# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sequential prepared-task scheduling within the controller budget."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kernelforge.kernel_rewrite_controller.contracts import (
    TASK_STATUS_FAILED,
    TASK_STATUS_SKIPPED,
    TASK_STATUS_SUCCEEDED,
    KernelRewriteTask,
)
from kernelforge.kernel_rewrite_controller.dispatcher import (
    SingleTaskResult,
    dispatch_single_task,
)
from kernelforge.kernel_rewrite_controller.paths import ControllerLayout
from kernelforge.kernel_rewrite_controller.state import TaskStateStore
from kernelforge.kernel_rewrite_controller.task import (
    discover_task_dirs,
    load_task,
    sort_tasks,
)

ANALYSIS_BUDGET_SEC = 60 * 60
FORGE_LOOP_BUDGET_SEC = 90 * 60
MIN_TASK_START_REMAINING_SEC = 30 * 60


@dataclass(frozen=True)
class ScheduleResult:
    """Aggregate outcome of one pass over all published tasks."""

    task_count: int
    results: tuple[SingleTaskResult, ...]
    stopped_for_budget: bool = False

    @property
    def succeeded_count(self) -> int:
        return sum(result.status == TASK_STATUS_SUCCEEDED for result in self.results)

    @property
    def failed_count(self) -> int:
        return sum(result.status == TASK_STATUS_FAILED for result in self.results)

    @property
    def skipped_count(self) -> int:
        return sum(result.status == TASK_STATUS_SKIPPED for result in self.results)


def _skipped_result(task, reason: str) -> SingleTaskResult:
    return SingleTaskResult(
        task=task,
        worktree=None,
        forge_outcome=None,
        patch_path=None,
        status=TASK_STATUS_SKIPPED,
        reason=reason,
    )


def _skip_task(task_dir: Path, task, reason: str) -> SingleTaskResult:
    TaskStateStore(task_dir).mark_skipped(reason)
    return _skipped_result(task, reason)


def dispatch_prepared_tasks(
    layout: ControllerLayout,
    *,
    controller_deadline_unix: float,
    clock: Callable[[], float] = time.time,
) -> ScheduleResult:
    """Validate and run all published tasks sequentially by priority."""
    task_dirs = discover_task_dirs(layout)
    parsed_by_id: dict[str, tuple[Path, KernelRewriteTask]] = {}
    results: list[SingleTaskResult] = []

    for task_dir in task_dirs:
        parsed = load_task(task_dir, record_state=True)
        if parsed.task is None:
            results.append(_skipped_result(None, parsed.reason))
            continue
        incumbent = parsed_by_id.get(parsed.task.operator_id)
        if incumbent is None or parsed.task.priority < incumbent[1].priority:
            parsed_by_id[parsed.task.operator_id] = (task_dir, parsed.task)
        else:
            results.append(
                _skip_task(
                    task_dir,
                    parsed.task,
                    f"duplicate operator identity: {parsed.task.operator_id}",
                )
            )

    tasks = sort_tasks([entry[1] for entry in parsed_by_id.values()])
    if not tasks:
        return ScheduleResult(
            task_count=len(task_dirs),
            results=tuple(results),
        )

    task_dirs_by_id = {task.operator_id: task_dir for task_dir, task in parsed_by_id.values()}
    shared_base_commit = tasks[0].base_commit
    shared_repo_root = tasks[0].repo_root
    stopped_for_budget = False

    for index, task in enumerate(tasks):
        task_dir = task_dirs_by_id[task.operator_id]
        if task.base_commit != shared_base_commit or task.repo_root != shared_repo_root:
            results.append(
                _skip_task(
                    task_dir,
                    task,
                    "task does not share the controller base commit and repository",
                )
            )
            continue

        now = float(clock())
        remaining = float(controller_deadline_unix) - now
        if remaining < MIN_TASK_START_REMAINING_SEC:
            stopped_for_budget = True
            for pending in tasks[index:]:
                pending_dir = task_dirs_by_id[pending.operator_id]
                results.append(
                    _skip_task(
                        pending_dir,
                        pending,
                        (
                            "insufficient controller time remaining: "
                            f"{max(0.0, remaining):.3f}s < {MIN_TASK_START_REMAINING_SEC}s"
                        ),
                    )
                )
            break

        task_deadline = min(
            float(controller_deadline_unix),
            now + FORGE_LOOP_BUDGET_SEC,
        )
        results.append(
            dispatch_single_task(
                task_dir,
                layout=layout,
                deadline_unix=task_deadline,
                expected_base_commit=shared_base_commit,
            )
        )

    return ScheduleResult(
        task_count=len(task_dirs),
        results=tuple(results),
        stopped_for_budget=stopped_for_budget,
    )


__all__ = [
    "ANALYSIS_BUDGET_SEC",
    "FORGE_LOOP_BUDGET_SEC",
    "MIN_TASK_START_REMAINING_SEC",
    "ScheduleResult",
    "dispatch_prepared_tasks",
]
