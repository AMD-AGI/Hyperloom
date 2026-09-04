# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sequential prepared-task scheduling within the controller budget."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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

log = logging.getLogger(__name__)

ANALYSIS_BUDGET_SEC = 60 * 60
FORGE_LOOP_BUDGET_SEC = 90 * 60
MIN_TASK_START_REMAINING_SEC = 30 * 60


@dataclass(frozen=True)
class ScheduleResult:
    """Aggregate outcome of one pass over all published tasks."""

    task_count: int
    results: tuple[SingleTaskResult, ...]
    stopped_for_budget: bool = False
    #: Base commit this run pinned each source repository to, keyed by absolute
    #: repository root. Reported so the campaign's baselines are auditable
    #: without reading every task.
    repository_pins: dict[str, str] = field(default_factory=dict)

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
    log.warning("skipping operator task %s: %s", task_dir.name, reason)
    TaskStateStore(task_dir).mark_skipped(reason)
    return _skipped_result(task, reason)


def dispatch_prepared_tasks(
    layout: ControllerLayout,
    *,
    controller_deadline_unix: float,
    clock: Callable[[], float] = time.time,
    on_progress: Callable[[ScheduleResult], None] | None = None,
) -> ScheduleResult:
    """Validate and run all published tasks sequentially by priority.

    ``on_progress`` receives the schedule so far after every task reaches a
    terminal state, so a caller can persist what a campaign has accounted for
    before it is killed rather than only when it returns.
    """
    task_dirs = discover_task_dirs(layout)
    parsed_by_id: dict[str, tuple[Path, KernelRewriteTask]] = {}
    results: list[SingleTaskResult] = []

    for task_dir in task_dirs:
        parsed = load_task(task_dir, record_state=True)
        if parsed.task is None:
            results.append(_skipped_result(None, parsed.reason))
            continue
        incumbent = parsed_by_id.get(parsed.task.operator_id)
        if incumbent is None:
            parsed_by_id[parsed.task.operator_id] = (task_dir, parsed.task)
        elif parsed.task.priority < incumbent[1].priority:
            # The displaced incumbent needs the same skip record its rival would
            # have got. Without one it stays `ready` and produces no result row,
            # while ``task_count`` still counts its directory -- so the run would
            # report one more task than it accounts for.
            results.append(
                _skip_task(
                    incumbent[0],
                    incumbent[1],
                    f"superseded by a higher-priority task for operator {parsed.task.operator_id}",
                )
            )
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
    # One base commit per repository rather than one repository per campaign.
    # Every patch is a diff from its own repository's pinned commit and is
    # applied to that repository alone, so two independent repositories cannot
    # conflict; only a second base within one repository can. The pin comes from
    # the highest-priority task naming that repository, which makes it a function
    # of the agent's own ranking rather than of publication order.
    pinned_bases: dict[Path, str] = {}
    stopped_for_budget = False

    def _snapshot() -> ScheduleResult:
        return ScheduleResult(
            task_count=len(task_dirs),
            results=tuple(results),
            stopped_for_budget=stopped_for_budget,
            repository_pins=dict(sorted((str(repo), commit) for repo, commit in pinned_bases.items())),
        )

    def _report() -> None:
        if on_progress is None:
            return
        # Suppressed on purpose. The callback exists so a campaign's accounting
        # survives being killed, and its own work -- scanning the patch directory
        # and rewriting two files on a shared filesystem -- can fail on its own.
        # Letting that failure out would abort a campaign that is making progress
        # and may already hold published patches, which is the opposite of what
        # recording the accounting is for.
        with contextlib.suppress(Exception):
            on_progress(_snapshot())

    for index, task in enumerate(tasks):
        task_dir = task_dirs_by_id[task.operator_id]
        pinned_base = pinned_bases.setdefault(task.repo_root, task.base_commit)
        if task.base_commit != pinned_base:
            results.append(
                _skip_task(
                    task_dir,
                    task,
                    f"repository {task.repo_root} is pinned to base commit {pinned_base}",
                )
            )
            _report()
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
            _report()
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
                expected_base_commit=pinned_base,
            )
        )
        _report()

    return _snapshot()


__all__ = [
    "ANALYSIS_BUDGET_SEC",
    "FORGE_LOOP_BUDGET_SEC",
    "MIN_TASK_START_REMAINING_SEC",
    "ScheduleResult",
    "dispatch_prepared_tasks",
]
