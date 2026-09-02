# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

from kernelforge.kernel_rewrite_controller import ControllerLayout, TaskStateStore
from kernelforge.kernel_rewrite_controller import scheduler
from kernelforge.kernel_rewrite_controller.dispatcher import SingleTaskResult
from kernelforge.knowledge.kernel_identity import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)


def _publish_task(
    layout: ControllerLayout,
    *,
    repo_root: Path,
    kernel_name: str,
    priority: int,
    base_commit: str = "a" * 40,
) -> Path:
    identity = {
        "producer": "forge-loop",
        "kernel_name": kernel_name,
        "framework": "sglang",
        "framework_version": "0.5.0",
        "backend": "triton",
        "gpu": "mi355x",
    }
    operator_id = kernel_recipe_canonical_id(KernelRecipeIdentity.from_mapping(identity))
    task_dir = layout.task_dir(operator_id)
    task_dir.mkdir(parents=True)
    (task_dir / "driver.py").write_text("print('SNR: 100 dB')\n", encoding="utf-8")
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": identity,
                "base_commit": base_commit,
                "repo_root": str(repo_root),
                "kernel_path": f"sglang/kernels/{kernel_name}.py",
                "operator_name": kernel_name,
                "driver_path": "driver.py",
                "source_files": [],
                "target_functions": [kernel_name],
                "shape_cases": [],
                "priority": priority,
                "reason": "",
                "evidence": [],
            }
        ),
        encoding="utf-8",
    )
    return task_dir


def _result(task, status: str) -> SingleTaskResult:
    return SingleTaskResult(
        task=task,
        worktree=None,
        forge_outcome=None,
        patch_path=None,
        status=status,
        reason="",
    )


def test_fixed_scheduler_budgets_match_the_design() -> None:
    assert scheduler.ANALYSIS_BUDGET_SEC == 60 * 60
    assert scheduler.FORGE_LOOP_BUDGET_SEC == 90 * 60
    assert scheduler.MIN_TASK_START_REMAINING_SEC == 30 * 60


def test_tasks_run_sequentially_by_priority_and_continue_after_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = ControllerLayout(tmp_path / "output")
    repo = tmp_path / "repo"
    repo.mkdir()
    _publish_task(layout, repo_root=repo, kernel_name="third", priority=2)
    _publish_task(layout, repo_root=repo, kernel_name="first", priority=0)
    _publish_task(layout, repo_root=repo, kernel_name="second", priority=1)
    calls: list[tuple[str, float]] = []

    def _dispatch(task_dir, *, deadline_unix, **_kwargs):
        task = scheduler.load_task(task_dir, record_state=False).task
        assert task is not None
        calls.append((task.identity.kernel_name, deadline_unix))
        return _result(task, "failed" if task.identity.kernel_name == "second" else "succeeded")

    monkeypatch.setattr(scheduler, "dispatch_single_task", _dispatch)

    result = scheduler.dispatch_prepared_tasks(
        layout,
        controller_deadline_unix=20_000,
        clock=lambda: 10_000,
    )

    assert [name for name, _ in calls] == ["first", "second", "third"]
    assert all(deadline == 10_000 + 90 * 60 for _, deadline in calls)
    assert result.task_count == 3
    assert result.succeeded_count == 2
    assert result.failed_count == 1
    assert result.stopped_for_budget is False


def test_task_deadline_is_capped_by_controller_remaining_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = ControllerLayout(tmp_path / "output")
    repo = tmp_path / "repo"
    repo.mkdir()
    _publish_task(layout, repo_root=repo, kernel_name="only", priority=0)
    deadlines: list[float] = []

    def _dispatch(task_dir, *, deadline_unix, **_kwargs):
        deadlines.append(deadline_unix)
        task = scheduler.load_task(task_dir, record_state=False).task
        assert task is not None
        return _result(task, "succeeded")

    monkeypatch.setattr(scheduler, "dispatch_single_task", _dispatch)

    scheduler.dispatch_prepared_tasks(
        layout,
        controller_deadline_unix=13_600,
        clock=lambda: 10_000,
    )

    assert deadlines == [13_600]


def test_less_than_thirty_minutes_skips_all_remaining_tasks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = ControllerLayout(tmp_path / "output")
    repo = tmp_path / "repo"
    repo.mkdir()
    task_dirs = [
        _publish_task(layout, repo_root=repo, kernel_name="first", priority=0),
        _publish_task(layout, repo_root=repo, kernel_name="second", priority=1),
    ]
    monkeypatch.setattr(
        scheduler,
        "dispatch_single_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("task must not start")),
    )

    result = scheduler.dispatch_prepared_tasks(
        layout,
        controller_deadline_unix=11_799,
        clock=lambda: 10_000,
    )

    assert result.stopped_for_budget is True
    assert result.skipped_count == 2
    assert [TaskStateStore(path).load().status for path in task_dirs] == ["skipped", "skipped"]  # type: ignore[union-attr]


def test_a_different_shared_base_is_skipped_without_blocking_siblings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = ControllerLayout(tmp_path / "output")
    repo = tmp_path / "repo"
    repo.mkdir()
    _publish_task(layout, repo_root=repo, kernel_name="first", priority=0)
    mismatched = _publish_task(
        layout,
        repo_root=repo,
        kernel_name="second",
        priority=1,
        base_commit="b" * 40,
    )
    calls: list[str] = []

    def _dispatch(task_dir, **_kwargs):
        task = scheduler.load_task(task_dir, record_state=False).task
        assert task is not None
        calls.append(task.identity.kernel_name)
        return _result(task, "succeeded")

    monkeypatch.setattr(scheduler, "dispatch_single_task", _dispatch)

    result = scheduler.dispatch_prepared_tasks(
        layout,
        controller_deadline_unix=20_000,
        clock=lambda: 10_000,
    )

    assert calls == ["first"]
    assert result.succeeded_count == 1
    assert result.skipped_count == 1
    assert TaskStateStore(mismatched).load().status == "skipped"  # type: ignore[union-attr]
