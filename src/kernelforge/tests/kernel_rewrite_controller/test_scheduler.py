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


def test_progress_is_reported_after_every_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A campaign killed mid-schedule must have already reported what it did.

    The caller persists this, and the write it used to rely on is the terminal
    one -- the one a hard timeout never reaches.
    """
    layout = ControllerLayout(tmp_path / "output")
    repo = tmp_path / "repo"
    repo.mkdir()
    _publish_task(layout, repo_root=repo, kernel_name="first", priority=0)
    _publish_task(layout, repo_root=repo, kernel_name="second", priority=1)

    def _dispatch(task_dir, **_kwargs):
        task = scheduler.load_task(task_dir, record_state=False).task
        assert task is not None
        return _result(task, "succeeded")

    monkeypatch.setattr(scheduler, "dispatch_single_task", _dispatch)
    seen: list[tuple[int, dict[str, str]]] = []

    scheduler.dispatch_prepared_tasks(
        layout,
        controller_deadline_unix=20_000,
        clock=lambda: 10_000,
        on_progress=lambda partial: seen.append((len(partial.results), dict(partial.repository_pins))),
    )

    assert [count for count, _ in seen] == [1, 2]
    assert all(pins == {str(repo): "a" * 40} for _count, pins in seen)


def test_a_failing_progress_report_does_not_end_the_campaign(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Recording the accounting must not be able to abort what it records.

    The callback scans the patch directory and rewrites two files on a shared
    filesystem, and it now runs at every task boundary, so its own failure is a
    real event -- and letting it out would abandon patches already published.
    """
    layout = ControllerLayout(tmp_path / "output")
    repo = tmp_path / "repo"
    repo.mkdir()
    _publish_task(layout, repo_root=repo, kernel_name="first", priority=0)
    _publish_task(layout, repo_root=repo, kernel_name="second", priority=1)

    def _dispatch(task_dir, **_kwargs):
        task = scheduler.load_task(task_dir, record_state=False).task
        assert task is not None
        return _result(task, "succeeded")

    def _explode(_partial) -> None:
        raise OSError("stale NFS file handle")

    monkeypatch.setattr(scheduler, "dispatch_single_task", _dispatch)

    result = scheduler.dispatch_prepared_tasks(
        layout,
        controller_deadline_unix=20_000,
        clock=lambda: 10_000,
        on_progress=_explode,
    )

    assert result.succeeded_count == 2
    assert result.task_count == 2


def test_a_superseded_duplicate_is_recorded_rather_than_dropped(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The loser of a dedup needs a skip record whichever side it is on.

    Without one it stays ``ready`` and never appears in results, while
    ``task_count`` still counts its directory.
    """
    layout = ControllerLayout(tmp_path / "output")
    repo = tmp_path / "repo"
    repo.mkdir()
    # Same identity, two directories: reachable only by writing the second one
    # under a name of its own, which is what a future non-identity layout would do.
    loser = _publish_task(layout, repo_root=repo, kernel_name="only", priority=5)
    # Sorted after the encoded identity directory, so the better priority is the
    # one that arrives second and displaces an incumbent.
    winner_dir = layout.tasks_root / "zz-better-priority"
    winner_dir.mkdir()
    for name in ("task.json", "driver.py"):
        (winner_dir / name).write_text((loser / name).read_text(encoding="utf-8"), encoding="utf-8")
    payload = json.loads((winner_dir / "task.json").read_text(encoding="utf-8"))
    payload["priority"] = 0
    (winner_dir / "task.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        scheduler,
        "dispatch_single_task",
        lambda task_dir, **_kwargs: _result(scheduler.load_task(task_dir, record_state=False).task, "succeeded"),
    )
    monkeypatch.setattr(scheduler, "load_task", _load_without_directory_identity)

    result = scheduler.dispatch_prepared_tasks(
        layout,
        controller_deadline_unix=20_000,
        clock=lambda: 10_000,
    )

    assert result.task_count == len(result.results)
    assert result.skipped_count == 1
    assert TaskStateStore(loser).load().status == "skipped"


def _load_without_directory_identity(task_dir, **kwargs):
    """Let a second directory hold the same identity, which the layout forbids.

    The production layout derives a task's directory name from its identity, so
    two directories cannot collide today. The dedup branch still has to record
    its loser, because that invariant lives in a different module.
    """
    from kernelforge.kernel_rewrite_controller import task as task_module

    payload = json.loads((Path(task_dir) / "task.json").read_text(encoding="utf-8"))
    parsed = task_module.parse_task_payload(
        payload,
        task_dir=task_dir,
        expected_base_commit=kwargs.get("expected_base_commit"),
        enforce_directory_identity=False,
    )
    store = TaskStateStore(task_dir)
    if kwargs.get("record_state", True) and store.load() is None:
        store.initialize_ready()
    return task_module.TaskParseResult(task=parsed)


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


def test_tasks_from_separate_repositories_each_get_their_own_base(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = ControllerLayout(tmp_path / "output")
    sglang = tmp_path / "sglang"
    aiter = tmp_path / "aiter"
    sglang.mkdir()
    aiter.mkdir()
    _publish_task(layout, repo_root=aiter, kernel_name="moe_stage1", priority=0, base_commit="a" * 40)
    _publish_task(layout, repo_root=sglang, kernel_name="rmsnorm", priority=1, base_commit="c" * 40)
    calls: list[str] = []

    def _dispatch(task_dir, **kwargs):
        task = scheduler.load_task(task_dir, record_state=False).task
        assert task is not None
        # Each task must be validated against its own repository's pin, not the
        # top-priority task's, or the second repository could never run.
        assert kwargs["expected_base_commit"] == task.base_commit
        calls.append(task.identity.kernel_name)
        return _result(task, "succeeded")

    monkeypatch.setattr(scheduler, "dispatch_single_task", _dispatch)

    result = scheduler.dispatch_prepared_tasks(
        layout,
        controller_deadline_unix=20_000,
        clock=lambda: 10_000,
    )

    assert calls == ["moe_stage1", "rmsnorm"]
    assert result.succeeded_count == 2
    assert result.skipped_count == 0
    assert result.repository_pins == {str(aiter): "a" * 40, str(sglang): "c" * 40}


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
