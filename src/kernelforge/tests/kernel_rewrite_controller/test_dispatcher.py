# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from kernelforge.kernel_rewrite_controller import ControllerLayout, TaskStateStore
from kernelforge.kernel_rewrite_controller import dispatcher
from kernelforge.kernel_rewrite_controller.forge_runner import ForgeLoopOutcome
from kernelforge.kernel_rewrite_controller.worktree import OperatorWorktree


def _fake_worktree(task, layout: ControllerLayout) -> OperatorWorktree:
    workspace = layout.workspace_dir(task.operator_id)
    kernel = workspace / task.kernel_path
    kernel.parent.mkdir(parents=True)
    kernel.write_text("VALUE = 1\n", encoding="utf-8")
    return OperatorWorktree(
        repo_root=task.repo_root,
        workspace=workspace,
        branch="forge/controller/test",
        base_commit=task.base_commit,
        kernel_path=kernel,
        source_files=(kernel,),
    )


def test_dispatch_single_task_records_success_and_controller_base_patch(
    tmp_path: Path,
    task_dir: Path,
    monkeypatch,
) -> None:
    layout = ControllerLayout(tmp_path / "output")
    monkeypatch.setattr(dispatcher, "create_operator_worktree", _fake_worktree)
    monkeypatch.setattr(
        dispatcher,
        "run_forge_loop",
        lambda invocation: ForgeLoopOutcome(
            returncode=0,
            stdout="",
            stderr="",
            result={"improved": True, "best_commit": "b" * 40},
            timed_out=False,
            command=invocation.command,
        ),
    )
    monkeypatch.setattr(
        dispatcher,
        "export_patch_from_base",
        lambda worktree, best_commit: "diff --git a/sglang/kernels/fused_moe.py b/sglang/kernels/fused_moe.py\n",
    )

    result = dispatcher.dispatch_single_task(
        task_dir,
        layout=layout,
        deadline_unix=10**12,
        expected_base_commit="a" * 40,
    )

    assert result.status == "succeeded"
    assert result.patch_path == task_dir / "controller-result.patch"
    assert result.patch_path.is_file()
    state = TaskStateStore(task_dir).load()
    assert state is not None
    assert state.status == "succeeded"
    assert state.workspace_dir == str(layout.workspace_dir(result.task.operator_id))  # type: ignore[union-attr]


def test_dispatch_single_task_contains_forge_failure(
    tmp_path: Path,
    task_dir: Path,
    monkeypatch,
) -> None:
    layout = ControllerLayout(tmp_path / "output")
    monkeypatch.setattr(dispatcher, "create_operator_worktree", _fake_worktree)
    monkeypatch.setattr(
        dispatcher,
        "run_forge_loop",
        lambda invocation: ForgeLoopOutcome(
            returncode=2,
            stdout="",
            stderr="driver preparation failed",
            result=None,
            timed_out=False,
            command=invocation.command,
        ),
    )

    result = dispatcher.dispatch_single_task(
        task_dir,
        layout=layout,
        deadline_unix=10**12,
        expected_base_commit="a" * 40,
    )

    assert result.status == "failed"
    assert "driver preparation failed" in result.reason
    assert result.patch_path is None
    state = TaskStateStore(task_dir).load()
    assert state is not None
    assert state.status == "failed"


def test_dispatch_single_task_skips_an_invalid_task_without_starting_forge(
    tmp_path: Path,
    task_dir: Path,
    monkeypatch,
) -> None:
    (task_dir / "task.json").write_text("{", encoding="utf-8")
    monkeypatch.setattr(
        dispatcher,
        "run_forge_loop",
        lambda invocation: (_ for _ in ()).throw(AssertionError("forge-loop must not start")),
    )

    result = dispatcher.dispatch_single_task(
        task_dir,
        layout=ControllerLayout(tmp_path / "output"),
        deadline_unix=10**12,
    )

    assert result.status == "skipped"
    assert TaskStateStore(task_dir).load().status == "skipped"  # type: ignore[union-attr]
