# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A task's driver has to run from inside the repository it measures.

The published task directory is not in the repository. A driver run from there
is external to the workspace, and forge-loop then hands its preparation agent a
staging directory instead of the workspace -- which the agent's safety guard
refuses, because it is not a Git checkout. Every task whose driver needs repair
dies there, and a collective task always needs repair: the analyst is told to
hand over a single-process driver and leave the ranks to the preparer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelforge.kernel_rewrite_controller.paths import ControllerLayout
from kernelforge.kernel_rewrite_controller.task import parse_task_payload
from kernelforge.kernel_rewrite_controller.worktree import (
    DRIVER_STAGE_PREFIX,
    OperatorWorktree,
    stage_operator_driver,
)
from kernelforge.loop.path_ownership import is_producer_owned_path

BASE_COMMIT = "a" * 40


@pytest.fixture
def staged(tmp_path: Path, task_dir: Path):
    """One task's driver, copied into a workspace standing in for the repo."""
    task = parse_task_payload(
        json.loads((task_dir / "task.json").read_text(encoding="utf-8")),
        task_dir=task_dir,
        expected_base_commit=BASE_COMMIT,
    )
    workspace = tmp_path / "workspace"
    kernel = workspace / task.kernel_path
    kernel.parent.mkdir(parents=True)
    kernel.write_text("VALUE = 1\n", encoding="utf-8")
    worktree = OperatorWorktree(
        repo_root=workspace,
        workspace=workspace,
        branch="forge/controller/test",
        base_commit=BASE_COMMIT,
        kernel_path=kernel,
        source_files=(kernel,),
    )
    return task, workspace, stage_operator_driver(task, task_dir, worktree)


def test_the_driver_runs_from_inside_the_workspace(staged) -> None:
    """External to the workspace is what stops the preparation agent starting."""
    _task, workspace, driver = staged

    assert driver.is_file()
    assert driver.resolve().is_relative_to(workspace.resolve())


def test_the_repository_root_is_one_directory_up(staged) -> None:
    """The rule the analyst prompt states, so a driver can find its own tree."""
    _task, workspace, driver = staged

    assert driver.resolve().parents[1] == workspace.resolve()


def test_the_copy_is_producer_state(staged) -> None:
    """What keeps it out of the exported patch and gets it cleaned up."""
    _task, workspace, driver = staged

    relative = driver.resolve().relative_to(workspace.resolve()).as_posix()
    assert relative.startswith(DRIVER_STAGE_PREFIX)
    assert is_producer_owned_path(relative)


def test_the_staging_directory_is_hidden_from_git(staged) -> None:
    """forge-loop rejects untracked paths its caller never declared.

    The helper modules the preparation agent writes beside the driver land here
    too, so the rule has to cover the directory rather than the one file.
    """
    _task, _workspace, driver = staged

    assert (driver.parent / ".gitignore").read_text(encoding="utf-8") == "*\n"


def test_the_published_task_file_is_left_alone(staged, task_dir: Path) -> None:
    """The task directory keeps the analyst's deliverable; the copy is the working one."""
    task, _workspace, driver = staged

    published = task_dir / task.driver_path
    assert published.is_file()
    assert published.read_text(encoding="utf-8") == driver.read_text(encoding="utf-8")
    assert published.resolve() != driver.resolve()


def test_two_operators_do_not_share_a_staging_directory(staged, tmp_path: Path, task_dir: Path) -> None:
    """One directory per operator, so concurrent tasks cannot overwrite each other."""
    task, workspace, driver = staged
    other = parse_task_payload(
        {
            **json.loads((task_dir / "task.json").read_text(encoding="utf-8")),
            "operator_name": "a_different_operator",
            "identity": {
                **json.loads((task_dir / "task.json").read_text(encoding="utf-8"))["identity"],
                "kernel_name": "a_different_operator",
            },
        },
        task_dir=task_dir,
        expected_base_commit=BASE_COMMIT,
        enforce_directory_identity=False,
    )
    worktree = OperatorWorktree(
        repo_root=workspace,
        workspace=workspace,
        branch="forge/controller/other",
        base_commit=BASE_COMMIT,
        kernel_path=workspace / task.kernel_path,
        source_files=(),
    )

    assert stage_operator_driver(other, task_dir, worktree).parent != driver.parent


def test_the_staging_directory_does_not_survive_the_borrow(staged, tmp_path: Path) -> None:
    """A borrowed repository is handed back as it was found.

    The driver copy cannot be found by the untracked scan that removes the rest
    of the campaign's leavings: that scan lists neither directories nor the
    ignored files inside one, and everything here is ignored by design.
    """
    from kernelforge.kernel_rewrite_controller import worktree as worktree_module

    _task, workspace, driver = staged
    assert driver.is_file()

    worktree_module._remove_producer_untracked(workspace)

    assert not driver.parent.exists()


def test_a_driver_left_by_a_killed_run_is_reclaimed(staged) -> None:
    """Cleanup is by prefix, so the next borrower does not inherit the last one's."""
    from kernelforge.kernel_rewrite_controller import worktree as worktree_module

    _task, workspace, _driver = staged
    abandoned = workspace / f"{DRIVER_STAGE_PREFIX}0123456789abcdef"
    abandoned.mkdir()
    (abandoned / "driver.py").write_text("stale\n", encoding="utf-8")

    worktree_module._remove_producer_untracked(workspace)

    assert not abandoned.exists()


def test_the_preparation_audit_is_kept_outside_the_workspace(tmp_path: Path) -> None:
    """Its home is deleted with the borrowed tree, and it is the only account
    of why a task that could not be prepared stopped."""
    layout = ControllerLayout(tmp_path / "out")

    audit = layout.preparation_audit_dir("kernel:forge-loop:k:aiter:0.1.0:aiter:mi355x")

    assert not audit.resolve().is_relative_to((tmp_path / "workspace").resolve())
    assert audit.resolve().is_relative_to(layout.controller_root.resolve())
