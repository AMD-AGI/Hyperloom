# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pinned Git worktrees for controller-owned operator campaigns."""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from kernelforge.kernel_rewrite_controller.contracts import KernelRewriteTask
from kernelforge.kernel_rewrite_controller.paths import ControllerLayout
from kernelforge.llm.git import GitError, git


class WorktreeError(RuntimeError):
    """An operator worktree could not be created or validated."""


@dataclass(frozen=True)
class OperatorWorktree:
    """One task's isolated checkout pinned to the shared base commit."""

    repo_root: Path
    workspace: Path
    branch: str
    base_commit: str
    kernel_path: Path
    source_files: tuple[Path, ...]


def _git_toplevel(repo_root: Path) -> Path:
    try:
        result = git("rev-parse", "--show-toplevel", cwd=repo_root)
    except GitError as error:
        raise WorktreeError(f"repo_root is not a Git checkout: {repo_root}: {error}") from error
    return Path(result.stdout.strip()).resolve()


def _require_commit(repo_root: Path, commit: str) -> None:
    result = git("cat-file", "-e", f"{commit}^{{commit}}", cwd=repo_root, check=False)
    if result.returncode != 0:
        raise WorktreeError(f"base commit does not exist in {repo_root}: {commit}")


def _branch_name(operator_id: str) -> str:
    digest = hashlib.sha256(operator_id.encode("utf-8")).hexdigest()[:16]
    return f"forge/controller/{digest}-{uuid.uuid4().hex[:8]}"


def _remove_partial_worktree(repo_root: Path, workspace: Path, branch: str) -> None:
    with contextlib.suppress(Exception):
        git("worktree", "remove", "--force", str(workspace), cwd=repo_root, check=False)
    shutil.rmtree(workspace, ignore_errors=True)
    with contextlib.suppress(Exception):
        git("branch", "-D", branch, cwd=repo_root, check=False)


def create_operator_worktree(
    task: KernelRewriteTask,
    layout: ControllerLayout,
) -> OperatorWorktree:
    """Create one fresh branch/worktree from the task's pinned base commit."""
    repo_root = task.repo_root.resolve()
    if _git_toplevel(repo_root) != repo_root:
        raise WorktreeError(f"repo_root must be the Git top-level directory: {repo_root}")
    _require_commit(repo_root, task.base_commit)

    workspace = layout.workspace_dir(task.operator_id)
    if workspace.exists():
        raise WorktreeError(f"operator workspace already exists and cannot be resumed: {workspace}")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    branch = _branch_name(task.operator_id)
    try:
        git("worktree", "prune", cwd=repo_root, check=False)
        git(
            "worktree",
            "add",
            "-b",
            branch,
            str(workspace),
            task.base_commit,
            cwd=repo_root,
        )
        actual_head = git("rev-parse", "HEAD", cwd=workspace).stdout.strip().lower()
        if actual_head != task.base_commit:
            raise WorktreeError(f"worktree HEAD mismatch: created {actual_head}, expected {task.base_commit}")
        kernel_path = (workspace / task.kernel_path).resolve()
        if not kernel_path.is_relative_to(workspace) or not kernel_path.is_file():
            raise WorktreeError(f"kernel path is not a file in the base commit: {task.kernel_path}")
        source_files = tuple((workspace / relative).resolve() for relative in task.source_files)
        for source_file in source_files:
            if not source_file.is_relative_to(workspace) or not source_file.is_file():
                raise WorktreeError(f"source file is not a file in the base commit: {source_file}")
        return OperatorWorktree(
            repo_root=repo_root,
            workspace=workspace,
            branch=branch,
            base_commit=task.base_commit,
            kernel_path=kernel_path,
            source_files=source_files,
        )
    except Exception:
        _remove_partial_worktree(repo_root, workspace, branch)
        raise


def export_patch_from_base(
    worktree: OperatorWorktree,
    *,
    best_commit: str,
) -> str:
    """Export the full binary diff from the controller base to a Forge KEEP."""
    best = str(best_commit or "").strip().lower()
    if not best:
        raise WorktreeError("forge-loop returned no best commit")
    _require_commit(worktree.workspace, best)
    ancestor = git(
        "merge-base",
        "--is-ancestor",
        worktree.base_commit,
        best,
        cwd=worktree.workspace,
        check=False,
    )
    if ancestor.returncode != 0:
        raise WorktreeError(f"best commit {best} is not based on controller base {worktree.base_commit}")
    return str(
        git(
            "diff",
            "--binary",
            f"{worktree.base_commit}..{best}",
            cwd=worktree.workspace,
        ).stdout
    )


__all__ = [
    "OperatorWorktree",
    "WorktreeError",
    "create_operator_worktree",
    "export_patch_from_base",
]
