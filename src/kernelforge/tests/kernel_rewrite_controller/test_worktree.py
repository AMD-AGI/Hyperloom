# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kernelforge.kernel_rewrite_controller import ControllerLayout, parse_task_payload
from kernelforge.kernel_rewrite_controller.worktree import (
    WorktreeError,
    create_operator_worktree,
    export_patch_from_base,
)
from kernelforge.knowledge.kernel_identity import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "controller-test",
    "GIT_AUTHOR_EMAIL": "controller-test@local",
    "GIT_COMMITTER_NAME": "controller-test",
    "GIT_COMMITTER_EMAIL": "controller-test@local",
}


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, **_GIT_IDENTITY},
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _source_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "source-repo"
    repo.mkdir()
    _git(repo, "init")
    kernel = repo / "sglang" / "kernels" / "fused_moe.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo, _git(repo, "rev-parse", "HEAD")


def _task(tmp_path: Path, repo: Path, base_commit: str):
    identity_mapping = {
        "producer": "forge-loop",
        "kernel_name": "fused_moe",
        "framework": "sglang",
        "framework_version": "0.5.0",
        "backend": "triton",
        "gpu": "mi355x",
    }
    operator_id = kernel_recipe_canonical_id(KernelRecipeIdentity.from_mapping(identity_mapping))
    task_dir = tmp_path / "output" / "controller" / "tasks" / operator_id
    task_dir.mkdir(parents=True)
    (task_dir / "driver.py").write_text("print('SNR: 100 dB')\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "identity": identity_mapping,
        "base_commit": base_commit,
        "repo_root": str(repo),
        "kernel_path": "sglang/kernels/fused_moe.py",
        "operator_name": "fused_moe",
        "driver_path": "driver.py",
        "source_files": ["sglang/kernels/fused_moe.py"],
        "target_functions": ["fused_moe"],
        "shape_cases": [],
        "priority": 0,
        "reason": "",
        "evidence": [],
    }
    return parse_task_payload(payload, task_dir=task_dir), task_dir


def test_create_operator_worktree_pins_the_shared_base_commit(tmp_path: Path) -> None:
    repo, base_commit = _source_repo(tmp_path)
    task, _ = _task(tmp_path, repo, base_commit)
    layout = ControllerLayout(tmp_path / "output")

    worktree = create_operator_worktree(task, layout)

    assert _git(worktree.workspace, "rev-parse", "HEAD") == base_commit
    assert worktree.kernel_path.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert worktree.source_files == (worktree.kernel_path,)
    assert not worktree.workspace.is_relative_to(repo)


def test_export_patch_uses_controller_base_and_excludes_external_driver(tmp_path: Path) -> None:
    repo, base_commit = _source_repo(tmp_path)
    task, task_dir = _task(tmp_path, repo, base_commit)
    worktree = create_operator_worktree(task, ControllerLayout(tmp_path / "output"))
    worktree.kernel_path.write_text("VALUE = 2\n", encoding="utf-8")
    _git(worktree.workspace, "add", ".")
    _git(worktree.workspace, "commit", "-m", "optimize kernel")
    best_commit = _git(worktree.workspace, "rev-parse", "HEAD")

    patch = export_patch_from_base(worktree, best_commit=best_commit)

    assert "VALUE = 2" in patch
    assert "fused_moe.py" in patch
    assert "driver.py" not in patch
    assert (task_dir / "driver.py").is_file()


def test_missing_kernel_at_base_removes_partial_worktree(tmp_path: Path) -> None:
    repo, base_commit = _source_repo(tmp_path)
    task, _ = _task(tmp_path, repo, base_commit)
    task = type(task)(
        **{
            **task.__dict__,
            "kernel_path": "sglang/kernels/missing.py",
        }
    )
    layout = ControllerLayout(tmp_path / "output")

    with pytest.raises(WorktreeError, match="kernel path is not a file"):
        create_operator_worktree(task, layout)

    assert not layout.workspace_dir(task.operator_id).exists()
