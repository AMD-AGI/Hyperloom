# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from kernelforge.kernel_rewrite_controller import (
    ControllerLayout,
    TaskStateStore,
    load_task,
)
from kernelforge.kernel_rewrite_controller.publisher import published_operator_dirs
from kernelforge.kernel_rewrite_controller.recovery import recover_task_result
from kernelforge.kernel_rewrite_controller.worktree import (
    create_operator_worktree,
    export_patch_from_base,
)
from kernelforge.knowledge.kernel_identity import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)
from kernelforge.loop.reporting import BestResultPublisher

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "controller-test",
    "GIT_AUTHOR_EMAIL": "controller-test@local",
    "GIT_COMMITTER_NAME": "controller-test",
    "GIT_COMMITTER_EMAIL": "controller-test@local",
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, **_GIT_IDENTITY},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _prepared_workspace(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    kernel = repo / "kernel.py"
    kernel.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    base_commit = _git(repo, "rev-parse", "HEAD")

    identity_mapping = {
        "producer": "forge-loop",
        "kernel_name": "kernel",
        "framework": "standalone",
        "framework_version": "unknown",
        "backend": "triton",
        "gpu": "mi355x",
    }
    operator_id = kernel_recipe_canonical_id(KernelRecipeIdentity.from_mapping(identity_mapping))
    layout = ControllerLayout(tmp_path / "output")
    task_dir = layout.task_dir(operator_id)
    task_dir.mkdir(parents=True)
    (task_dir / "driver.py").write_text("print('SNR: 100 dB')\n", encoding="utf-8")
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": identity_mapping,
                "base_commit": base_commit,
                "repo_root": str(repo),
                "kernel_path": "kernel.py",
                "operator_name": "kernel",
                "driver_path": "driver.py",
                "source_files": ["kernel.py"],
                "target_functions": ["kernel"],
                "shape_cases": [],
                "priority": 0,
                "reason": "",
                "evidence": [],
            }
        ),
        encoding="utf-8",
    )
    task = load_task(task_dir).task
    assert task is not None
    worktree = create_operator_worktree(task, layout)
    worktree.kernel_path.write_text("VALUE = 2\n", encoding="utf-8")
    _git(worktree.workspace, "add", ".")
    _git(worktree.workspace, "commit", "-m", "optimized")
    best_commit = _git(worktree.workspace, "rev-parse", "HEAD")
    return layout, task_dir, worktree, best_commit


def test_recovery_publishes_a_validated_result_sidecar(tmp_path: Path) -> None:
    layout, task_dir, _worktree, best_commit = _prepared_workspace(tmp_path)
    (task_dir / "forge-result.json").write_text(
        json.dumps(
            {
                "improved": True,
                "best_commit": best_commit,
                "mean_case_speedup": 1.2,
            }
        ),
        encoding="utf-8",
    )

    recovered = recover_task_result(layout, task_dir)

    assert recovered.published is True
    assert recovered.patch_dir is not None
    assert "VALUE = 2" in (recovered.patch_dir / "change.patch").read_text(encoding="utf-8")
    state = TaskStateStore(task_dir).load()
    assert state is not None
    assert state.status == "succeeded"
    assert state.result_patch_dir == str(recovered.patch_dir)


def test_recovery_is_idempotent_for_an_already_published_best(tmp_path: Path) -> None:
    layout, task_dir, _worktree, best_commit = _prepared_workspace(tmp_path)
    (task_dir / "forge-result.json").write_text(
        json.dumps({"improved": True, "best_commit": best_commit}),
        encoding="utf-8",
    )

    first = recover_task_result(layout, task_dir)
    second = recover_task_result(layout, task_dir)

    assert first.published is True
    assert second.published is False
    assert second.patch_dir == first.patch_dir
    assert second.reason == "best result already published"
    assert len(published_operator_dirs(layout)) == 1


def test_recovery_publishes_a_complete_forge_best_manifest(tmp_path: Path) -> None:
    layout, task_dir, worktree, best_commit = _prepared_workspace(tmp_path)
    forge_patch = export_patch_from_base(worktree, best_commit=best_commit)
    BestResultPublisher(str(worktree.workspace)).publish(
        campaign_id="campaign",
        session_index=0,
        experiment_id="experiment",
        iteration=1,
        commit_hash=best_commit,
        plan="optimize",
        baseline_wall_ms=2.0,
        search_start_ms=2.0,
        best_wall_ms=1.0,
        mean_case_speedup=2.0,
        search_start_mean_case_speedup=1.0,
        snr_db=100.0,
        validation_text="PASS\n",
        benchmark={"success": True},
        changed_files=["kernel.py"],
        patch=forge_patch,
    )

    recovered = recover_task_result(layout, task_dir)

    assert recovered.published is True
    assert recovered.best_commit == best_commit
    assert recovered.patch_dir is not None
    assert "VALUE = 2" in (recovered.patch_dir / "change.patch").read_text(encoding="utf-8")


def test_recovery_ignores_uncommitted_workspace_edits(tmp_path: Path) -> None:
    layout, task_dir, worktree, _best_commit = _prepared_workspace(tmp_path)
    worktree.kernel_path.write_text("VALUE = 3\n", encoding="utf-8")

    recovered = recover_task_result(layout, task_dir)

    assert recovered.published is False
    assert recovered.reason == "no trusted forge-loop best result"
    assert published_operator_dirs(layout) == ()
