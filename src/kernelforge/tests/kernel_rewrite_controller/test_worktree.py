# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kernelforge.kernel_rewrite_controller import ControllerLayout, parse_task_payload
from kernelforge.kernel_rewrite_controller.paths import operator_directory_name
import kernelforge.kernel_rewrite_controller.worktree as worktree_module
from kernelforge.kernel_rewrite_controller.worktree import (
    CAMPAIGN_BRANCH_PREFIX,
    FORGE_LOOP_OUTPUT_DIRNAME,
    WorktreeError,
    create_operator_worktree,
    export_patch_from_base,
    reclaim_campaign_branch,
    release_operator_worktree,
    untracked_paths,
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


def _task_payload(tmp_path: Path, repo: Path, base_commit: str):
    identity_mapping = {
        "producer": "forge-loop",
        "kernel_name": "fused_moe",
        "framework": "sglang",
        "framework_version": "0.5.0",
        "backend": "triton",
        "gpu": "mi355x",
    }
    operator_id = kernel_recipe_canonical_id(KernelRecipeIdentity.from_mapping(identity_mapping))
    task_dir = tmp_path / "output" / "controller" / "tasks" / operator_directory_name(operator_id)
    task_dir.mkdir(parents=True)
    (task_dir / "driver.py").write_text("print('SNR: 100 dB')\n", encoding="utf-8")
    payload = {
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
    return payload, task_dir


def _task(tmp_path: Path, repo: Path, base_commit: str):
    payload, task_dir = _task_payload(tmp_path, repo, base_commit)
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


def test_forge_loop_output_is_invisible_to_the_workspace_guard(tmp_path: Path) -> None:
    repo, base_commit = _source_repo(tmp_path)
    task, _ = _task(tmp_path, repo, base_commit)

    worktree = create_operator_worktree(task, ControllerLayout(tmp_path / "output"))

    # The JIT cache reaches this depth, and the guard asks git for new paths with
    # exactly this command, so a shallower assertion would not cover the failure.
    jit_artifact = worktree.workspace / "forge_experiments" / "aiter_cache" / "sources" / "abc" / "launch_moe"
    jit_artifact.parent.mkdir(parents=True)
    jit_artifact.write_text("compiled\n", encoding="utf-8")

    untracked = _git(worktree.workspace, "ls-files", "--others", "--exclude-standard")

    assert untracked == ""


def test_forge_loop_output_ignore_rule_stays_out_of_the_patch(tmp_path: Path) -> None:
    repo, base_commit = _source_repo(tmp_path)
    task, _ = _task(tmp_path, repo, base_commit)
    worktree = create_operator_worktree(task, ControllerLayout(tmp_path / "output"))
    worktree.kernel_path.write_text("VALUE = 2\n", encoding="utf-8")
    _git(worktree.workspace, "add", "-A")
    _git(worktree.workspace, "commit", "-m", "optimize kernel")

    patch = export_patch_from_base(
        worktree,
        best_commit=_git(worktree.workspace, "rev-parse", "HEAD"),
    )

    assert "VALUE = 2" in patch
    assert "forge_experiments" not in patch


def test_forge_loop_output_ignore_does_not_leak_into_the_source_repo(tmp_path: Path) -> None:
    repo, base_commit = _source_repo(tmp_path)
    task, _ = _task(tmp_path, repo, base_commit)

    create_operator_worktree(task, ControllerLayout(tmp_path / "output"))
    (repo / "forge_experiments").mkdir()
    (repo / "forge_experiments" / "stray").write_text("x\n", encoding="utf-8")

    assert _git(repo, "status", "--porcelain") == "?? forge_experiments/"


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


def test_a_repo_root_that_is_not_a_git_checkout_is_refused(tmp_path: Path) -> None:
    """A worktree can only be cut from a repository."""
    plain = tmp_path / "not-a-repo"
    (plain / "sglang" / "kernels").mkdir(parents=True)
    (plain / "sglang" / "kernels" / "fused_moe.py").write_text("VALUE = 1\n", encoding="utf-8")
    task, _ = _task(tmp_path, plain, "a" * 40)
    layout = ControllerLayout(tmp_path / "output")

    with pytest.raises(WorktreeError, match="not a Git checkout"):
        create_operator_worktree(task, layout)


def test_a_repo_root_below_the_top_level_is_refused(tmp_path: Path) -> None:
    """A subdirectory would pin the base to the wrong tree's HEAD."""
    repo, base_commit = _source_repo(tmp_path)
    task, _ = _task(tmp_path, repo / "sglang", base_commit)
    layout = ControllerLayout(tmp_path / "output")

    with pytest.raises(WorktreeError, match="must be the Git top-level directory"):
        create_operator_worktree(task, layout)


def test_a_base_commit_the_repository_does_not_have_is_refused(tmp_path: Path) -> None:
    """The pin has to name a commit, or every patch is cut against nothing."""
    repo, _base_commit = _source_repo(tmp_path)
    task, _ = _task(tmp_path, repo, "b" * 40)
    layout = ControllerLayout(tmp_path / "output")

    with pytest.raises(WorktreeError, match="base commit does not exist"):
        create_operator_worktree(task, layout)


def test_an_existing_workspace_is_refused_rather_than_resumed(tmp_path: Path) -> None:
    """Reusing a workspace would measure against a tree of unknown provenance."""
    repo, base_commit = _source_repo(tmp_path)
    task, _ = _task(tmp_path, repo, base_commit)
    layout = ControllerLayout(tmp_path / "output")
    squatter = layout.workspace_dir(task.operator_id)
    squatter.mkdir(parents=True)

    with pytest.raises(WorktreeError, match="cannot be resumed"):
        create_operator_worktree(task, layout)


def test_a_source_file_absent_from_the_base_commit_is_refused(tmp_path: Path) -> None:
    """A source file the base does not carry cannot be what forge-loop edits."""
    repo, base_commit = _source_repo(tmp_path)
    payload, task_dir = _task_payload(tmp_path, repo, base_commit)
    task = parse_task_payload(
        {**payload, "source_files": ["sglang/kernels/fused_moe.py", "sglang/kernels/absent.py"]},
        task_dir=task_dir,
    )
    layout = ControllerLayout(tmp_path / "output")

    with pytest.raises(WorktreeError, match="source file is not a file in the base commit"):
        create_operator_worktree(task, layout)
    assert not layout.workspace_dir(task.operator_id).exists()


@pytest.fixture
def editable(monkeypatch: pytest.MonkeyPatch):
    """Make one repository read as an editable install for the code under test."""

    def _apply(repo: Path) -> None:
        monkeypatch.setattr(
            worktree_module,
            "needs_inplace",
            lambda candidate: Path(candidate).resolve() == repo.resolve(),
        )

    return _apply


def test_an_editable_repository_is_borrowed_rather_than_copied(
    tmp_path: Path,
    editable,
) -> None:
    """A private checkout of an editable install is edited and never imported."""
    repo, base_commit = _source_repo(tmp_path)
    editable(repo)
    task, _ = _task(tmp_path, repo, base_commit)
    layout = ControllerLayout(tmp_path / "output")

    borrowed = create_operator_worktree(task, layout)
    try:
        assert borrowed.inplace is True
        assert borrowed.workspace == repo.resolve()
        assert not layout.workspace_dir(task.operator_id).exists()
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").startswith(CAMPAIGN_BRANCH_PREFIX)
        assert _git(repo, "rev-parse", "HEAD") == base_commit
    finally:
        release_operator_worktree(borrowed)


def test_releasing_a_borrowed_repository_undoes_the_campaign(tmp_path: Path, editable) -> None:
    """The patch is already published, so the tree it was built in is disposable."""
    repo, base_commit = _source_repo(tmp_path)
    # Written before the borrow, which is what makes it the operator's: after it,
    # an untracked file is indistinguishable from one the campaign added.
    keepsake = repo / "operator-notes.txt"
    keepsake.write_text("mine\n", encoding="utf-8")
    editable(repo)
    task, _ = _task(tmp_path, repo, base_commit)
    layout = ControllerLayout(tmp_path / "output")
    origin_ref = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")

    borrowed = create_operator_worktree(task, layout)
    borrowed.kernel_path.write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "forge_experiments").mkdir(exist_ok=True)
    (repo / "forge_experiments" / "iteration.json").write_text("{}", encoding="utf-8")

    release_operator_worktree(borrowed)

    assert (repo / "sglang" / "kernels" / "fused_moe.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (repo / "forge_experiments").exists()
    assert keepsake.read_text(encoding="utf-8") == "mine\n"
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == origin_ref
    assert _git(repo, "status", "--porcelain", "--untracked-files=no") == ""
    assert borrowed.branch not in _git(repo, "branch", "--list", borrowed.branch)


def test_a_second_campaign_can_borrow_the_repository_after_release(tmp_path: Path, editable) -> None:
    """The lock has to come off, or the next operator never starts."""
    repo, base_commit = _source_repo(tmp_path)
    editable(repo)
    task, _ = _task(tmp_path, repo, base_commit)
    layout = ControllerLayout(tmp_path / "output")

    release_operator_worktree(create_operator_worktree(task, layout))
    again = create_operator_worktree(task, layout)

    try:
        assert again.inplace is True
    finally:
        release_operator_worktree(again)


def test_a_repository_already_borrowed_is_refused(tmp_path: Path, editable) -> None:
    repo, base_commit = _source_repo(tmp_path)
    editable(repo)
    task, _ = _task(tmp_path, repo, base_commit)
    layout = ControllerLayout(tmp_path / "output")
    held = create_operator_worktree(task, layout)

    try:
        with pytest.raises(WorktreeError, match="already holds"):
            create_operator_worktree(task, layout)
    finally:
        release_operator_worktree(held)


def test_a_repository_that_is_not_the_base_commit_is_refused(tmp_path: Path, editable) -> None:
    """Borrowing a dirty tree would fold someone else's edit into this patch."""
    repo, base_commit = _source_repo(tmp_path)
    editable(repo)
    task, _ = _task(tmp_path, repo, base_commit)
    layout = ControllerLayout(tmp_path / "output")
    (repo / "sglang" / "kernels" / "fused_moe.py").write_text("VALUE = 99\n", encoding="utf-8")

    with pytest.raises(WorktreeError, match="carries uncommitted changes against base commit"):
        create_operator_worktree(task, layout)


def test_a_campaign_branch_a_killed_run_left_behind_is_reclaimed(tmp_path: Path, editable) -> None:
    """Nothing in this process runs when the host kills the controller outright."""
    repo, base_commit = _source_repo(tmp_path)
    editable(repo)
    task, _ = _task(tmp_path, repo, base_commit)
    layout = ControllerLayout(tmp_path / "output")
    _git(repo, "checkout", "-b", f"{CAMPAIGN_BRANCH_PREFIX}abandoned", base_commit)
    (repo / "sglang" / "kernels" / "fused_moe.py").write_text("half-finished\n", encoding="utf-8")

    borrowed = create_operator_worktree(task, layout)

    try:
        assert borrowed.inplace is True
        assert borrowed.kernel_path.read_text(encoding="utf-8") == "VALUE = 1\n"
    finally:
        release_operator_worktree(borrowed)


def test_release_removes_what_the_campaign_created_and_keeps_what_it_found(
    tmp_path: Path,
    editable,
) -> None:
    """A file the campaign committed survives `checkout <base> -- <path>`.

    The base does not carry the path, so that restore cannot touch it; it only
    reads as untracked once HEAD and the index have moved. Removing it needs an
    inventory of what was untracked before the borrow -- by name it is
    indistinguishable from a file the repository's owner keeps.
    """
    repo, base_commit = _source_repo(tmp_path)
    keepsake = repo / "operator-notes.txt"
    keepsake.write_text("mine\n", encoding="utf-8")
    editable(repo)
    task, _ = _task(tmp_path, repo, base_commit)
    layout = ControllerLayout(tmp_path / "output")

    borrowed = create_operator_worktree(task, layout)
    borrowed.kernel_path.write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "new_kernel_helper.py").write_text("helper\n", encoding="utf-8")
    (repo / "generated").mkdir()
    (repo / "generated" / "kernel.h").write_text("h\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=c", "-c", "user.email=c@l", "commit", "-m", "campaign work")
    (repo / "scratch.tmp").write_text("x\n", encoding="utf-8")

    release_operator_worktree(borrowed)

    assert borrowed.kernel_path.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (repo / "new_kernel_helper.py").exists()
    assert not (repo / "generated").exists()
    assert not (repo / "scratch.tmp").exists()
    assert keepsake.read_text(encoding="utf-8") == "mine\n"
    assert _git(repo, "status", "--porcelain") == "?? operator-notes.txt"


def test_release_keeps_the_campaign_bookkeeping_out_of_the_repository(
    tmp_path: Path,
    editable,
) -> None:
    """Every in-place task in one repository is handed the same directory.

    So it cannot be left there -- the next campaign would resume this one's
    state -- but a run that published no patch has nothing else to be read from.
    """
    repo, base_commit = _source_repo(tmp_path)
    editable(repo)
    task, _ = _task(tmp_path, repo, base_commit)
    layout = ControllerLayout(tmp_path / "output")

    borrowed = create_operator_worktree(task, layout)
    (repo / FORGE_LOOP_OUTPUT_DIRNAME / "best_result.json").write_text("{}", encoding="utf-8")

    release_operator_worktree(borrowed)

    assert not (repo / FORGE_LOOP_OUTPUT_DIRNAME).exists()
    archived = layout.workspace_dir(task.operator_id) / FORGE_LOOP_OUTPUT_DIRNAME
    assert (archived / "best_result.json").read_text(encoding="utf-8") == "{}"


def test_reclaiming_a_branch_also_removes_what_that_campaign_created(tmp_path: Path) -> None:
    """`checkout --force` restores tracked content and leaves the rest behind."""
    repo, base_commit = _source_repo(tmp_path)
    baseline = untracked_paths(repo)
    _git(repo, "checkout", "-b", f"{CAMPAIGN_BRANCH_PREFIX}killed", base_commit)
    (repo / "left_behind.py").write_text("x\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=c", "-c", "user.email=c@l", "commit", "-m", "killed campaign")

    branch = reclaim_campaign_branch(repo, base_commit, baseline_untracked=baseline)

    assert branch == f"{CAMPAIGN_BRANCH_PREFIX}killed"
    assert not (repo / "left_behind.py").exists()
    assert _git(repo, "status", "--porcelain") == ""


def test_releasing_a_borrowed_repository_twice_is_harmless(tmp_path: Path, editable) -> None:
    """Three lanes take this lock and each releases from a finally."""
    repo, base_commit = _source_repo(tmp_path)
    editable(repo)
    task, _ = _task(tmp_path, repo, base_commit)
    layout = ControllerLayout(tmp_path / "output")

    borrowed = create_operator_worktree(task, layout)
    release_operator_worktree(borrowed)
    release_operator_worktree(borrowed)

    again = create_operator_worktree(task, layout)
    assert again.inplace is True
    release_operator_worktree(again)
