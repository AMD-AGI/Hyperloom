# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

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
    read_campaign_baseline,
    release_operator_worktree,
    untracked_paths,
)
from kernelforge.loop.editable_repo import release_repo_lock
from kernelforge.knowledge.kernel_identity import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)
from kernelforge.tests.kernel_rewrite_controller.conftest import _git


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

    # The JIT cache reaches this depth, and the guard asks git for new paths with exactly this command, so a shallower
    # assertion would not cover the failure.
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


def _abandon_campaign(repo: Path, base_commit: str, *, stage: str) -> None:
    """Leave the repository as a killed campaign would have."""
    _git(repo, "checkout", "-b", f"{CAMPAIGN_BRANCH_PREFIX}killed", base_commit)
    (repo / "sglang" / "kernels" / "fused_moe.py").write_text("CAMPAIGN\n", encoding="utf-8")
    (repo / "left_behind.py").write_text("x\n", encoding="utf-8")
    _git(repo, "add", stage)
    _git(repo, "-c", "user.name=c", "-c", "user.email=c@l", "commit", "-m", "killed campaign")


def test_reclaiming_a_branch_removes_what_that_campaign_created(tmp_path: Path) -> None:
    repo, base_commit = _source_repo(tmp_path)
    baseline = untracked_paths(repo)
    _abandon_campaign(repo, base_commit, stage="--all")

    branch = reclaim_campaign_branch(repo, base_commit, baseline_untracked=baseline)

    assert branch == f"{CAMPAIGN_BRANCH_PREFIX}killed"
    assert not (repo / "left_behind.py").exists()
    assert (repo / "sglang" / "kernels" / "fused_moe.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert _git(repo, "status", "--porcelain") == ""


@pytest.mark.parametrize("stage", ["--update", "--all"])
def test_reclaiming_keeps_untracked_files_the_campaign_did_not_write(
    tmp_path: Path,
    stage: str,
) -> None:
    """A serving tree holds files no lane tracked -- fusion writes one.

    The seal commits with ``add --update``, so a module another lane created is
    untracked for the whole session. It also means a campaign that committed
    with ``add --all`` swept it onto its own branch, which is why the restore
    cannot be ``checkout --force``: that would delete it for being absent from
    the base rather than leave it untracked for the inventory to judge.
    """
    repo, base_commit = _source_repo(tmp_path)
    theirs = repo / "glm4_moe_fused_llm_allreduce.py"
    theirs.write_text("another lane wrote this\n", encoding="utf-8")
    baseline = untracked_paths(repo)
    _abandon_campaign(repo, base_commit, stage=stage)

    reclaim_campaign_branch(repo, base_commit, baseline_untracked=baseline)

    assert theirs.read_text(encoding="utf-8") == "another lane wrote this\n"
    assert not (repo / "left_behind.py").exists()


def test_reclaiming_without_an_inventory_deletes_nothing(tmp_path: Path) -> None:
    """An empty inventory would read as "every untracked path is the campaign's".

    Reached when the controller was killed and Hyperloom's own reclaim did not
    run either, so no caller has one and none was recorded. Unknown has to mean
    unknown: guessing here removes files nothing in this system wrote.
    """
    repo, base_commit = _source_repo(tmp_path)
    theirs = repo / "glm4_moe_fused_llm_allreduce.py"
    theirs.write_text("another lane wrote this\n", encoding="utf-8")
    _abandon_campaign(repo, base_commit, stage="--update")

    reclaim_campaign_branch(repo, base_commit)

    assert theirs.read_text(encoding="utf-8") == "another lane wrote this\n"
    assert (repo / "left_behind.py").exists()
    assert (repo / "sglang" / "kernels" / "fused_moe.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_reclaiming_with_neither_a_record_nor_a_base_commit_changes_nothing(
    tmp_path: Path,
) -> None:
    """Restoring to a commit nothing vouches for is worse than not restoring.

    The caller that seals has no base commit of its own -- the whole point of
    sealing is to establish one -- so when no record survived either, there is
    no answer, and inventing one would make some arbitrary commit the baseline
    every measurement afterwards is taken against.
    """
    repo, base_commit = _source_repo(tmp_path)
    _abandon_campaign(repo, base_commit, stage="--all")
    campaign_head = _git(repo, "rev-parse", "HEAD")

    assert reclaim_campaign_branch(repo) == ""

    assert _git(repo, "rev-parse", "HEAD") == campaign_head
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == f"{CAMPAIGN_BRANCH_PREFIX}killed"
    assert (repo / "sglang" / "kernels" / "fused_moe.py").read_text(encoding="utf-8") == "CAMPAIGN\n"


def test_a_recorded_base_commit_outranks_the_one_the_caller_offers(tmp_path: Path) -> None:
    """The borrowing process is the only one that saw the repository before.

    A caller reaching this after a kill knows the commit it sealed at, which is
    the same answer while one session is running. Across sessions it is not: the
    next seal has only what the dead campaign wrote down.
    """
    repo, base_commit = _source_repo(tmp_path)
    (repo / "sglang" / "kernels" / "fused_moe.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "--update")
    _git(repo, "-c", "user.name=c", "-c", "user.email=c@l", "commit", "-m", "later")
    recorded_base = _git(repo, "rev-parse", "HEAD")
    worktree_module.record_campaign_baseline(repo, recorded_base, frozenset(), origin_ref="master")
    _abandon_campaign(repo, recorded_base, stage="--all")

    assert reclaim_campaign_branch(repo, base_commit) == f"{CAMPAIGN_BRANCH_PREFIX}killed"

    assert _git(repo, "rev-parse", "HEAD") == recorded_base
    assert (repo / "sglang" / "kernels" / "fused_moe.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_a_borrow_records_the_inventory_for_whoever_has_to_hand_it_back(
    tmp_path: Path,
    editable,
) -> None:
    """Held in memory it dies with the process the host kills."""
    repo, base_commit = _source_repo(tmp_path)
    theirs = repo / "glm4_moe_fused_llm_allreduce.py"
    theirs.write_text("another lane wrote this\n", encoding="utf-8")
    editable(repo)
    task, _ = _task(tmp_path, repo, base_commit)
    layout = ControllerLayout(tmp_path / "output")

    borrowed = create_operator_worktree(task, layout)
    recorded = read_campaign_baseline(repo)

    assert recorded is not None
    assert recorded.base_commit == base_commit
    assert recorded.origin_ref == "master"
    assert "glm4_moe_fused_llm_allreduce.py" in recorded.untracked

    release_operator_worktree(borrowed)
    # Dropped once the repository is actually back, or a later borrow would read
    # it as the account of a campaign that never returned.
    assert read_campaign_baseline(repo) is None


def test_a_later_borrow_reclaims_a_killed_campaign_from_the_record(
    tmp_path: Path,
    editable,
) -> None:
    repo, base_commit = _source_repo(tmp_path)
    theirs = repo / "glm4_moe_fused_llm_allreduce.py"
    theirs.write_text("another lane wrote this\n", encoding="utf-8")
    editable(repo)
    task, _ = _task(tmp_path, repo, base_commit)
    layout = ControllerLayout(tmp_path / "output")

    # A borrow the host killed: the record is on disk, nothing released.
    killed = create_operator_worktree(task, layout)
    (repo / "left_behind.py").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "--all")
    _git(repo, "-c", "user.name=c", "-c", "user.email=c@l", "commit", "-m", "killed campaign")
    release_repo_lock(killed.lock)

    borrowed = create_operator_worktree(task, layout)

    try:
        assert not (repo / "left_behind.py").exists()
        assert theirs.read_text(encoding="utf-8") == "another lane wrote this\n"
    finally:
        release_operator_worktree(borrowed)


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
