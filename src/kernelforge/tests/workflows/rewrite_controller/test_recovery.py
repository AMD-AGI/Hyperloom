# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from kernelforge.workflows.rewrite_controller import (
    ControllerLayout,
    TaskStateStore,
    load_task,
)
from kernelforge.workflows.rewrite_controller import recovery
from kernelforge.workflows.rewrite_controller.contracts import TaskStateError
from kernelforge.workflows.rewrite_controller.publisher import PublicationError, published_operator_dirs
from kernelforge.workflows.rewrite_controller.worktree import (
    WorktreeError,
    create_operator_worktree,
    export_patch_from_base,
)
from kernelforge.knowledge.kb_store.identity.kernel_recipe import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)
from kernelforge.workflows.forge_loop.campaign.reporting import BestResultPublisher, KeepBundle, KeepIdentity, KeepScore
from kernelforge.tests.workflows.rewrite_controller.conftest import _git

recover_task_result = recovery.recover_task_result


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


def _publish_best(worktree, best_commit, patch):
    """Publish iteration 1 of the test campaign as its best result."""
    BestResultPublisher(str(worktree.workspace)).publish(
        KeepIdentity("campaign", 0, "experiment", 1, best_commit, "optimize"),
        KeepScore(2.0, 1.0, 2.0, 1.0, snr_db=100.0, search_start_ms=2.0),
        KeepBundle("PASS\n", {"success": True}, ["kernel.py"], patch),
    )


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


def _recover_with_publication_failure(tmp_path: Path, monkeypatch, failure: Exception):
    layout, task_dir, _worktree, best_commit = _prepared_workspace(tmp_path)
    (task_dir / "forge-result.json").write_text(
        json.dumps({"improved": True, "best_commit": best_commit}),
        encoding="utf-8",
    )

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(recovery, "publish_operator_result", fail)
    return layout, best_commit, recover_task_result(layout, task_dir)


@pytest.mark.parametrize(
    "failure",
    [
        PublicationError("operator publication conflicts with existing version"),
        WorktreeError("best commit is not based on controller base"),
        TaskStateError("task state must be initialized before transition"),
        subprocess.CalledProcessError(128, ["git", "diff"], stderr="fatal: bad object"),
        OSError("disk full"),
        UnicodeDecodeError("utf-8", b"\xfc", 0, 1, "invalid start byte"),
    ],
    ids=lambda failure: type(failure).__name__,
)
def test_an_expected_publication_failure_is_reported_not_raised(tmp_path: Path, monkeypatch, failure) -> None:
    layout, best_commit, recovered = _recover_with_publication_failure(tmp_path, monkeypatch, failure)

    assert recovered == recovery.RecoveryResult(
        operator_id=recovered.operator_id,
        published=False,
        best_commit=best_commit,
        reason=f"could not publish forge result sidecar: {failure}",
    )
    assert not published_operator_dirs(layout)


def test_an_unexpected_publication_error_propagates(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(TypeError, match="^publication fields drifted$"):
        _recover_with_publication_failure(tmp_path, monkeypatch, TypeError("publication fields drifted"))


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
    _publish_best(worktree, best_commit, forge_patch)

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


def test_a_stale_sidecar_does_not_pull_the_published_patch_backwards(tmp_path: Path) -> None:
    # The two views are not equally attested: a manifest is trusted only once describes_current_best has confirmed a
    # complete bundle, while the sidecar needs an improved flag and a commit.
    layout, task_dir, worktree, older_commit = _prepared_workspace(tmp_path)
    worktree.kernel_path.write_text("VALUE = 3\n", encoding="utf-8")
    _git(worktree.workspace, "add", ".")
    _git(worktree.workspace, "commit", "-m", "optimized again")
    newer_commit = _git(worktree.workspace, "rev-parse", "HEAD")

    BestResultPublisher(str(worktree.workspace)).publish(
        KeepIdentity(
            campaign_id="campaign",
            session_index=0,
            experiment_id="experiment",
            iteration=5,
            commit_hash=newer_commit,
            plan="optimize",
        ),
        KeepScore(
            baseline_wall_ms=2.0,
            best_wall_ms=0.5,
            mean_case_speedup=4.0,
            search_start_mean_case_speedup=1.0,
            snr_db=100.0,
            search_start_ms=2.0,
        ),
        KeepBundle(
            validation_text="PASS\n",
            benchmark={"success": True},
            changed_files=["kernel.py"],
            patch=export_patch_from_base(worktree, best_commit=newer_commit),
        ),
    )
    (task_dir / "forge-result.json").write_text(
        json.dumps({"improved": True, "best_commit": older_commit, "best_iteration": 3}),
        encoding="utf-8",
    )

    recovered = recover_task_result(layout, task_dir)

    assert recovered.published is True
    assert recovered.best_commit == newer_commit
    assert recovered.patch_dir is not None
    assert "VALUE = 3" in (recovered.patch_dir / "change.patch").read_text(encoding="utf-8")


def test_a_newer_sidecar_still_wins_over_an_older_manifest(tmp_path: Path) -> None:
    layout, task_dir, worktree, older_commit = _prepared_workspace(tmp_path)
    worktree.kernel_path.write_text("VALUE = 3\n", encoding="utf-8")
    _git(worktree.workspace, "add", ".")
    _git(worktree.workspace, "commit", "-m", "optimized again")
    newer_commit = _git(worktree.workspace, "rev-parse", "HEAD")

    BestResultPublisher(str(worktree.workspace)).publish(
        KeepIdentity(
            campaign_id="campaign",
            session_index=0,
            experiment_id="experiment",
            iteration=3,
            commit_hash=older_commit,
            plan="optimize",
        ),
        KeepScore(
            baseline_wall_ms=2.0,
            best_wall_ms=1.0,
            mean_case_speedup=2.0,
            search_start_mean_case_speedup=1.0,
            snr_db=100.0,
            search_start_ms=2.0,
        ),
        KeepBundle(
            validation_text="PASS\n",
            benchmark={"success": True},
            changed_files=["kernel.py"],
            patch=export_patch_from_base(worktree, best_commit=older_commit),
        ),
    )
    (task_dir / "forge-result.json").write_text(
        json.dumps({"improved": True, "best_commit": newer_commit, "best_iteration": 5}),
        encoding="utf-8",
    )

    recovered = recover_task_result(layout, task_dir)

    assert recovered.published is True
    assert recovered.best_commit == newer_commit


def test_a_sidecar_that_did_not_improve_is_not_published(tmp_path: Path) -> None:
    """``improved`` is the whole claim; without it there is nothing to integrate."""
    layout, task_dir, _worktree, best_commit = _prepared_workspace(tmp_path)
    (task_dir / "forge-result.json").write_text(
        json.dumps({"improved": False, "best_commit": best_commit}),
        encoding="utf-8",
    )

    recovered = recover_task_result(layout, task_dir)

    assert recovered.published is False
    assert not published_operator_dirs(layout)


def test_a_sidecar_without_a_best_commit_is_not_published(tmp_path: Path) -> None:
    """An improvement with no commit names no patch."""
    layout, task_dir, _worktree, _best_commit = _prepared_workspace(tmp_path)
    (task_dir / "forge-result.json").write_text(
        json.dumps({"improved": True, "best_commit": ""}),
        encoding="utf-8",
    )

    recovered = recover_task_result(layout, task_dir)

    assert recovered.published is False
    assert not published_operator_dirs(layout)


def test_an_unvalidated_checkpoint_cannot_stand_in_for_a_best_commit(tmp_path: Path) -> None:
    """The checkpoint fallback is gated on validation, not on carrying a commit."""
    layout, task_dir, _worktree, best_commit = _prepared_workspace(tmp_path)
    (task_dir / "forge-result.json").write_text(
        json.dumps(
            {
                "improved": True,
                "best_commit": "",
                "checkpoint": {"validation_passed": False, "best_commit": best_commit},
            }
        ),
        encoding="utf-8",
    )

    recovered = recover_task_result(layout, task_dir)

    assert recovered.published is False
    assert not published_operator_dirs(layout)


def test_a_validated_checkpoint_supplies_the_missing_best_commit(tmp_path: Path) -> None:
    """A forge-loop killed after validation still has a result worth keeping."""
    layout, task_dir, _worktree, best_commit = _prepared_workspace(tmp_path)
    (task_dir / "forge-result.json").write_text(
        json.dumps(
            {
                "improved": True,
                "best_commit": "",
                "checkpoint": {"validation_passed": True, "best_commit": best_commit},
            }
        ),
        encoding="utf-8",
    )

    recovered = recover_task_result(layout, task_dir)

    assert recovered.published is True
    assert recovered.patch_dir is not None


def test_a_task_with_no_forge_output_at_all_publishes_nothing(tmp_path: Path) -> None:
    layout, task_dir, _worktree, _best_commit = _prepared_workspace(tmp_path)

    recovered = recover_task_result(layout, task_dir)

    assert recovered.published is False
    assert recovered.reason == "no trusted forge-loop best result"
    assert not published_operator_dirs(layout)


def test_a_leftover_campaign_manifest_is_not_taken_for_this_task(tmp_path: Path) -> None:
    """An in-place repository hands every task the same experiments directory.

    A run the host killed leaves its best-result bundle there, naming a commit its
    branch took with it on release. Trusting it reports the dead campaign's commit as
    a missing base commit and loses the task with a reason that has nothing to do with
    what it did.
    """
    layout, task_dir, worktree, best_commit = _prepared_workspace(tmp_path)
    forge_patch = export_patch_from_base(worktree, best_commit=best_commit)
    _publish_best(worktree, best_commit, forge_patch)
    # The commit a released campaign leaves behind: recorded in the manifest, gone
    # from the repository.
    manifest_path = BestResultPublisher(str(worktree.workspace)).manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["commit_hash"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    recovered = recover_task_result(layout, task_dir)

    assert recovered.published is False
    assert recovered.reason == "no trusted forge-loop best result"
    assert published_operator_dirs(layout) == ()
