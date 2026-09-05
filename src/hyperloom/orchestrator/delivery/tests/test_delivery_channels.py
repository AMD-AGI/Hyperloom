# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Harvest, revert and replay all agree about which tree kind they are talking to."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.reference_script import render_reference_script
from hyperloom.orchestrator.actions.executors._nogit_patch import (
    _apply_patch_no_git,
    _reverse_applies_cleanly,
    _revert_patches_no_git,
)
from hyperloom.orchestrator.bringup.trees import VCS_GIT, VCS_NONE
from hyperloom.orchestrator.delivery import (
    file_digest,
    ledger,
    post_images_from_diff,
    write_post_images,
)
from hyperloom.orchestrator.specialists.subprocess_ import (
    SpecialistSubprocessDispatcher,
    _declared_targets,
)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def test_the_harvest_excludes_the_specialists_own_scratch_copies() -> None:
    pathspec = SpecialistSubprocessDispatcher._harvest_pathspec(())
    assert pathspec[0] == "."
    assert ":(exclude)patches" in pathspec
    assert ":(exclude)artifacts" in pathspec


def test_declared_targets_scope_the_harvest_pathspec() -> None:
    assert SpecialistSubprocessDispatcher._harvest_pathspec(["pkg/mod.py"]) == ["pkg/mod.py"]


def test_the_round_declaration_scopes_the_harvest() -> None:
    assert _declared_targets({"deliverable": {"tree_id": "t", "targets": ["pkg/wide.py"]}}) == ("pkg/wide.py",)
    assert _declared_targets({}) == ()


def test_an_uncommitted_worktree_edit_is_still_harvested(tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    (worktree / "pkg").mkdir(parents=True)
    (worktree / "pkg" / "mod.py").write_text("base\n", encoding="utf-8")
    _git(worktree, "init", "-q")
    _git(worktree, "config", "user.email", "t@example.invalid")
    _git(worktree, "config", "user.name", "t")
    _git(worktree, "add", "-A")
    _git(worktree, "commit", "-qm", "base")
    base = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    # The specialist edits and adds, and commits neither.
    (worktree / "pkg" / "mod.py").write_text("changed\n", encoding="utf-8")
    (worktree / "pkg" / "added.py").write_text("new\n", encoding="utf-8")
    # ... and leaves whole-file copies aside while comparing revisions.
    (worktree / "patches").mkdir()
    (worktree / "patches" / "copy_of_mod.py").write_text("base\n", encoding="utf-8")

    diff = SpecialistSubprocessDispatcher._harvest_worktree_diff(worktree, base=base)
    assert "pkg/mod.py" in diff
    assert "pkg/added.py" in diff
    assert "copy_of_mod.py" not in diff


@pytest.mark.skipif(shutil.which("patch") is None, reason="POSIX patch is not installed")
def test_the_non_git_channel_records_every_backup_before_it_mutates(tmp_path: Path) -> None:
    root = tmp_path / "wheel"
    root.mkdir()
    target = root / "mod.py"
    target.write_text("one\n", encoding="utf-8")
    pre_image = file_digest(target)
    patch = tmp_path / "p.diff"
    patch.write_text(
        "--- a/mod.py\n+++ b/mod.py\n@@ -1 +1 @@\n-one\n+two\n",
        encoding="utf-8",
    )
    backup_root = tmp_path / "backups"

    ok, err, backups, _ = _apply_patch_no_git(root, patch, backup_root)
    assert ok, err
    assert target.read_text(encoding="utf-8") == "two\n"

    # The record is on disk, not only in the list the caller happens to hold.
    persisted = ledger.load_records(backup_root)
    assert [r["target"] for r in persisted] == [str(target)]
    assert persisted[0]["pre_image_sha256"] == pre_image

    # A process that lost the in-memory records still restores the tree.
    reverted_ok, errors = _revert_patches_no_git([], backup_root=backup_root)
    assert reverted_ok, errors
    assert target.read_text(encoding="utf-8") == "one\n"


@pytest.mark.skipif(shutil.which("patch") is None, reason="POSIX patch is not installed")
def test_a_near_miss_is_not_accepted_as_already_applied(tmp_path: Path) -> None:
    root = tmp_path / "wheel"
    root.mkdir()
    target = root / "mod.py"
    patch = tmp_path / "p.diff"
    patch.write_text(
        "--- a/mod.py\n+++ b/mod.py\n@@ -1,3 +1,3 @@\n ctx\n-one\n+two\n ctx2\n",
        encoding="utf-8",
    )

    target.write_text("ctx\ntwo\nctx2\n", encoding="utf-8")
    exact = file_digest(target)
    assert _reverse_applies_cleanly(root, patch, post_images={"mod.py": exact})

    # The same file with an unrelated extra line is not the post-image, however
    # willing ``patch`` is to match it at an offset.
    target.write_text("extra\nctx\ntwo\nctx2\n", encoding="utf-8")
    assert _reverse_applies_cleanly(root, patch), "the probe alone accepts the near miss"
    assert not _reverse_applies_cleanly(root, patch, post_images={"mod.py": exact})


@pytest.mark.skipif(shutil.which("patch") is None, reason="POSIX patch is not installed")
def test_the_apply_reads_the_digests_frozen_beside_the_patch(tmp_path: Path) -> None:
    root = tmp_path / "wheel"
    root.mkdir()
    target = root / "mod.py"
    patch = tmp_path / "p.diff"
    patch.write_text(
        "--- a/mod.py\n+++ b/mod.py\n@@ -1,3 +1,3 @@\n ctx\n-one\n+two\n ctx2\n",
        encoding="utf-8",
    )
    target.write_text("ctx\ntwo\nctx2\n", encoding="utf-8")
    write_post_images(patch, {"mod.py": file_digest(target)})

    # Exactly the validated post-state: the forward apply is a satisfied no-op.
    ok, err, backups, _ = _apply_patch_no_git(root, patch, tmp_path / "b1")
    assert ok, err
    assert backups == []

    # A tree that merely resembles it is not what the round validated, so the
    # apply must not bank it as already applied.
    target.write_text("extra\nctx\ntwo\nctx2\n", encoding="utf-8")
    ok, err, _, _ = _apply_patch_no_git(root, patch, tmp_path / "b2")
    assert not ok
    assert "all strip levels" in err


@pytest.mark.skipif(shutil.which("patch") is None, reason="POSIX patch is not installed")
def test_a_sanitized_patch_still_finds_its_frozen_digests(tmp_path: Path) -> None:
    # A harvested git diff always carries index lines, so the apply feeds the
    # CLI a rewritten copy. The digests sit beside the authored patch.
    root = tmp_path / "wheel"
    root.mkdir()
    target = root / "mod.py"
    patch = tmp_path / "p.diff"
    patch.write_text(
        "diff --git a/mod.py b/mod.py\n"
        "index 0000000..0000000 100644\n"
        "--- a/mod.py\n+++ b/mod.py\n@@ -1,3 +1,3 @@\n ctx\n-one\n+two\n ctx2\n",
        encoding="utf-8",
    )
    target.write_text("ctx\ntwo\nctx2\n", encoding="utf-8")
    write_post_images(patch, {"mod.py": file_digest(target)})

    target.write_text("extra\nctx\ntwo\nctx2\n", encoding="utf-8")
    ok, err, _, _ = _apply_patch_no_git(root, patch, tmp_path / "b")
    assert not ok, "the near miss was banked as already applied"


def test_the_post_images_are_hashed_where_the_work_was_validated(tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    (worktree / "pkg").mkdir(parents=True)
    (worktree / "pkg" / "mod.py").write_text("validated\n", encoding="utf-8")
    diff = "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1 +1 @@\n-base\n+validated\n"

    images = post_images_from_diff(diff, worktree)
    assert images == {"pkg/mod.py": file_digest(worktree / "pkg" / "mod.py")}

    # A deletion has no post-image to freeze.
    assert post_images_from_diff("--- a/pkg/gone.py\n+++ /dev/null\n", worktree) == {}


def test_the_replay_script_matches_the_tree_kind() -> None:
    rounds = [{"patches": ["patches/001_fix.patch"], "artifacts": []}]
    common = {
        "framework": "sglang",
        "server_args": "",
        "framework_root": "/opt/sglang",
        "rounds": rounds,
    }

    git_script = render_reference_script(**common, framework_root_vcs=VCS_GIT)
    assert 'git -C "$FRAMEWORK_ROOT" apply' in git_script
    assert "patch -p" not in git_script

    nogit_script = render_reference_script(**common, framework_root_vcs=VCS_NONE)
    # Under ``set -e`` a git ladder against a wheel aborts before the launch line.
    assert "set -euo pipefail" in nogit_script
    assert "git -C" not in nogit_script
    assert 'patch -p"$lvl" --fuzz=0 -d "$FRAMEWORK_ROOT"' in nogit_script
    assert nogit_script.rstrip().endswith("python3 -m sglang.launch_server --model-path=$MODEL")


def test_an_install_that_does_not_match_the_frozen_digest_is_a_failure(tmp_path: Path) -> None:
    from hyperloom.orchestrator.actions.executors.integrate_patch import (
        _ArtifactSpec,
        _installed_digest_mismatch,
    )
    from hyperloom.orchestrator.delivery import Artifact, Deliverable, file_digest

    root = tmp_path / "fw"
    root.mkdir()
    target = root / "cfg.json"
    target.write_text("validated\n", encoding="utf-8")
    spec = _ArtifactSpec(
        source=tmp_path / "cfg.json",
        target=target,
        rel_target="cfg.json",
        root=root,
    )
    frozen = Deliverable(
        tree_id="t",
        artifacts=(
            Artifact(
                target="cfg.json",
                tree_id="t",
                source=str(tmp_path / "cfg.json"),
                source_sha256=file_digest(target),
                pre_image_sha256="absent",
            ),
        ),
    )
    assert _installed_digest_mismatch(frozen, spec) == ""

    target.write_text("something the round never validated\n", encoding="utf-8")
    assert "not what was validated" in _installed_digest_mismatch(frozen, spec)


def test_a_revert_that_leaves_a_patched_file_behind_is_named(tmp_path: Path, caplog) -> None:
    # A non-git tree's patch targets are only known once a strip level is
    # detected, so the pre-round baseline cannot cover them. The apply's ledger
    # is the record that can.
    from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor

    root = tmp_path / "wheel"
    root.mkdir()
    target = root / "mod.py"
    target.write_text("pre-round\n", encoding="utf-8")
    backup_root = tmp_path / "backups"
    ledger.append_record(
        backup_root,
        {
            "target": str(target),
            "existed": True,
            "backup_path": str(backup_root / "mod.bak"),
            "revert_action": "restore",
            "pre_image_sha256": file_digest(target),
        },
    )

    executor = IntegratePatchExecutor(session_dir=tmp_path)
    executor._nogit_backup_root = backup_root

    with caplog.at_level("ERROR"):
        executor._log_residual_drift(root)
    assert not caplog.records, "a tree restored to its pre-image was reported as drifted"

    target.write_text("what the revert failed to undo\n", encoding="utf-8")
    with caplog.at_level("ERROR"):
        executor._log_residual_drift(root)
    assert any("still differs from its pre-round state" in r.getMessage() for r in caplog.records)
