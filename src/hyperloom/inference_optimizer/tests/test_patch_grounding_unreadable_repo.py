# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A repository git cannot open says nothing about the patch; a patch git rejects is still stale."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from hyperloom.orchestrator.specialists import patch_safety as ps

_PATCH = (
    "diff --git a/f.txt b/f.txt\n"
    "index 1111111..2222222 100644\n"
    "--- a/f.txt\n"
    "+++ b/f.txt\n"
    "@@ -1 +1 @@\n"
    "-{old}\n"
    "+changed\n"
)

# The hunk header promises three lines the body does not carry: ``git apply`` exits 128 on it.
_CORRUPT_PATCH = "diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n-original\n+changed\n"


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git("init", "-q", ".", cwd=root)
    (root / "f.txt").write_text("original\n", encoding="utf-8")
    _git("add", "f.txt", cwd=root)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init", cwd=root)
    return root


def _ground(patch_text: str, base: Path) -> str:
    patch = base.parent / "candidate.diff"
    patch.write_text(patch_text, encoding="utf-8")
    _kept, _ungrounded, grounding, _multi_root = ps.vet_patches([str(patch)], base_checkout=base)
    return grounding[str(patch)]


def test_a_patch_that_does_not_apply_is_stale(tmp_path):
    root = _repo(tmp_path)

    assert _ground(_PATCH.format(old="SOMETHING-ELSE"), root) == ps.GROUND_STALE


def test_a_patch_that_applies_is_grounded(tmp_path):
    root = _repo(tmp_path)

    assert _ground(_PATCH.format(old="original"), root) == ps.GROUND_APPLIES


def test_a_worktree_whose_parent_is_gone_is_unchecked_not_stale(tmp_path):
    root = _repo(tmp_path)
    worktree = tmp_path / "wt"
    _git("worktree", "add", "-q", str(worktree), "-b", "wtb", cwd=root)
    shutil.rmtree(root / ".git" / "worktrees")

    assert _ground(_PATCH.format(old="SOMETHING-ELSE"), worktree) == ps.GROUND_UNCHECKED


def test_a_corrupt_patch_on_a_usable_repo_is_stale(tmp_path):
    root = _repo(tmp_path)

    assert _ground(_CORRUPT_PATCH, root) == ps.GROUND_STALE


def test_a_patch_that_does_not_apply_to_a_plain_tree_is_stale(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "f.txt").write_text("original\n", encoding="utf-8")

    assert _ground(_PATCH.format(old="SOMETHING-ELSE"), tree) == ps.GROUND_STALE
