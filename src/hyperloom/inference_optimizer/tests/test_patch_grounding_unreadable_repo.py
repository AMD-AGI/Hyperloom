# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A repository git cannot read says nothing about the patch it was asked to check."""

from __future__ import annotations

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


def _ground(patch_text: str, base: Path) -> ps.PatchGroundingResult:
    return ps.ground_patch_text(patch_text, base_checkout=base)


def test_a_patch_that_does_not_apply_is_stale(tmp_path):
    root = _repo(tmp_path)

    result = _ground(_PATCH.format(old="SOMETHING-ELSE"), root)

    assert result.verdict == ps.GROUND_STALE


def test_a_patch_that_applies_is_grounded(tmp_path):
    root = _repo(tmp_path)

    result = _ground(_PATCH.format(old="original"), root)

    assert result.verdict == ps.GROUND_APPLIES


def test_a_worktree_whose_parent_is_gone_is_unchecked_not_stale(tmp_path):
    """The specialist worktrees are git worktrees of a checkout inside the pod.

    Read anywhere else -- an archived session, a later leg, an operator on the
    login node -- git exits 128 before it reads the patch. Recorded as stale,
    every patch in the set is labelled broken on evidence about the machine,
    and the note reaches whoever decides whether the round is worth running.
    """
    root = _repo(tmp_path)
    worktree = tmp_path / "wt"
    _git("worktree", "add", "-q", str(worktree), "-b", "wtb", cwd=root)
    # What preemption leaves behind: the worktree's files, and no parent to resolve.
    subprocess.run(["rm", "-rf", str(root / ".git" / "worktrees")], check=True)

    result = _ground(_PATCH.format(old="original"), worktree)

    assert result.verdict == ps.GROUND_UNCHECKED
    assert "not a git repository" in result.detail
