# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the specialist worktree git helpers.

The legacy ``run_bench`` micro-bench surface has been removed (GPU specialists
now run real serving / benchmark / autotune loops on their own leased cards),
so only the worktree git helpers remain here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


from inference_optimizer.orchestrator import specialist_bench as sb


def _git(repo: Path, *args: str) -> None:
    """Run a git command in repo, raising on failure."""
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(repo: Path) -> None:
    """Initialize a git repo with one committed file f.txt='a\\n'."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.com")
    _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("a\n", encoding="utf-8")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "init")


# ---- envelopes ----


def test_error_and_ok():
    e = sb._error("bad", x=1)
    assert e == {"ok": False, "reason": "bad", "x": 1}
    assert sb._ok() == {"ok": True}
    assert sb._ok({"a": 2}) == {"ok": True, "a": 2}


# ---- apply_patch_in_worktree ----


def test_apply_patch_empty():
    assert sb.apply_patch_in_worktree(Path("/tmp"), "")["reason"] == "empty_patch"


def test_apply_patch_worktree_missing(tmp_path):
    out = sb.apply_patch_in_worktree(tmp_path / "nope", "diff --git a/x b/x\n")
    assert out["reason"] == "worktree_missing"


def test_apply_patch_path_escape(tmp_path):
    _init_repo(tmp_path)
    patch = "--- a/../escape\n+++ b/../escape\n"
    out = sb.apply_patch_in_worktree(tmp_path, patch)
    assert out["reason"] == "patch_path_escapes_worktree"


def test_apply_patch_success(tmp_path):
    _init_repo(tmp_path)
    # Build a real patch via git, then revert so apply re-applies it.
    (tmp_path / "f.txt").write_text("b\n", encoding="utf-8")
    diff = subprocess.run(
        ["git", "-C", str(tmp_path), "diff"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    _git(tmp_path, "checkout", "--", "f.txt")
    out = sb.apply_patch_in_worktree(tmp_path, diff)
    assert out["ok"] is True
    assert (tmp_path / "f.txt").read_text() == "b\n"


def test_apply_patch_rejected(tmp_path):
    _init_repo(tmp_path)
    bad = "diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-does-not-match\n+new\n"
    out = sb.apply_patch_in_worktree(tmp_path, bad)
    assert out["reason"] == "git_apply_rejected"


# ---- capture_worktree_cumulative_diff ----


def test_capture_diff_not_a_dir(tmp_path):
    assert sb.capture_worktree_cumulative_diff(tmp_path / "nope") is None


def test_capture_diff_clean_repo(tmp_path):
    _init_repo(tmp_path)
    assert sb.capture_worktree_cumulative_diff(tmp_path) == ""


def test_capture_diff_dirty_repo(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("changed\n", encoding="utf-8")
    diff = sb.capture_worktree_cumulative_diff(tmp_path)
    assert diff and "changed" in diff


def test_capture_diff_non_repo(tmp_path):
    # An existing dir that is not a git repo -> git fails -> None.
    assert sb.capture_worktree_cumulative_diff(tmp_path) is None


# ---- reset_worktree ----


def test_reset_worktree_not_a_dir(tmp_path):
    # Should not raise.
    sb.reset_worktree(tmp_path / "nope")


def test_reset_worktree_restores(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("dirty\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("x\n", encoding="utf-8")
    sb.reset_worktree(tmp_path)
    assert (tmp_path / "f.txt").read_text() == "a\n"
    assert not (tmp_path / "untracked.txt").exists()
