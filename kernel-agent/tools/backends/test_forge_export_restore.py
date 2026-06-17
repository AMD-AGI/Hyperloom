#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression test for forge in-place export/restore covering ALL changed files.

Reproduces the run6 bug (Issue 5): the agent's winning change landed in a
SIBLING tracked file (a *_config.py defaults module), not in source_file. The
old code only exported/restored source_file, so:
  - the exported artifact was byte-identical to the original (no optimization),
  - the sibling file was left dirty in the live repo.

These tests drive forge_submit's _prepare_inplace / _export_best_artifacts /
_restore_inplace against a synthetic git repo (no GPU / no claude) and assert:
  1. export captures the sibling file + a non-empty forge.patch,
  2. restore reverts BOTH the kernel and the sibling file to pre-forge content,
  3. a pre-existing dirty tracked file is preserved (still dirty) after restore,
  4. an untracked file is never touched,
  5. the repo ends on its original branch with no forge/* temp branch.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forge_submit  # noqa: E402


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args],
                          capture_output=True, text=True).stdout.strip()


def _commit_all(repo: str, msg: str) -> None:
    """Mirror IterationLoop._git_commit (git add -u + commit)."""
    subprocess.run(["git", "-C", repo, "add", "-u"], capture_output=True)
    subprocess.run(["git", "-C", repo, "commit", "-m", msg], capture_output=True)


def _make_repo(tmp: Path) -> dict:
    repo = tmp / "repo"
    repo.mkdir()
    r = str(repo)
    _git(r, "init", "-b", "main")
    subprocess.run(["git", "-C", r, "config", "user.name", "t"], capture_output=True)
    subprocess.run(["git", "-C", r, "config", "user.email", "t@t"], capture_output=True)
    kernel = repo / "kernel.py"
    config = repo / "kernel_config.py"
    other = repo / "other.txt"
    kernel.write_text("KERNEL_ORIGINAL\n")
    config.write_text("BLOCK_SIZE = 64\n")
    other.write_text("committed_v1\n")
    subprocess.run(["git", "-C", r, "add", "."], capture_output=True)
    subprocess.run(["git", "-C", r, "commit", "-m", "initial"], capture_output=True)
    # Pre-existing dirty tracked file (must survive a forge run untouched).
    other.write_text("committed_v1\nPRE_EXISTING_DIRTY\n")
    # Pre-existing untracked file (must never be touched).
    (repo / "untracked.txt").write_text("untracked\n")
    return {"repo": r, "kernel": kernel, "config": config, "other": other,
            "untracked": repo / "untracked.txt"}


def test_export_and_restore_cover_sibling_file():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        env = _make_repo(tmp)
        repo, kernel, config, other, untracked = (
            env["repo"], env["kernel"], env["config"], env["other"], env["untracked"])
        out = tmp / "out"
        out.mkdir()
        branch = "forge/test/kernel"

        orig_branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        assert orig_branch == "main"

        prep = forge_submit._prepare_inplace(str(kernel), repo, branch)
        assert prep is not None, "prepare_inplace should succeed on a clean git repo"
        workspace, worktree_kernel, restore = prep
        base_commit = restore["base_commit"]
        # base_commit must have absorbed the pre-existing dirty file.
        assert base_commit, "base_commit must be set"

        # --- simulate the loop: winning edit lands in the SIBLING config file ---
        # iter1 (kept): config 64 -> 128, kernel untouched.
        config.write_text("BLOCK_SIZE = 128\n")
        _commit_all(repo, "iter1: tune config")
        # iter2 (reverted): config 128 -> 256, then revert.
        config.write_text("BLOCK_SIZE = 256\n")
        _commit_all(repo, "iter2: worse")
        subprocess.run(["git", "-C", repo, "revert", "--no-edit", "HEAD"],
                       capture_output=True)
        # best-kept state on disk now: config == 128.
        assert "128" in config.read_text()

        # --- export ---
        primary, changed = forge_submit._export_best_artifacts(
            workspace, base_commit, str(kernel), str(kernel), out)
        # The sibling config file must be in the changed set + exported.
        assert "kernel_config.py" in changed, f"changed should include config: {changed}"
        exported_cfg = out / "optimized_versions" / "files" / "kernel_config.py"
        assert exported_cfg.is_file(), "config file must be exported under files/"
        assert "128" in exported_cfg.read_text(), "exported config must carry the optimization"
        patch = (out / "optimized_versions" / "forge.patch").read_text()
        assert "BLOCK_SIZE" in patch and "128" in patch, "patch must contain the real change"

        # --- restore ---
        forge_submit._restore_inplace(restore)

        # 1. kernel + config reverted to pre-forge content.
        assert kernel.read_text() == "KERNEL_ORIGINAL\n"
        assert config.read_text() == "BLOCK_SIZE = 64\n", "config must be restored to pre-forge"
        # 2. pre-existing dirty file preserved (still dirty).
        assert other.read_text() == "committed_v1\nPRE_EXISTING_DIRTY\n"
        status = _git(repo, "status", "--porcelain")
        assert "other.txt" in status, f"pre-existing dirty must remain dirty: {status!r}"
        assert "kernel_config.py" not in status, f"config must be clean after restore: {status!r}"
        # 3. untracked file untouched.
        assert untracked.read_text() == "untracked\n"
        # 4. back on original branch, no forge temp branch.
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
        branches = _git(repo, "branch", "--list", branch)
        assert branches.strip() == "", f"temp branch must be deleted: {branches!r}"
        print("PASS test_export_and_restore_cover_sibling_file")


def test_restore_from_detached_head_preserves_dirty():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        env = _make_repo(tmp)
        repo, kernel, config, other = (
            env["repo"], env["kernel"], env["config"], env["other"])
        branch = "forge/test/kernel"
        # Detach HEAD at the current commit (the run6 sglang scenario).
        head = _git(repo, "rev-parse", "HEAD")
        subprocess.run(["git", "-C", repo, "checkout", "--detach", head],
                       capture_output=True)
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"

        prep = forge_submit._prepare_inplace(str(kernel), repo, branch)
        assert prep is not None
        _, _, restore = prep
        assert restore["orig_branch"] == "HEAD"

        config.write_text("BLOCK_SIZE = 128\n")
        _commit_all(repo, "iter1")

        forge_submit._restore_inplace(restore)

        # Restored content + detached at original commit + dirty preserved.
        assert config.read_text() == "BLOCK_SIZE = 64\n"
        assert _git(repo, "rev-parse", "HEAD") == head
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
        assert "PRE_EXISTING_DIRTY" in other.read_text()
        assert _git(repo, "branch", "--list", branch).strip() == ""
        print("PASS test_restore_from_detached_head_preserves_dirty")


def test_default_branch_resolves_main():
    with tempfile.TemporaryDirectory() as td:
        env = _make_repo(Path(td))
        # No origin/HEAD configured -> falls back to a local 'main'/'master'.
        assert forge_submit._default_branch(env["repo"]) == "main"
        print("PASS test_default_branch_resolves_main")


def test_prepare_inplace_autorecovers_from_stale_forge_branch():
    """A prior run SIGKILL'd before _restore_inplace leaves the repo stranded on
    its forge/<ts>/... temp branch with an un-reverted optimization. The next
    _prepare_inplace must AUTO-RECOVER (force-checkout the default branch, drop
    the stale branch, snapshot a pristine baseline) instead of refusing."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        env = _make_repo(tmp)
        repo, kernel = env["repo"], env["kernel"]

        # Simulate the crashed run: strand HEAD on a forge/ temp branch whose tip
        # carries a leftover (never-restored) optimization.
        stale = "forge/20990101T000000Z/kernel"
        subprocess.run(["git", "-C", repo, "checkout", "-b", stale], capture_output=True)
        kernel.write_text("KERNEL_OPTIMIZED_LEFTOVER\n")
        _commit_all(repo, "forge: leftover optimization from crashed run")
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == stale

        branch = "forge/newrun/kernel"
        prep = forge_submit._prepare_inplace(str(kernel), repo, branch)
        assert prep is not None, "auto-recover must yield a usable prep, not None"
        _, _, restore = prep

        # Recovery force-checked-out main, so the leftover optimization is gone
        # and the baseline snapshot is pristine.
        assert kernel.read_text() == "KERNEL_ORIGINAL\n", "must recover to pristine kernel"
        # Stale forge branch deleted; restore targets the recovered default branch.
        assert _git(repo, "branch", "--list", stale).strip() == "", "stale forge branch must be deleted"
        assert restore["orig_branch"] == "main", f"orig_branch should be recovered main: {restore}"

        forge_submit._restore_inplace(restore)
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
        assert _git(repo, "branch", "--list", branch).strip() == "", "run temp branch must be deleted"
        print("PASS test_prepare_inplace_autorecovers_from_stale_forge_branch")


if __name__ == "__main__":
    test_export_and_restore_cover_sibling_file()
    test_restore_from_detached_head_preserves_dirty()
    test_default_branch_resolves_main()
    test_prepare_inplace_autorecovers_from_stale_forge_branch()
    print("ALL TESTS PASSED")
