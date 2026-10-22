# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A second checkout of the campaign workspace at another commit, for timing that commit beside the candidate.

It lives next to the workspace, never inside it: lanes copy the whole workspace, and an agent may edit anything
inside, so a tree there would be copied into every lane and could be changed by the session it measures.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from kernelforge.loop.editable_repo import needs_inplace
from kernelforge.loop.jit_rebuild import _CPP_EXTS

# Python bytecode the driver writes when it imports the kernel; anything else untracked means the tree was touched.
_SCRATCH = ("__pycache__", ".pyc")


def incumbent_tree_path(workspace: str | Path) -> Path:
    """Where the incumbent's tree of ``workspace`` lives: a sibling folder."""
    workspace = Path(workspace).resolve()
    return workspace.parent / f"{workspace.name}.forge-incumbent"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True).stdout


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DetachedWorktree:
    """A detached git worktree of ``repo`` at ``path``, kept exactly at one commit."""

    def __init__(self, repo: str | Path, path: str | Path) -> None:
        self.repo = Path(repo).resolve()
        self.path = Path(path).resolve()

    def _dirty(self) -> bool:
        lines = _git(self.path, "status", "--porcelain", "--untracked-files=all").splitlines()
        return any(not any(mark in line for mark in _SCRATCH) for line in lines)

    def ensure_at(self, commit: str) -> str:
        """Bring the tree to ``commit`` with nothing else in it; return why it had to be reset, or "" if it was not."""
        if not (self.path / ".git").exists():
            _git(self.repo, "worktree", "prune")
            _git(self.repo, "worktree", "add", "--detach", "--force", str(self.path), commit)
            return "created"
        moved = _git(self.path, "rev-parse", "HEAD").strip() != commit
        dirty = self._dirty()
        if not moved and not dirty:
            return ""
        _git(self.path, "checkout", "--detach", "--force", commit)
        _git(self.path, "clean", "-ffdx")
        return "moved to the new incumbent" if moved and not dirty else "found changed and restored"

    def remove(self) -> None:
        if (self.path / ".git").exists():
            _git(self.repo, "worktree", "remove", "--force", str(self.path))


def tree_copy_refusal(*, workspace: str | Path, driver: str | Path, source_files: list[str]) -> str:
    """Why a second checkout would not run the kernel it holds, or "" when it would.

    An editable install loads the workspace's package whatever tree the driver sits in; a driver outside the
    workspace, or one git does not track, is not in the checkout to run; a C++ JIT source is keyed by environment
    the loop owns for the workspace.
    """
    workspace, driver = Path(workspace).resolve(), Path(driver).resolve()
    if needs_inplace(str(workspace)):
        return "the workspace is an editable install, which every tree would import"
    if not driver.is_relative_to(workspace):
        return f"the driver lives outside the workspace ({driver}), so every tree would run the same one"
    tracked = subprocess.run(
        ["git", "-C", str(workspace), "ls-files", "--error-unmatch", str(driver.relative_to(workspace))],
        capture_output=True,
    )
    if tracked.returncode != 0:
        return "the driver is not committed, so a checkout of another commit would not hold it"
    native = [path for path in source_files if str(path).endswith(_CPP_EXTS)]
    if native:
        return f"C/C++ sources are JIT-built for the workspace: {native[:3]}"
    return ""
