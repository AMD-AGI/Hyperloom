# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Detached worktrees of the campaign repository, one per start node of a round.

A start worktree is where that start's specialists probe, what its lanes are copied from, and where its candidates are
measured and committed. It shares the campaign repository's object store and references, so a commit made in it is a
commit of the campaign repository.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from kernelforge.tools.git import ensure_commit_identity, git

WORKTREE_ROOT_SUFFIX = ".tree-starts"
_DIR_COMMIT_CHARS = 12


class StartWorktrees:
    """The start worktrees of one campaign, kept beside its workspace."""

    def __init__(self, workspace_dir: str | Path):
        self.repo = Path(workspace_dir).resolve()
        self.root = self.repo.parent / f"{self.repo.name}{WORKTREE_ROOT_SUFFIX}"

    def path_of(self, commit: str) -> Path:
        return self.root / commit[:_DIR_COMMIT_CHARS]

    def ensure(self, commit: str) -> Path:
        """A clean worktree checked out at ``commit``, reusing the one a previous round left there."""
        path = self.path_of(commit)
        if path.is_dir() and _head(path) == commit:
            self.restore(path, commit)
            return path
        if path.exists():
            self._remove(path)
        self.root.mkdir(parents=True, exist_ok=True)
        self._worktree("add", "--detach", str(path), commit)
        return path

    def restore(self, path: Path, commit: str) -> None:
        """Put a worktree back on ``commit`` with no tracked or untracked edits; ignored build output stays."""
        git("checkout", "--quiet", "--detach", "--force", commit, cwd=path)
        git("clean", "-fdq", cwd=path)

    def apply(self, path: Path, patch: str) -> bool:
        """Stage one candidate's diff in a worktree, reporting whether it applied cleanly."""
        if not patch.strip():
            return False
        text = patch if patch.endswith("\n") else patch + "\n"
        if git("apply", "--index", "--check", cwd=path, input=text, check=False).returncode != 0:
            return False
        git("apply", "--index", cwd=path, input=text)
        return True

    def staged_tree(self, path: Path) -> str:
        """The file tree a commit of ``path`` would record now."""
        _stage_tracked(path)
        return git("write-tree", cwd=path).stdout.strip()

    def commit(self, path: Path, message: str) -> str:
        """Commit every staged and tracked edit of a worktree and return the new commit."""
        ensure_commit_identity(path)
        before = _head(path)
        _stage_tracked(path)
        git("commit", "--quiet", "-m", message, cwd=path)
        after = _head(path)
        if after == before:
            raise RuntimeError(f"git commit did not advance {path}")
        return after

    def release(self, keep: Iterable[str]) -> None:
        """Remove every start worktree whose commit is not in ``keep``."""
        kept = {self.path_of(commit) for commit in keep}
        for path in self._existing():
            if path not in kept:
                self._remove(path)

    def clear(self) -> None:
        """Remove every start worktree, including those an interrupted session left behind."""
        for path in self._existing():
            self._remove(path)
        self._worktree("prune")

    def _existing(self) -> list[Path]:
        return sorted(item for item in self.root.iterdir() if item.is_dir()) if self.root.is_dir() else []

    def _remove(self, path: Path) -> None:
        self._worktree("remove", "--force", str(path), check=False)
        if path.exists():
            shutil.rmtree(path)
        self._worktree("prune")

    def _worktree(self, *args: str, check: bool = True) -> None:
        git("worktree", *args, cwd=self.repo, check=check)


def _head(path: Path) -> str:
    return git("rev-parse", "HEAD", cwd=path).stdout.strip()


def _stage_tracked(path: Path) -> None:
    """Stage every edit of a tracked file; a new file is never a node's."""
    git("add", "-u", cwd=path)


def untracked_outside(workspace_dir: str | Path, allowed: Iterable[str]) -> list[str]:
    """Untracked, unignored paths of a workspace outside ``allowed``; a worktree would not see them."""
    listed = git("ls-files", "--others", "--exclude-standard", cwd=workspace_dir).stdout
    prefixes = tuple(f"{item.rstrip('/')}/" for item in allowed)
    return [line for line in listed.splitlines() if line and not line.startswith(prefixes)]


__all__ = ["StartWorktrees", "WORKTREE_ROOT_SUFFIX", "untracked_outside"]
