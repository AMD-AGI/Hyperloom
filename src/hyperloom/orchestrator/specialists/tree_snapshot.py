# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session-owned git history for a framework tree that has none of its own."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

#: Build output and caches: the specialist edits source, and a worktree carrying
#: a framework's shared objects would cost gigabytes on every dispatch.
_EXCLUDES: tuple[str, ...] = ("__pycache__/", "*.pyc", "*.pyo", "*.so", "*.so.*")

_IDENTITY: tuple[str, ...] = ("-c", "user.name=hyperloom", "-c", "user.email=hyperloom@localhost")

_GIT_TIMEOUT_SEC = 600.0

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(repo: Path) -> threading.Lock:
    """The lock serialising snapshots into ``repo``; specialists dispatch concurrently."""
    with _locks_guard:
        return _locks.setdefault(str(repo), threading.Lock())


def _git(repo: Path, tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git against ``repo`` with ``tree`` as its work tree."""
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "-c",
            f"safe.directory={tree}",
            "--git-dir",
            str(repo),
            "--work-tree",
            str(tree),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SEC,
        check=False,
    )


def _failure(action: str, cp: subprocess.CompletedProcess[str]) -> str:
    """Describe a failed git step."""
    return f"git {action} rc={cp.returncode}: {(cp.stderr or cp.stdout).strip()[:400]}"


def snapshot_tree(tree: Path, repo: Path, *, refresh: bool) -> tuple[Path | None, str]:
    """Commit ``tree`` as it is now into the bare repository ``repo``.

    The repository lives outside ``tree`` and nothing is written into the tree,
    so it keeps reading as the plain directory every other consumer takes it
    for. A worktree branched off ``repo`` holds the tree's files at the same
    relative paths, so a diff harvested from it applies to ``tree`` itself.

    Args:
        tree: The directory to snapshot.
        repo: The bare repository holding the history; created on first use.
        refresh: Commit ``tree``'s current content. When false, an existing
            snapshot is branched off as it stands: the tree is mid-integrate
            and would hand the specialist a candidate no decision has kept.

    Returns:
        ``(repo, "")`` once ``repo`` has a commit to branch off, else
        ``(None, error)``.
    """
    with _lock(repo):
        try:
            fresh = not (repo / "HEAD").is_file()
            if fresh and not refresh:
                return None, f"{tree} is being integrated into and has no snapshot to branch off yet"
            if not refresh:
                return repo, ""
            if fresh:
                repo.parent.mkdir(parents=True, exist_ok=True)
                cp = subprocess.run(
                    ["git", "init", "--bare", "-q", str(repo)],
                    capture_output=True,
                    text=True,
                    timeout=_GIT_TIMEOUT_SEC,
                    check=False,
                )
                if cp.returncode != 0:
                    return None, _failure("init", cp)
                (repo / "info").mkdir(parents=True, exist_ok=True)
                (repo / "info" / "exclude").write_text("".join(f"{p}\n" for p in _EXCLUDES), encoding="utf-8")
            cp = _git(repo, tree, "add", "-A", "--", ".")
            if cp.returncode != 0:
                return None, _failure("add", cp)
            staged = _git(repo, tree, "diff", "--cached", "--quiet")
            if fresh or staged.returncode != 0:
                cp = _git(repo, tree, *_IDENTITY, "commit", "-q", "--no-verify", "--allow-empty", "-m", "snapshot")
                if cp.returncode != 0:
                    return None, _failure("commit", cp)
        except (OSError, subprocess.SubprocessError) as exc:
            return None, f"snapshot of {tree} failed: {exc!r}"
    return repo, ""


__all__ = ["snapshot_tree"]
