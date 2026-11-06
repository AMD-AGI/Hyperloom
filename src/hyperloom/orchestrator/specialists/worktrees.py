# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist's per-task git worktree, and the admin directory the app's own git goes through.

The app's git over a worktree a specialist wrote (the commit it starts from, the harvested patch) reads the admin
directory the repository recorded when the worktree was made, never the one the worktree's ``.git`` names: the
specialist can rewrite that file, and where it runs as the app's own uid (a bubblewrap sandbox, or unconfined) no
ownership check tells an admin directory it made from the app's.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from hyperloom_common.git_safety import recorded_admin_dir

__all__ = ["LinkedWorktree", "PinnedGit", "link_worktree"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinkedWorktree:
    """A per-task worktree (``path``; ``None`` when there is none to use), its admin directory as its repository
    records it, and why there is none."""

    path: Path | None
    admin: Path | None = None
    error: str = ""


@dataclass(frozen=True)
class PinnedGit:
    """How the app's git reads a specialist's worktree: through ``admin``, its admin directory as its repository
    records it, from ``start``, the commit recorded before the specialist ran (empty: that admin directory's HEAD)."""

    admin: Path
    start: str = ""


def link_worktree(base: Path, path: Path, branch: str) -> LinkedWorktree:
    """The worktree at ``path`` on a new ``branch`` off ``base`` (one already there is reused), with its admin
    directory as ``base``'s repository records it; none when the repository records no worktree there, or several."""
    made, error = _setup_worktree(base, path, branch)
    if made is None:
        return LinkedWorktree(None, error=error)
    try:
        return LinkedWorktree(made, recorded_admin_dir(base, made))
    except (OSError, subprocess.SubprocessError) as exc:
        return LinkedWorktree(None, error=f"{made} is not a worktree {base} records: {exc}")


def _setup_worktree(
    base: Path,
    worktree_path: Path,
    branch: str,
) -> tuple[Path | None, str]:
    """Create a fresh git worktree at ``worktree_path`` branched off
    ``base``'s HEAD.

    Best-effort: on git error returns ``(None, err)`` so the caller can
    proceed without isolation or hard-fail.

    Args:
        base: Git checkout the worktree is branched off of.
        worktree_path: Destination path for the new worktree.
        branch: Branch name to create for the worktree.

    Returns:
        A ``(worktree_path, "")`` tuple on success, or ``(None, error)`` on
        git failure.
    """
    if worktree_path.exists():
        # Resume / retry: reuse an existing worktree.
        log.warning(
            "specialist worktree already exists at %s; reusing",
            worktree_path,
        )
        return worktree_path, ""
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git",
        "-C",
        str(base),
        "worktree",
        "add",
        "-b",
        branch,
        str(worktree_path),
    ]
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
            # The ref, gitfile and admin dir git makes here must stay readable by a demoted specialist,
            # whatever umask a launch set on the app.
            umask=0o022,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, f"git worktree add failed to spawn: {exc!r}"
    if cp.returncode != 0:
        return None, (f"git worktree add rc={cp.returncode}: stderr={cp.stderr.strip()[:400]!r}")
    return worktree_path, ""
