# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist's per-task git worktree, and the admin directory the app's own git goes through.

The app's git over a worktree a specialist wrote (the commit it starts from, the harvested patch) reads the admin
directory the repository recorded when the worktree was made, never the one the worktree's ``.git`` names: the
specialist can rewrite that file, and where it runs as the app's own uid (a bubblewrap sandbox, or unconfined) no
ownership check tells an admin directory it made from the app's.
"""

from __future__ import annotations

import errno
import logging
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from hyperloom_common.git_safety import recorded_admin_dir, worktree_head

__all__ = ["HeldWorktree", "LinkedWorktree", "PinnedGit", "link_worktree"]

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


@dataclass(frozen=True)
class HeldWorktree:
    """A specialist's worktree as it stood before the agent ran: its path, the device and inode of the directory there
    (``None`` when the path was not a directory of its own: missing, or a link), and ``git``, how the app's git reads
    it (``None``: it does not).

    The app's git, its patch scan and its patch write reach the worktree by path, as the app and outside any sandbox.
    An agent of the app's own uid writes the workspace the worktree sits in, so it can rename the worktree and leave a
    link to any directory of the host in its place, which the app would then diff, scan and write. So the app reads
    the worktree after the run only while its path still names the directory recorded here.
    """

    path: Path
    inode: tuple[int, int] | None
    git: PinnedGit | None = None

    @classmethod
    def before_launch(cls, path: Path, admin: Path | None = None) -> HeldWorktree:
        """Record the directory at ``path`` and, through ``admin`` (its admin directory as its repository records it)
        when given, the commit it sits at, before anything of the agent's runs. No git runs over a path that is not a
        directory of its own."""
        inode = _directory_inode(path)
        if admin is None or inode is None:
            return cls(path, inode)
        return cls(path, inode, PinnedGit(admin, worktree_head(path, admin=admin)))

    def intact(self) -> bool:
        """Whether ``path`` still names the directory recorded."""
        return self.inode is not None and _directory_inode(self.path) == self.inode

    def read_text(self, name: str) -> str:
        """The UTF-8 text of the regular file ``name`` in the worktree, read through the directory recorded and no
        other: the worktree is opened without following a link and that open directory checked, so a swap after any
        earlier check cannot send the read elsewhere.

        Raises:
            OSError: ``path`` no longer names the directory recorded (``ESTALE``), or ``name`` is not a regular file
                there.
        """
        dir_fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(dir_fd)
            if (info.st_dev, info.st_ino) != self.inode:
                raise OSError(errno.ESTALE, "not the worktree the specialist was given", str(self.path))
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dir_fd)
        finally:
            os.close(dir_fd)
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError(errno.EINVAL, "not a regular file", str(self.path / name))
            return handle.read().decode("utf-8")

    def refusal(self) -> str:
        """``""`` while ``path`` names the directory recorded; else why nothing may be read through it.

        A link found at ``path`` then is removed (never what it names), so nothing that reads the workspace later
        reaches the directory it named either.
        """
        if self.intact():
            return ""
        if self.inode is None:
            reason = f"worktree_refused: {self.path} was not a directory of its own when the specialist started"
        else:
            reason = f"worktree_refused: {self.path} is no longer the directory the specialist was given"
        if _remove_link(self.path):
            reason += "; the link left in its place was removed"
        log.warning("specialist %s; nothing is collected from it", reason)
        return reason


def _directory_inode(path: Path) -> tuple[int, int] | None:
    """The device and inode of the directory at ``path`` itself, or ``None`` when there is none (a link is none)."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return (info.st_dev, info.st_ino) if stat.S_ISDIR(info.st_mode) else None


def _remove_link(path: Path) -> bool:
    """Remove ``path`` when it is a symbolic link; whether one was removed."""
    try:
        if not stat.S_ISLNK(os.lstat(path).st_mode):
            return False
        os.unlink(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        log.warning("specialist worktree: cannot remove the link at %s: %r", path, exc)
        return False
    return True


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
