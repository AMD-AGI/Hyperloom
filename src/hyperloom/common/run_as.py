# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Start a child process as a separate, non-root user.

Env scrubbing keeps credentials out of a specialist's environment; this module
adds the OS boundary: a specialist that runs as its own user cannot read the
app's ``/proc/<pid>/environ``, state directory or key files.

The host names the user with ``HYPERLOOM_SPECIALIST_USER``. Unset or blank
means no change at all: :func:`specialist_run_as` returns ``None`` and a
launcher starts the child exactly as it did before.

A requested demote that cannot be honoured is an error, never a quiet fallback
to the app's own user: an unknown user, uid 0, a primary or supplementary group
of gid 0 and the app's own user are all refused with :class:`RunAsError`.

The switch is made by ``subprocess.Popen(user=, group=, extra_groups=)``
(:meth:`RunAs.popen_kwargs`), which runs in the forked child before ``exec``
without a Python ``preexec_fn``; ``preexec_fn`` is unsafe in a process that
runs threads, and the app runs MCP and event threads.
"""

from __future__ import annotations

import grp
import os
import pwd
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, MutableMapping

from hyperloom.common.env import env_str
from hyperloom.common.nofollow import read_text_nofollow

SPECIALIST_USER_ENV = "HYPERLOOM_SPECIALIST_USER"
_GIT_NAME = ".git"

# New files made by the child are group-writable, so the app (a member of the
# specialist's group, or root) can still read and rewrite them.
_CHILD_UMASK = 0o007
_GROUP_RW = stat.S_IRGRP | stat.S_IWGRP
_GROUP_RWX_SETGID = _GROUP_RW | stat.S_IXGRP | stat.S_ISGID
_GROUP_R = stat.S_IRGRP
_GROUP_RX_SETGID = _GROUP_R | stat.S_IXGRP | stat.S_ISGID
# The app's own umask once a specialist user is set: the group may read and write,
# others get nothing. What the app creates (a sync's reset and apply rebuild
# worktree files as the app) stays group-writable.
_APP_UMASK = _CHILD_UMASK


class RunAsError(ValueError):
    """The requested specialist user cannot be used; the launch must not proceed."""


@dataclass(frozen=True)
class RunAs:
    """What a launcher needs to start a child as one named user."""

    user: str
    uid: int
    gid: int
    groups: tuple[int, ...]
    home: str

    def popen_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``subprocess.Popen`` that start the child as this user."""
        return {
            "user": self.uid,
            "group": self.gid,
            "extra_groups": list(self.groups),
            "umask": _CHILD_UMASK,
        }

    def apply_env(self, env: MutableMapping[str, str], *, safe_dirs: Iterable[Path] = ()) -> None:
        """Point the identity variables at this user, in place.

        The app's ``HOME`` is not readable by this user. ``safe_dirs`` are
        marked ``safe.directory`` for git (command-line scope, the only scope
        git honours for it), because the child is not the owner of a tree the
        app created.
        """
        env["HOME"] = self.home
        env["USER"] = env["LOGNAME"] = self.user
        if "USERNAME" in env:
            env["USERNAME"] = self.user
        safe = [str(p) for p in safe_dirs]
        env["GIT_CONFIG_COUNT"] = str(len(safe))
        for i, path in enumerate(safe):
            env[f"GIT_CONFIG_KEY_{i}"] = "safe.directory"
            env[f"GIT_CONFIG_VALUE_{i}"] = path

    def share_worktree_git(self, worktree: Path) -> None:
        """Let this user read the git metadata of a linked worktree, and nothing more.

        ``git worktree add`` creates the ``<worktree>/.git`` gitfile and its
        ``<repo>/.git/worktrees/<name>`` admin dir under the app's group and,
        once the app's umask is 007, without other-read, so the child's git
        would fail. Both are shared read-only, as is search (not write) on the
        ``worktrees`` directory above the admin dir. A worktree with no gitfile (a snapshot)
        needs nothing.

        The gitfile sits in a tree the child can rewrite, so its ``gitdir:``
        line is believed only when the admin dir it names points back at this
        worktree's gitfile (the admin dir is the app's); a gitfile that lies is
        refused with ``OSError`` rather than shared.
        """
        gitfile = worktree / _GIT_NAME
        if not gitfile.is_file() or gitfile.is_symlink():
            return
        first = read_text_nofollow(gitfile).strip()
        if not first.startswith("gitdir:"):
            raise OSError(f"refusing to share {gitfile}: not a gitdir pointer")
        admin = Path(first.removeprefix("gitdir:").strip())
        back = read_text_nofollow(admin / "gitdir").strip()
        if admin.parent.name != "worktrees" or os.path.realpath(back) != os.path.realpath(gitfile):
            raise OSError(f"refusing to share {admin}: it is not the admin dir of {worktree}")
        self.share_tree(gitfile, write=False)
        self.share_tree(admin, write=False)
        parent = admin.parent
        os.chown(parent, -1, self.gid)
        # Search, not write: a group that could rename entries here could break every other task's git.
        parent.chmod((stat.S_IMODE(parent.stat().st_mode) | stat.S_IRGRP | stat.S_IXGRP) & ~stat.S_IWGRP)

    def share_tree(self, root: Path, *, sticky: bool = False, write: bool = True) -> None:
        """Give this user's group read-write access to everything under ``root``.

        The tree stays owned by the app. Its group becomes this user's group,
        directories get the setgid bit (new entries keep the group) and every
        entry gets group read and write (``write=False``: group read only,
        plus search on directories, for a tree the child only reads, such as
        a worktree's git metadata). ``sticky`` additionally stops the child
        deleting the app's own files in the top directory. Symlinks are
        not followed, and a ``root`` that is itself a symlink is refused.
        ``.git`` is left alone: the child does no git writes. Needs root or
        membership in the group; raises ``OSError``
        otherwise, so a launcher cannot start a child that cannot write.
        """
        if root.is_symlink():
            raise OSError(f"refusing to share {root}: it is a symlink")
        for path in _walk(root):
            st = path.lstat()
            if st.st_gid != self.gid:
                os.chown(path, -1, self.gid, follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode):
                continue
            if stat.S_ISDIR(st.st_mode):
                add = _GROUP_RWX_SETGID if write else _GROUP_RX_SETGID
            else:
                add = _GROUP_RW if write else _GROUP_R
            if sticky and path == root:
                add |= stat.S_ISVTX
            mode = stat.S_IMODE(st.st_mode) | add
            if not write:
                mode &= ~stat.S_IWGRP  # read-only means the group cannot write what an earlier umask let it
            if mode != stat.S_IMODE(st.st_mode):
                path.chmod(mode)


def _walk(root: Path) -> Iterable[Path]:
    """``root`` and everything below it, never following a link and never entering or touching ``.git``."""
    yield root
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d != _GIT_NAME]
        base = Path(dirpath)
        for name in (*dirnames, *(f for f in filenames if f != _GIT_NAME)):
            yield base / name


def resolve_run_as(user: str | None) -> RunAs | None:
    """Look ``user`` up now and return how to run a child as it; ``None`` for no demote.

    Raises:
        RunAsError: The user does not exist, is uid 0, is the app's own user,
            or belongs to group 0.
    """
    name = (user or "").strip()
    if not name:
        return None
    try:
        entry = pwd.getpwnam(name)
    except KeyError:
        raise RunAsError(f"{SPECIALIST_USER_ENV}={name!r}: no such user") from None
    if entry.pw_uid == 0:
        raise RunAsError(f"{SPECIALIST_USER_ENV}={name!r} is uid 0; specialists must not run as root")
    if entry.pw_uid == os.geteuid():
        raise RunAsError(f"{SPECIALIST_USER_ENV}={name!r} is the app's own user; no boundary would exist")
    groups = tuple(os.getgrouplist(entry.pw_name, entry.pw_gid))
    if 0 in groups:
        root_group = grp.getgrgid(0).gr_name
        raise RunAsError(f"{SPECIALIST_USER_ENV}={name!r} is in group {root_group!r} (gid 0)")
    return RunAs(user=entry.pw_name, uid=entry.pw_uid, gid=entry.pw_gid, groups=groups, home=entry.pw_dir)


def specialist_run_as() -> RunAs | None:
    """Resolve ``HYPERLOOM_SPECIALIST_USER``; ``None`` (today's behaviour) when unset or blank. No side effects."""
    return resolve_run_as(env_str(SPECIALIST_USER_ENV))


def set_group_shared_umask() -> None:
    """Make this process's own umask grant its group full access and others none (0o007).

    A launcher calls it once it has decided to demote, so files the app itself
    (re)creates in a specialist's worktree, such as a sync's reset and apply,
    stay group-writable. Process-wide; never call it when no user was named.
    """
    os.umask(_APP_UMASK)


_CAPABILITIES = {"CAP_KILL": 5, "CAP_SETGID": 6, "CAP_SETUID": 7}


def missing_capabilities(effective: int) -> list[str]:
    """The launch capabilities absent from the effective capability bitmask ``effective``."""
    return [name for name, bit in _CAPABILITIES.items() if not effective >> bit & 1]


def missing_launch_capabilities() -> list[str]:
    """Capabilities the app lacks to start, and later stop, a child that runs as another user.

    ``setuid``/``setgid`` start it; ``kill`` is what lets the reap paths (deadline,
    stale heartbeat, done file, cancel) signal a process of another uid, and
    without it every reap ends unconfirmed. Root has all three unless they were
    dropped, so the effective set is read, not inferred from the uid. When it
    cannot be read, root is assumed to have them and anyone else to have none.
    """
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        effective = next(int(line.split()[1], 16) for line in status.splitlines() if line.startswith("CapEff:"))
    except (OSError, StopIteration, ValueError, IndexError):
        return [] if os.geteuid() == 0 else list(_CAPABILITIES)
    return missing_capabilities(effective)
