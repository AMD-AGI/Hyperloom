# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the app's own git against a worktree a less-privileged account can rewrite.

The account that runs a specialist owns the group-write bit on its worktree, so
it can replace the ``.git`` gitfile, or turn it into a directory, and fill it
with a repository whose configuration names programs (``core.fsmonitor``,
hooks, filter and textconv drivers). ``git -C <worktree>`` run by the app would
execute them as the app, which is often root.

:func:`app_git` therefore never lets git discover the repository from the
worktree. It names the worktree's admin directory itself (``--git-dir``) after
:func:`trusted_admin_dir` has shown that the directory is the app's: a real
directory owned by the app, under a ``worktrees`` directory of a repository
the app owns, whose ``gitdir`` file points back at this worktree's gitfile.
The account cannot create such a directory, because it cannot create
app-owned files. Config scopes the account can reach are switched off, and
programs git would still run from configuration (the fsmonitor hook, hooks, an
external diff) are set to nothing on the command line.
"""

from __future__ import annotations

import errno
import os
import stat
import subprocess
from pathlib import Path

from hyperloom.common.nofollow import read_text_nofollow

_GITFILE = ".git"
_DEFAULT_TIMEOUT_SEC = 120.0
# Command-line scope outranks every file: nothing a repository's config says can start a program through these.
_NO_PROGRAMS = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "diff.external=",
    "-c",
    "core.attributesFile=/dev/null",
    "-c",
    "core.pager=cat",
)


def _app_owned_dir(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise OSError(errno.EPERM, "not a directory owned by the app", str(path))


def trusted_admin_dir(worktree: Path) -> Path:
    """The admin directory of the linked worktree ``worktree``, proven to be the app's.

    Raises:
        OSError: ``.git`` is not a gitfile, or the directory it names is not an
            app-owned admin directory of a repository whose ``worktrees`` entry
            points back at this gitfile.
    """
    gitfile = worktree / _GITFILE
    first = read_text_nofollow(gitfile).strip()
    if not first.startswith("gitdir:"):
        raise OSError(errno.EINVAL, "not a gitdir pointer", str(gitfile))
    admin = Path(first.removeprefix("gitdir:").strip())
    if not admin.is_absolute():
        admin = worktree / admin
    _app_owned_dir(admin)
    _app_owned_dir(admin.parent)
    _app_owned_dir(admin.parent.parent)
    back = admin / "gitdir"
    if admin.parent.name != "worktrees" or back.lstat().st_uid != os.geteuid():
        raise OSError(errno.EPERM, "not an admin dir of a linked worktree", str(admin))
    if os.path.realpath(read_text_nofollow(back).strip()) != os.path.realpath(gitfile):
        raise OSError(errno.EPERM, "the admin dir does not belong to this worktree", str(admin))
    return admin


def app_git(worktree: Path, *args: str, timeout: float = _DEFAULT_TIMEOUT_SEC) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` in ``worktree`` against its trusted admin dir, never against what the worktree holds.

    Raises:
        OSError: The worktree's git metadata is not the app's (nothing was run).
    """
    admin = trusted_admin_dir(worktree)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_ATTR_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
    )
    argv = [
        "git",
        "--no-pager",
        "--git-dir",
        str(admin),
        "--work-tree",
        str(worktree),
        "-c",
        f"safe.directory={worktree}",
        "-c",
        f"safe.directory={admin}",
        *_NO_PROGRAMS,
        *args,
    ]
    return subprocess.run(argv, cwd=worktree, env=env, capture_output=True, text=True, timeout=timeout, check=False)
