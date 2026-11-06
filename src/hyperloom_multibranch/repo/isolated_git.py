# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How this package runs git: what git does in the session's repository and its scratch trees depends on nothing the
operator set up, as ``hyperloom_common.git_safety`` keeps it out of app-owned worktrees, and starts no program that a
repository's own config names.

An inherited ``GIT_*`` (``GIT_DIR`` inside a hook or ``git rebase --exec``) would aim a command at another repository;
the system and global config could change the object format, hooks could veto a ref move, and ignore or attributes
files could drop or hide a file's change.

The session's source base, its worktrees and an agent's checkout lie on storage that the sandboxes of the run's group
write, so their ``.git/config``, their hooks and their ``.gitattributes`` are not the app's: a config could name a
hook, an fsmonitor, a pager, an external diff, a signing program or a filter driver, which git would run as the app.
Every git command of the app pins the ones with a fixed name (:data:`NO_PROGRAMS`), and one that reads or writes a work
tree also blanks every filter driver that the config it reads defines (:func:`blank_filter_drivers`), since an
attributes file in the tree can apply a driver of any name to any path. No object is fetched lazily
(:data:`NO_LAZY_FETCH`): a config can turn a repository into a partial clone whose promisor remote's transport is a
program. Nothing here trusts a repository by its owner or its mode.
"""

from __future__ import annotations

import errno
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

# Command-line scope outranks every config file. The excludes file defaults to $XDG_CONFIG_HOME/git/ignore even when
# no global config is read.
_NO_OPERATOR_FILES = (
    "-c",
    f"core.excludesFile={os.devnull}",
    "-c",
    f"core.attributesFile={os.devnull}",
)
#: Every program with a fixed config name that a repository's own config could have git start on the app's commands:
#: hooks, the fsmonitor, an external diff, the pager, a signing program on commit and the ``ext::`` transport.
NO_PROGRAMS = (
    "-c",
    f"core.hooksPath={os.devnull}",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "diff.external=",
    "-c",
    "core.pager=cat",
    "-c",
    "commit.gpgSign=false",
    "-c",
    "protocol.ext.allow=never",
)
#: No ssh program either: the session's own repositories never reach a remote, and only a promisor remote a
#: config names would make git fetch.
_NO_TRANSPORT = ("-c", "core.sshCommand=false")
#: No object is fetched behind a command's back: a config can make a repository a partial clone whose promisor
#: remote's transport (an ssh command, a local ``uploadpack`` program) runs when an object is missing.
NO_LAZY_FETCH = {"GIT_NO_LAZY_FETCH": "1"}
#: ``git init`` with no template directory, so no hook or ``info/exclude`` is copied in.
INIT = ("init", "--template=")
#: What a filter driver may run: a filter whose programs are all empty runs nothing.
FILTER_PROGRAMS = ("clean", "smudge", "process")
_FILTER_KEYS = rf"^filter\..*\.({'|'.join(FILTER_PROGRAMS)})$"
_NO_SUCH_KEY = 1  # git config's exit status when no key matches
_CONFIG_TIMEOUT_S = 60.0


def git_argv(*args: str) -> list[str]:
    """The command line of ``git <args>``."""
    return ["git", *_NO_OPERATOR_FILES, *NO_PROGRAMS, *_NO_TRANSPORT, *args]


def git_env(**pinned: str) -> dict[str, str]:
    """The environment of a git command: the process's without any ``GIT_*``, then ``pinned``."""
    inherited = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    return {
        **inherited,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "LC_ALL": "C",
        **NO_LAZY_FETCH,
        **pinned,
    }


def leading_options(args: Sequence[str]) -> list[str]:
    """The ``-c <setting>`` pairs ``args`` starts with, before its subcommand: a ``safe.directory`` among them decides
    which repository's config git reads, so :func:`blank_filter_drivers` is given them too."""
    count = 0
    while count + 1 < len(args) and args[count] == "-c":
        count += 2
    return list(args[:count])


def trusting(directory: Path) -> tuple[str, str]:
    """The ``-c`` pair that has git use the repository at ``directory`` whoever owns it (``safe.directory``): a tree on
    the shared storage may have been made by another account of the group, and no program its config names runs."""
    return ("-c", f"safe.directory={os.path.realpath(directory)}")


def blank_filter_drivers(cwd: Path, env: Mapping[str, str] | None = None, *prefix: str) -> list[str]:
    """The ``-c filter.<name>.<program>=`` pairs that blank every filter driver the config git reads in ``cwd`` under
    ``env`` defines (``prefix`` are the options the command passes before its subcommand, which may name the
    repository); ``[]`` when it defines none.

    The command these go to reads the config again when it starts, so a driver another account adds in between is not
    blanked: the config is the group's to write, and this narrows what it can start to that moment.

    Raises:
        OSError: The config could not be read, or names a driver ``-c`` cannot address (its name holds ``=``).
    """
    done = subprocess.run(
        [
            "git",
            *_NO_OPERATOR_FILES,
            *NO_PROGRAMS,
            *prefix,
            "config",
            "--null",
            "--name-only",
            "--get-regexp",
            _FILTER_KEYS,
        ],
        cwd=cwd,
        env=None if env is None else dict(env),
        capture_output=True,
        timeout=_CONFIG_TIMEOUT_S,
        check=False,
    )
    if done.returncode not in (0, _NO_SUCH_KEY):
        raise OSError(errno.EIO, f"git config: {os.fsdecode(done.stderr).strip()}", str(cwd))
    names = {os.fsdecode(key).rpartition(".")[0].partition(".")[2] for key in done.stdout.split(b"\0") if key}
    if any("=" in name for name in names):
        raise OSError(errno.EINVAL, "a filter driver name holds '=', which git -c cannot override", str(cwd))
    return [arg for name in sorted(names) for program in FILTER_PROGRAMS for arg in ("-c", f"filter.{name}.{program}=")]


__all__ = [
    "FILTER_PROGRAMS",
    "INIT",
    "NO_LAZY_FETCH",
    "NO_PROGRAMS",
    "blank_filter_drivers",
    "git_argv",
    "git_env",
    "leading_options",
    "trusting",
]
