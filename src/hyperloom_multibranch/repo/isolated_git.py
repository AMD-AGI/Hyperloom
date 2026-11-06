# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How this package runs git: what git does in the session's repository and its scratch trees depends on nothing the
operator set up, as ``hyperloom_common.git_safety`` keeps it out of app-owned worktrees, and starts no program that a
repository's own config names.

An inherited ``GIT_*`` (``GIT_DIR`` inside a hook or ``git rebase --exec``) would aim a command at another repository;
the system and global config could change the object format, hooks could veto a ref move, and ignore or attributes
files could drop or hide a file's change.

The session's source base, its worktrees and an agent's checkout lie on storage that the sandboxes of the run's group
write, so their ``.git/config``, their hooks and their ``.gitattributes`` are not the app's. Every git command of the
app pins what :data:`~hyperloom_common.git_safety.NO_PROGRAMS` pins for the single-lineage app's git, derived from every
config key git starts a program from, and one that reads or writes a work tree also blanks every driver program the
config it reads defines (:func:`blank_driver_programs`), since an attributes file in the tree can apply a driver of any
name to any path. Git is allowed no transport (:data:`~hyperloom_common.git_safety.NO_TRANSPORT`) and fetches no object
lazily (:data:`NO_LAZY_FETCH`): a config can turn a repository into a partial clone whose promisor remote's transport is
a program. A repository nested in a tree is another config these do not reach; the callers keep git out of one. Nothing
here trusts a repository by its owner or its mode.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from hyperloom_common import git_safety
from hyperloom_common.git_safety import NO_PROGRAMS, NO_TRANSPORT

# Command-line scope outranks every config file. The excludes file defaults to $XDG_CONFIG_HOME/git/ignore even when
# no global config is read.
_NO_OPERATOR_FILES = (
    "-c",
    f"core.excludesFile={os.devnull}",
    "-c",
    f"core.attributesFile={os.devnull}",
)
#: No object is fetched behind a command's back: a config can make a repository a partial clone whose promisor
#: remote's transport (an ssh command, a local ``uploadpack`` program) runs when an object is missing.
NO_LAZY_FETCH = {"GIT_NO_LAZY_FETCH": "1"}
#: ``git init`` with no template directory, so no hook or ``info/exclude`` is copied in.
INIT = ("init", "--template=")
_CONFIG_TIMEOUT_S = 60.0


def git_argv(*args: str) -> list[str]:
    """The command line of ``git <args>``."""
    return ["git", *_NO_OPERATOR_FILES, *NO_PROGRAMS, *args]


def git_env(**pinned: str) -> dict[str, str]:
    """The environment of a git command: the process's without any ``GIT_*``, allowing no transport, then ``pinned``."""
    inherited = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    return {
        **inherited,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "LC_ALL": "C",
        **NO_LAZY_FETCH,
        **NO_TRANSPORT,
        **pinned,
    }


def leading_options(args: Sequence[str]) -> list[str]:
    """The ``-c <setting>`` pairs ``args`` starts with, before its subcommand: a ``safe.directory`` among them decides
    which repository's config git reads, so :func:`blank_driver_programs` is given them too."""
    count = 0
    while count + 1 < len(args) and args[count] == "-c":
        count += 2
    return list(args[:count])


def trusting(directory: Path) -> tuple[str, str]:
    """The ``-c`` pair that has git use the repository at ``directory`` whoever owns it (``safe.directory``): a tree on
    the shared storage may have been made by another account of the group, and no program its config names runs."""
    return ("-c", f"safe.directory={os.path.realpath(directory)}")


def blank_driver_programs(cwd: Path, env: Mapping[str, str], *prefix: str) -> list[str]:
    """The ``-c <key>=`` pairs that blank every driver program (a filter's, a diff's, a merge's, a trailer's) the config
    git reads in ``cwd`` under ``env`` defines, includes followed (``prefix`` are the options the command passes before
    its subcommand, which may name the repository); ``[]`` when it defines none.

    The command these go to reads the config again when it starts, so a driver another account adds in between is not
    blanked: the config is the group's to write, and this narrows what it can start to that moment.

    Raises:
        OSError: The config could not be read, or names a driver ``-c`` cannot address (its name holds ``=``).
    """
    git = ["git", *_NO_OPERATOR_FILES, *NO_PROGRAMS, *prefix]
    return git_safety.blank_driver_programs(git, cwd, dict(env), _CONFIG_TIMEOUT_S)


__all__ = [
    "INIT",
    "NO_LAZY_FETCH",
    "blank_driver_programs",
    "git_argv",
    "git_env",
    "leading_options",
    "trusting",
]
