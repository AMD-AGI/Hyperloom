# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How this package runs git: what git does in the session's repository and its scratch trees depends on nothing the
operator set up, as ``hyperloom_common.git_safety`` keeps it out of app-owned worktrees.

An inherited ``GIT_*`` (``GIT_DIR`` inside a hook or ``git rebase --exec``) would aim a command at another repository;
the system and global config could change the object format, hooks could veto a ref move, and ignore or attributes
files could drop or hide a file's change.
"""

from __future__ import annotations

import os
from pathlib import Path

# Command-line scope outranks every config file. The excludes file defaults to $XDG_CONFIG_HOME/git/ignore even when
# no global config is read.
_NO_OPERATOR_FILES = (
    "-c",
    f"core.hooksPath={os.devnull}",
    "-c",
    f"core.excludesFile={os.devnull}",
    "-c",
    f"core.attributesFile={os.devnull}",
)
#: ``git init`` with no template directory, so no hook or ``info/exclude`` is copied in.
INIT = ("init", "--template=")


def git_argv(*args: str) -> list[str]:
    """The command line of ``git <args>``."""
    return ["git", *_NO_OPERATOR_FILES, *args]


def git_env(**pinned: str) -> dict[str, str]:
    """The environment of a git command: the process's without any ``GIT_*``, then ``pinned``."""
    inherited = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    return {
        **inherited,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "LC_ALL": "C",
        **pinned,
    }


def trusting(directory: Path) -> tuple[str, str]:
    """The ``-c`` pair that has git use the repository at ``directory`` whoever owns it (``safe.directory``): a tree may
    have been made by another account, and no program its config names runs."""
    return ("-c", f"safe.directory={os.path.realpath(directory)}")


__all__ = ["INIT", "git_argv", "git_env", "trusting"]
