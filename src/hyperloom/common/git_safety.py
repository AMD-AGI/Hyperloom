# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``safe.directory`` handling for git subprocesses.

The documented container recipe bind-mounts the repo at the same path, so the
checkouts the optimizer reads and patches are routinely owned by a different uid
than the process. git then refuses every operation on them, reads included.

The exception travels with the argv instead of the user's gitconfig: running an
optimization should not change how git behaves elsewhere on the host.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["repo_root", "safe_directory_args"]


def repo_root(target: str | Path) -> str | None:
    """Nearest ancestor of ``target`` holding a ``.git`` entry, else None.

    ``.git`` is a file in linked worktrees, so existence is the test. None means
    the target is outside any checkout, where no exception should be invented.
    """
    try:
        current = Path(target).expanduser().resolve()
    except OSError:
        return None
    for candidate in (current, *current.parents):
        try:
            if (candidate / ".git").exists():
                return str(candidate)
        except OSError:
            continue
    return None


def safe_directory_args(args: list[str], *, cwd: str | Path | None = None) -> list[str]:
    """Prepend a ``safe.directory`` exception for the repo ``args`` targets.

    ``args`` excludes the ``git`` executable. The repo is located from ``-C``
    when present, else from ``cwd``; callers use both forms. Three git rules
    shape this: ownership resolves against the repository *root*, so naming a
    subdirectory is ignored; ``-c`` after the subcommand is treated as the
    subcommand's own argument; and command config is protected config, so
    ``safe.directory`` is honoured there though repository config is not.
    """
    target: str | Path | None = cwd
    try:
        target = args[args.index("-C") + 1]
    except (ValueError, IndexError):
        pass
    if target is None:
        return args
    root = repo_root(target)
    if root is None:
        return args
    return ["-c", f"safe.directory={root}", *args]
