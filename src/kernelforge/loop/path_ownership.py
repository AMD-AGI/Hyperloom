# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Authoritative registry of path ownership inside a Forge workspace.

Every file or directory in a workspace belongs to one of three owners:

framework
    Package sources the agent is optimizing.  Protected by the workspace guard;
    restored verbatim on rollback.  Must never appear in a published patch as
    anything other than the intended source-level delta.

producer
    Forge's own bookkeeping: experiment logs, measurement drivers, candidate
    artefacts.  These are the working outputs of a forge run and must not
    propagate into a framework patch (``git add -u`` already excludes them, but
    ``is_producer_owned_path`` is the explicit gate in the apply-back path).

runtime
    Machine-generated caches and compiled binaries that the framework regenerates
    from source on import.  Never staged, never conflict-checked; including them
    in a git index inflates it by gigabytes and causes spurious integrity
    violations when a compile during the turn creates new entries.
"""

from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath


# ---------------------------------------------------------------------------
# Producer-owned: forge bookkeeping that must not reach a framework patch.
# ---------------------------------------------------------------------------

#: Basename patterns marking forge bookkeeping paths.
#: Each pattern is matched against *every component* of a repo-relative path
#: so a deeply nested file whose ancestor matches is also considered producer-
#: owned.  See :func:`is_producer_owned_path`.
PRODUCER_PATH_PATTERNS: tuple[str, ...] = (
    "forge_experiments",
    ".forge_rewrite",
    ".forge_driver_*",
    "optimization_report.md",
    "optimized_versions",
)

#: Legacy name kept for callers that imported from ``rewrite_by_flydsl.protocol``.
PRODUCER_OWNED_PATH_PATTERNS = PRODUCER_PATH_PATTERNS


def is_producer_owned_path(path: str) -> bool:
    """True when a repository-relative *path* holds forge bookkeeping.

    Matching is per path component so a framework file whose name merely starts
    with a producer prefix stays framework-owned.

    Args:
        path: Repository-relative POSIX path string.

    Returns:
        True if any path component matches a producer pattern.
    """
    for part in PurePosixPath(str(path).strip()).parts:
        if any(fnmatch.fnmatchcase(part, pattern) for pattern in PRODUCER_PATH_PATTERNS):
            return True
    return False


# ---------------------------------------------------------------------------
# Runtime-owned: machine-generated caches and compiled binaries.
# ---------------------------------------------------------------------------

#: Directory basenames that hold machine-generated artefacts.
#: Used by the staging transaction (external_artifacts) to avoid hashing or
#: copying build caches, and as the source for :func:`runtime_gitignore_globs`.
RUNTIME_DIRECTORY_NAMES: frozenset[str] = frozenset(
    {
        "__pycache__",
        "flydsl_cache",
        "jit_cache",
        "jit",
        "build",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)

#: File suffixes that are always runtime artefacts (bytecode, compiled objects).
RUNTIME_FILE_SUFFIXES: frozenset[str] = frozenset({".pyc", ".pyo", ".so"})


def runtime_gitignore_globs() -> tuple[str, ...]:
    """Return gitignore-style glob patterns covering all runtime artefacts.

    Suitable for writing into ``.git/info/exclude`` before a baseline ``git add``
    so that compiled binaries and caches are never hashed or committed.
    """
    dir_globs = tuple(f"{name}/" for name in sorted(RUNTIME_DIRECTORY_NAMES))
    ext_globs = tuple(f"*{suffix}" for suffix in sorted(RUNTIME_FILE_SUFFIXES))
    return dir_globs + ext_globs
