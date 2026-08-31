# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Authoritative registry of path ownership inside a Forge workspace.

Every path belongs to one of three owners. *framework* is the package source
under optimization: the workspace guard protects it and a patch carries its
delta. *producer* is forge's own bookkeeping, which must never reach a patch.
*runtime* is machine-generated cache and compiled output, which must never be
staged: a compile during a turn adds entries and the guard reads them as the
turn creating files.
"""

from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath

#: Attempt-scoped scratch root the rewrite producer creates in a workspace.
ATTEMPT_ROOT_DIR = ".forge_rewrite"

#: Path components marking forge bookkeeping.
PRODUCER_PATH_PATTERNS: tuple[str, ...] = (
    "forge_experiments",
    ATTEMPT_ROOT_DIR,
    ".forge_driver_*",
    "optimization_report.md",
    "optimized_versions",
)

#: Directory basenames holding machine-generated artefacts.
RUNTIME_DIRECTORY_NAMES: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
        "flydsl_cache",
        "jit",
        "jit_cache",
    }
)

#: Directory patterns that only a glob can express.
RUNTIME_DIRECTORY_GLOBS: tuple[str, ...] = ("*.egg-info",)

#: Suffixes that are always build output.
RUNTIME_FILE_SUFFIXES: frozenset[str] = frozenset({".pyc", ".pyo", ".so"})


def is_producer_owned_path(path: str) -> bool:
    """True when a repository-relative path holds forge bookkeeping.

    Matching is per path component, so a framework file whose name merely
    starts with a producer prefix stays framework-owned.

    Args:
        path: Repository-relative POSIX path.

    Returns:
        True when any component matches a producer pattern.
    """
    for part in PurePosixPath(str(path).strip()).parts:
        if any(fnmatch.fnmatchcase(part, pattern) for pattern in PRODUCER_PATH_PATTERNS):
            return True
    return False


def runtime_gitignore_globs() -> tuple[str, ...]:
    """Gitignore patterns covering every runtime artefact.

    Written into a repository's exclude file before a baseline ``git add`` so
    compiled output is never hashed.

    Returns:
        Directory and suffix patterns in gitignore syntax.
    """
    directories = sorted(RUNTIME_DIRECTORY_NAMES) + list(RUNTIME_DIRECTORY_GLOBS)
    return tuple(f"{name}/" for name in directories) + tuple(
        f"*{suffix}" for suffix in sorted(RUNTIME_FILE_SUFFIXES)
    )
