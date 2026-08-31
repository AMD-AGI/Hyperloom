# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Authoritative registry of path ownership inside a Forge workspace.

Every path belongs to one of three owners. *framework* is the package source
under optimization: the workspace guard protects it and a patch carries its
delta. *producer* is forge's own bookkeeping, which must never reach a patch.
*runtime* is machine-generated cache and compiled output, which must never be
staged: a compile during a turn adds entries and the guard reads them as the
turn creating files.

Runtime splits by consumer. A git-index exclusion and a copy-filter exclusion
are different demands. Compiled artefacts must stay in the index so a git
revert can restore them — a stale binary influences measurement even after a
source revert — but must be excluded from copies because a scratch copy shadows
the installed package and imports directly from it.
"""

from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath

#: Attempt-scoped scratch root the rewrite producer creates in a workspace.
ATTEMPT_ROOT_DIR = ".forge_rewrite"

#: Path components marking forge bookkeeping.
#:
#: The apply-back gate turns a match into a refused patch, so a pattern earns
#: its place only by naming state that exists. The loop's report and candidates
#: live under ``forge_experiments``, which the first entry already covers.
PRODUCER_PATH_PATTERNS: tuple[str, ...] = (
    "forge_experiments",
    ATTEMPT_ROOT_DIR,
    ".forge_driver_*",
)

#: Directory basenames excluded from both copies and the git index.
#:
#: A name here must never be one a package imports from. ``aiter`` alone rules
#: out ``jit`` and ``dist``: ``aiter/jit`` holds ``core.py`` and the extension
#: modules ``import aiter`` loads, and ``aiter/dist`` holds its distributed
#: sources.
RUNTIME_DIRECTORY_NAMES: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "flydsl_cache",
        "jit_cache",
    }
)

#: Directory basenames excluded from copies but kept in the git index.
#:
#: A revert must be able to remove these; excluding them from the index would
#: leave stale build output that influences measurement after a source revert.
COPY_FILTER_DIRECTORY_NAMES: frozenset[str] = frozenset({"build"})

#: Directory patterns that only a glob can express.
RUNTIME_DIRECTORY_GLOBS: tuple[str, ...] = ("*.egg-info",)

#: Suffixes worth neither copying nor hashing.
RUNTIME_FILE_SUFFIXES: frozenset[str] = frozenset({".pyc", ".pyo"})

#: Suffixes a copy must carry and an index must not hash.
#:
#: A scratch copy shadows the installed package outright, so dropping the
#: extension modules leaves nothing to import and nothing to fall back to.
#: Hashing them instead costs the loop a binary in every diff it takes.
COMPILED_FILE_SUFFIXES: frozenset[str] = frozenset({".so"})


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
    """Gitignore patterns for artefacts that are safe to leave after a revert.

    Written into a repository's exclude file before a baseline ``git add``.
    Compiled artefacts (.so, build/) are intentionally excluded from this list:
    they must stay tracked so ``git revert`` can restore them and so a stale
    binary from a rolled-back iteration does not corrupt the next measurement.

    Returns:
        Directory and suffix patterns in gitignore syntax.
    """
    directories = sorted(RUNTIME_DIRECTORY_NAMES) + list(RUNTIME_DIRECTORY_GLOBS)
    suffixes = sorted(RUNTIME_FILE_SUFFIXES)
    return tuple(f"{name}/" for name in directories) + tuple(f"*{suffix}" for suffix in suffixes)
