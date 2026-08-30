# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical workspace edit policy shared by Forge agent backends and the loop."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import stat
from typing import Iterable
from kernelforge.llm.git import git


PROTECTED_GLOBS = (
    "*harness*.py",
    "config.yaml",
    "config.yml",
    "forge_driver.py",
    "task_runner.py",
    "cal_kernel_perf.py",
    "performance_utils*.py",
    "test_*.py",
    "*_test.py",
    "*_test.cpp",
    "*_test.cu",
    "*_test.hip",
    "*_ref.py",
    "*_reference.py",
    "conftest.py",
)

PROTECTED_DIRS = frozenset(
    {
        "benchmark",
        "benchmarks",
        "script",
        "scripts",
        "test",
        "tests",
        "perf",
    }
)


def is_protected_path(
    path: str | Path,
    *,
    workspace: str | Path | None = None,
    exact_paths: Iterable[str | Path] = (),
    extra_globs: Iterable[str] = (),
) -> bool:
    """Return whether ``path`` belongs to the authoritative measurement surface."""

    raw = Path(path).expanduser()
    root = Path(workspace).expanduser().resolve() if workspace else None
    absolute = raw.resolve() if raw.is_absolute() else ((root / raw).resolve() if root else raw.resolve())
    protected_abs: set[Path] = set()
    for item in exact_paths:
        if not str(item or "").strip():
            continue
        candidate = Path(item).expanduser()
        protected_abs.add(
            candidate.resolve()
            if candidate.is_absolute()
            else ((root / candidate).resolve() if root else candidate.resolve())
        )
    if absolute in protected_abs:
        return True

    relative = raw
    if root is not None:
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            relative = raw
    if any(part.lower() in PROTECTED_DIRS for part in relative.parts[:-1]):
        return True
    patterns = (*PROTECTED_GLOBS, *tuple(extra_globs))
    relative_posix = relative.as_posix()
    return any(
        fnmatch.fnmatch(relative.name, pattern) or fnmatch.fnmatch(relative_posix, pattern) for pattern in patterns
    )


def protected_path_inventory(
    workspace: str | Path,
    *,
    exact_paths: Iterable[str | Path] = (),
    extra_globs: Iterable[str] = (),
) -> tuple[Path, ...]:
    """Return every protected filesystem entry under ``workspace`` recursively.

    Exact paths are included even when they are absent so callers can detect a
    protected file created during a session. Every discovered entry is classified
    by :func:`is_protected_path`; inventory and write-policy rules therefore cannot
    drift on nested basename globs, path globs, or protected directories.

    Traversal and metadata errors are intentionally propagated. An integrity
    checker cannot treat an unreadable part of the measurement surface as absent.
    """

    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise OSError(f"protected inventory workspace is not a directory: {root}")
    globs = tuple(extra_globs)
    exact = {
        (
            Path(item).expanduser().resolve()
            if Path(item).expanduser().is_absolute()
            else (root / Path(item).expanduser()).resolve()
        )
        for item in exact_paths
        if str(item or "").strip()
    }
    inventory = set(exact)

    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, dirnames, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        # The repository's own bookkeeping is not part of the measurement
        # surface, and it moves on its own: git rewrites the index whenever a
        # stat cache goes cold, which a build alone is enough to cause. A guard
        # holding its bytes would reject the session for git's housekeeping.
        # What the repository state must not do is checked semantically instead
        # -- HEAD, the active branch, the refs, and the index entries.
        dirnames[:] = [name for name in dirnames if name != ".git"]
        parent = Path(directory)
        entries = [*filenames]
        for dirname in dirnames:
            candidate = parent / dirname
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                entries.append(dirname)
        for name in entries:
            candidate = parent / name
            metadata = candidate.lstat()
            if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)):
                continue
            if is_protected_path(
                candidate,
                workspace=root,
                exact_paths=exact,
                extra_globs=globs,
            ):
                inventory.add(Path(os.path.abspath(candidate)))
    return tuple(sorted(inventory, key=str))


def tracked_editable_paths(
    workspace: str | Path,
    *,
    exact_protected_paths: Iterable[str | Path] = (),
    extra_protected_globs: Iterable[str] = (),
) -> set[str]:
    """Return every tracked workspace path outside the protected measurement set."""

    root = Path(workspace).expanduser().resolve()
    result = git("ls-files", "-z", cwd=root, check=False, text=False)
    if result.returncode != 0:
        return set()
    editable: set[str] = set()
    for encoded in result.stdout.split(b"\0"):
        if not encoded:
            continue
        relative = encoded.decode(errors="surrogateescape")
        if not is_protected_path(
            relative,
            workspace=root,
            exact_paths=exact_protected_paths,
            extra_globs=extra_protected_globs,
        ):
            editable.add(Path(relative).as_posix())
    return editable
