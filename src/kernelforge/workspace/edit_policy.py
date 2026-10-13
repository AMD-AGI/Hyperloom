# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical workspace edit policy shared by Forge agent backends and the loop."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import stat
from typing import Iterable
from kernelforge.tools.git import git_bytes


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

# Untracked files the toolchain drops in a workspace on its own while a kernel runs. Only these named outputs are
# ignored; any other new file a session leaves remains a safety violation.
TOOL_OWNED_UNTRACKED_GLOBS = (
    # rocprof runs below the git root, so cover root and nested directories.
    ".rocprofv3/*",
    "*/.rocprofv3/*",
    "*_results.db",
    # AITER may create JIT shards during a turn; its configured root always ends
    # in ``aiter_cache``, regardless of the experiments directory.
    "aiter_cache/*",
    "*/aiter_cache/*",
)

FORGE_LOOP_OUTPUT_DIRNAME = "forge_experiments"

_BOOKKEEPING_DIRNAMES = frozenset({".git", FORGE_LOOP_OUTPUT_DIRNAME})

# A file the caller declares editable can share a name with a test: ROCmBench colocates each kernel with its tests
# in one test_*.py. Only these name patterns yield to a declared target. The rest of the measurement surface stays
# protected however it is declared, and the caller polices what inside such a file may change.
TARGET_RECLAIMABLE_GLOBS = frozenset({"test_*.py", "*_test.py", "*_test.cpp", "*_test.cu", "*_test.hip"})

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
    editable_paths: Iterable[str | Path] = (),
) -> bool:
    """Return whether ``path`` belongs to the authoritative measurement surface.

    ``editable_paths`` are the files the caller declares as targets; one of them escapes only the test-name
    patterns in ``TARGET_RECLAIMABLE_GLOBS``, never an exact path, a protected directory or another pattern.
    """

    raw = Path(path).expanduser()
    root = Path(workspace).expanduser().resolve() if workspace else None
    absolute = raw.resolve() if raw.is_absolute() else ((root / raw).resolve() if root else raw.resolve())

    def resolved(items: Iterable[str | Path]) -> set[Path]:
        paths: set[Path] = set()
        for item in items:
            if not str(item or "").strip():
                continue
            candidate = Path(item).expanduser()
            paths.add(
                candidate.resolve()
                if candidate.is_absolute()
                else ((root / candidate).resolve() if root else candidate.resolve())
            )
        return paths

    protected_abs = resolved(exact_paths)
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
    if absolute in resolved(editable_paths):
        patterns = tuple(pattern for pattern in patterns if pattern not in TARGET_RECLAIMABLE_GLOBS)
    relative_posix = relative.as_posix()
    return any(
        fnmatch.fnmatch(relative.name, pattern) or fnmatch.fnmatch(relative_posix, pattern) for pattern in patterns
    )


def protected_path_inventory(
    workspace: str | Path,
    *,
    exact_paths: Iterable[str | Path] = (),
    extra_globs: Iterable[str] = (),
    editable_paths: Iterable[str | Path] = (),
) -> tuple[Path, ...]:
    """Return every protected filesystem entry under ``workspace`` recursively."""
    editable = tuple(editable_paths)

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
        # The repository's own bookkeeping is not part of the measurement surface, and it moves on its own: git
        # rewrites the index whenever a stat cache goes cold, which a build alone is enough to cause.
        dirnames[:] = [name for name in dirnames if name not in _BOOKKEEPING_DIRNAMES]
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
                editable_paths=editable,
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
    result = git_bytes("ls-files", "-z", cwd=root, check=False)
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
