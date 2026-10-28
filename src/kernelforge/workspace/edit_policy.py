# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical workspace edit policy Forge's sessions and the loop are held to."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from kernelforge.tools.git import git_bytes
from kernelforge.tools.protected_paths import ProtectedPathRules


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

# The authoritative measurement surface of a Forge workspace. The repository's own bookkeeping is not part of it, and
# it moves on its own: git rewrites the index whenever a stat cache goes cold, which a build alone is enough to cause.
FORGE_PROTECTED_PATHS = ProtectedPathRules(
    globs=PROTECTED_GLOBS,
    dirs=PROTECTED_DIRS,
    unscanned_dirs=frozenset({".git", FORGE_LOOP_OUTPUT_DIRNAME}),
)


def hide_forge_loop_output(workspace: str | Path) -> Path:
    """Create forge-loop's output directory under ``workspace``, hide it from Git, and return it.

    A ``.gitignore`` holding ``*`` ignores everything beside it, itself included, so Git never lists the directory:
    nothing the loop or a session writes there is committed or reported as a new file, whatever the workspace ignores.
    """
    output_root = Path(workspace) / FORGE_LOOP_OUTPUT_DIRNAME
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / ".gitignore").write_text("*\n", encoding="utf-8")
    return output_root


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
        if not FORGE_PROTECTED_PATHS.is_protected(
            relative,
            workspace=root,
            exact_paths=exact_protected_paths,
            extra_globs=extra_protected_globs,
        ):
            editable.add(Path(relative).as_posix())
    return editable
