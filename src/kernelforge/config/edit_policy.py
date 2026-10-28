# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a Forge workspace protects from a candidate: the measurement surface every agent session is held to."""

from __future__ import annotations

from kernelforge.utils.protected_paths import ProtectedPathRules


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
