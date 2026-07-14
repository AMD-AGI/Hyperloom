###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Independent benchmark/test-file discovery for the bypass analysis backend.

Populates ``benchmark_files`` on routable hot-kernel candidates so the shared
GEAK harness generator (``harness_generator.maybe_generate_harness``) can build
a runnable per-kernel benchmark when downstream tools need one.

Compact, independent reimplementation of the discovery *logic* used by
``tracelens_analysis.find_benchmark_files`` (content-grep over the kernel repo's
benchmark/test subdirs). It does **not** import TraceLens and does not carry
TraceLens' curated op->file mapping — a content match finds tests even when the
file name does not contain the op (e.g. ``silu_and_mul`` lives in
``test_activation.py``), which a name-only match would miss.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

# Repo subdirs that conventionally hold tests/benchmarks (aiter/sglang/vllm).
_BENCHMARK_DIRS = ("op_tests", "tests", "benchmarks", "benchmark", "test", "perf")
# Only files whose name looks like a test/benchmark are trusted as harness seeds.
_BENCH_NAME_HINTS = ("test_", "_test", "bench", "benchmark")
# Multi-GPU / distributed harnesses are demoted (a single-GPU rocprof run cannot
# drive them and they mismeasure the isolated kernel).
_MULTIGPU_RE = re.compile(r"(?i)multi_?gpu|distributed|_dist\b|all_?reduce|all_?gather|world_size")


def repo_root_from_source(source_file: str) -> str:
    """Best-effort kernel-repo root for a resolved source path.

    Walks up from ``source_file`` until an ancestor directory contains one of
    :data:`_BENCHMARK_DIRS` (e.g. ``/sgl-workspace/aiter/csrc/kernels/x.cu`` ->
    ``/sgl-workspace/aiter`` because it has ``op_tests/``).

    Args:
        source_file: Resolved kernel source path.

    Returns:
        The repo root path, or ``""`` when none is found / the path is empty.
    """
    if not source_file:
        return ""
    try:
        p = Path(source_file).resolve()
    except (OSError, RuntimeError):
        return ""
    for anc in [p, *p.parents]:
        try:
            if anc.is_dir() and any((anc / d).is_dir() for d in _BENCHMARK_DIRS):
                return str(anc)
        except OSError:
            continue
    return ""


def _keywords(op_name: str, source_file: str) -> list[str]:
    """Search keywords derived from the op name + source stem (deduped, >=4 chars)."""
    kws: list[str] = []
    base = (op_name or "").split("::")[-1].strip()
    if base:
        kws.append(base)
    if source_file:
        stem = Path(source_file).stem
        if stem:
            kws.append(stem)
    out: list[str] = []
    for k in kws:
        k = k.strip()
        if len(k) >= 4 and k not in out:
            out.append(k)
    return out


def find_benchmark_files(op_name: str, source_file: str, *, max_files: int = 10) -> list[str]:
    """Find on-disk benchmark/test ``.py`` files for a kernel via content grep.

    Args:
        op_name: Launching op name (e.g. ``aiter::rmsnorm``).
        source_file: Resolved kernel source path (repo root + keyword source).
        max_files: Cap on returned paths.

    Returns:
        Absolute benchmark-file paths (multi-GPU harnesses demoted), or ``[]``.
    """
    root = repo_root_from_source(source_file)
    if not root:
        return []
    keywords = _keywords(op_name, source_file)
    if not keywords:
        return []
    grep = shutil.which("grep")
    if not grep:
        return []
    rootp = Path(root)
    found: list[str] = []
    seen: set[str] = set()
    for sub in _BENCHMARK_DIRS:
        subp = rootp / sub
        if not subp.is_dir():
            continue
        for kw in keywords:
            try:
                proc = subprocess.run(
                    [grep, "-rlnI", "--include=*.py", kw, str(subp)],
                    text=True, capture_output=True, timeout=15,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if proc.returncode not in (0, 1):  # 0=match, 1=no match; 2+=error
                continue
            for line in (proc.stdout or "").splitlines():
                line = line.strip()
                if not line or line in seen:
                    continue
                base = Path(line).name.lower()
                if not any(hint in base for hint in _BENCH_NAME_HINTS):
                    continue
                seen.add(line)
                found.append(line)
    # Stable order: single-GPU harnesses first (multi-GPU can't be profiled solo).
    found.sort(key=lambda s: 1 if _MULTIGPU_RE.search(s) else 0)
    return found[:max_files]


__all__ = ["find_benchmark_files", "repo_root_from_source"]
