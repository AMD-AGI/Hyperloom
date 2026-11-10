# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Finding the repository and the benchmark harness the traced run came from."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ._kernel_search import _candidate_keywords, _harness_search_bases


def find_repo_root(source_file: str) -> str:
    """Walk upward from source_file until we find a .git/ dir; return the dir.

    Returns "" when no git repo root is found.

    Args:
        source_file (str): Path to a file inside a (possibly) git repo.

    Returns:
        str: The directory containing the nearest ``.git`` ancestor, or
            ``""`` when none is found.
    """
    if not source_file:
        return ""
    p = Path(source_file).expanduser().resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return str(parent)
    return ""


_BENCHMARK_DIRS = ("op_tests", "tests", "benchmarks", "benchmark", "test", "perf")

#: Curated harness lookups, keyed by marker substrings in a kernel's name or
#: source path. Paths are *checkout-relative* on purpose: the same harness sits
#: at ``/sgl-workspace/aiter/op_tests/...`` in a serving image and is absent from
#: a wheel install, so a pinned absolute path is either right on one host or a
#: fabrication on every other.
_KNOWN_HARNESS_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    # --- Normalization ---
    (
        (
            "rmsnorm_quant",
            "add_rmsnorm_quant",
            "rmsnorm",
            "add_rmsnorm",
        ),
        (
            "aiter/op_tests/test_rmsnorm2dFusedAddQuant.py",
            "aiter/op_tests/test_rmsnorm2d.py",
            "aiter/op_tests/op_benchmarks/triton/bench_rmsnorm.py",
            "sglang/sgl-kernel/benchmark/bench_rmsnorm.py",
            "aiter/op_tests/triton_tests/normalization/test_rmsnorm.py",
            "aiter/op_tests/triton_tests/normalization/test_fused_add_rmsnorm_pad.py",
        ),
    ),
    # --- Activation ---
    (
        (
            "activation",
            "act_and_mul",
            "silu",
        ),
        (
            "aiter/op_tests/test_activation.py",
            "aiter/op_tests/op_benchmarks/triton/bench_ff_a16w16_fused.py",
            "sglang/sgl-kernel/tests/test_activation.py",
            "sglang/sgl-kernel/benchmark/bench_activation.py",
            "sglang/python/sglang/jit_kernel/tests/test_activation.py",
            "sglang/python/sglang/jit_kernel/benchmark/bench_activation.py",
        ),
    ),
    # --- Attention ---
    (
        (
            "paged_attention",
            "fmha",
            "attention",
        ),
        (
            "aiter/op_tests/test_pa.py",
            "aiter/op_tests/op_benchmarks/triton/bench_pa_decode.py",
            "aiter/op_tests/op_benchmarks/triton/bench_pa_prefill.py",
        ),
    ),
    # --- MLA decode ---
    (
        (
            "mla_decode",
            "pseudo_mla",
            "mla_persistent",
        ),
        (
            "aiter/op_tests/test_mla.py",
            "aiter/op_tests/test_mla_persistent.py",
            "aiter/op_tests/op_benchmarks/triton/bench_mla_decode.py",
        ),
    ),
    # --- MoE CK two-stage ---
    (
        (
            "ck_moe_stage",
            "moe_2stage",
            "moe_stage1",
            "moe_stage2",
        ),
        (
            "aiter/op_tests/test_moe_2stage.py",
            "aiter/op_tests/op_benchmarks/triton/bench_moe.py",
        ),
    ),
    # --- MoE FP8 blockscale (ASM) ---
    (
        (
            "fmoe_fp8_blockscale",
            "moe_blockscale",
        ),
        (
            "aiter/op_tests/test_moe_blockscale.py",
            "aiter/op_tests/triton_tests/moe/test_moe_gemm_a8w8_blockscale.py",
        ),
    ),
    # --- GEMM A8W8 blockscale ---
    (
        ("gemm_a8w8_blockscale",),
        (
            "aiter/op_tests/test_gemm_a8w8_blockscale.py",
            "aiter/op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py",
        ),
    ),
    # --- Quantization ---
    (
        (
            "dynamic_per_token_scaled_quant",
            "per_token_quant",
        ),
        (
            "aiter/op_tests/test_quant.py",
            "aiter/op_tests/triton_tests/quant/test_quant.py",
        ),
    ),
    # --- Batch-invariant addmm (Triton) ---
    (
        (
            "batch_invariant",
            "addmm",
        ),
        ("sglang/test/registered/unit/batch_invariant_ops/test_batch_invariant_ops.py",),
    ),
)

_BENCHMARK_GREP_INCLUDES = tuple(f"--include=*.{ext}" for ext in ("py", "cpp", "cu", "cuh", "hip", "sh"))

_MULTIGPU_PATH_TAGS = ("multigpu", "multi_gpu", "multinode", "/dist/", "_dist_")


def _files_mentioning(keyword: str, root: Path) -> list[Path]:
    """List the existing source and script files under ``root`` that contain ``keyword``.

    A grep that cannot start, times out, or exits with an error finds nothing.
    """
    try:
        proc = subprocess.run(
            ["grep", "-rln", *_BENCHMARK_GREP_INCLUDES, "--", keyword, str(root)],
            text=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode not in (0, 1):
        return []
    paths = [Path(line) for line in map(str.strip, proc.stdout.splitlines()) if line]
    return [path for path in paths if path.exists()]


def _known_harness_files(name: str, source_file: str) -> list[Path]:
    """Return curated benchmark/test harnesses matching a kernel.

    Resolves each checkout-relative hint against the bases that exist here and
    keeps only files actually present. A list naming paths that cannot be
    opened is worse than an empty one, because every reader downstream -- the
    dispatch prompt included -- treats a non-empty list as a harness it can run.

    Args:
        name (str): Kernel symbol/name.
        source_file (str): Resolved source-file path (may be empty).

    Returns:
        list[Path]: Existing curated harness files, possibly empty.
    """
    blob = f"{name} {source_file}".lower()
    out: list[Path] = []
    seen: set[str] = set()
    bases = _harness_search_bases()
    for markers, relatives in _KNOWN_HARNESS_HINTS:
        if not any(marker in blob for marker in markers):
            continue
        for relative in relatives:
            for base in bases:
                candidate = os.path.join(base, relative)
                if candidate not in seen and os.path.isfile(candidate):
                    seen.add(candidate)
                    out.append(Path(candidate))
                    break
    return out


_LIBRARY_TOKENS = ("aiter", "sglang", "vllm", "flashinfer", "sgl-kernel", "sgl_kernel")


def _library_token(path: str) -> str:
    """Return the library a path belongs to (aiter/sglang/...), else "".

    Used to keep kernel↔benchmark pairing within one library. sgl-kernel and
    sgl_kernel normalize to "sglang" since they are the sglang kernel package.
    """
    low = (path or "").lower()
    for tok in _LIBRARY_TOKENS:
        if f"/{tok}/" in low or f"/{tok}." in low:
            return "sglang" if tok in ("sgl-kernel", "sgl_kernel") else tok
    return ""


def find_benchmark_files(name: str, repo_root: str, source_file: str = "") -> list[str]:
    """Find test/benchmark files matching a kernel under a repo's subdirs.

    Searches ``repo_root``'s known benchmark/test subdirectories for the
    kernel keywords, returning absolute paths.

    Args:
        name: Kernel symbol/name.
        repo_root: Repo root to search; empty returns only curated harnesses.
        source_file: Resolved source-file path, used to derive extra keywords.

    Returns:
        Up to ten matching harness paths, with multi-GPU tests demoted.
    """
    known = _known_harness_files(name, source_file)
    if not repo_root:
        return [str(p) for p in known[:10]]
    keywords = _candidate_keywords(name)
    # Add the source stem (and no-underscore variant) for repos that name tests differently.
    stem = Path(source_file).stem
    if stem and stem not in keywords:
        keywords.append(stem)
    no_us = stem.replace("_", "")
    if len(no_us) >= 6 and no_us not in keywords:
        keywords.append(no_us)
    if not keywords:
        return []
    root = Path(repo_root)
    found: list[Path] = list(known)
    for sub in _BENCHMARK_DIRS:
        sub_root = root / sub
        if not sub_root.exists():
            continue
        for keyword in keywords:
            found.extend(_files_mentioning(keyword, sub_root))
    unique = list(dict.fromkeys(str(p) for p in found))
    # Never pair a kernel from one library with a benchmark from another (a sglang sgl-kernel .cuh with an aiter
    # op_test): validating an edit against an unrelated library's op always fails the smoke test.
    src_lib = _library_token(source_file)
    if src_lib:
        same_lib = [s for s in unique if _library_token(s) in (src_lib, "")]
        if same_lib:
            unique = same_lib
    # Demote multi-GPU / distributed tests to the end: backends running on a
    # single Ray worker can't satisfy them, and they tend to make agents bail.
    unique.sort(key=lambda path: any(tag in path.lower() for tag in _MULTIGPU_PATH_TAGS))
    return unique[:10]
