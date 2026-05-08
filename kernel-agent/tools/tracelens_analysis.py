#!/usr/bin/env python3
"""TraceLens analysis tool for the resident Kernel Agent skill.

This tool is intentionally conservative: it records every step, writes a stable
artifact set, supports TraceLens capture directories, and has a dry-run path for
local validation without requiring TraceLens to be installed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KERNEL_HINTS = (
    "kernel", "triton", "hip", "cuda", "rocblas", "hipblas", "aiter",
    "fmha", "gemm", "attention", "moe", "rmsnorm", "layernorm",
)
RUNTIME_API_NAMES = {
    "hipeventsynchronize",
    "hipdevicesynchronize",
    "hipstreamsynchronize",
    "hipgraphlaunch",
    "hiplaunchkernel",
    "hipmodulelaunchkernel",
    "hipmemcpy",
    "hipmemset",
    "cudaeventsynchronize",
    "cudadevicesynchronize",
    "cudastreamsynchronize",
}
DEFAULT_TRACELENS_ROOT = "/wekafs/hyperloom/TraceLens-internal"
LOCAL_BUNDLE_TRACELENS_ROOT = "/wekafs/fully-local/TraceLens-internal"

# TraceLens ships two perf-report CLIs:
#   - `..._inference` is the correct entry for vLLM/SGLang inference traces
#     (issue #124 Bug 1). It assumes graph-replay execution and emits the
#     fields that downstream fusion / roofline analysis expects.
#   - `..._pytorch` is the legacy / training default. We keep it as a
#     fallback so older TraceLens installs (without the inference variant)
#     still work.
INFERENCE_PERF_CLI = "TraceLens_generate_perf_report_pytorch_inference"
LEGACY_PERF_CLI = "TraceLens_generate_perf_report_pytorch"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as tmp:
        json.dump(data, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def read_last_lines(log_path: Path, limit: int = 20) -> list[str]:
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


def update_status(
    status_path: Path,
    *,
    state: str,
    current_step: str,
    log_path: Path,
    artifact_paths: dict[str, str],
    run_id: str,
    started_at: str,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "tool": "tracelens_analysis",
        "run_id": run_id,
        "state": state,
        "current_step": current_step,
        "pid": os.getpid(),
        "started_at": started_at,
        "updated_at": utc_now(),
        "log_path": str(log_path),
        "artifact_paths": artifact_paths,
        "offset_bytes": log_path.stat().st_size if log_path.exists() else 0,
        "last_lines": read_last_lines(log_path),
    }
    if error:
        payload["error"] = error
    atomic_write_json(status_path, payload)


def open_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


def discover_trace_inputs(trace_input: Path) -> tuple[str, list[Path]]:
    if trace_input.is_file():
        return "file", [trace_input]
    if not trace_input.is_dir():
        raise FileNotFoundError(f"trace_input does not exist: {trace_input}")

    traces: list[Path] = []
    for pattern in ("*.json", "*.json.gz", "*.trace", "*.trace.json", "*.trace.json.gz"):
        traces.extend(sorted(trace_input.rglob(pattern)))
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique = []
    for trace in traces:
        if trace not in seen:
            seen.add(trace)
            unique.append(trace)
    if not unique:
        raise FileNotFoundError(f"no trace files found under capture directory: {trace_input}")
    return "capture_dir", unique


def is_kernel_event(event: dict[str, Any]) -> bool:
    """Strict GPU-kernel filter for raw torch_profiler events.

    PyTorch profiler tags real GPU kernel launches with ``cat == 'kernel'``.
    Everything else (``python_function`` / ``cuda_runtime`` / ``cpu_op``)
    is host-side activity even when the symbol name happens to contain
    "cuda" / "hip" / "synchronize" — including these via fuzzy matching
    causes ``torch/cuda/streams.py(222): synchronize`` (the CPU wait that
    accumulates the ENTIRE GPU duration of the wrapped enqueue burst) to
    eclipse all real kernels in the top-K hot list, which then makes
    every downstream step (source resolver, GEAK / Codex / Claude
    backend dispatch) operate on a phantom kernel.
    """
    cat = str(event.get("cat") or event.get("category") or "").lower()
    if cat != "kernel":
        return False
    name = str(event.get("name") or event.get("kernel_name") or "")
    if name.lower() in RUNTIME_API_NAMES:
        return False
    return True


def extract_shape(event: dict[str, Any]) -> dict[str, Any] | None:
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    for key in ("shape", "shapes", "input_shape", "trace_shapes"):
        if key in args:
            return {key: args[key]}
        if key in event:
            return {key: event[key]}
    return None


def extract_source_file(event: dict[str, Any]) -> str:
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    for key in ("source_file", "file", "filename", "path"):
        value = args.get(key) or event.get(key)
        if value:
            return str(value)
    return ""


def source_type_for(name: str, source_file: str) -> str:
    lower_name = name.lower()
    lower_file = source_file.lower()
    if is_runtime_generated_kernel(name, source_file):
        return "runtime_generated"
    if source_file.endswith((".cu", ".cuh", ".hip", ".cpp", ".h", ".hpp")):
        return "hip_cpp"
    if "triton" in lower_name and source_file.endswith(".py"):
        return "triton"
    if source_file.endswith(".py"):
        return "python"
    if "hipblas" in lower_name or "rocblas" in lower_name:
        return "vendor_binary"
    return "unknown"


_RUNTIME_GENERATED_SOURCE_MARKERS = (
    "/tmp/torchinductor",
    "/torchinductor_",
    "/.cache/torch/inductor",
    "/.triton/cache",
    "/triton/cache",
)
_COMPILE_GENERATED_NAME_MARKERS = (
    "triton_poi_",
    "triton_red_",
    "triton_tem_",
    "torchinductor",
    "inductor",
)
_REUSABLE_SOURCE_ROOTS = (
    "/sgl-workspace/aiter/",
    "/sgl-workspace/sglang/",
    "/sgl-workspace/vllm/",
    "/opt/venv/lib/python3.10/site-packages/aiter/",
    "/opt/venv/lib/python3.10/site-packages/sglang/",
    "/opt/venv/lib/python3.10/site-packages/vllm/",
)


def is_runtime_generated_kernel(name: str, source_file: str) -> bool:
    """Return True for torch.compile / Inductor / cache-generated kernels.

    Kernel-opt must target reusable native sources. Runtime-generated files
    under torchinductor or Triton caches are tied to a specific compile graph,
    shape, and cache state; patching them is not portable across serving runs.
    """
    lower_name = (name or "").lower()
    lower_file = (source_file or "").lower()
    if any(marker in lower_file for marker in _RUNTIME_GENERATED_SOURCE_MARKERS):
        return True
    if any(marker in lower_name for marker in _COMPILE_GENERATED_NAME_MARKERS):
        # A stable in-repo SGLang/vLLM Triton source can still be reusable.
        return not any(root in lower_file for root in _REUSABLE_SOURCE_ROOTS)
    return False


def is_reusable_native_kernel(candidate: dict[str, Any]) -> bool:
    """Whether a candidate is safe to send to kernel optimization backends."""
    source_file = str(candidate.get("source_file") or "")
    if not source_file:
        return False
    if candidate.get("source_type") == "vendor_binary":
        return False
    if candidate.get("vendor_dispatch_wrapper"):
        return False
    if is_runtime_generated_kernel(str(candidate.get("name") or ""), source_file):
        return False
    lower_file = source_file.lower()
    if not any(root in lower_file for root in _REUSABLE_SOURCE_ROOTS):
        return False
    return candidate.get("source_type") in {"hip_cpp", "triton", "python"}


# Wrapper TUs that just dispatch to a precompiled .so / .co (no device body
# we can rewrite) — agents waste their budget grepping but produce no real
# patch. Detected by file size + content signature, similar to
# `_is_pybind_shim` for pybind glue but broader: also catches Python
# dispatch wrappers and the few "ctypes load + call" shims that aren't
# strictly pybind. Kept conservative so we don't drop legitimate small
# kernels.
_VENDOR_DISPATCH_SIGS = (
    "ctypes.CDLL",  # pure-Python wrapper around .so
    "torch.ops.aiter.",  # registered aten op forwarding
    "_C_aiter.",  # bound C extension forwarding
    "module_name = ",  # aiter jit module loaders
    "AITER_JIT_LOAD",  # aiter macro
    "hipModuleLoad",  # raw .co loader
    "AiterAsmKernel",  # ASM dispatch wrapper
)
_VENDOR_KEYWORD_NAMES = (
    "hipblaslt", "rocblaslt", "miopen", "ck_kernels",
)


def is_vendor_dispatch_wrapper(name: str, source_file: str) -> bool:
    """Heuristic: True when source_file is a thin dispatch wrapper around a
    precompiled vendor binary (.so/.co), i.e. nothing for a kernel agent
    to rewrite. Distinct from `_is_pybind_shim` (which catches PYBIND11
    registration TUs); this catches Python wrappers + ctypes/jit-load
    style C++ shims + vendor BLAS names.
    """
    nm = (name or "").lower()
    if any(kw in nm for kw in _VENDOR_KEYWORD_NAMES):
        return True
    if not source_file:
        return False
    p = Path(source_file)
    try:
        if not p.is_file():
            return False
        # Heuristic threshold: real device kernels (Triton .py rms_norm
        # ~38 KB, HIP `.cuh` custom_all_reduce ~110 KB, attention kernels
        # ~30+ KB) are almost always > 16 KB; ASM dispatch wrappers like
        # asm_gemm_a16w16.cu (~10 KB) and pybind shims (~250 B) sit
        # well below. Anything > 16 KB is presumed real and never marked
        # vendor by this check.
        if p.stat().st_size > 16 * 1024:
            return False
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return any(sig in text for sig in _VENDOR_DISPATCH_SIGS)


KNOWN_SEARCH_ROOTS = (
    "/sgl-workspace/aiter",
    "/sgl-workspace/sglang/sgl-kernel",
    "/sgl-workspace/sglang/python/sglang",
    "/sgl-workspace/vllm",
    "/opt/venv/lib/python3.10/site-packages/sglang",
    "/opt/venv/lib/python3.10/site-packages/aiter",
    "/opt/venv/lib/python3.10/site-packages/vllm",
)
SOURCE_EXTENSIONS = (".cuh", ".cu", ".hip", ".cpp", ".h", ".hpp", ".py")


def _strip_template_args(symbol: str) -> str:
    out: list[str] = []
    depth = 0
    for ch in symbol:
        if ch == "<":
            depth += 1
            continue
        if ch == ">":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


_NAMESPACE_BLOCKLIST = {
    "aiter", "sglang", "vllm", "torch", "ck_tile", "ck", "pybind",
    "RankData", "RankSignals", "Signal", "module", "namespace",
}
_TYPE_BLOCKLIST = {
    "void", "int", "float", "char", "long", "short", "bool", "unsigned", "string",
}


def _candidate_keywords(name: str) -> list[str]:
    """Pick stable search keywords from a kernel symbol.

    Prefers descriptive identifiers (e.g. cross_device_reduce_2stage, gemm_a16w16)
    over namespace/type tokens (aiter, vllm, RankData) that match too widely.
    """
    cleaned = name.strip()
    if cleaned.startswith("_Z"):
        # Itanium ABI uses <len><name>; walk through and slice manually so
        # consecutive segments (e.g. 5aiter26cross_device_reduce_2stage...) are
        # parsed as separate identifiers.
        import re
        tokens = []
        pos = 0
        while pos < len(cleaned):
            m = re.match(r"(\d+)", cleaned[pos:])
            if not m:
                pos += 1
                continue
            length = int(m.group(1))
            start = pos + m.end()
            ident = cleaned[start:start + length]
            if ident and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", ident):
                tokens.append(ident)
                pos = start + length
            else:
                pos = start + 1
        if not tokens:
            tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{4,}", cleaned)
    else:
        cleaned = _strip_template_args(cleaned)
        if "::" in cleaned:
            cleaned = cleaned.split("::")[-1]
        tokens = [cleaned]
    seen: set[str] = set()
    raw: list[str] = []
    for tok in tokens:
        tok = tok.strip("_")
        if not tok or tok in seen:
            continue
        if tok in _TYPE_BLOCKLIST:
            continue
        if len(tok) < 5:
            continue
        seen.add(tok)
        raw.append(tok)
    if not raw:
        return []
    # Prefer multi-segment identifiers (snake_case / longer) and drop
    # well-known namespace tokens that match too many files.
    descriptive = [t for t in raw if t not in _NAMESPACE_BLOCKLIST]
    if descriptive:
        descriptive.sort(key=lambda t: (-t.count("_"), -len(t)))
        return descriptive[:3]
    raw.sort(key=lambda t: (-t.count("_"), -len(t)))
    return raw[:3]


_GREP_CACHE: dict[tuple[str, str], list[Path]] = {}


def _grep_for_keyword(keyword: str, root: Path) -> list[Path]:
    if not root.exists():
        return []
    cache_key = (keyword, str(root))
    if cache_key in _GREP_CACHE:
        return _GREP_CACHE[cache_key]
    cmd = [
        "grep", "-rln",
        "--include=*.cuh", "--include=*.cu", "--include=*.hip",
        "--include=*.cpp", "--include=*.h", "--include=*.hpp",
        "--include=*.py",
        keyword, str(root),
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=15)
    except Exception:
        _GREP_CACHE[cache_key] = []
        return []
    if proc.returncode not in (0, 1):
        _GREP_CACHE[cache_key] = []
        return []
    paths: list[Path] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        path = Path(line)
        if path.exists() and path.suffix in SOURCE_EXTENSIONS:
            paths.append(path)
    _GREP_CACHE[cache_key] = paths
    return paths


def _rank_paths(paths: list[Path]) -> list[Path]:
    def score(path: Path) -> tuple[int, int, int]:
        s = str(path)
        # Prefer real source repos over installed wheels and over optimized variants.
        depth_penalty = s.count("/")
        kind_score = 0
        if "/csrc/" in s:
            kind_score -= 3
        if "/optimized_versions/" in s or "/build/" in s:
            kind_score += 5
        if "/site-packages/" in s:
            kind_score += 2
        ext_score = {".cuh": 0, ".cu": 0, ".hip": 0, ".cpp": 1, ".h": 2, ".hpp": 2, ".py": 3}.get(path.suffix, 4)
        return (kind_score, ext_score, depth_penalty)

    return sorted(paths, key=score)


def locate_source_via_grep(name: str) -> str:
    """Locate a kernel source file by grepping known repos.

    Returns "" when no confident match exists. Never fabricates a path.
    """
    keywords = _candidate_keywords(name)
    if not keywords:
        return ""
    for keyword in keywords:
        hits: list[Path] = []
        for root in KNOWN_SEARCH_ROOTS:
            hits.extend(_grep_for_keyword(keyword, Path(root)))
        if hits:
            ranked = _rank_paths(hits)
            return str(ranked[0])
    return ""


def find_repo_root(source_file: str) -> str:
    """Walk upward from source_file until we find a .git/ dir; return the dir.

    Returns "" when no git repo root is found.
    """
    if not source_file:
        return ""
    p = Path(source_file).expanduser().resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return str(parent)
    return ""


_BENCHMARK_DIRS = ("op_tests", "tests", "benchmarks", "benchmark", "test", "perf")


_KNOWN_HARNESS_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("rmsnorm_quant", "add_rmsnorm_quant", "rmsnorm"),
        (
            "/sgl-workspace/aiter/op_tests/test_rmsnorm2dFusedAddQuant.py",
            "/sgl-workspace/aiter/op_tests/test_rmsnorm2d.py",
            "/sgl-workspace/aiter/op_tests/op_benchmarks/triton/bench_rmsnorm.py",
            "/sgl-workspace/sglang/sgl-kernel/benchmark/bench_rmsnorm.py",
        ),
    ),
    (
        ("activation", "act_and_mul", "silu"),
        (
            "/sgl-workspace/aiter/op_tests/test_activation.py",
            "/sgl-workspace/aiter/op_tests/op_benchmarks/triton/bench_ff_a16w16_fused.py",
        ),
    ),
    (
        ("paged_attention", "fmha", "attention"),
        (
            "/sgl-workspace/aiter/op_tests/test_pa.py",
            "/sgl-workspace/aiter/op_tests/op_benchmarks/triton/bench_pa_decode.py",
            "/sgl-workspace/aiter/op_tests/op_benchmarks/triton/bench_pa_prefill.py",
        ),
    ),
)


def _known_harness_files(name: str, source_file: str) -> list[Path]:
    blob = f"{name} {source_file}".lower()
    out: list[Path] = []
    for markers, paths in _KNOWN_HARNESS_HINTS:
        if any(marker in blob for marker in markers):
            out.extend(Path(p) for p in paths if Path(p).exists())
    return out


def find_benchmark_files(name: str, repo_root: str, source_file: str = "") -> list[str]:
    """Look for Python/cpp test/benchmark files matching the kernel keywords
    inside well-known sub-directories of *repo_root*. Returns absolute paths.
    """
    known = _known_harness_files(name, source_file)
    if not repo_root:
        return [str(p) for p in known[:10]]
    keywords = _candidate_keywords(name)
    # Source-file stem and a no-underscore variant catch repos that name tests
    # slightly differently from the kernel symbol (e.g. cross_device_reduce vs
    # custom_allreduce in aiter).
    if source_file:
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
            try:
                proc = subprocess.run(
                    [
                        "grep", "-rln",
                        "--include=*.py", "--include=*.cpp", "--include=*.cu",
                        "--include=*.cuh", "--include=*.hip", "--include=*.sh",
                        keyword, str(sub_root),
                    ],
                    text=True, capture_output=True, timeout=15,
                )
            except Exception:
                continue
            if proc.returncode not in (0, 1):
                continue
            for line in (proc.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if not p.exists():
                    continue
                base = p.name.lower()
                # Prefer files clearly named test/benchmark/bench
                if any(tag in base for tag in ("test_", "_test.", "bench", "benchmark")):
                    found.append(p)
                else:
                    found.append(p)
    seen: set[str] = set()
    unique: list[str] = []
    for p in found:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        unique.append(s)
    # Demote multi-GPU / distributed tests to the end: backends running on a
    # single Ray worker can't satisfy them, and they tend to make agents bail.
    def _is_multigpu(path_str: str) -> bool:
        low = path_str.lower()
        return any(tag in low for tag in ("multigpu", "multi_gpu", "multinode", "/dist/", "_dist_"))
    unique.sort(key=_is_multigpu)
    return unique[:10]


_PYBIND_PARENT_DIRS = ("csrc/pybind", "csrc/python", "python_bindings")
# A pybind11 registration shim is typically <2KB and contains nothing but
# `PYBIND11_MODULE(...) { ... }`. Trace events for ASM-implemented kernels
# point at this shim instead of the device code, which makes optimization
# pointless (r17 GEAK selected a 233-byte file and correctly returned 1.00x
# after concluding "no device code here"). Promote shims to real .cu/.cuh.
def _is_pybind_shim(source_file: str) -> bool:
    if not source_file:
        return False
    p = Path(source_file)
    if not any(d in source_file for d in _PYBIND_PARENT_DIRS):
        return False
    if not source_file.endswith((".cu", ".cpp", ".cc")):
        return False
    try:
        if p.stat().st_size > 2048:
            return False
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return "PYBIND11_MODULE" in text or "pybind11" in text


def upgrade_pybind_shim_source(source_file: str, kernel_name: str,
                               kernel_repo: str) -> str:
    """If `source_file` is a tiny pybind11 registration TU, walk the repo to
    find the real device-code .cu/.cuh that implements `kernel_name`. Returns
    the upgraded path, or `source_file` unchanged if no better target is
    found.

    The selection prefers `csrc/py_itfs_cu/*<stem>*.cu` and
    `csrc/include/*<stem>*.cuh` (where `<stem>` is the pybind file name with
    `_pybind` stripped), then falls back to a grep for the kernel symbol
    name in any .cu/.cuh under the repo.
    """
    if not _is_pybind_shim(source_file):
        return source_file
    repo = Path(kernel_repo) if kernel_repo else Path(source_file).parent.parent.parent
    if not repo.is_dir():
        return source_file
    stem = Path(source_file).stem.replace("_pybind", "").replace("_asm_pybind", "")
    # Strategy 1: same-stem file under py_itfs_cu / kernels / include.
    for sub in ("csrc/py_itfs_cu", "csrc/kernels", "csrc/include", "csrc/asm"):
        for ext in (".cu", ".cuh", ".cpp", ".h", ".hpp"):
            candidates = list((repo / sub).glob(f"*{stem}*{ext}")) if (repo / sub).is_dir() else []
            for c in candidates:
                # Skip another pybind shim if we hit one.
                if _is_pybind_shim(str(c)):
                    continue
                if c.stat().st_size > 2048:
                    return str(c)
    # Strategy 2: ripgrep the kernel symbol name. Demangle keywords help.
    sym = kernel_name.split("(")[0].split("<")[0].split("::")[-1]
    if sym and len(sym) >= 4:
        for ext in ("*.cu", "*.cuh", "*.hip"):
            for f in repo.rglob(ext):
                if _is_pybind_shim(str(f)):
                    continue
                try:
                    if sym in f.read_text(encoding="utf-8", errors="replace"):
                        if f.stat().st_size > 2048:
                            return str(f)
                except Exception:
                    continue
    return source_file


# ---------------------------------------------------------------------------
# B path: prefer TraceLens-generated kernel_summary csv
# ---------------------------------------------------------------------------
# TraceLens already does the right thing — it uses the cuda graph parent +
# hipGraphLaunch parent_cpu_op join to count REAL device time per kernel
# launch (and skips the host-side `cuda::synchronize` python-frame events
# that confused our raw parser). Reading TraceLens's own csv output is
# both (a) more accurate and (b) auto-tracks future TraceLens improvements.
# Schema (from `--enable_kernel_summary --output_csvs_dir <dir>`):
#   columns: Parent op category | Parent cpu_op | Kernel name |
#            Kernel stream | Kernel duration (µs)_sum |
#            Kernel duration (µs)_count | Kernel duration (µs)_mean | ...
# We pick top_k by sum-duration and aggregate variants whose names
# differ only by template parameters (Cijk_*MT256x16x64 vs Cijk_*MT16x16x512
# stay separate — they ARE different rocBLAS variants — but the same
# bf16 add_rmsnorm_quant_kernel called from 2 stream_ids merges).

_TRACELENS_REQUIRED_COLS = (
    "Kernel name",
    "Kernel duration (\u00b5s)_sum",
    "Kernel duration (\u00b5s)_count",
)


def parse_tracelens_kernel_summary(
    csv_path: Path, top_k: int,
) -> list[dict[str, Any]] | None:
    """Read TraceLens's ``kernel_summary.csv`` → wrapper schema (top_k).

    Returns ``None`` if the csv doesn't exist, is missing required columns,
    or contains zero kernel rows — the caller falls back to the raw parser.
    """
    import csv as _csv  # localised; we don't want a hard import at module load
    if not csv_path.exists():
        return None
    try:
        with csv_path.open(encoding="utf-8") as f:
            rdr = _csv.DictReader(f)
            field_lookup = {h.lower(): h for h in (rdr.fieldnames or [])}
            for required in _TRACELENS_REQUIRED_COLS:
                if required.lower() not in field_lookup:
                    return None
            name_col = field_lookup["kernel name"]
            sum_col = field_lookup["kernel duration (\u00b5s)_sum"]
            count_col = field_lookup["kernel duration (\u00b5s)_count"]
            # #125: Parent op category column is optional — older TraceLens
            # builds may not emit it; we degrade gracefully when missing.
            cat_col = field_lookup.get("parent op category")
            rows = []
            for r in rdr:
                name = (r.get(name_col) or "").strip()
                if not name:
                    continue
                try:
                    dur = float(r.get(sum_col) or 0)
                except ValueError:
                    continue
                try:
                    cnt = int(float(r.get(count_col) or 0))
                except ValueError:
                    cnt = 0
                if dur <= 0:
                    continue
                row = {"name": name, "duration_us": dur, "call_count": cnt,
                       "tracelens_category": ""}
                if cat_col:
                    row["tracelens_category"] = (r.get(cat_col) or "").strip()
                rows.append(row)
    except Exception:
        return None
    if not rows:
        return None
    # Aggregate same-name rows (same kernel launched on multiple streams).
    agg: dict[str, dict[str, Any]] = {}
    for r in rows:
        bucket = agg.setdefault(
            r["name"],
            {"name": r["name"], "duration_us": 0.0, "call_count": 0,
             "source_file": "", "source_type": "unknown", "shapes": [],
             "tracelens_category": ""},
        )
        bucket["duration_us"] += r["duration_us"]
        bucket["call_count"] += r["call_count"]
        # First non-empty category wins (variants of same kernel rarely
        # straddle different framework-side categories).
        if not bucket["tracelens_category"] and r.get("tracelens_category"):
            bucket["tracelens_category"] = r["tracelens_category"]
    total_dur = sum(c["duration_us"] for c in agg.values())
    top = sorted(agg.values(), key=lambda x: x["duration_us"], reverse=True)[:top_k]
    return _finalize_candidates(top, total_dur=total_dur)


# ---------------------------------------------------------------------------
# #125: TraceLens orchestrator structured outputs (category_data/*.json)
# ---------------------------------------------------------------------------
# When the standalone-analysis-orchestrator skill runs (Step 5+), TraceLens
# emits per-category JSON files carrying the GEAK-required triple
# (kernel_category, kernel_shape, kernel_path) directly. We consume them
# when present; otherwise we fall back to the kernel_summary.csv parser
# (with shape backfill from raw trace) or, as a last resort, the legacy
# raw-trace parser.
#
# The schema is still being negotiated with the TraceLens team
# (traceLens-issue.md §3.4.5 / §7.2). We accept three observed layouts:
#
#   (1) {"category": "GEMM", "kernels": [{name, duration_us, shape,
#                                          source_path, ...}, ...]}
#   (2) {"name": "GEMM", "items": [...]}                # alt key names
#   (3) {"GEMM": [...], "SDPA": [...]}                  # flat dict-of-lists


def _extract_category_kernels(payload: Any) -> list[dict[str, Any]]:
    """Best-effort kernel list extractor for category_data layouts."""
    out: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        cat = (payload.get("category") or payload.get("name") or "").strip()
        kernels = payload.get("kernels") or payload.get("items") or payload.get("entries")
        if isinstance(kernels, list):
            for k in kernels:
                if isinstance(k, dict):
                    item = dict(k)
                    if cat and not item.get("category"):
                        item["category"] = cat
                    out.append(item)
            return out
        # Flat dict-of-lists layout.
        for k, v in payload.items():
            if isinstance(v, list):
                for entry in v:
                    if isinstance(entry, dict):
                        item = dict(entry)
                        item.setdefault("category", k)
                        out.append(item)
    elif isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict):
                out.append(entry)
    return out


def parse_tracelens_category_data(
    category_dir: Path, top_k: int,
) -> list[dict[str, Any]] | None:
    """Read TraceLens ``category_data/*.json`` → candidates with the GEAK
    (kernel_category, shape, source_path) triple (#125).

    Returns ``None`` when the directory doesn't exist, is empty, or yields
    no parseable kernels. Caller falls back to the csv parser.
    """
    if not category_dir.exists() or not category_dir.is_dir():
        return None

    rows: list[dict[str, Any]] = []
    for jp in sorted(category_dir.glob("*.json")):
        try:
            payload = json.loads(jp.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for entry in _extract_category_kernels(payload):
            name = str(entry.get("name") or entry.get("kernel_name") or "").strip()
            if not name:
                continue
            try:
                dur = float(
                    entry.get("duration_us")
                    or entry.get("duration")
                    or entry.get("sum_duration_us")
                    or 0
                )
            except (TypeError, ValueError):
                continue
            if dur <= 0:
                continue
            try:
                cnt = int(float(entry.get("call_count") or entry.get("count") or 0))
            except (TypeError, ValueError):
                cnt = 0
            shape = entry.get("shape") or entry.get("input_shape") or entry.get("shapes")
            shapes_field: list[Any] = []
            if isinstance(shape, list):
                shapes_field = [shape] if shape and not isinstance(shape[0], (list, dict)) else list(shape)
            elif shape:
                shapes_field = [shape]
            source_path = str(
                entry.get("source_path") or entry.get("source_file")
                or entry.get("path") or ""
            ).strip()
            rows.append({
                "name": name,
                "duration_us": dur,
                "call_count": cnt,
                "tracelens_category": str(entry.get("category") or "").strip(),
                "shapes": shapes_field,
                "source_file": source_path,
            })

    if not rows:
        return None

    # Aggregate same-name rows (rare across category files but defensive).
    agg: dict[str, dict[str, Any]] = {}
    for r in rows:
        bucket = agg.setdefault(
            r["name"],
            {"name": r["name"], "duration_us": 0.0, "call_count": 0,
             "source_file": r["source_file"],
             "source_type": "unknown",
             "shapes": [],
             "tracelens_category": r["tracelens_category"]},
        )
        bucket["duration_us"] += r["duration_us"]
        bucket["call_count"] += r["call_count"]
        for sh in r["shapes"]:
            if sh not in bucket["shapes"]:
                bucket["shapes"].append(sh)
        if not bucket["source_file"] and r["source_file"]:
            bucket["source_file"] = r["source_file"]
        if not bucket["tracelens_category"] and r["tracelens_category"]:
            bucket["tracelens_category"] = r["tracelens_category"]

    total_dur = sum(c["duration_us"] for c in agg.values())
    top = sorted(agg.values(), key=lambda x: x["duration_us"], reverse=True)[:top_k]
    return _finalize_candidates(top, total_dur=total_dur)


def augment_csv_candidates_with_raw_shapes(
    candidates: list[dict[str, Any]], trace_files: list[Path],
) -> None:
    """Best-effort shape backfill for csv-only candidates (#125).

    The csv parser doesn't carry shape info because ``kernel_summary.csv``
    aggregates by kernel name and drops per-launch ``Input Dims``. We mine
    the raw trace once for any candidate whose ``shapes`` list is still
    empty. Failures are silent — this is enrichment, not a hard requirement.
    """
    needs_shape = [c for c in candidates if not c.get("shapes")]
    if not needs_shape:
        return
    name_set = {c["name"] for c in needs_shape}
    found: dict[str, list[Any]] = {}
    for tf in trace_files:
        try:
            payload = open_json(tf)
        except Exception:
            continue
        events = payload.get("traceEvents") if isinstance(payload, dict) else None
        if not isinstance(events, list):
            continue
        for ev in events:
            if not isinstance(ev, dict) or not is_kernel_event(ev):
                continue
            ename = str(ev.get("kernel_name") or ev.get("name") or "")
            if ename not in name_set:
                continue
            shape = extract_shape(ev)
            if not shape:
                continue
            bucket = found.setdefault(ename, [])
            if shape not in bucket:
                bucket.append(shape)
    for cand in needs_shape:
        sh = found.get(cand["name"])
        if sh:
            cand["shapes"] = sh


def derive_kernel_category(candidate: dict[str, Any]) -> str:
    """Map a candidate to its GEAK-facing kernel category (#125).

    Priority:
      1. Explicit category from TraceLens (csv ``Parent op category`` or
         category_data ``category``)
      2. Heuristic from kernel name (gemm / attn / norm / activation / …)
      3. ``unknown``
    """
    cat = (candidate.get("tracelens_category") or "").strip()
    if cat:
        return cat
    name = str(candidate.get("name") or "").lower()
    if any(t in name for t in ("gemm", "matmul", "rocblas", "hipblas",
                                "cijk", "sgemm", "hgemm")):
        return "GEMM"
    if any(t in name for t in ("attention", "attn", "fmha",
                                "paged_attention", "flash")):
        return "SDPA"
    if "rmsnorm" in name or "layernorm" in name or "norm_kernel" in name:
        return "LayerNorm"
    if "act_and_mul" in name or "silu" in name or "gelu" in name or "activation" in name:
        return "Activation"
    if "moe" in name or "topk" in name or "expert" in name:
        return "MoE"
    if "softmax" in name:
        return "Softmax"
    if "embed" in name:
        return "Embedding"
    if "reduce" in name or "all_reduce" in name or "all_gather" in name:
        return "Communication"
    if "triton" in name:
        return "Triton"
    if "elementwise" in name or "binary" in name:
        return "Elementwise"
    return "unknown"


def is_multigpu_kernel(name: str, source_file: str) -> bool:
    """Heuristic: kernel is a multi-GPU collective if name/source hints it."""
    blob = f"{name} {source_file}".lower()
    return any(tag in blob for tag in (
        "all_reduce", "allreduce", "all_gather", "allgather",
        "reduce_scatter", "broadcast", "p2p", "send_recv",
        "cross_device", "rank_signal", "ranksignals",
        "/dist/", "dist/", "communicator",
    ))


def analyze_trace_files(trace_files: list[Path], top_k: int) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    total_dur = 0.0

    for trace_file in trace_files:
        try:
            payload = open_json(trace_file)
        except Exception:
            continue

        if isinstance(payload.get("kernels"), list):
            events = payload["kernels"]
        else:
            events = payload.get("traceEvents", [])
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict) or not is_kernel_event(event):
                continue
            name = str(event.get("kernel_name") or event.get("name") or "unknown_kernel")
            dur = float(event.get("dur") or event.get("duration_us") or event.get("duration") or 0)
            if dur <= 0:
                continue
            total_dur += dur
            item = aggregates.setdefault(
                name,
                {
                    "name": name,
                    "duration_us": 0.0,
                    "call_count": 0,
                    "source_file": "",
                    "source_type": "unknown",
                    "shapes": [],
                },
            )
            item["duration_us"] += dur
            item["call_count"] += 1
            if not item.get("_extracted_source_checked"):
                item["source_file"] = extract_source_file(event)
                item["_extracted_source_checked"] = True
            shape = extract_shape(event)
            if shape and shape not in item["shapes"]:
                item["shapes"].append(shape)

    candidates = sorted(aggregates.values(), key=lambda x: x["duration_us"], reverse=True)
    top = candidates[:top_k]
    return _finalize_candidates(top, total_dur=total_dur)


def _finalize_candidates(
    top: list[dict[str, Any]], *, total_dur: float | None = None,
) -> list[dict[str, Any]]:
    """Apply source resolution / pybind upgrade / backend recommend / notes.

    Shared post-processing for both the raw-trace parser
    (``analyze_trace_files``) and the TraceLens csv parser
    (``parse_tracelens_kernel_summary``). Mutates ``top`` in place.
    """
    sum_dur = total_dur if total_dur is not None else sum(it.get("duration_us", 0.0) for it in top)
    sum_dur = sum_dur or 1.0
    for idx, item in enumerate(top, 1):
        item.pop("_extracted_source_checked", None)
        item.setdefault("source_file", "")
        item.setdefault("source_type", "unknown")
        item.setdefault("shapes", [])
        item["kernel_id"] = f"k{idx:03d}"
        # Honour pre-computed gpu_pct (B path), else compute now.
        if not item.get("gpu_pct"):
            item["gpu_pct"] = round(item["duration_us"] / sum_dur * 100.0, 3)
        item["duration_us"] = round(item["duration_us"], 3)
        if not item.get("source_file"):
            item["source_file"] = locate_source_via_grep(item["name"])
        # Trace events for ASM-implemented kernels point at a tiny pybind11
        # shim TU (e.g. csrc/pybind/gemm_a16w16_asm_pybind.cu, 233 B). Try to
        # promote that to the real device code so optimization is meaningful.
        item["kernel_repo"] = find_repo_root(item.get("source_file", ""))
        item["source_file"] = upgrade_pybind_shim_source(
            item.get("source_file", ""), item["name"], item.get("kernel_repo", "")
        )
        # Re-resolve repo in case the upgraded path lives in a different repo
        # (rare, but defensive).
        item["kernel_repo"] = find_repo_root(item.get("source_file", "")) or item["kernel_repo"]
        item["source_type"] = source_type_for(item["name"], item.get("source_file", ""))
        # Re-classify thin vendor / dispatch wrappers (the .so/.co loader
        # shims that have no rewritable kernel body). Even when the file
        # extension is .cu/.py and source_type would otherwise be hip_cpp/
        # python, downgrade to vendor_binary so recommend_backends() drops
        # this candidate (or the runner skips it entirely).
        if (item["source_type"] != "vendor_binary"
                and is_vendor_dispatch_wrapper(item["name"], item.get("source_file", ""))):
            item["source_type"] = "vendor_binary"
            item["vendor_dispatch_wrapper"] = True
        item["runtime_generated_kernel"] = is_runtime_generated_kernel(
            item["name"], item.get("source_file", "")
        )
        item["reusable_native_kernel"] = is_reusable_native_kernel(item)
        item["benchmark_files"] = find_benchmark_files(
            item["name"], item.get("kernel_repo", ""), item.get("source_file", "")
        )
        item["is_multigpu"] = is_multigpu_kernel(item["name"], item.get("source_file", ""))
        # Communication / collective kernels need real multi-GPU launches to
        # measure XGMI / RDMA paths; single-GPU "rank-slice surrogate"
        # microbenchmarks only exercise LDS+L2 and produce misleading speedups.
        # Default to 2 GPUs for multi-GPU kernels (sufficient for most all-reduce
        # / all-gather / send-recv shapes); compute kernels stay at 1.
        item["num_gpus_recommended"] = 2 if item["is_multigpu"] else 1
        item["recommended_backends"] = recommend_backends(item)
        item["optimization_notes"] = build_notes(item)
        # #125: surface a stable kernel_category for GEAK to dispatch on,
        # plus source_path mirror for parity with TraceLens category_data.
        # shape is already populated in `shapes`.
        item["kernel_category"] = derive_kernel_category(item)
        item.setdefault("source_path", item.get("source_file", ""))
    return top


def recommend_backends(candidate: dict[str, Any]) -> list[str]:
    source_type = candidate.get("source_type")
    if not candidate.get("source_file"):
        return []
    if not candidate.get("reusable_native_kernel", is_reusable_native_kernel(candidate)):
        return []
    if source_type == "vendor_binary":
        return []
    if source_type == "runtime_generated":
        return []
    if source_type == "hip_cpp":
        return ["geak", "claude", "codex"]
    if source_type == "triton":
        return ["geak", "claude", "codex"]
    if source_type == "python":
        return ["claude", "codex"]
    return ["claude", "codex"]


def build_notes(candidate: dict[str, Any]) -> str:
    if not candidate.get("source_file"):
        return "source file not resolved; backend dispatch will be skipped"
    if candidate.get("runtime_generated_kernel", is_runtime_generated_kernel(
        str(candidate.get("name") or ""), str(candidate.get("source_file") or "")
    )):
        return (
            "runtime-generated torch.compile/Inductor kernel; not reusable, "
            "kernel-opt disabled"
        )
    if not candidate.get("reusable_native_kernel", is_reusable_native_kernel(candidate)):
        return "not a reusable native source; kernel-opt disabled"
    return f"resolved source: {candidate['source_file']}"


def select_perf_report_cli(log_path: Path) -> str:
    """Pick the TraceLens perf-report CLI for the current install (#124).

    Prefers ``TraceLens_generate_perf_report_pytorch_inference`` (the correct
    entry for vLLM/SGLang inference traces) and falls back to the legacy
    ``TraceLens_generate_perf_report_pytorch`` when only an older TraceLens
    build is on PATH. Raises if neither is available.
    """
    if shutil.which(INFERENCE_PERF_CLI):
        append_log(log_path, f"perf report CLI: {INFERENCE_PERF_CLI} (TraceLens #124)")
        return INFERENCE_PERF_CLI
    if shutil.which(LEGACY_PERF_CLI):
        append_log(
            log_path,
            f"WARNING: {INFERENCE_PERF_CLI} not on PATH; falling back to "
            f"{LEGACY_PERF_CLI} (legacy TraceLens build)",
        )
        return LEGACY_PERF_CLI
    raise RuntimeError(
        f"No TraceLens perf-report CLI found on PATH. "
        f"Looked for {INFERENCE_PERF_CLI!r} (preferred, #124) and "
        f"{LEGACY_PERF_CLI!r} (legacy fallback)."
    )


def run_command(cmd: list[str], *, cwd: Path | None, log_path: Path, timeout_s: int) -> int:
    append_log(log_path, f"$ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    append_log(log_path, proc.stdout or "")
    append_log(log_path, f"[exit_code] {proc.returncode}")
    return proc.returncode


def roofline_match_key(name: str) -> str:
    """Normalize trace and rocprof names enough to join roofline data."""
    raw = name or ""
    lower = raw.lower()
    if "cijk_" in lower:
        return "hipblaslt_gemm"
    if "gemm_a16w16_asm" in lower or "a16w16" in lower:
        return "aiter_asm_gemm"
    if "attn_fwd" in lower or "flash_attn" in lower:
        return "attention"
    if "moe_ck2stages" in lower or "moe_ck_tile" in lower:
        return "moe_gemm"
    if "vectorized_layer_norm" in lower or "rms_norm" in lower:
        return "rms_norm"
    if "topk" in lower:
        return "topk"
    if "rope" in lower or "rotary" in lower:
        return "rope"
    if "nccl" in lower or "allreduce" in lower:
        return "allreduce"
    if "copy" in lower or "memcpy" in lower:
        return "memcpy"
    if "softmax" in lower:
        return "softmax"
    if "skinny" in lower:
        return "skinny_gemm"
    return lower[:80]


def load_roofline_results(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("name"):
            continue
        out[roofline_match_key(str(row["name"]))] = row
    return out


def merge_roofline_into_candidates(
    candidates: list[dict[str, Any]],
    roofline_by_name: dict[str, dict[str, Any]],
) -> None:
    for item in candidates:
        if not isinstance(item, dict):
            continue
        roofline = roofline_by_name.get(roofline_match_key(str(item.get("name") or "")))
        if roofline:
            item["bottleneck"] = roofline.get("bottleneck", "unknown")
            item["arithmetic_intensity"] = roofline.get("arithmetic_intensity")
            item["compute_utilization_pct"] = roofline.get("compute_utilization_pct", 0.0)
            item["bandwidth_utilization_pct"] = roofline.get("bandwidth_utilization_pct", 0.0)
            item["suggestion"] = roofline.get("suggestion", "")
            item["recommended_actions"] = roofline.get("recommended_actions") or []
            item["roofline_name"] = roofline.get("name")
        else:
            item.setdefault("bottleneck", "unknown")
            item.setdefault("arithmetic_intensity", None)
            item.setdefault("compute_utilization_pct", 0.0)
            item.setdefault("bandwidth_utilization_pct", 0.0)
            item.setdefault("recommended_actions", [])


def write_reports(
    run_dir: Path,
    *,
    trace_input_type: str,
    trace_files: list[Path],
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, str]:
    tracelens_dir = run_dir / "tracelens"
    (tracelens_dir / "system_findings").mkdir(parents=True, exist_ok=True)
    (tracelens_dir / "category_findings").mkdir(parents=True, exist_ok=True)

    manifest = {
        "trace_input": str(Path(args.trace_input).resolve()),
        "trace_input_type": trace_input_type,
        "trace_files": [str(p) for p in trace_files],
        "created_at": utc_now(),
    }
    report = {
        "model_name": args.model_name,
        "framework": args.framework,
        "target_platform": args.target_platform,
        "analysis_mode": args.analysis_mode,
        "runtime_env": args.runtime_env,
        "trace_input_type": trace_input_type,
        "hot_kernels": candidates,
        "source": "tracelens_analysis",
        "dry_run": args.dry_run,
    }
    atomic_write_json(run_dir / "trace_input_manifest.json", manifest)
    atomic_write_json(tracelens_dir / "tracelens_report.json", report)
    atomic_write_json(run_dir / "kernel_candidates.json", {"hot_kernels": candidates, **report})

    md_path = tracelens_dir / "standalone_analysis.md"
    lines = [
        "# TraceLens Standalone Analysis",
        "",
        f"- Model: {args.model_name}",
        f"- Framework: {args.framework}",
        f"- Target platform: {args.target_platform}",
        f"- Trace input type: {trace_input_type}",
        "",
        "## Hot Kernels",
        "",
    ]
    for item in candidates:
        bottleneck = item.get("bottleneck") or "unknown"
        lines.append(
            f"- `{item['kernel_id']}` `{item['name']}`: {item['gpu_pct']}% GPU, "
            f"{item['call_count']} calls, bottleneck `{bottleneck}`, "
            f"source `{item.get('source_file') or 'unresolved'}`"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    readable_report = tracelens_dir / "tracelens_report.md"
    readable_report.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    if args.compat_report_path:
        compat_path = Path(args.compat_report_path)
        compat_path.parent.mkdir(parents=True, exist_ok=True)
        compat_path.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "trace_input_manifest": str(run_dir / "trace_input_manifest.json"),
        "kernel_candidates": str(run_dir / "kernel_candidates.json"),
        "tracelens_report_json": str(tracelens_dir / "tracelens_report.json"),
        "trace_report_path": str(md_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Kernel Agent TraceLens analysis tool")
    parser.add_argument("--trace-input", required=True)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--framework", default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--target-platform", default="MI355X")
    parser.add_argument("--analysis-mode", default="default")
    parser.add_argument("--runtime-env", default="local")
    parser.add_argument("--workspace-path", default=os.environ.get("WORKSPACE_PATH", "/workspace"))
    parser.add_argument("--tracelens-root", default=os.environ.get("TRACELENS_ROOT", DEFAULT_TRACELENS_ROOT))
    parser.add_argument("--roofline-json", default="")
    parser.add_argument("--compat-report-path", default="")
    parser.add_argument("--budget-minutes", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help=(
            "Disable TraceLens trace splitting (#127). When set, the raw "
            "filtered trace is fed directly to TraceLens; useful for debugging "
            "or when the splitter binary isn't available."
        ),
    )
    parser.add_argument(
        "--split-num-steps",
        type=int,
        default=int(os.environ.get("TRACELENS_SPLIT_NUM_STEPS", "32") or 32),
        help=(
            "Number of steady-state iterations for the splitter to extract "
            "(#127). Maps to --num-steps on TraceLens.TraceUtils."
            "split_inference_trace_annotation."
        ),
    )
    parser.add_argument(
        "--split-conc",
        default=os.environ.get("TRACELENS_SPLIT_CONC", "") or os.environ.get("CONC", ""),
        help=(
            "Expected peak concurrency for the splitter (#127). Maps to "
            "--CONC. Defaults to $CONC when set."
        ),
    )
    parser.add_argument(
        "--split-osl",
        default=os.environ.get("TRACELENS_SPLIT_OSL", "") or os.environ.get("OSL", ""),
        help=(
            "Maximum output sequence length hint for the splitter (#127). "
            "Maps to --OSL. Defaults to $OSL when set."
        ),
    )
    args = parser.parse_args()

    session_id = args.session_id or uuid.uuid4().hex[:12]
    run_id = f"tl-{uuid.uuid4().hex[:8]}"
    started_at = utc_now()
    root = Path(args.workspace_path) / "kernel-agent"
    run_dir = root / "runs" / session_id
    log_path = run_dir / "logs" / "tracelens_analysis" / f"{run_id}.log"
    status_path = run_dir / "status" / "tracelens_analysis" / f"{run_id}.json"
    artifacts: dict[str, str] = {}

    try:
        update_status(status_path, state="running", current_step="discover_trace_input",
                      log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                      started_at=started_at)
        trace_input = Path(args.trace_input).expanduser().resolve()
        trace_input_type, trace_files = discover_trace_inputs(trace_input)
        append_log(log_path, f"trace_input_type={trace_input_type}")
        append_log(log_path, f"trace_files={len(trace_files)}")

        if not args.dry_run:
            update_status(status_path, state="running", current_step="install_tracelens",
                          log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                          started_at=started_at)
            tl_root = Path(args.tracelens_root)
            if not tl_root.exists() and Path(LOCAL_BUNDLE_TRACELENS_ROOT).exists():
                tl_root = Path(LOCAL_BUNDLE_TRACELENS_ROOT)
            if not tl_root.exists():
                raise FileNotFoundError(f"TraceLens root not found: {tl_root}")
            run_command([sys.executable, "-m", "pip", "install", "-e", "."],
                        cwd=tl_root, log_path=log_path,
                        timeout_s=max(60, int(args.budget_minutes * 60)))
            perf_cli = select_perf_report_cli(log_path)
            rc = run_command([perf_cli, "--help"],
                             cwd=tl_root, log_path=log_path, timeout_s=60)
            if rc != 0:
                raise RuntimeError(f"{perf_cli} --help failed")
            skill = tl_root / "TraceLens/AgenticMode/Standalone/.cursor/skills/standalone-analysis-orchestrator.md"
            if not skill.exists():
                raise FileNotFoundError(f"TraceLens standalone skill not found: {skill}")
            append_log(log_path, f"TraceLens skill: {skill}")

            tracelens_dir = run_dir / "tracelens"
            tracelens_dir.mkdir(parents=True, exist_ok=True)

            # ---- #127: split inference trace into steady-state chunks ----
            # The filtered trace from vLLM/SGLang spans the full benchmark
            # window (warmup + tear-down + steady-state mixed together).
            # TraceLens's perf report expects a single steady-state chunk.
            # Use TraceLens's own splitter to produce
            # mixed_steady_state_*_trace.json.gz, then feed the first chunk
            # to TraceLens_generate_perf_report_pytorch_inference. Fail-soft:
            # if the splitter is unavailable or produces no output, fall back
            # to the original filtered trace (legacy behaviour).
            cli_trace_path = trace_files[0]
            if not args.skip_split:
                update_status(status_path, state="running", current_step="split_trace",
                              log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                              started_at=started_at)
                split_dir = tracelens_dir / "trace_split"
                split_dir.mkdir(parents=True, exist_ok=True)
                # TraceLens splitter CLI (real interface):
                #   python -m TraceLens.TraceUtils.split_inference_trace_annotation
                #     <trace_path> -o <output_dir> --find-steady-state
                #     [--num-steps N] [--CONC C] [--OSL O]
                # `--platform` does not exist; --find-steady-state writes
                # mixed_steady_state_* / decode_only_steady_state_* /
                # prefilldecode_steady_state_* into output_dir.
                split_cmd = [
                    sys.executable, "-m",
                    "TraceLens.TraceUtils.split_inference_trace_annotation",
                    str(trace_files[0]),
                    "-o", str(split_dir),
                    "--find-steady-state",
                    "--num-steps", str(max(8, int(args.split_num_steps or 32))),
                ]
                conc = args.split_conc or os.environ.get("CONC", "").strip()
                if str(conc).strip():
                    split_cmd += ["--CONC", str(conc).strip()]
                osl = args.split_osl or os.environ.get("OSL", "").strip()
                if str(osl).strip():
                    split_cmd += ["--OSL", str(osl).strip()]
                split_rc = run_command(
                    split_cmd,
                    cwd=tl_root,
                    log_path=log_path,
                    timeout_s=max(60, int(args.budget_minutes * 60)),
                )
                # Splitter writes <type>_steady_state_*.json[.gz]; accept any of
                # the three windows (mixed first, then decode_only, then
                # prefilldecode) and prefer mixed for perf-report consumption.
                def _collect(prefix: str) -> list[Path]:
                    out: list[Path] = []
                    for ext in ("trace.json.gz", "json.gz", "trace.json", "json"):
                        out.extend(sorted(split_dir.rglob(f"{prefix}_steady_state_*.{ext}")))
                    return out

                mixed_chunks = _collect("mixed")
                decode_chunks = _collect("decode_only")
                prefill_chunks = _collect("prefilldecode")
                steady_chunks = mixed_chunks or decode_chunks or prefill_chunks
                if split_rc == 0 and steady_chunks:
                    cli_trace_path = steady_chunks[0]
                    artifacts["tracelens_trace_split_dir"] = str(split_dir)
                    artifacts["tracelens_steady_state_trace"] = str(cli_trace_path)
                    append_log(
                        log_path,
                        f"trace split OK: mixed={len(mixed_chunks)} "
                        f"decode_only={len(decode_chunks)} "
                        f"prefilldecode={len(prefill_chunks)}; "
                        f"using {cli_trace_path.name} for perf report",
                    )
                else:
                    append_log(
                        log_path,
                        f"WARNING: trace split unavailable "
                        f"(rc={split_rc}, mixed={len(mixed_chunks)}, "
                        f"decode_only={len(decode_chunks)}, "
                        f"prefilldecode={len(prefill_chunks)}); "
                        f"falling back to filtered trace {trace_files[0].name}",
                    )

            update_status(status_path, state="running", current_step="run_tracelens_cli",
                          log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                          started_at=started_at)
            csv_dir = tracelens_dir / "csvs"
            xlsx_path = tracelens_dir / "perf_report.xlsx"
            rc = run_command([
                perf_cli,
                "--profile_json_path", str(cli_trace_path),
                "--output_xlsx_path", str(xlsx_path),
                "--output_csvs_dir", str(csv_dir),
                "--include_unlinked_kernels",
                "--enable_kernel_summary",
            ], cwd=None, log_path=log_path,
                timeout_s=max(60, int(args.budget_minutes * 60)))
            if rc != 0:
                append_log(log_path, "WARNING: TraceLens report CLI failed; falling back to raw trace parser")
            else:
                artifacts["tracelens_xlsx"] = str(xlsx_path)
                artifacts["tracelens_csv_dir"] = str(csv_dir)
        else:
            append_log(log_path, "[dry-run] skipping TraceLens install and external CLI")

        update_status(status_path, state="running", current_step="extract_hot_kernels",
                      log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                      started_at=started_at)
        # Hot-kernel extraction priority (#125):
        #   1. TraceLens orchestrator's category_data/*.json — the GEAK
        #      (kernel_category, shape, source_path) triple is on disk
        #   2. TraceLens kernel_summary.csv — name + duration + framework
        #      category; we backfill shape from the raw trace
        #   3. Raw-trace parser — last-resort legacy path
        candidates = None
        tl_csv_dir_local = artifacts.get("tracelens_csv_dir")
        if tl_csv_dir_local:
            # category_data may sit peer-of-csv or inside-csv depending on
            # the TraceLens release; probe both.
            for cat_dir_candidate in (
                Path(tl_csv_dir_local).parent / "category_data",
                Path(tl_csv_dir_local) / "category_data",
            ):
                if cat_dir_candidate.exists() and cat_dir_candidate.is_dir():
                    cat_candidates = parse_tracelens_category_data(
                        cat_dir_candidate, args.top_k,
                    )
                    if cat_candidates:
                        candidates = cat_candidates
                        artifacts["tracelens_category_dir"] = str(cat_dir_candidate)
                        append_log(
                            log_path,
                            f"hot kernels from TraceLens category_data "
                            f"({len(candidates)}, src={cat_dir_candidate})",
                        )
                        break
        if not candidates and tl_csv_dir_local:
            tl_csv = Path(tl_csv_dir_local) / "kernel_summary.csv"
            candidates = parse_tracelens_kernel_summary(tl_csv, args.top_k)
            if candidates:
                # #125: csv has no shape info; mine raw trace as best-effort.
                augment_csv_candidates_with_raw_shapes(candidates, trace_files)
                append_log(log_path,
                           f"hot kernels from TraceLens csv "
                           f"({len(candidates)}, src={tl_csv}); "
                           f"shape backfill from raw trace")
        if not candidates:
            append_log(log_path,
                       "fallback: raw trace parser (TraceLens outputs unavailable)")
            candidates = analyze_trace_files(trace_files, args.top_k)
        roofline_by_name = load_roofline_results(args.roofline_json)
        if roofline_by_name:
            append_log(log_path, f"merged roofline results: {len(roofline_by_name)} kernels")
        merge_roofline_into_candidates(candidates, roofline_by_name)
        artifacts.update(write_reports(run_dir, trace_input_type=trace_input_type,
                                       trace_files=trace_files, candidates=candidates,
                                       args=args))
        if args.roofline_json:
            artifacts["roofline_json"] = str(Path(args.roofline_json).expanduser())
        artifacts["cli_log_path"] = str(log_path)
        artifacts["status_path"] = str(status_path)

        result = {
            "tool": "tracelens_analysis",
            "session_id": session_id,
            "run_id": run_id,
            "trace_input_type": trace_input_type,
            "hot_kernels": candidates,
            "trace_report_path": artifacts["trace_report_path"],
            "cli_log_path": str(log_path),
            "status_path": str(status_path),
            "artifact_paths": artifacts,
        }
        atomic_write_json(run_dir / "session_state.json", {
            "session_id": session_id,
            "last_tool": "tracelens_analysis",
            "last_run_id": run_id,
            "updated_at": utc_now(),
            "model_name": args.model_name,
            "framework": args.framework,
        })
        update_status(status_path, state="succeeded", current_step="done",
                      log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                      started_at=started_at)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        append_log(log_path, f"[error] {type(exc).__name__}: {exc}")
        update_status(status_path, state="failed", current_step="failed",
                      log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                      started_at=started_at, error=f"{type(exc).__name__}: {exc}")
        print(json.dumps({
            "tool": "tracelens_analysis",
            "session_id": session_id,
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "cli_log_path": str(log_path),
            "status_path": str(status_path),
        }, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
