#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""TraceLens analysis tool for the resident Kernel Agent skill.

Conservative: records every step, writes a stable artifact set, supports TraceLens
capture directories, and has a dry-run path that works without TraceLens installed.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tracelens_skill_runner import (
    aggregate_by_source_function,
    discover_capture_folder,
    extract_idle_pct_from_analysis_md,
    normalize_upstream_category,
    parse_analysis_md,
    run_tracelens_skill,
)

# Standalone-tool workspace-root resolver (cannot import inference_optimizer.paths; see _paths.py).
from _paths import workspace_root


HIGH_IDLE_PCT_THRESHOLD_DEFAULT = 80.0
HIGH_IDLE_PCT_THRESHOLD_ENV = "HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD"


def _resolve_idle_pct_threshold() -> float:
    """Return the idle-percent gate threshold (default 80.0%).

    Relaxed from the docx §2 ~20% target to 80% because real SGLang inference traces
    sit structurally at ~50-60% idle (host scheduling + JIT/launch overhead), which the
    20% gate over-suppressed. Pin via ``HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD``.
    """
    raw = os.environ.get(HIGH_IDLE_PCT_THRESHOLD_ENV, "").strip()
    if not raw:
        return HIGH_IDLE_PCT_THRESHOLD_DEFAULT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return HIGH_IDLE_PCT_THRESHOLD_DEFAULT
    if value < 0.0:
        return HIGH_IDLE_PCT_THRESHOLD_DEFAULT
    return value


def _build_high_idle_warning(
    *, idle_pct: float, threshold_pct: float, report_path: Path,
) -> dict[str, Any]:
    """Build the ``trace_health_warnings[]`` entry for a high-idle trace (consumed by trace_analyze_handler T4 to route to param optimization)."""
    return {
        "code": "high_gpu_idle_pct",
        "severity": "warning",
        "idle_pct": round(idle_pct, 2),
        "threshold_pct": round(threshold_pct, 2),
        "source": str(report_path),
        "message": (
            f"GPU was idle {idle_pct:.2f}% of trace wall time (threshold "
            f"{threshold_pct:.2f}%). Per Report_Interfacing.docx §2 "
            "(idle-gate sanity check in Possible Approach (Hyperloom v3)), "
            "kernel-level rewriting is unlikely to improve end-to-end "
            "latency in this regime — recommend parameter optimization "
            "(batch size, KV-cache shape, prefill/decode split) over "
            "per-kernel rewrites. Hyperloom is suppressing the hot-kernel "
            "candidate list and surfacing this warning so the Coordinator "
            "can route to params/backends."
        ),
    }


def _build_trace_split_warning(
    *, trace_input: Path, split_dir: Path, split_rc: int,
    mixed_count: int, decode_count: int, prefilldecode_count: int,
) -> dict[str, Any]:
    return {
        "code": "trace_split_no_steady_state",
        "severity": "warning",
        "trace_input": str(trace_input),
        "split_dir": str(split_dir),
        "split_returncode": split_rc,
        "mixed_count": mixed_count,
        "decode_only_count": decode_count,
        "prefilldecode_count": prefilldecode_count,
        "message": (
            "TraceLens splitter produced no steady-state chunks; refusing "
            "to analyze the raw trace because that can report misleading "
            "high idle and suppress valid kernel opportunities. Verify the "
            "profile request used TraceLens-compatible annotations and enough "
            "NUM_PROMPTS to reach the requested start_step/num_steps window."
        ),
    }


def _check_selected_chunk_has_gpu_events(
    *,
    split_dir: Path,
    selected_chunk: Path,
    mode: str,
    available_modes: "dict[str, tuple[str, list[Path]]]",
) -> "dict[str, Any] | None":
    """Verify the ``--steady-state-mode``-selected chunk actually contains GPU events (N25, SOLAR-10.7B case).

    Reads the splitter's ``execution_details.csv`` (num_gpu_events / gpu_busy_duration);
    returns ``None`` when the chunk carries real GPU work, else a ``steady_state_chunk_empty``
    warning the caller appends and raises on. A data-validity gate, not a reorder — falling
    back to another mode is the operator's call (never a silent swap).
    """
    import csv

    details_path = split_dir / "execution_details.csv"
    if not details_path.is_file():
        # No CSV (older TraceLens): let the chunk through (T3 idle gate still applies).
        return None
    try:
        with details_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error):
        return None

    selected_resolved = str(selected_chunk.resolve())
    selected_row: dict[str, str] | None = None
    for row in rows:
        out_path = row.get("output_path", "")
        if not out_path:
            continue
        try:
            if str(Path(out_path).resolve()) == selected_resolved:
                selected_row = row
                break
        except (OSError, ValueError):
            continue
    if selected_row is None:
        return None

    def _f(name: str) -> float:
        try:
            return float(selected_row.get(name) or "0") or 0.0
        except (TypeError, ValueError):
            return 0.0

    num_gpu_events = int(_f("num_gpu_events"))
    gpu_busy_duration = _f("gpu_busy_duration")
    if num_gpu_events > 0 and gpu_busy_duration > 0.0:
        return None  # chunk carries real GPU work -- proceed.

    # Empty: surface which other modes' chunks DO have gpu events for re-issue.
    non_empty_modes: list[str] = []
    for other_mode, (label, chunks) in available_modes.items():
        if other_mode == mode or not chunks:
            continue
        other_resolved = str(chunks[0].resolve())
        for row in rows:
            try:
                if str(Path(row.get("output_path", "")).resolve()) != other_resolved:
                    continue
            except (OSError, ValueError):
                continue
            try:
                other_events = int(float(row.get("num_gpu_events") or "0"))
                other_busy = float(row.get("gpu_busy_duration") or "0")
            except (TypeError, ValueError):
                other_events, other_busy = 0, 0.0
            if other_events > 0 and other_busy > 0.0:
                non_empty_modes.append(other_mode)
            break

    return {
        "code": "steady_state_chunk_empty",
        "severity": "blocking",
        "requested_mode": mode,
        "selected_chunk": str(selected_chunk),
        "num_gpu_events": num_gpu_events,
        "gpu_busy_duration": gpu_busy_duration,
        "non_empty_modes": non_empty_modes,
        "remediation": (
            "Re-issue roofline with env "
            "INFERENCE_OPTIMIZER_STEADY_STATE_MODE set to one of "
            f"{non_empty_modes or ['(none of the splitter outputs has GPU events; re-profile required)']}. "
            "Most common cause: short / batched workload (e.g. "
            "NUM_PROMPTS<=CONC*OSL/2) where prefill is burst-shaped so "
            "the mixed window degenerates to PD=0; switching to "
            "'prefilldecode' picks up the real GEMM/attention region."
        ),
        "message": (
            f"TraceLens splitter selected chunk ({mode}) has "
            f"num_gpu_events={num_gpu_events}, "
            f"gpu_busy_duration={gpu_busy_duration:.1f}us -- structurally "
            "empty. Refusing to feed it into TraceLens analysis (would "
            "produce a misleading high-idle Executive Summary). The "
            "coordinator should re-issue roofline with a different "
            "--steady-state-mode per the 'remediation' field."
        ),
    }


# N36 chunk-quality gate (busy_ratio threshold + alternate-mode lookup).
# Complements the N25 structural gate: a structurally-non-empty chunk can still be
# garbage (e.g. 0.063% busy). Emits ``steady_state_chunk_low_quality`` (retry allowlist)
# when an alternate mode is materially better; returns None otherwise (avoids retry-loop).
_DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO = 0.05  # 5%
# Alternate must beat the requested mode by this margin to avoid thrashing between equally-bad modes.
_CHUNK_QUALITY_ALTERNATE_MARGIN = 0.10  # 10 ppt


def _resolve_min_busy_ratio() -> float:
    raw = os.environ.get(
        "INFERENCE_OPTIMIZER_CHUNK_QUALITY_MIN_BUSY_RATIO", "",
    ).strip()
    if not raw:
        return _DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO
    try:
        v = float(raw)
        return v if 0.0 <= v <= 1.0 else _DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO
    except ValueError:
        return _DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO


def _busy_ratio(num_events: float, busy_us: float, dur_us: float) -> float | None:
    """Return ``busy_us / dur_us`` or ``None`` when undefined (caller defers to N25)."""
    if dur_us <= 0.0 or num_events <= 0:
        return None
    return max(0.0, min(1.0, busy_us / dur_us))


def _check_selected_chunk_has_gpu_events_quality(
    *,
    split_dir: "Path",
    selected_chunk: "Path",
    mode: str,
    available_modes: "dict[str, tuple[str, list[Path]]]",
) -> "dict[str, Any] | None":
    """Quality gate complementing N25's structural gate.

    ``None`` when the chunk is acceptable (busy_ratio >= threshold), no alternate is
    materially better, or the CSV/row is absent. Otherwise a ``steady_state_chunk_low_quality``
    warning (same shape as N25 for the N26 retry path). See the module-level N36 comment.
    """
    import csv as _csv

    details_path = split_dir / "execution_details.csv"
    if not details_path.is_file():
        return None
    try:
        with details_path.open("r", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
    except (OSError, _csv.Error):
        return None

    def _row_for(chunk_path: "Path") -> "dict[str, str] | None":
        resolved = str(chunk_path.resolve())
        for row in rows:
            out_path = row.get("output_path", "")
            if not out_path:
                continue
            try:
                if str(Path(out_path).resolve()) == resolved:
                    return row
            except (OSError, ValueError):
                continue
        return None

    def _stats(row: "dict[str, str] | None") -> "tuple[int, float, float]":
        if row is None:
            return 0, 0.0, 0.0
        def _f(k: str) -> float:
            try:
                return float(row.get(k) or "0") or 0.0
            except (TypeError, ValueError):
                return 0.0
        return int(_f("num_gpu_events")), _f("gpu_busy_duration"), _f("gpu_duration")

    selected_row = _row_for(selected_chunk)
    if selected_row is None:
        return None
    sel_events, sel_busy, sel_dur = _stats(selected_row)
    sel_ratio = _busy_ratio(sel_events, sel_busy, sel_dur)
    if sel_ratio is None:
        # Can't measure ratio; structural-empty case already covered. Defer.
        return None
    threshold = _resolve_min_busy_ratio()
    if sel_ratio >= threshold:
        return None

    # Below threshold: look for an alternate mode with materially higher busy_ratio.
    alternates: list[tuple[str, float]] = []
    for other_mode, (_label, chunks) in available_modes.items():
        if other_mode == mode or not chunks:
            continue
        other_row = _row_for(chunks[0])
        if other_row is None:
            continue
        oth_events, oth_busy, oth_dur = _stats(other_row)
        oth_ratio = _busy_ratio(oth_events, oth_busy, oth_dur)
        if oth_ratio is None:
            continue
        if oth_ratio >= threshold and (oth_ratio - sel_ratio) >= _CHUNK_QUALITY_ALTERNATE_MARGIN:
            alternates.append((other_mode, oth_ratio))
    if not alternates:
        return None  # No better mode exists; let roofline_failure_streak path handle.

    # Best alternate first (the retry path picks the head of non_empty_modes).
    alternates.sort(key=lambda mr: -mr[1])
    non_empty_modes = [m for m, _r in alternates]
    return {
        "code": "steady_state_chunk_low_quality",
        "severity": "blocking",
        "requested_mode": mode,
        "selected_chunk": str(selected_chunk),
        "num_gpu_events": sel_events,
        "gpu_busy_duration": sel_busy,
        "gpu_duration": sel_dur,
        "busy_ratio": sel_ratio,
        "threshold": threshold,
        "non_empty_modes": non_empty_modes,
        "alternate_busy_ratios": dict(alternates),
        "remediation": (
            "Re-issue roofline with env "
            "INFERENCE_OPTIMIZER_STEADY_STATE_MODE set to one of "
            f"{non_empty_modes}. The TraceLens splitter chunk for the "
            f"requested mode '{mode}' is {sel_ratio*100:.2f}% busy "
            f"(threshold {threshold*100:.0f}%) -- non-empty but "
            "substantively garbage. Most common cause for prefill-"
            "heavy workloads: profile window misalignment "
            "(_workload_envs.delay_iters formula only considers OSL, "
            "so high-ISL workloads land in pure-decode windows)."
        ),
        "message": (
            f"TraceLens splitter selected chunk ({mode}) busy_ratio="
            f"{sel_ratio*100:.3f}% (events={sel_events}, "
            f"busy={sel_busy:.1f}us / dur={sel_dur:.1f}us) -- below "
            f"the {threshold*100:.0f}% threshold and alternate "
            f"modes have higher busy_ratio. Refusing to feed it into "
            "TraceLens analysis (would produce a misleading analysis.md "
            "with reusable_native_kernel_ids=[] and stall the "
            "optimization loop, per DSR1-0528 10k/1k case)."
        ),
    }


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
# No in-process default for the TraceLens roots: TRACELENS_ROOT comes from env / --tracelens-root
# (fail loudly if absent), and the internal extension is opt-in via TRACELENS_INTERNAL_ROOT.
DEFAULT_TRACELENS_INTERNAL_ROOT = ""


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
    updated_at = utc_now()
    payload: dict[str, Any] = {
        "tool": "tracelens_analysis",
        "run_id": run_id,
        "state": state,
        "current_step": current_step,
        "pid": os.getpid(),
        "started_at": started_at,
        "updated_at": updated_at,
        "log_path": str(log_path),
        "artifact_paths": artifact_paths,
        "offset_bytes": log_path.stat().st_size if log_path.exists() else 0,
        "last_lines": read_last_lines(log_path),
    }
    # Emit ended_at + duration_seconds on terminal states for the session_breakdown timeline (additive).
    if state in ("succeeded", "failed", "aborted", "cancelled"):
        payload["ended_at"] = updated_at
        try:
            start_dt = datetime.fromisoformat(started_at)
            end_dt = datetime.fromisoformat(updated_at)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            payload["duration_seconds"] = max(
                0.0, (end_dt - start_dt).total_seconds(),
            )
        except (ValueError, TypeError):
            payload["duration_seconds"] = None
    if error:
        payload["error"] = error
    atomic_write_json(status_path, payload)


def open_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


def count_gpu_kernel_events(trace_file: Path, max_events: int = 1_000_000) -> int:
    """Count GPU kernel events in a torch_profiler trace (pre-flight check for CPU-only traces).

    Counts only real GPU kernels via :func:`is_kernel_event`, not host-side wrappers.
    """
    try:
        payload = open_json(trace_file)
    except Exception:
        return 0
    events = payload.get("traceEvents") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return 0
    count = 0
    for ev in events:
        if isinstance(ev, dict) and is_kernel_event(ev):
            count += 1
            if count >= max_events:
                break
    return count


def _trace_input_sort_key(path: Path) -> tuple[int, str]:
    """Prefer the merged annotated trace over rank/phase shards during directory discovery (TraceLens splitter needs the large trace)."""
    name = path.name
    if name.startswith("merged-"):
        return (0, name)
    if re.search(r"TP-\d+-DECODE\.trace\.json(?:\.gz)?$", name):
        return (2, name)
    if name.startswith("bs_") or name.startswith("graph_capture"):
        return (3, name)
    return (1, name)


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
    unique.sort(key=_trace_input_sort_key)
    if not unique:
        raise FileNotFoundError(f"no trace files found under capture directory: {trace_input}")
    return "capture_dir", unique


def is_kernel_event(event: dict[str, Any]) -> bool:
    """Strict GPU-kernel filter: only ``cat == 'kernel'`` events (excludes host-side sync/launch wrappers that would eclipse real kernels)."""
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


_FLYDSL_SOURCE_MARKERS = (
    "import flydsl",
    "from flydsl",
    "@flyc.kernel",
    "@flyc.jit",
    "flydsl.compiler",
    "flydsl.expr",
)
_FLYDSL_SCAN_BYTES = 4096


def _looks_like_flydsl_source(source_file: str) -> bool:
    """Return True when ``source_file`` is a FlyDSL kernel source (content-sniff first 4 KiB for FlyDSL markers)."""
    if not source_file or not source_file.endswith(".py"):
        return False
    try:
        with open(source_file, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(_FLYDSL_SCAN_BYTES)
    except OSError:
        return False
    return any(marker in head for marker in _FLYDSL_SOURCE_MARKERS)


_FLYDSL_PSEUDO_OP_NAME_MARKERS = (
    "pseudo_op::moe_flydsl_",
    "pseudo_op::flydsl_",
)


def source_type_for(name: str, source_file: str) -> str:
    lower_name = name.lower()
    lower_file = source_file.lower()
    # PR #668: synthetic ``pseudo_op::*flydsl_*`` carry no source_file; match the name prefix directly.
    if any(marker in lower_name for marker in _FLYDSL_PSEUDO_OP_NAME_MARKERS):
        return "flydsl"
    if is_runtime_generated_kernel(name, source_file):
        return "runtime_generated"
    if source_file.endswith((".cu", ".cuh", ".hip", ".cpp", ".h", ".hpp")):
        return "hip_cpp"
    if "triton" in lower_name and source_file.endswith(".py"):
        return "triton"
    if _looks_like_flydsl_source(source_file):
        return "flydsl"
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
@functools.lru_cache(maxsize=1)
def _framework_patch_roots() -> tuple[str, ...]:
    """Framework install roots (from framework_paths.resolve_patch_target_roots); also emits a lower-case variant of each (``/app/ATOM/atom/`` -> ``/app/atom/atom/``) for case-insensitive matching."""
    try:
        from inference_optimizer.orchestrator.framework_paths import (
            resolve_patch_target_roots,
        )

        roots = resolve_patch_target_roots()
    except ImportError:
        from apply_kernel_patch import known_target_roots

        roots = known_target_roots()
    out: list[str] = []
    seen: set[str] = set()
    for root in roots:
        for variant in (root, root.lower()):
            if variant and variant not in seen:
                seen.add(variant)
                out.append(variant)
    return tuple(out)


@functools.lru_cache(maxsize=1)
def _aiter_csrc_root() -> str:
    """aiter's device-source root (``.../aiter_meta/csrc/``) from the installed package; empty if aiter is unimportable."""
    try:
        import aiter.jit.core as _jc  # type: ignore
    except Exception:
        return ""
    raw = (getattr(_jc, "AITER_CSRC_DIR", "") or "").replace(os.sep, "/")
    return (raw.rstrip("/") + "/") if raw else ""


def _flydsl_reusable_roots() -> tuple[str, ...]:
    """FlyDSL checkout root(s) for PR #668 moe_flydsl pseudo-ops, lower-cased ($DSL2_ROOT/$FLYDSL_ROOT take precedence over the WekaFS default)."""
    out: list[str] = []
    for env_key in ("DSL2_ROOT", "FLYDSL_ROOT"):
        val = (os.environ.get(env_key, "") or "").strip()
        if val:
            out.append((val.rstrip("/") + "/").lower())
    for default in ("/wekafs/yunkai/flydsl/", "/sgl-workspace/flydsl/"):
        if default not in out:
            out.append(default)
    return tuple(out)


def _reusable_roots() -> tuple[str, ...]:
    """Discovered framework roots + aiter csrc + FlyDSL checkout (dynamic)."""
    roots = _framework_patch_roots()
    csrc = _aiter_csrc_root()
    if csrc and csrc not in roots:
        roots = roots + (csrc,)
    for fly in _flydsl_reusable_roots():
        if fly not in roots:
            roots = roots + (fly,)
    return roots
# Kernel-name substrings marking an op non-patchable regardless of source: vendor BLAS, collectives, copies.
_NON_PATCHABLE_NAME_MARKERS: tuple[str, ...] = (
    "rocblas",
    "hipblas",
    "hipblaslt",
    "rocblaslt",
    "tensile",
    "miopen",
    "ck_kernels",
    "nccl",
    "rccl",
    "hipmemcpy",
    "__amd_rocclr_copybuffer",
    "aten::copy",
)

# Vendor BLAS / closed-source compute backends. A candidate whose runtime
# implementation is one of these has *no rewritable source* regardless of
# which Python file the symbol resolves to: the device body lives in a
# precompiled vendor binary (Tensile/hipBLASLt/rocBLAS/CK kernels), and the
# attributed ``source_file`` is only the framework dispatch site. Matched
# against the candidate's ``library`` field (case-insensitive).
_VENDOR_BACKEND_LIBRARIES: frozenset[str] = frozenset(
    {
        "tensile",
        "hipblas",
        "hipblaslt",
        "rocblas",
        "rocblaslt",
        "ck",
        "composable_kernel",
        "ck_kernels",
        "miopen",
    }
)

# torch ``__torch_function__`` / ``__torch_dispatch__`` interception shims.
# These files intercept a tensor op and forward it to a vendor backend
# (e.g. vLLM ``parameter.py`` forwards ``rocm_unquantized_gemm`` to Tensile);
# the file itself contains no rewritable device kernel, so a symbol that
# resolves here is a dispatch stub, not an editable kernel. Matched as a
# POSIX path suffix against the resolved ``source_file``.
_TORCH_DISPATCH_SHIM_SOURCES: tuple[str, ...] = (
    "vllm/model_executor/parameter.py",
)


def is_torch_dispatch_shim_source(source_file: str) -> bool:
    """True when ``source_file`` is a known torch-dispatch interception shim.

    These ``__torch_function__`` / ``__torch_dispatch__`` files forward a
    tensor op to a vendor backend and hold no rewritable device kernel, so a
    symbol attributed here must be treated as a non-reusable dispatch stub
    (same handling as ``@compile_ops`` JIT stubs in ``aiter/ops/moe_op.py``).
    """
    posix = str(source_file or "").replace("\\", "/")
    return any(posix.endswith(suffix) for suffix in _TORCH_DISPATCH_SHIM_SOURCES)


def is_runtime_generated_kernel(name: str, source_file: str) -> bool:
    """Return True for torch.compile / Inductor / cache-generated kernels (not portable across serving runs)."""
    lower_name = (name or "").lower()
    lower_file = (source_file or "").lower()
    if any(marker in lower_file for marker in _RUNTIME_GENERATED_SOURCE_MARKERS):
        return True
    if any(marker in lower_name for marker in _COMPILE_GENERATED_NAME_MARKERS):
        # A stable in-repo Triton source can still be reusable.
        return not any(root in lower_file for root in _reusable_roots())
    return False


def classify_patchability(candidate: dict[str, Any]) -> tuple[bool, str]:
    """Return ``(reusable, skip_reason)`` for a hot-kernel candidate.

    Single source of truth for the kernel-opt routing gate; ``skip_reason`` is empty
    when reusable, else a short audit explanation. Also rejects vendor/collective/native-op
    name markers (:data:`_NON_PATCHABLE_NAME_MARKERS`) and library-less ``aten::*`` ops.
    """
    source_file = str(candidate.get("source_file") or "")
    name = str(candidate.get("name") or "")
    lower_name = name.lower()
    if not source_file:
        return False, "source file not resolved"
    if candidate.get("source_type") == "vendor_binary":
        return False, "vendor binary (no rewritable source)"
    if candidate.get("vendor_dispatch_wrapper"):
        return False, f"vendor dispatch wrapper at {source_file}"
    if is_torch_dispatch_shim_source(source_file):
        return False, (
            f"torch dispatch shim (no rewritable kernel body): {source_file}"
        )
    for marker in _NON_PATCHABLE_NAME_MARKERS:
        if marker in lower_name:
            return False, (
                f"non-patchable kernel name marker '{marker}' in {name!r}"
            )
    library = str(candidate.get("library") or "").strip().lower()
    if library in _VENDOR_BACKEND_LIBRARIES:
        return False, (
            f"vendor backend library {candidate.get('library')!r} "
            "(precompiled binary, no rewritable source)"
        )
    if name.startswith("aten::"):
        if not library or library in {"tensile", "pytorch native"}:
            return False, (
                f"PyTorch native op {name!r} backed by "
                f"{candidate.get('library') or 'unknown'} library "
                "(typically Tensile / vendor backend)"
            )
    if is_runtime_generated_kernel(name, source_file):
        return False, (
            f"runtime-generated (torch.compile / Inductor cache): {source_file}"
        )
    lower_file = source_file.lower()
    if not any(root in lower_file for root in _reusable_roots()):
        return False, (
            f"source not under a reusable framework root: {source_file}"
        )
    source_type = candidate.get("source_type")
    if source_type not in {"hip_cpp", "triton", "python", "flydsl"}:
        return False, (
            f"source_type={source_type!r} not in {{hip_cpp, triton, python, flydsl}}"
        )
    return True, ""


def is_reusable_native_kernel(candidate: dict[str, Any]) -> bool:
    """Whether a candidate is safe to send to kernel optimization backends.

    Thin wrapper over :func:`classify_patchability` kept for backward
    compatibility with downstream consumers that only need the bool.
    """
    return classify_patchability(candidate)[0]


# Wrapper TUs that just dispatch to a precompiled .so/.co (no rewritable device body),
# detected by small file size + content signature. Conservative so real small kernels survive.
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
    """Heuristic: True when source_file is a thin dispatch wrapper around a precompiled vendor binary (.so/.co); nothing to rewrite."""
    nm = (name or "").lower()
    if any(kw in nm for kw in _VENDOR_KEYWORD_NAMES):
        return True
    if not source_file:
        return False
    p = Path(source_file)
    try:
        if not p.is_file():
            return False
        # >16 KB is presumed a real device kernel; ASM wrappers / pybind shims sit well below.
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
        # Itanium ABI <len><name>: slice manually so consecutive segments parse separately.
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
    """Find test/benchmark files matching the kernel keywords under *repo_root*'s known subdirs (absolute paths)."""
    known = _known_harness_files(name, source_file)
    if not repo_root:
        return [str(p) for p in known[:10]]
    keywords = _candidate_keywords(name)
    # Add the source stem (and no-underscore variant) for repos that name tests differently.
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
# A pybind11 registration shim (<2KB, just PYBIND11_MODULE) has no device code; detect so callers promote it to the real .cu/.cuh.
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


# ---------------------------------------------------------------------------
# PR-K: aiter @compile_ops launcher → device source promotion.
#
# aiter ships ``@compile_ops("module_<x>", gen_func=...)`` decorators on its
# top-level Python wrappers under ``aiter/ops/`` (e.g. ``aiter/ops/moe_op.py``
# for the ``ck_moe_stage1/2`` family). Trace events name the wrapper as the
# call-site, so torch.profiler / TraceLens propagate ``aiter/ops/moe_op.py``
# as the kernel's ``source_file`` — but the actual compute lives in
# ``csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`` (codegen entry
# that hipcc compiles into ``module_moe_ck2stages_*.so`` under
# ``<aiter>/jit/build/``). Rewriting the wrapper is a no-op at runtime
# because the compiled .so bypasses the wrapper via the @compile_ops
# dispatch path. Hyperloom Qwen3-30B-A3B-Base sessions burned 5+ rounds on
# wrapper rewrites that GEAK/Codex correctly compiled but had zero E2E
# effect (REVERT @-2.66%); the fix is to promote the wrapper to the
# device-source ``.cu`` BEFORE handing the candidate to the LLM.
#
# Scope is intentionally narrow: only kernels whose name matches one of the
# entries in :data:`_AITER_COMPILE_OPS_PROMOTIONS` get promoted, and only
# when the corresponding ``.cu`` exists on disk under the resolved
# ``kernel_repo``. Anything else falls through with the wrapper unchanged
# (LLM still gets the original signal — better than a fabricated guess).
# ---------------------------------------------------------------------------
# Each entry: (kernel_name_substring_lowercase, ordered_csrc_relpaths_to_try).
# First on-disk match wins. Order matters when a kernel name matches multiple
# patterns or when several ``.cu`` files implement variants of the same op.
_AITER_COMPILE_OPS_PROMOTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # ck_moe_stage1/2 — the .cu is the @compile_ops codegen entry (jit/build invalidated by PR-K before rebuild).
    ("ck_moe_stage", (
        "csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu",
    )),
    # topk_softmax decode kernels.
    ("topk_softmax_group", ("csrc/kernels/topk_softmax_kernels_group.cu",)),
    ("topk_softmax", ("csrc/kernels/topk_softmax_kernels.cu",)),
    # moe_align_block_size — pre-GEMM expert routing prep.
    ("moe_align_block_size", ("csrc/kernels/moe_align_block_size_kernels.cu",)),
    # moe_fused_gate — gating + top-k in fused form.
    ("moe_fused_gate", ("csrc/kernels/moe_fused_gate.cu",)),
)

# Fallback aiter editable-checkout root when find_repo_root can't resolve from a wheel-install wrapper (no csrc/).
_AITER_FALLBACK_REPO = "/sgl-workspace/aiter"


def upgrade_aiter_compile_ops_launcher(
    source_file: str, kernel_name: str, kernel_repo: str,
) -> str:
    """Promote an aiter ``@compile_ops`` Python wrapper to the device ``.cu``.

    Promotes when source_file is a ``.py`` under ``aiter/ops/``, kernel_name matches a
    :data:`_AITER_COMPILE_OPS_PROMOTIONS` pattern, and the ``.cu`` exists under kernel_repo
    (or :data:`_AITER_FALLBACK_REPO`). Otherwise returns source_file unchanged.
    """
    if not source_file or not kernel_name:
        return source_file
    s = source_file.replace(os.sep, "/")
    if "/aiter/ops/" not in s or not s.endswith(".py"):
        return source_file

    name_lower = kernel_name.lower()
    matched_pattern: str | None = None
    matched_relpaths: tuple[str, ...] = ()
    for pattern, relpaths in _AITER_COMPILE_OPS_PROMOTIONS:
        if pattern in name_lower:
            matched_pattern = pattern
            matched_relpaths = relpaths
            break
    if matched_pattern is None:
        return source_file

    candidate_repos: list[str] = []
    if kernel_repo:
        candidate_repos.append(kernel_repo)
    elif source_file:
        derived = find_repo_root(source_file)
        if derived:
            candidate_repos.append(derived)
    if _AITER_FALLBACK_REPO not in candidate_repos:
        candidate_repos.append(_AITER_FALLBACK_REPO)

    seen: set[str] = set()
    for repo_str in candidate_repos:
        if not repo_str or repo_str in seen:
            continue
        seen.add(repo_str)
        repo = Path(repo_str)
        if not repo.is_dir():
            continue
        for relpath in matched_relpaths:
            candidate = repo / relpath
            if candidate.is_file():
                return str(candidate)
    return source_file


def upgrade_pybind_shim_source(source_file: str, kernel_name: str,
                               kernel_repo: str) -> str:
    """If `source_file` is a tiny pybind11 shim, find the real device .cu/.cuh implementing `kernel_name`.

    Prefers a same-stem file under csrc/py_itfs_cu|kernels|include, then greps the symbol.
    Returns `source_file` unchanged if no better target is found.
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
                if _is_pybind_shim(str(c)):
                    continue
                if c.stat().st_size > 2048:
                    return str(c)
    # Strategy 2: grep the kernel symbol name.
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


def _coerce_count(value: Any) -> int | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.startswith("np.float64(") and text.endswith(")"):
        text = text[len("np.float64("):-1]
    try:
        count = int(float(text))
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def _merge_shape_call(target: list[Any], shape: Any, call_num: int) -> None:
    for entry in target:
        if isinstance(entry, dict) and entry.get("shape") == shape:
            entry["call_num"] = int(entry.get("call_num") or 0) + call_num
            return
    target.append({"call_num": call_num, "shape": shape})


def _shape_call_entries(shapes: Any, call_num: Any = None) -> list[dict[str, Any]]:
    if not isinstance(shapes, list):
        return []
    count = _coerce_count(call_num) or 1
    entries: list[dict[str, Any]] = []
    for shape in shapes:
        if isinstance(shape, dict):
            value = shape.get("shape")
            shape_count = _coerce_count(shape.get("call_num")) or count
        else:
            value = shape
            shape_count = count
        if value in (None, "", [], ()):
            continue
        _merge_shape_call(entries, value, shape_count)
    return entries


def derive_kernel_category(candidate: dict[str, Any]) -> str:
    """Map a candidate to its GEAK-facing kernel category (#125).

    Priority: explicit TraceLens category (normalized), then a kernel-name heuristic, then ``unknown``.
    """
    cat = (candidate.get("tracelens_category") or "").strip()
    if cat:
        return normalize_upstream_category(cat)
    if candidate.get("source_type") == "flydsl":
        return "FlyDSL"
    name = str(candidate.get("name") or "").lower()
    if any(t in name for t in ("gemm", "matmul", "rocblas", "hipblas",
                                "cijk", "sgemm", "hgemm",
                                # PyTorch op-name variants for the raw-trace path.
                                "::mm", "::addmm", "::bmm")):
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


def load_op_category_map(
    perf_report_csv_dir: Path | str,
) -> dict[str, str]:
    """Read ``{name: raw TraceLens op category}`` from unified_perf_summary.csv (first non-empty per name; ``{}`` when absent)."""
    csv_path = Path(perf_report_csv_dir) / "unified_perf_summary.csv"
    if not csv_path.is_file():
        return {}
    import csv as _csv
    out: dict[str, str] = {}
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                name = str(row.get("name") or "").strip()
                cat = str(row.get("op category") or "").strip()
                if name and cat and name not in out:
                    out[name] = cat
    except (OSError, _csv.Error):
        return {}
    return out


def _finalize_candidates(
    top: list[dict[str, Any]], *, total_dur: float | None = None,
    perf_report_csv_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Shared post-processing for parsed candidate rows (source resolution / pybind upgrade / backend recommend / notes); mutates ``top`` in place.

    When ``perf_report_csv_dir`` is given, populates each item's ``tracelens_category`` from the CSV.
    """
    op_cat_map = (
        load_op_category_map(perf_report_csv_dir)
        if perf_report_csv_dir is not None else {}
    )
    sum_dur = total_dur if total_dur is not None else sum(it.get("duration_us", 0.0) for it in top)
    sum_dur = sum_dur or 1.0
    for idx, item in enumerate(top, 1):
        item.pop("_extracted_source_checked", None)
        item.setdefault("source_file", "")
        item.setdefault("source_type", "unknown")
        item.setdefault("shapes", [])
        # Mark trace-extracted shapes so the dispatch-time validator can tell their provenance.
        if item.get("shapes"):
            item.setdefault("shape_provenance", "torch_trace")
        item["kernel_id"] = f"k{idx:03d}"
        if not item.get("gpu_pct"):
            item["gpu_pct"] = round(item["duration_us"] / sum_dur * 100.0, 3)
        item["duration_us"] = round(item["duration_us"], 3)
        if not item.get("source_file"):
            item["source_file"] = locate_source_via_grep(item["name"])
        # Promote a tiny pybind shim TU to the real device code.
        item["kernel_repo"] = find_repo_root(item.get("source_file", ""))
        item["source_file"] = upgrade_pybind_shim_source(
            item.get("source_file", ""), item["name"], item.get("kernel_repo", "")
        )
        # PR-K: aiter @compile_ops launcher → device source promotion. Capture the wrapper
        # before the upgrade; only set launcher_source_file when promotion changed the path.
        wrapper_before_promotion = item.get("source_file", "")
        item["source_file"] = upgrade_aiter_compile_ops_launcher(
            wrapper_before_promotion, item["name"], item.get("kernel_repo", "")
        )
        if item["source_file"] != wrapper_before_promotion:
            item["launcher_source_file"] = wrapper_before_promotion
            item["source_promoted_from_launcher"] = True
        # Re-resolve repo in case the upgraded path lives in a different repo.
        item["kernel_repo"] = find_repo_root(item.get("source_file", "")) or item["kernel_repo"]
        item["source_type"] = source_type_for(item["name"], item.get("source_file", ""))
        # PR #668 FlyDSL pseudo-ops carry no real source_file; inject the real FlyDSL
        # MoE kernel source before the patchability gate so FlyDSL routes to GEAK.
        if item["source_type"] == "flydsl":
            _sf = str(item.get("source_file") or "").strip()
            if (not _sf) or (not os.path.isfile(_sf)):
                _fb = _resolve_flydsl_source_fallback()
                if _fb:
                    item["source_file"] = _fb
                    item["kernel_repo"] = (
                        find_repo_root(_fb) or item.get("kernel_repo", "")
                    )
                    item["flydsl_source_from_fallback"] = True
        # Downgrade thin .so/.co dispatch wrappers to vendor_binary so recommend_backends() drops them.
        if (item["source_type"] != "vendor_binary"
                and is_vendor_dispatch_wrapper(item["name"], item.get("source_file", ""))):
            item["source_type"] = "vendor_binary"
            item["vendor_dispatch_wrapper"] = True
        item["runtime_generated_kernel"] = is_runtime_generated_kernel(
            item["name"], item.get("source_file", "")
        )
        # One classify_patchability call yields both the routing bool and the audit skip_reason.
        reusable, skip_reason = classify_patchability(item)
        item["reusable_native_kernel"] = reusable
        item["skip_reason"] = skip_reason
        item["benchmark_files"] = find_benchmark_files(
            item["name"], item.get("kernel_repo", ""), item.get("source_file", "")
        )
        item["is_multigpu"] = is_multigpu_kernel(item["name"], item.get("source_file", ""))
        # Collective kernels need real multi-GPU launches; default to 2, compute kernels stay at 1.
        item["num_gpus_recommended"] = 2 if item["is_multigpu"] else 1
        item["recommended_backends"] = recommend_backends(item)
        item["optimization_notes"] = build_notes(item)
        # CSV category only fills in when tracelens_category wasn't already set (analysis.md path).
        if op_cat_map and not str(item.get("tracelens_category") or "").strip():
            csv_cat = op_cat_map.get(str(item.get("name") or ""))
            if csv_cat:
                item["tracelens_category"] = csv_cat
        # Stable kernel_category for GEAK dispatch + source_path mirror for consumers.
        item["kernel_category"] = derive_kernel_category(item)
        item.setdefault("source_path", item.get("source_file", ""))
    return top


def recommend_backends(candidate: dict[str, Any]) -> list[str]:
    """Recommend a backend ladder (GEAK first, then claude/codex) for a reusable native kernel.

    Returns ``[]`` for unresolved source, non-reusable, vendor-binary, or runtime-generated kernels.
    """
    source_type = candidate.get("source_type")
    if not candidate.get("source_file"):
        return []
    if not candidate.get("reusable_native_kernel", is_reusable_native_kernel(candidate)):
        return []
    if source_type == "vendor_binary":
        return []
    if source_type == "runtime_generated":
        return []
    # GEAK first; append cursor only when CURSOR_API_KEY is provisioned.
    cursor_tail = ["cursor"] if os.environ.get("CURSOR_API_KEY", "").strip() else []
    return ["geak", "claude", "codex"] + cursor_tail


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
    lower = (name or "").lower()
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
        if isinstance(row, dict) and row.get("name"):
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
            item.setdefault("compute_utilization_pct", None)
            item.setdefault("bandwidth_utilization_pct", None)
            item.setdefault("recommended_actions", [])


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _kernel_roofline_row(candidate: dict[str, Any]) -> dict[str, Any]:
    """Project one hot-kernel candidate into the kernel-roofline view."""
    arithmetic_intensity = _first_non_empty(
        candidate.get("arithmetic_intensity"),
        candidate.get("flops_per_byte"),
    )
    return {
        "kernel_id": candidate.get("kernel_id"),
        "name": candidate.get("name"),
        "gpu_pct": candidate.get("gpu_pct"),
        "duration_us": candidate.get("duration_us"),
        "call_count": candidate.get("call_count"),
        "kernel_category": candidate.get("kernel_category"),
        "source_file": candidate.get("source_file"),
        "bottleneck": _first_non_empty(
            candidate.get("bottleneck"),
            candidate.get("bound_type"),
        ),
        "bound_type": candidate.get("bound_type"),
        "arithmetic_intensity": arithmetic_intensity,
        "flops_per_byte": candidate.get("flops_per_byte"),
        "efficiency_percent": candidate.get("efficiency_percent"),
        "compute_utilization_pct": candidate.get("compute_utilization_pct"),
        "bandwidth_utilization_pct": candidate.get("bandwidth_utilization_pct"),
        "suggestion": candidate.get("suggestion") or "",
        "roofline_name": candidate.get("roofline_name"),
        "recommended_actions": list(candidate.get("recommended_actions") or []),
        "reusable_native_kernel": bool(candidate.get("reusable_native_kernel")),
        "rocprof_roofline": candidate.get("rocprof_roofline"),    }


def build_kernel_roofline_payload(
    *,
    trace_input: str,
    trace_input_type: str,
    analysis_md_path: str,
    kernel_candidates_path: str,
    roofline_json_path: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the per-kernel roofline sidecar (a view over candidates + optional --roofline-json; missing counters stay null)."""
    rows = [
        _kernel_roofline_row(candidate)
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    return {
        "schema_version": 1,
        "source": "tracelens_analysis",
        "trace_input": trace_input,
        "trace_input_type": trace_input_type,
        "analysis_md_path": analysis_md_path,
        "kernel_candidates_path": kernel_candidates_path,
        "roofline_json_path": roofline_json_path,
        "kernels": rows,
    }


def kernel_roofline_path_for_run(run_dir: Path) -> Path:
    """Return the session-level kernel roofline report path (stable dashboard pointer).

    PR-C layout is ``.../runs/<session_id>/<ts>_<run_id>/``; the pre-PR-C
    ``.../runs/<session_id>/`` layout is still handled by the fallback branch.
    """
    try:
        # PR-C layout: .../runs/<session_id>/<ts>_<run_id>/
        session_sub = run_dir.parent
        runs_dir = session_sub.parent
        kernel_agent_dir = runs_dir.parent
    except (IndexError, AttributeError):
        return run_dir / "reports" / "kernel_roofline.json"
    if runs_dir.name == "runs" and kernel_agent_dir.name == "kernel-agent":
        session_dir = kernel_agent_dir.parent
        return session_dir / "reports" / "kernel_roofline.json"
    # Backward-compat: pre-PR-C layout (.../runs/<session_id>/)
    try:
        runs_dir_legacy = run_dir.parent
        kernel_agent_dir_legacy = runs_dir_legacy.parent
    except (IndexError, AttributeError):
        return run_dir / "reports" / "kernel_roofline.json"
    if (
        runs_dir_legacy.name == "runs"
        and kernel_agent_dir_legacy.name == "kernel-agent"
    ):
        session_dir = kernel_agent_dir_legacy.parent
        return session_dir / "reports" / "kernel_roofline.json"
    return run_dir / "reports" / "kernel_roofline.json"


def _candidate_model_config_paths(model_name: str) -> list[Path]:
    text = str(model_name or "").strip()
    if not text:
        return []
    raw = Path(text).expanduser()
    candidates: list[Path] = []
    if raw.suffix == ".json":
        candidates.append(raw)
    candidates.append(raw / "config.json")
    models_root = Path(os.environ.get("HYPERLOOM_MODELS_ROOT", "/wekafs/models"))
    candidates.append(models_root / text / "config.json")
    candidates.append(models_root / raw.name / "config.json")
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def load_model_kernel_params(model_name: str) -> dict[str, Any]:
    """Read HF config.json and return attention parameters relevant to GEAK."""
    for config_path in _candidate_model_config_paths(model_name):
        if not config_path.is_file():
            continue
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        params: dict[str, Any] = {
            "MODEL_CONFIG_PATH": str(config_path),
        }
        has_mla_dims = any(
            cfg.get(key) is not None
            for key in ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
        )
        if cfg.get("head_dim") is not None:
            params["HEAD_SIZE"] = cfg.get("head_dim")
        elif not has_mla_dims:
            hidden = cfg.get("hidden_size")
            heads = cfg.get("num_attention_heads")
            if isinstance(hidden, int) and isinstance(heads, int) and heads > 0 and hidden % heads == 0:
                params["HEAD_SIZE"] = hidden // heads
        for src, dst in (
            ("qk_nope_head_dim", "QK_NOPE_HEAD_DIM"),
            ("qk_rope_head_dim", "QK_ROPE_HEAD_DIM"),
            ("v_head_dim", "V_HEAD_DIM"),
            ("kv_lora_rank", "KV_LORA_RANK"),
            ("num_attention_heads", "NUM_ATTENTION_HEADS"),
            ("num_key_value_heads", "NUM_KEY_VALUE_HEADS"),
            ("hidden_size", "HIDDEN_SIZE"),
        ):
            if cfg.get(src) is not None:
                params[dst] = cfg[src]
        return params
    return {}


_FLYDSL_TARGET_ARCH_BY_PLATFORM = {
    "mi300x": "gfx942",
    "mi308x": "gfx942",
    "mi325x": "gfx942",
    "mi355x": "gfx950",
}
_FLYDSL_SMEM_MARKERS = ("SmemAllocator", "SmemPtr", "smem_alloc")
_FLYDSL_BUFFER_LOAD_MARKERS = (
    "make_buffer_tensor", "BufferCopy", "rocdl", "buffer_load",
)


def _resolve_flydsl_source_fallback() -> str:
    """Resolve the real FlyDSL MoE kernel source ($DSL2_ROOT/kernels/moe_gemm_2stage.py) for synthetic PR #668 pseudo-ops; first existing path or ""."""
    roots = [
        os.environ.get("DSL2_ROOT", "").strip(),
        os.environ.get("FLYDSL_ROOT", "").strip(),
        "/wekafs/yunkai/FlyDSL",
        "/sgl-workspace/flydsl",
    ]
    for root in roots:
        if not root:
            continue
        cand = os.path.join(root, "kernels", "moe_gemm_2stage.py")
        if os.path.isfile(cand):
            return cand
    return ""


def _flydsl_kernel_params(
    source_file: str, target_platform: str,
) -> dict[str, Any]:
    """Return FlyDSL-specific kernel_params (target arch, JIT cache state, smem/buffer-load usage); best-effort, never raises."""
    params: dict[str, Any] = {}
    arch = _FLYDSL_TARGET_ARCH_BY_PLATFORM.get(
        (target_platform or "").strip().lower(),
    )
    if arch:
        params["FLYDSL_TARGET_ARCH"] = arch
    cache_dir = os.environ.get("FLYDSL_AUTOTUNE_CACHE_DIR", "").strip()
    if cache_dir:
        params["FLYDSL_AUTOTUNE_CACHE_DIR"] = cache_dir
    enable_cache = os.environ.get("FLYDSL_RUNTIME_ENABLE_CACHE", "").strip()
    if enable_cache:
        params["FLYDSL_RUNTIME_ENABLE_CACHE"] = enable_cache
    if source_file:
        try:
            with open(source_file, "r", encoding="utf-8", errors="replace") as fh:
                head = fh.read(_FLYDSL_SCAN_BYTES)
        except OSError:
            head = ""
        if head:
            if any(m in head for m in _FLYDSL_SMEM_MARKERS):
                params["FLYDSL_USES_SMEM"] = True
            if any(m in head for m in _FLYDSL_BUFFER_LOAD_MARKERS):
                params["FLYDSL_USES_BUFFER_LOAD"] = True
    return params


def enrich_candidates_with_runtime_metadata(
    candidates: list[dict[str, Any]], args: argparse.Namespace,
) -> None:
    """Attach stable runtime metadata fields before GEAK prompt generation."""
    framework = str(getattr(args, "framework", "") or "").strip()
    model_params = load_model_kernel_params(str(getattr(args, "model_name", "") or ""))
    target_platform = str(getattr(args, "target_platform", "") or "")
    runtime_flags = {
        "analysis_mode": getattr(args, "analysis_mode", ""),
        "runtime_env": getattr(args, "runtime_env", ""),
        "target_platform": target_platform,
    }
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if framework:
            item.setdefault("framework", framework)
            item.setdefault("backend", framework)
        if "input_shapes" not in item:
            item["input_shapes"] = _shape_call_entries(
                item.get("shapes", []) or [], item.get("call_count"),
            )
        item.setdefault("output_shapes", [])
        item.setdefault("input_dtypes", item.get("dtypes", []) or [])
        item.setdefault("output_dtypes", [])
        item.setdefault("runtime_args", {})
        item.setdefault("env_vars", {})
        item.setdefault("kernel_params", {})
        if model_params:
            params = item["kernel_params"]
            if isinstance(params, dict):
                for key, value in model_params.items():
                    params.setdefault(key, value)
        if item.get("source_type") == "flydsl":
            # PR #668 pseudo-ops carry no real source_file; inject the FlyDSL MoE kernel source.
            _sf2 = str(item.get("source_file") or "").strip()
            if (not _sf2) or (not os.path.isfile(_sf2)):
                fb = _resolve_flydsl_source_fallback()
                if fb:
                    item["source_file"] = fb
            flydsl_params = _flydsl_kernel_params(
                str(item.get("source_file") or ""), target_platform,
            )
            params = item["kernel_params"]
            if isinstance(params, dict):
                for key, value in flydsl_params.items():
                    params.setdefault(key, value)
        flags = item.get("runtime_flags")
        if not isinstance(flags, dict):
            flags = {}
            item["runtime_flags"] = flags
        for key, value in runtime_flags.items():
            if value not in (None, ""):
                flags.setdefault(key, value)
        flags.setdefault("is_multigpu", bool(item.get("is_multigpu")))
        flags.setdefault("num_gpus_recommended", item.get("num_gpus_recommended"))


def build_task_groups(
    candidates: list[dict[str, Any]],
    *,
    source_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate reusable candidates by AST-resolved source function (wrapper over aggregate_by_source_function).

    Only reusable_native_kernel candidates are grouped. Returns ``[]`` when none carries a
    parseable launcher path, so callers fall through to per-kernel dispatch.
    """
    reusable = [c for c in candidates if isinstance(c, dict) and c.get("reusable_native_kernel")]
    if not reusable:
        return []
    return aggregate_by_source_function(reusable, source_root=source_root)


def build_audit_summary(
    candidates: list[dict[str, Any]],
    *,
    trace_input: str,
    framework: str = "",
    target_platform: str = "",
    task_groups: list[dict[str, Any]] | None = None,
    trace_health_warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the ``tracelens/summary.json`` payload, splitting candidates into routable ``tasks`` and ``skipped`` (each with ``skip_reason``), preserving priority order. Pure function."""
    tasks: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        reusable = bool(cand.get("reusable_native_kernel"))
        compact = {
            "kernel_id":         cand.get("kernel_id"),
            "name":              cand.get("name"),
            "source_file":       cand.get("source_file") or "",
            "source_type":       cand.get("source_type") or "",
            "kernel_category":   cand.get("kernel_category") or "",
            "gpu_pct":           cand.get("gpu_pct"),
            "duration_us":       cand.get("duration_us"),
            "call_count":        cand.get("call_count"),
            "tracelens_pitem_rank":  cand.get("tracelens_pitem_rank"),
            "tracelens_pitem_title": cand.get("tracelens_pitem_title"),
            "bound_type":        cand.get("bound_type") or "",
        }
        if reusable:
            compact["recommended_backends"] = list(
                cand.get("recommended_backends") or []
            )
            tasks.append(compact)
        else:
            compact["skip_reason"] = cand.get("skip_reason") or "unknown"
            skipped.append(compact)
    # PR-B §1: compact task_group projections for the audit view (full rows live on kernel_candidates.json).
    group_entries: list[dict[str, Any]] = []
    for group in task_groups or []:
        if not isinstance(group, dict):
            continue
        group_entries.append({
            "task_group_id":        group.get("task_group_id"),
            "source_path":          group.get("source_path"),
            "definition_line":      group.get("definition_line"),
            "function_name":        group.get("function_name"),
            "ast_resolved":         bool(group.get("ast_resolved")),
            "primary_kernel_id":    group.get("primary_kernel_id"),
            "kernel_ids":           list(group.get("kernel_ids") or []),
            "row_count":            len(group.get("rows") or []),
            "aggregate_duration_us": group.get("aggregate_duration_us"),
            "aggregate_call_count": group.get("aggregate_call_count"),
            "aggregate_gpu_pct":    group.get("aggregate_gpu_pct"),
        })
    return {
        "generated_at":    utc_now(),
        "trace_input":     trace_input,
        "framework":       framework,
        "target_platform": target_platform,
        "task_count":      len(tasks),
        "skipped_count":   len(skipped),
        "task_group_count": len(group_entries),
        "tasks":           tasks,
        "skipped":         skipped,
        "task_groups":     group_entries,
        # T3: trace-quality findings (empty = healthy; non-empty explains an empty ``tasks``).
        "trace_health_warnings": list(trace_health_warnings or []),
    }


def write_reports(
    run_dir: Path,
    *,
    trace_input_type: str,
    trace_files: list[Path],
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
    existing_report_path: Path | None = None,
    trace_health_warnings: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Write Hyperloom-owned sidecar JSONs and surface the upstream ``analysis.md``.

    ``analysis.md`` is owned by the TraceLens SDK orchestrator and not copied/aliased (#203).
    Raises ``RuntimeError`` rather than fabricating a report when the orchestrator didn't produce it.
    """
    tracelens_dir = run_dir / "tracelens"
    (tracelens_dir / "system_findings").mkdir(parents=True, exist_ok=True)
    (tracelens_dir / "category_findings").mkdir(parents=True, exist_ok=True)
    enrich_candidates_with_runtime_metadata(candidates, args)

    manifest = {
        "trace_input": str(Path(args.trace_input).resolve()),
        "trace_input_type": trace_input_type,
        "trace_files": [str(p) for p in trace_files],
        "created_at": utc_now(),
    }
    # PR-B §1: aggregate reusable candidates into source-function task groups (additive; hot_kernels keeps full order).
    source_root_str = getattr(args, "source_root", None)
    task_groups = build_task_groups(candidates, source_root=source_root_str)
    report = {
        "model_name": args.model_name,
        "framework": args.framework,
        "target_platform": args.target_platform,
        "analysis_mode": args.analysis_mode,
        "runtime_env": args.runtime_env,
        "trace_input_type": trace_input_type,
        "hot_kernels": candidates,
        "task_groups": task_groups,
        "source": "tracelens_analysis",
        "dry_run": args.dry_run,
    }
    atomic_write_json(run_dir / "trace_input_manifest.json", manifest)
    atomic_write_json(tracelens_dir / "tracelens_report.json", report)
    kernel_candidates_path = run_dir / "kernel_candidates.json"
    # Hyperloom#314: hot_kernels is the dispatch payload (routable only); skipped_kernels keeps
    # full dicts so direct lookups can still resolve non-routable kernels. Full list stays in tracelens_report.json.
    routable_candidates = [
        c for c in candidates
        if isinstance(c, dict) and c.get("reusable_native_kernel") is True
    ]
    skipped_kernels = [
        c for c in candidates
        if isinstance(c, dict) and c.get("reusable_native_kernel") is not True
    ]
    atomic_write_json(
        kernel_candidates_path,
        {
            **report,
            "hot_kernels": routable_candidates,
            "skipped_kernels": skipped_kernels,
            "task_groups": task_groups,
        },
    )

    # PR-A §3: per-run audit sidecar (tasks routed vs skipped w/ reason); PR-B adds task_groups[].
    summary = build_audit_summary(
        candidates,
        trace_input=str(Path(args.trace_input).resolve()),
        framework=str(args.framework or ""),
        target_platform=str(args.target_platform or ""),
        task_groups=task_groups,
        trace_health_warnings=trace_health_warnings,
    )
    summary_path = tracelens_dir / "summary.json"
    atomic_write_json(summary_path, summary)

    missing_trace_report = (
        existing_report_path is None or not existing_report_path.exists()
    )
    trace_quality_blocked = any(
        isinstance(w, dict) and w.get("code") == "trace_split_no_steady_state"
        for w in (trace_health_warnings or [])
    )
    if missing_trace_report:
        if not getattr(args, "dry_run", False):
            if trace_quality_blocked:
                # Intentionally refused to run on a raw/non-steady trace; leave trace_report_path empty.
                existing_report_path = None
            else:
                raise RuntimeError(
                    "TraceLens SDK orchestrator did not produce analysis.md "
                    f"(expected at {existing_report_path}); refusing to "
                    "fabricate a Markdown report. Inspect the TraceLens skill "
                    "log and report upstream if this is reproducible."
                )
        else:
            # ``--dry-run``: synthesize a tiny stub so trace_report_path existence checks pass.
            stub_md = tracelens_dir / "analysis.md"
            stub_md.write_text(
                "# TraceLens dry-run stub (no SDK orchestrator output)\n",
                encoding="utf-8",
            )
            existing_report_path = stub_md

    kernel_roofline_path = kernel_roofline_path_for_run(run_dir)
    kernel_roofline_payload = build_kernel_roofline_payload(
        trace_input=str(Path(args.trace_input).resolve()),
        trace_input_type=trace_input_type,
        analysis_md_path=(
            str(existing_report_path) if existing_report_path else ""
        ),
        kernel_candidates_path=str(kernel_candidates_path),
        roofline_json_path=(
            str(Path(args.roofline_json).expanduser())
            if getattr(args, "roofline_json", "") else ""
        ),
        candidates=candidates,
    )
    atomic_write_json(kernel_roofline_path, kernel_roofline_payload)

    # Batch rocprof-compute enrichment is opt-in because it can profile many kernels.
    # Kernel-opt still profiles the selected kernel on demand.
    enrich_value = os.environ.get("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", "0").strip().lower()
    if enrich_value in {"1", "true", "yes", "on"}:
        try:
            tools_dir = str(Path(__file__).resolve().parent)
            if tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
            from rocprof_roofline import enrich_kernel_roofline_sidecar  # noqa: WPS433
            enrich_summary = enrich_kernel_roofline_sidecar(
                sidecar_path=str(kernel_roofline_path),
                candidates_path=str(kernel_candidates_path),
                workdir=str(run_dir),
                timeout_sec_per_kernel=int(
                    os.environ.get("HYPERLOOM_ROCPROF_ROOFLINE_TIMEOUT_SEC", "1800") or 1800
                ),
                log_fn=None,
            )
            print(
                "[rocprof_enrich] "
                f"matched={enrich_summary.get('matched', 0)} "
                f"skipped={enrich_summary.get('skipped', 0)} "
                f"failed={enrich_summary.get('failed', 0)} "
                f"rows={enrich_summary.get('rows', 0)}"
            )
        except Exception as exc:  # pragma: no cover - guard against import cycles
            print(f"[rocprof_enrich] skipped: {type(exc).__name__}: {exc}")

    artifact_paths = {
        "trace_input_manifest": str(run_dir / "trace_input_manifest.json"),
        "kernel_candidates": str(kernel_candidates_path),
        "kernel_roofline": str(kernel_roofline_path),
        "tracelens_report_json": str(tracelens_dir / "tracelens_report.json"),
        # Canonical Markdown exit is the orchestrator's analysis.md (surfaced, not aliased; #203/#217).
        "trace_report_path": str(existing_report_path) if existing_report_path else "",
        "tracelens_summary": str(summary_path),
    }
    return artifact_paths


def _default_workspace_path() -> str:
    """Resolve the default workspace root for ``--workspace-path``.

    Fallback order: ``$USER_DATA_PATH``, then legacy ``$WORKSPACE_PATH``, then
    ``_paths.workspace_root()`` (which warns once when ``$USER_DATA_PATH`` is unset).
    """
    user_data = os.environ.get("USER_DATA_PATH")
    if user_data:
        return user_data
    workspace = os.environ.get("WORKSPACE_PATH")
    if workspace:
        return workspace
    # Neither env set: route through the shared helper so the one-shot
    # "USER_DATA_PATH unset" warning fires and we still return the same
    # /workspace/hyperloom default the orchestrator uses.
    return workspace_root()


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
    parser.add_argument(
        "--source-root",
        default=os.environ.get("TRACELENS_SOURCE_ROOT", "") or None,
        help=(
            "Optional root directory against which TraceLens launcher "
            "paths (e.g. ``aiter/ops/rmsnorm.py(76): rmsnorm``) are "
            "resolved for AST-based function-line lookup. Used only by "
            "the PR-B source-function aggregation pass; absolute paths "
            "in the report don't need this. Defaults to "
            "$TRACELENS_SOURCE_ROOT when set."
        ),
    )
    parser.add_argument(
        "--workspace-path",
        default=_default_workspace_path(),
        help=(
            "Root the tool writes under (output lands at "
            "<workspace_path>/kernel-agent/runs/<session_id>/...). "
            "Defaults to $USER_DATA_PATH so every kernel-agent artefact "
            "stays inside the session dir; falls back to $WORKSPACE_PATH "
            "for legacy launchers, then to /workspace/hyperloom."
        ),
    )
    parser.add_argument("--tracelens-root", default=os.environ.get("TRACELENS_ROOT", ""),
                        help="TraceLens public checkout (TRACELENS_ROOT). Required: "
                             "kernel-agent/scripts/install.sh exports it from "
                             "kernel-agent.env.sh; pass --tracelens-root only when "
                             "running outside the installer-managed env.")
    parser.add_argument("--tracelens-internal-root",
                        default=os.environ.get("TRACELENS_INTERNAL_ROOT", DEFAULT_TRACELENS_INTERNAL_ROOT),
                        help="Optional TraceLens-internal checkout (TRACELENS_INTERNAL_ROOT). "
                             "Rehydration module; plumbed to run_tracelens_skill. "
                             "Leave empty for the open-source-only report.")
    parser.add_argument("--roofline-json", default="")
    parser.add_argument(
        "--capture-folder",
        default=os.environ.get("TRACELENS_CAPTURE_FOLDER", ""),
        help=(
            "Optional graph-capture folder for TraceLens inference graph "
            "replay analysis. Also accepts env TRACELENS_CAPTURE_FOLDER."
        ),
    )
    parser.add_argument("--budget-minutes", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    default_llm_orchestrator = os.environ.get(
        "KERNEL_AGENT_USE_LLM_ORCHESTRATOR", "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    parser.add_argument(
        "--use-llm-orchestrator",
        dest="use_llm_orchestrator",
        action="store_true",
        default=default_llm_orchestrator,
        help=(
            "Run TraceLens analysis-orchestrator skill through "
            "claude_agent_sdk before falling back to the deterministic parser "
            "(default: env KERNEL_AGENT_USE_LLM_ORCHESTRATOR, on)."
        ),
    )
    parser.add_argument(
        "--no-llm-orchestrator",
        dest="use_llm_orchestrator",
        action="store_false",
        help=(
            "Disable the Claude SDK TraceLens skill runner. Production runs "
            "will fail rather than falling back to intermediate/CSV candidate "
            "parsers; --dry-run still uses the test-only raw parser."
        ),
    )
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
    parser.add_argument(
        "--split-r",
        default=(
            os.environ.get("TRACELENS_SPLIT_R", "")
            or os.environ.get("RANDOM_RANGE_RATIO", "")
        ),
        help=(
            "OSL window ratio R for the splitter (#194 §3). Maps to "
            "--R on TraceLens.TraceUtils.split_inference_trace_annotation. "
            "Pairs with --CONC / --OSL so mixed-window selection uses the "
            "benchmark-contract PD ratio instead of an empirical default. "
            "Defaults to $RANDOM_RANGE_RATIO when set; leave empty to let "
            "the splitter fall back to its built-in heuristic."
        ),
    )
    parser.add_argument(
        "--steady-state-mode",
        choices=("mixed", "decode_only", "prefilldecode"),
        default=(
            os.environ.get("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", "").strip()
            or "mixed"
        ),
        help=(
            "Which of TraceLens splitter's three steady-state chunks to "
            "consume for the perf report (see docs/Inference_analysis.md "
            "in TraceLens-internal). The splitter always produces all "
            "three (mixed / decode_only / prefilldecode); this flag picks "
            "ONE per TraceLens's design that the chunks are parallel "
            "view-of-the-same-trace, not a fallback ladder. "
            "Defaults to 'mixed' (representative DO:PD mix at ~max "
            "concurrency) which matches roofline-v2's default profiling "
            "intent. Switch to 'prefilldecode' when the workload is "
            "short / batched (NUM_PROMPTS << CONC*OSL) so prefill is "
            "burst-shaped and the mixed window degenerates to PD=0 -- "
            "TP=1 + CUDA-graph traces frequently hit this corner case "
            "because the decode region's GPU work is fully inside the "
            "graph replay and rocprofiler-sdk doesn't emit aggregate "
            "Dispatch Task events outside TP-multi-stream contexts, so "
            "the mixed chunk looks 99%% idle while the prefilldecode "
            "chunk carries the real GEMM / attention kernels. "
            "Switch to 'decode_only' when you specifically want the "
            "longest pure-decode region (decode-perf comparison runs). "
            "May also be set via env "
            "INFERENCE_OPTIMIZER_STEADY_STATE_MODE so the coordinator "
            "can re-issue roofline with a different mode after a "
            "steady_state_chunk_empty warning lands."
        ),
    )
    args = parser.parse_args()

    session_id = args.session_id or uuid.uuid4().hex[:12]
    run_id = f"tl-{uuid.uuid4().hex[:8]}"
    started_at = utc_now()
    # PR-C: per-invocation sub-directory ``<compact_timestamp>_<run_id>`` so each run keeps its own artifacts.
    ts_compact = started_at.replace("-", "").replace(":", "").split(".")[0]
    if not ts_compact.endswith("Z"):
        ts_compact = ts_compact + "Z"
    sub_dir = f"{ts_compact}_{run_id}"
    root = Path(args.workspace_path) / "kernel-agent"
    run_dir = root / "runs" / session_id / sub_dir
    log_path = run_dir / "logs" / "tracelens_analysis" / f"{run_id}.log"
    status_path = run_dir / "status" / "tracelens_analysis" / f"{run_id}.json"
    artifacts: dict[str, str] = {}
    agent_candidates: list[dict[str, Any]] | None = None
    agent_report_path: Path | None = None
    allow_empty_candidates = False
    orchestrator_mode = "inline"
    orchestrator_error = ""
    # T3: structured trace-health findings surfaced to the Coordinator (e.g. Idle % gate).
    trace_health_warnings: list[dict[str, Any]] = []

    try:
        update_status(status_path, state="running", current_step="discover_trace_input",
                      log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                      started_at=started_at)
        trace_input = Path(args.trace_input).expanduser().resolve()
        trace_input_type, trace_files = discover_trace_inputs(trace_input)
        append_log(log_path, f"trace_input_type={trace_input_type}")
        append_log(log_path, f"trace_files={len(trace_files)}")

        # Fail-fast on CPU-only traces: nothing downstream works without GPU kernel events.
        if not args.dry_run and trace_files:
            kernel_event_count = count_gpu_kernel_events(trace_files[0])
            append_log(
                log_path,
                f"trace_gpu_kernel_events={kernel_event_count} "
                f"(probe={trace_files[0].name})",
            )
            if kernel_event_count == 0:
                raise RuntimeError(
                    "Trace contains zero GPU kernel events "
                    f"({trace_files[0]}); the upstream profile run "
                    "captured CPU-only activity. Re-run profile with the "
                    "torch.profiler GPU activities enabled (no LD_PRELOAD "
                    "competing for ROCprofiler-SDK) before invoking "
                    "tracelens_analysis."
                )

        if not args.dry_run:
            update_status(status_path, state="running", current_step="install_tracelens",
                          log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                          started_at=started_at)
            tl_root_arg = (args.tracelens_root or "").strip()
            if not tl_root_arg:
                raise SystemExit(
                    "TraceLens root not provided: set TRACELENS_ROOT in env "
                    "(kernel-agent/scripts/install.sh writes it to "
                    "kernel-agent.env.sh) or pass --tracelens-root."
                )
            tl_root = Path(tl_root_arg)
            # Internal extension is opt-in (non-empty --tracelens-internal-root / env).
            internal_root_arg = (args.tracelens_internal_root or "").strip()
            tl_internal_root: Path | None = (
                Path(internal_root_arg) if internal_root_arg else None
            )
            if not tl_root.exists():
                raise FileNotFoundError(
                    f"TraceLens root not found: {tl_root} "
                    "(set TRACELENS_ROOT or pass --tracelens-root)"
                )
            if tl_internal_root is not None and not tl_internal_root.exists():
                append_log(log_path,
                    f"TraceLens-internal root not found: {tl_internal_root}; "
                    "falling back to open-source-only "
                    "(provide an existing internal checkout to enable)")
                tl_internal_root = None
            if tl_internal_root is None:
                append_log(log_path,
                    "TraceLens-internal: not provided "
                    "(open-source-only; set TRACELENS_INTERNAL_ROOT to enable)")
                os.environ.pop("TL_EXTENSION", None)
            run_command([sys.executable, "-m", "pip", "install", "-e", "."],
                        cwd=tl_root, log_path=log_path,
                        timeout_s=max(60, int(args.budget_minutes * 60)))
            # TraceLens v0.3 (#148): analysis-orchestrator.md under Agent/Analysis/, with the legacy path as fallback below.
            skill = tl_root / "TraceLens/Agent/Analysis/.cursor/skills/analysis-orchestrator.md"
            if not skill.exists():
                skill = tl_root / "TraceLens/AgenticMode/Standalone/.cursor/skills/standalone-analysis-orchestrator.md"
            if not skill.exists():
                raise FileNotFoundError(f"TraceLens standalone skill not found (tried Agent/Analysis and AgenticMode/Standalone paths): {skill}")
            append_log(log_path, f"TraceLens skill: {skill}")

            tracelens_dir = run_dir / "tracelens"
            tracelens_dir.mkdir(parents=True, exist_ok=True)

            # #127: split the full-window trace into steady-state chunks via TraceLens's splitter.
            cli_trace_path = trace_files[0]
            trace_split_blocked = False
            if not args.skip_split:
                update_status(status_path, state="running", current_step="split_trace",
                              log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                              started_at=started_at)
                split_dir = tracelens_dir / "trace_split"
                split_dir.mkdir(parents=True, exist_ok=True)
                # --find-steady-state writes the three *_steady_state_* chunks; --R (#194 §3) feeds PD-ratio selection.
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
                # Only pass --R when provided so the splitter's default keeps working for legacy traces.
                r_raw = args.split_r or os.environ.get(
                    "RANDOM_RANGE_RATIO", "",
                )
                r_str = str(r_raw).strip()
                if r_str:
                    try:
                        float(r_str)
                    except ValueError:
                        append_log(
                            log_path,
                            f"split_trace: ignoring non-numeric --R={r_str!r}",
                        )
                    else:
                        split_cmd += ["--R", r_str]
                split_rc = run_command(
                    split_cmd,
                    cwd=tl_root,
                    log_path=log_path,
                    timeout_s=max(60, int(args.budget_minutes * 60)),
                )
                # The three chunks are parallel views, not a fallback ladder; the consumer picks ONE
                # via --steady-state-mode (default 'mixed') and we hard-fail when the selected chunk is missing/empty.
                def _collect(prefix: str) -> list[Path]:
                    out: list[Path] = []
                    for ext in ("trace.json.gz", "json.gz", "trace.json", "json"):
                        out.extend(sorted(split_dir.rglob(f"{prefix}_steady_state_*.{ext}")))
                    return out

                mixed_chunks = _collect("mixed")
                decode_chunks = _collect("decode_only")
                prefill_chunks = _collect("prefilldecode")
                # Splitter produced nothing -> trace_split_no_steady_state failure (operator must re-profile).
                if split_rc != 0 or not (mixed_chunks or decode_chunks or prefill_chunks):
                    warning = _build_trace_split_warning(
                        trace_input=trace_files[0],
                        split_dir=split_dir,
                        split_rc=split_rc,
                        mixed_count=len(mixed_chunks),
                        decode_count=len(decode_chunks),
                        prefilldecode_count=len(prefill_chunks),
                    )
                    trace_health_warnings.append(warning)
                    append_log(
                        log_path,
                        f"WARNING: trace split unavailable "
                        f"(rc={split_rc}, mixed={len(mixed_chunks)}, "
                        f"decode_only={len(decode_chunks)}, "
                        f"prefilldecode={len(prefill_chunks)}); "
                        "refusing raw-trace fallback and returning "
                        "trace_split_no_steady_state warning",
                    )
                    raise RuntimeError(
                        "trace_split_no_steady_state: TraceLens splitter "
                        "produced no steady-state chunks; refusing to run "
                        "TraceLens analysis on the raw trace"
                    )

                _mode_to_chunks = {
                    "mixed": ("mixed_steady_state", mixed_chunks),
                    "decode_only": ("decode_only_steady_state", decode_chunks),
                    "prefilldecode": ("prefilldecode_steady_state", prefill_chunks),
                }
                chunk_label, selected_chunks = _mode_to_chunks[args.steady_state_mode]
                if not selected_chunks:
                    # Requested mode produced no chunk; emit a warning (no silent fallback) for re-issue.
                    warning = {
                        "code": "steady_state_chunk_missing",
                        "severity": "blocking",
                        "requested_mode": args.steady_state_mode,
                        "requested_chunk_label": chunk_label,
                        "available_modes": [
                            m for m, (_, ch) in _mode_to_chunks.items() if ch
                        ],
                        "remediation": (
                            "Re-issue roofline with env "
                            "INFERENCE_OPTIMIZER_STEADY_STATE_MODE set to one "
                            "of the available_modes (or pass --steady-state-mode "
                            "directly when invoking tracelens_analysis.py)."
                        ),
                        "trace_input": str(trace_files[0]),
                        "split_dir": str(split_dir),
                    }
                    trace_health_warnings.append(warning)
                    append_log(
                        log_path,
                        f"ERROR: --steady-state-mode={args.steady_state_mode} "
                        f"requested but no {chunk_label}_*.json[.gz] in "
                        f"{split_dir} (mixed={len(mixed_chunks)}, "
                        f"decode_only={len(decode_chunks)}, "
                        f"prefilldecode={len(prefill_chunks)}); refusing "
                        "silent fallback per TraceLens parallel-chunk design",
                    )
                    raise RuntimeError(
                        f"steady_state_chunk_missing: requested "
                        f"--steady-state-mode={args.steady_state_mode} but "
                        f"splitter produced no matching chunk under "
                        f"{split_dir}"
                    )

                # N25 data-validity gate: the selected chunk must have observable GPU work
                # (else re-issue with a different --steady-state-mode). See _check_selected_chunk_has_gpu_events.
                cli_trace_path = selected_chunks[0]
                empty_chunk_warning = _check_selected_chunk_has_gpu_events(
                    split_dir=split_dir,
                    selected_chunk=cli_trace_path,
                    mode=args.steady_state_mode,
                    available_modes=_mode_to_chunks,
                )
                if empty_chunk_warning is not None:
                    trace_health_warnings.append(empty_chunk_warning)
                    append_log(
                        log_path,
                        f"ERROR: --steady-state-mode={args.steady_state_mode} "
                        f"selected chunk {cli_trace_path.name} has "
                        f"num_gpu_events={empty_chunk_warning['num_gpu_events']} "
                        f"/ gpu_busy_duration={empty_chunk_warning['gpu_busy_duration']}"
                        f"; refusing to feed an empty chunk to TraceLens "
                        "analysis (would produce misleading "
                        "'Compute %=~0, Idle %=~100' Executive Summary)",
                    )
                    raise RuntimeError(
                        f"steady_state_chunk_empty: requested "
                        f"--steady-state-mode={args.steady_state_mode} but the "
                        f"selected chunk has zero GPU events; available "
                        f"non-empty modes: "
                        f"{empty_chunk_warning['non_empty_modes']}"
                    )

                # N36 quality gate: a structurally-non-empty but low-busy chunk emits
                # steady_state_chunk_low_quality for the same retry path. See the module comment.
                low_quality_warning = _check_selected_chunk_has_gpu_events_quality(
                    split_dir=split_dir,
                    selected_chunk=cli_trace_path,
                    mode=args.steady_state_mode,
                    available_modes=_mode_to_chunks,
                )
                if low_quality_warning is not None:
                    trace_health_warnings.append(low_quality_warning)
                    append_log(
                        log_path,
                        f"ERROR: --steady-state-mode={args.steady_state_mode} "
                        f"selected chunk {cli_trace_path.name} is "
                        f"non-empty but low-quality: busy_ratio="
                        f"{low_quality_warning['busy_ratio']*100:.3f}% "
                        f"(threshold "
                        f"{low_quality_warning['threshold']*100:.0f}%); "
                        f"alternate modes with higher busy_ratio: "
                        f"{low_quality_warning['non_empty_modes']}. "
                        "Refusing to analyze (would yield misleading "
                        "high-idle Executive Summary).",
                    )
                    raise RuntimeError(
                        f"steady_state_chunk_low_quality: requested "
                        f"--steady-state-mode={args.steady_state_mode} chunk "
                        f"busy_ratio="
                        f"{low_quality_warning['busy_ratio']*100:.3f}%; "
                        f"better alternates: "
                        f"{low_quality_warning['non_empty_modes']}"
                    )

                artifacts["tracelens_trace_split_dir"] = str(split_dir)
                artifacts["tracelens_steady_state_trace"] = str(cli_trace_path)
                append_log(
                    log_path,
                    f"trace split OK: mixed={len(mixed_chunks)} "
                    f"decode_only={len(decode_chunks)} "
                    f"prefilldecode={len(prefill_chunks)}; "
                    f"--steady-state-mode={args.steady_state_mode} -> "
                    f"using {cli_trace_path.name} for perf report",
                )

            if args.use_llm_orchestrator and not trace_split_blocked:
                update_status(status_path, state="running",
                              current_step="run_tracelens_sdk_orchestrator",
                              log_path=log_path, artifact_paths=artifacts,
                              run_id=run_id, started_at=started_at)
                try:
                    trace_input_path = Path(args.trace_input).expanduser().resolve()
                    capture_folder = (
                        Path(args.capture_folder).expanduser().resolve()
                        if args.capture_folder else
                        discover_capture_folder(trace_input_path, trace_files)
                    )
                    skill_result = asyncio.run(run_tracelens_skill(
                        skill_path=skill,
                        trace_path=cli_trace_path,
                        output_dir=tracelens_dir,
                        tracelens_root=tl_root,
                        tracelens_internal_root=tl_internal_root,
                        platform=args.target_platform,
                        framework=args.framework,
                        analysis_mode=args.analysis_mode,
                        capture_folder=capture_folder,
                        budget_minutes=args.budget_minutes,
                        model=os.environ.get("ANTHROPIC_MODEL", ""),
                        log=lambda msg: append_log(log_path, msg),
                    ))
                    artifacts.update(skill_result.artifact_paths)
                    agent_report_path = skill_result.report_path
                    orchestrator_mode = "claude_agent_sdk"

                    # T3: gate on the Executive Summary's Idle % before consuming candidates.
                    # High idle => empty hot_kernels[] + a trace_health_warnings entry (T4 routes to params).
                    raw_agent_candidates = []
                    report_source = ""
                    idle_pct_value = extract_idle_pct_from_analysis_md(
                        skill_result.report_path,
                    )
                    idle_pct_threshold = _resolve_idle_pct_threshold()
                    high_idle_detected = (
                        idle_pct_value is not None
                        and idle_pct_value > idle_pct_threshold
                    )
                    if high_idle_detected:
                        assert idle_pct_value is not None
                        agent_candidates = []
                        allow_empty_candidates = True
                        trace_health_warnings.append(
                            _build_high_idle_warning(
                                idle_pct=idle_pct_value,
                                threshold_pct=idle_pct_threshold,
                                report_path=skill_result.report_path,
                            )
                        )
                        report_source = "skipped:high_gpu_idle_pct"
                        append_log(
                            log_path,
                            f"TraceLens Executive Summary reports "
                            f"Idle % = {idle_pct_value:.2f}% (threshold "
                            f"{idle_pct_threshold:.2f}%); suppressing "
                            "hot_kernels[] — kernel rewriting cannot move "
                            "end-to-end latency in the high-idle regime. "
                            "Coordinator will see this in "
                            "trace_health_warnings[] and route to "
                            "parameter optimization.",
                        )
                    else:
                        if idle_pct_value is not None:
                            append_log(
                                log_path,
                                f"TraceLens Executive Summary: "
                                f"Idle % = {idle_pct_value:.2f}% "
                                f"(threshold {idle_pct_threshold:.2f}%) — "
                                "below gate, continuing with kernel "
                                "candidate extraction",
                            )
                        report_cands = parse_analysis_md(
                            skill_result.report_path, args.top_k,
                        )
                        if report_cands:
                            raw_agent_candidates = report_cands
                            report_source = "analysis.md"
                        else:
                            agent_candidates = []
                            allow_empty_candidates = True
                            append_log(
                                log_path,
                                "TraceLens analysis.md had no Detailed "
                                "Analysis compute candidate blocks "
                                "(v0.3 contract: analysis.md is the single "
                                "source of truth)."
                                " Producing empty hot_kernels[] — "
                                "downstream Coordinator will route to "
                                "params/backends.",
                            )

                    if raw_agent_candidates:
                        total_dur = sum(
                            float(c.get("duration_us") or 0)
                            for c in raw_agent_candidates
                        )
                        agent_candidates = _finalize_candidates(
                            raw_agent_candidates,
                            total_dur=total_dur or None,
                            perf_report_csv_dir=(
                                skill_result.output_dir / "perf_report_csvs"
                            ),
                        )
                        append_log(
                            log_path,
                            f"TraceLens SDK orchestrator produced "
                            f"{len(agent_candidates)} hot kernels "
                            f"(source={report_source})",
                        )
                except Exception as exc:  # noqa: BLE001
                    orchestrator_error = f"{type(exc).__name__}: {exc}"
                    append_log(
                        log_path,
                        f"WARNING: TraceLens SDK orchestrator failed; "
                        f"not falling back to intermediate/CSV candidate "
                        f"parsers: {type(exc).__name__}: {exc}",
                    )

            if agent_candidates is None:
                raise RuntimeError(
                    "TraceLens analysis.md was not produced; refusing to "
                    "fall back to priority_data/category_data/CSV candidate "
                    "parsers because analysis.md is the single source of truth."
                )
        else:
            append_log(log_path, "[dry-run] skipping TraceLens install and external CLI")

        update_status(status_path, state="running", current_step="extract_hot_kernels",
                      log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                      started_at=started_at)
        # Production candidate extraction is analysis.md-only (no sidecar/CSV fallback).
        candidates = agent_candidates
        if candidates:
            append_log(
                log_path,
                f"hot kernels from TraceLens SDK orchestrator ({len(candidates)})",
            )
        if not candidates:
            if allow_empty_candidates:
                # Routing signal (high idle / TraceLens failure): keep candidates empty so the
                # Coordinator pivots to params/backends. NEVER fall through to analyze_trace_files here.
                candidates = []
                append_log(
                    log_path,
                    "TraceLens produced no kernel candidates; returning "
                    "empty hot_kernels[] without fallback so params/backends "
                    "optimization can continue.",
                )
            elif args.dry_run:
                # Test-only path: parse the raw trace so unit tests can exercise extraction without TraceLens.
                append_log(
                    log_path,
                    "dry-run: parsing raw trace for hot kernels "
                    "(production code path raises here — see #203)",
                )
                candidates = analyze_trace_files(trace_files, args.top_k)
            else:
                raise RuntimeError(
                    "No hot-kernel candidates produced by any TraceLens "
                    "analysis.md path. Refusing intermediate/CSV/raw-trace "
                    "fallbacks because analysis.md is the single source of "
                    "truth. Inspect the TraceLens skill log and report "
                    "upstream if reproducible."
                )
        roofline_by_name = load_roofline_results(args.roofline_json)
        if roofline_by_name:
            append_log(log_path, f"merged roofline results: {len(roofline_by_name)} kernels")
        merge_roofline_into_candidates(candidates, roofline_by_name)
        artifacts.update(write_reports(run_dir, trace_input_type=trace_input_type,
                                       trace_files=trace_files, candidates=candidates,
                                       args=args,
                                       existing_report_path=agent_report_path,
                                       trace_health_warnings=trace_health_warnings))
        if args.roofline_json:
            artifacts["roofline_json"] = str(Path(args.roofline_json).expanduser())
        artifacts["cli_log_path"] = str(log_path)
        artifacts["status_path"] = str(status_path)

        # Surface the contracted ``analysis.md`` exit path so consumers can read it alongside hot_kernels (PR #155).
        analysis_report_path = ""
        for cand_key in ("tracelens_agent_report", "trace_report_path"):
            if artifacts.get(cand_key):
                analysis_report_path = str(artifacts[cand_key])
                break

        result = {
            "tool": "tracelens_analysis",
            "session_id": session_id,
            "run_id": run_id,
            "trace_input_type": trace_input_type,
            "hot_kernels": candidates,
            "trace_report_path": artifacts["trace_report_path"],
            "analysis_report_path": analysis_report_path,
            "cli_log_path": str(log_path),
            "status_path": str(status_path),
            "artifact_paths": artifacts,
            "orchestrator_mode": orchestrator_mode,
            "orchestrator_error": orchestrator_error,
            # T3: trace-quality findings surfaced to the Coordinator (empty = nothing wrong).
            "trace_health_warnings": trace_health_warnings,
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
        # Include trace_health_warnings accumulated pre-exception so the Coordinator can auto-recover.
        print(json.dumps({
            "tool": "tracelens_analysis",
            "session_id": session_id,
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "cli_log_path": str(log_path),
            "status_path": str(status_path),
            "trace_health_warnings": trace_health_warnings,
        }, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
