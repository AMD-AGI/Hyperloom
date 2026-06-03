#!/usr/bin/env python3
"""TraceLens analysis tool for the resident Kernel Agent skill.

This tool is intentionally conservative: it records every step, writes a stable
artifact set, supports TraceLens capture directories, and has a dry-run path for
local validation without requiring TraceLens to be installed.
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


HIGH_IDLE_PCT_THRESHOLD_DEFAULT = 80.0
HIGH_IDLE_PCT_THRESHOLD_ENV = "HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD"


def _resolve_idle_pct_threshold() -> float:
    """Return the idle-percent gate threshold (default 80.0%).

    Report_Interfacing.docx §2 (idle-gate sanity check) lists ``<10-20%``
    as a *rough* target. We initially picked the upper edge (20%) of
    that band as the gate. Empirically every production-scale run we
    measured on Qwen3-32B (formal cases A–D in
    issue_bak/tracelens-profile-debug-20260516) reports
    ``Idle % ∈ [48%, 60%]``, so the 20% gate suppressed kernel
    rewriting on every real workload and the whole TraceLens →
    GEAK pipeline never reached ``run_optimization``. After confirming
    with the TraceLens team that this idle floor is structural to
    SGLang inference traces (host-side scheduling + JIT/launch
    overhead that the docx envisioned at GEMM-microbench scale never
    accounts for), the gate is relaxed to **80%** — kernel rewriting
    is still suppressed when the GPU is essentially never on
    (``Idle % > 80%``), but realistic ``Idle % ≈ 50–60%`` traces are
    no longer treated as fatally host-bound. Operators with workloads
    that need the original conservative gate can pin it via the
    ``HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD`` environment variable;
    an unparseable or negative value falls back to the default.
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
    """Build the structured ``trace_health_warnings[]`` entry for a high-idle trace.

    The entry is consumed by ``kernel_request_handlers.trace_analyze_handler``
    (T4) which uses it to route to parameter optimization instead of
    GEAK kernel rewriting. The shape is deliberately minimal and
    JSON-serializable so it can be written verbatim into the audit
    summary, the orchestrator result, and any operator-facing surfaces.
    """
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
    """Verify the splitter-produced chunk selected by ``--steady-state-mode``
    actually contains GPU events.

    The TraceLens splitter writes one row per produced chunk to
    ``execution_details.csv`` with the columns ``output_path``,
    ``num_gpu_events``, ``gpu_duration``, ``gpu_busy_duration`` (see
    ``TraceLens/TraceUtils/split_inference_trace_annotation.py:1324-1331``).
    Returns ``None`` when the row says the chunk carries real GPU work,
    otherwise returns a ``steady_state_chunk_empty`` warning dict the
    caller should ``trace_health_warnings.append`` and raise on.

    This is a data-validity gate, not a heuristic chunk-reordering rule:
    TraceLens analysis is only meaningful when the chunk has events to
    analyze. Falling back to a different mode is the OPERATOR's call
    (re-issue with a different ``--steady-state-mode`` after seeing the
    warning); we never silently swap.

    The empirical case that drove this gate is SOLAR-10.7B TP=1 BF16,
    where:
      - mixed_steady_state_*: num_gpu_events=160, gpu_busy_duration=1,428 us
        out of gpu_duration=1,118,730 us (0.13% busy) -- 96 sampler kernels
        only, forward fully inside CUDA graph + rocprofiler-sdk emits no
        Dispatch Task aggregate without TP-multi-stream sync;
      - prefilldecode_steady_state_*: num_gpu_events=2,790,
        gpu_busy_duration=2,723,452 us out of 4,538,984 us (60% busy) --
        480 Tensile GEMM + 240 paged_attention + 480 add_rmsnorm_quant
        (the real workload).
    Pre-N25 we silently consumed the mixed chunk and produced an
    Executive Summary saying "Compute %=0.18%, Idle %=99.77%" that
    misled the LLM into selecting host-bound params variants. N25 hard-
    fails here and the coordinator re-issues with
    ``INFERENCE_OPTIMIZER_STEADY_STATE_MODE=prefilldecode``.
    """
    import csv

    details_path = split_dir / "execution_details.csv"
    if not details_path.is_file():
        # Splitter didn't emit the CSV (older TraceLens?) -- cannot
        # validate; let the chunk through and hope for the best. The
        # downstream Executive Summary idle gate (T3) still catches
        # high-idle traces, just less specifically.
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

    # Selected chunk is empty. Surface which OTHER modes' chunks DO have
    # gpu events so the coordinator / operator knows what to re-issue
    # with.
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


# ---------------------------------------------------------------------------
# N36 — chunk-quality gate (busy_ratio threshold + alternate-mode lookup)
# ---------------------------------------------------------------------------
# Background: N25 deliberately stays a STRUCTURAL gate (num_gpu_events>0 AND
# gpu_busy_duration>0). The DSR1-0528 (671B FP8 MoE) TP=8 10k/1k production
# run on 2026-05-21 exposed a gap: TraceLens' splitter happily produced a
# ``mixed_steady_state`` chunk with 160 events / 2053us busy out of 3.26s
# (0.063% busy) -- structurally non-empty so N25 passed -- but
# substantively garbage. Downstream analysis.md reported "Compute %=0.09%
# / Idle %=99.90%" with ``reusable_native_kernel_ids=[]`` and the LLM
# was left with nothing to feed GEAK with.
#
# Root cause: the profile window calibration in ``_workload_envs``
# computes ``delay_iters = OSL * (R+1) * 3 - max_iters/2`` -- it only
# considers OSL, so a 10k/1k workload (10x ISL prefill) lands at the
# same ``start_step=6016`` as a 1k/1k workload. With CONC=64, by step
# 6016 every batch has finished its single prefill iter and the
# profiler captures only decode iters where the 8x MI300X is sparse.
#
# N36 closes that gap: this helper checks busy_ratio AND looks for an
# alternate mode with materially higher busy_ratio. When such an
# alternate exists we emit ``steady_state_chunk_low_quality`` (in the
# N26 retry allowlist) so the coordinator re-issues trace_analyze
# automatically. When NO mode is better we return ``None`` -- emitting
# a retry-warning would spin the same bad trace forever; the
# ``roofline_failure_streak`` path handles that case (N27 fallback).
_DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO = 0.05  # 5%
# Alternate mode must beat the requested mode by at least this margin
# for the auto-retry to be worth it. Otherwise we'd thrash between
# equally-bad modes.
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
    """Return ``busy_us / dur_us`` or ``None`` when undefined (zero
    duration). Caller treats ``None`` as "no signal, defer to N25".
    """
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
    """Quality gate that complements N25's structural gate.

    Returns ``None`` when the selected chunk is acceptable (busy_ratio
    >= threshold) OR no alternate mode is materially better OR the CSV
    is missing / row not found. Returns a structured warning dict
    (``code=steady_state_chunk_low_quality``, same shape as N25 for
    drop-in compatibility with the N26 retry path) when an alternate
    mode with materially higher busy_ratio exists.

    See module-level N36 comment for the empirical case + design.
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
        # Can't measure ratio; N25 already covers the structural empty
        # case. Defer.
        return None
    threshold = _resolve_min_busy_ratio()
    if sel_ratio >= threshold:
        return None

    # Selected chunk is below threshold. Look for an alternate mode
    # whose chunk has materially higher busy_ratio (otherwise retrying
    # is pointless).
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

    # Sort by descending busy_ratio so the best alternate is first --
    # roofline._extract_steady_state_retry_mode picks the head of the
    # non_empty_modes list.
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
DEFAULT_TRACELENS_ROOT = "/wekafs/hyperloom/TraceLens"
# Internal extension is opt-in: no default path. It is used only when
# TRACELENS_INTERNAL_ROOT (env) or --tracelens-internal-root is set; an empty
# value keeps Hyperloom on the open-source-only report.
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
    # Hyperloom P2-3: emit ``ended_at`` + ``duration_seconds`` once the
    # run reaches a terminal state so the session_breakdown collector
    # can fill the ``tracelens_analysis`` timeline event with a real
    # wall-clock duration (previously always None). Both fields are
    # purely additive — older callers/readers that don't know about
    # them ignore the keys, preserving back-compat with status JSONs
    # written by previous kernel-agent revisions.
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
    """Return the number of GPU kernel events in a torch_profiler trace.

    Used as a fast pre-flight check so we fail loudly when the upstream
    profile produced a CPU-only trace (e.g. when a tool such as PMC's
    LD_PRELOAD steals the rocprofiler-sdk slot from torch.profiler / kineto
    and leaves the dump with zero ``cat == 'kernel'`` events). Counts only
    real GPU kernels — not host-side ``cuda_runtime`` / ``hipLaunchKernel``
    wrappers — by deferring to :func:`is_kernel_event`.

    The counter stops once it has confirmed at least one GPU kernel; we
    only need to know whether GPU activity is present, not the exact
    count, so the helper finishes in a fraction of a second on multi-GB
    traces.
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
    """Prefer TraceLens-friendly traces when a directory is supplied.

    Magpie/SGLang profile directories contain rank/phase shards such as
    `TP-0-DECODE.trace.json.gz` as well as a `merged-*.trace.json.gz`.
    TraceLens's splitter expects the large annotated trace; feeding the first
    lexicographic shard gives it a tiny single-rank decode slice. Explicit file
    inputs remain honoured by the caller; this ordering only affects directory
    discovery.
    """
    name = path.name
    if name.startswith("merged-"):
        return (0, name)
    if re.search(r"TP-\d+-DECODE\.trace\.json(?:\.gz)?$", name):
        return (2, name)
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
    """Return True when ``source_file`` is a FlyDSL kernel source.

    Detection is content-based: FlyDSL kernels are plain ``.py`` files
    so neither extension nor path is reliable. Sniff the first 4 KiB
    for FlyDSL import / decorator markers — same signals GEAK's
    ``task_generator._infer_kernel_type`` uses upstream.
    """
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
    # PR #668 (TraceLens) injects ``pseudo_op::moe_flydsl_stage1`` /
    # ``pseudo_op::moe_flydsl_stage2`` above ``aiter::fused_moe_`` events
    # whose subtree contains FlyDSL stage markers. The pseudo-op carries
    # no source_file (it is synthetic) so content sniffing returns
    # ``unknown``. Match the upstream pseudo-op name prefix directly so
    # the candidate gets ``source_type=flydsl`` without needing a
    # resolvable .py path.
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
    """Framework install roots (shared with PolicyGate / apply_kernel_patch).

    Sourced from ``framework_paths.resolve_patch_target_roots`` (importlib +
    glob discovery + static fallbacks, incl. atom). Callers below lower-case
    the source path before the substring check, so we also emit a lower-case
    variant of every root (``/app/ATOM/atom/`` -> ``/app/atom/atom/``) to keep
    the case-insensitive match working.
    """
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
    """aiter's own device-source root (e.g. ``.../aiter_meta/csrc/``),
    resolved from the installed package. Empty when aiter is not importable.
    Cached once per process."""
    try:
        import aiter.jit.core as _jc  # type: ignore
    except Exception:
        return ""
    raw = (getattr(_jc, "AITER_CSRC_DIR", "") or "").replace(os.sep, "/")
    return (raw.rstrip("/") + "/") if raw else ""


def _flydsl_reusable_roots() -> tuple[str, ...]:
    """FlyDSL kernel checkout root(s) for PR #668 moe_flydsl pseudo-ops.

    The real rewritable source for ``pseudo_op::moe_flydsl_*`` lives under
    ``$DSL2_ROOT/kernels/`` (e.g. moe_gemm_2stage.py). The dynamic
    framework-root discovery does not know about the FlyDSL checkout, so
    without this the candidate is rejected "source not under a reusable
    framework root" and never reaches GEAK. Lower-cased because
    classify_patchability matches against ``lower_file``. ``$DSL2_ROOT`` /
    ``$FLYDSL_ROOT`` take precedence; the WekaFS checkout is the default.
    """
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
# Kernel-name substrings that mark an operation as non-patchable regardless
# of source-file resolution: vendor BLAS routines, RCCL/NCCL collectives,
# raw memcpy/copy ops, and PyTorch native copy. Folded from the feature
# branch's ``tracelens_geak_task_parser._NON_PATCHABLE_MARKERS`` so the
# unified ``classify_patchability`` gate emits a clean rejection reason
# for the same kernels that parser used to drop.
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
        return not any(root in lower_file for root in _reusable_roots())
    return False


def classify_patchability(candidate: dict[str, Any]) -> tuple[bool, str]:
    """Return ``(reusable, skip_reason)`` for a hot-kernel candidate.

    Single source of truth for the kernel-opt routing gate. ``skip_reason``
    is empty when the candidate is reusable; otherwise it is a short
    human-readable explanation suitable for the summary.json audit
    sidecar. :func:`is_reusable_native_kernel` returns the boolean half
    so existing callers keep working.

    Compared to the legacy logic this also rejects:

    * kernels whose name contains a vendor BLAS / collective / native-op
      marker from :data:`_NON_PATCHABLE_NAME_MARKERS`, even when the
      reported source file resolves to a reusable framework root;
    * ``aten::*`` PyTorch native ops without a library hint (these
      typically point at Tensile / vendor backends and have no rewritable
      Python source).
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
    for marker in _NON_PATCHABLE_NAME_MARKERS:
        if marker in lower_name:
            return False, (
                f"non-patchable kernel name marker '{marker}' in {name!r}"
            )
    if name.startswith("aten::"):
        library = str(candidate.get("library") or "").strip().lower()
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
    # ck_moe_stage1 / ck_moe_stage2 — @compile_ops("module_moe_ck2stages",
    # gen_func=cmdGenFunc_ck_moe_stage). The ``.cu`` here is the codegen
    # entry; hipcc compiles per-(dtype, quant, act) instances into
    # module_moe_ck2stages_*.so under <aiter>/jit/build/. PR-K's
    # apply_kernel_patch invalidates that jit/build/ before rebuild so the
    # patched .cu actually takes effect on the next import.
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

# Fallback aiter editable-checkout root for cases where ``find_repo_root``
# cannot resolve from the wrapper path (e.g. wrapper is at
# ``/usr/local/lib/python3.12/dist-packages/aiter/ops/moe_op.py`` from a
# wheel install — wheel layouts strip ``csrc/`` so we have to look at the
# co-located editable repo). Keep aligned with ``KNOWN_SEARCH_ROOTS``.
_AITER_FALLBACK_REPO = "/sgl-workspace/aiter"


def upgrade_aiter_compile_ops_launcher(
    source_file: str, kernel_name: str, kernel_repo: str,
) -> str:
    """Promote an aiter ``@compile_ops`` Python wrapper to the device ``.cu``.

    Returns the promoted absolute path when:
      * ``source_file`` is a Python file under any ``aiter/ops/`` tree
        (matches both editable-checkout and dist-packages layouts);
      * ``kernel_name`` lowercased contains one of the
        :data:`_AITER_COMPILE_OPS_PROMOTIONS` substring patterns;
      * the corresponding ``.cu`` exists on disk under ``kernel_repo``
        (or under :data:`_AITER_FALLBACK_REPO` when the wrapper lives in
        a wheel install whose layout has no ``csrc/``).

    Otherwise returns ``source_file`` unchanged so the LLM still gets a
    valid (if suboptimal) signal — the caller will note the promotion
    miss in ``optimization_notes`` so operators can audit it.
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

    Priority:
      1. Explicit category from TraceLens (the ``analysis.md`` 9-column table
         in the Detailed Analysis section).
         Mapped via the upstream ``CATEGORY_SKILL_MAP`` keyset
         (``orchestrator_prepare.py``) — see PR #155 review comment from
         @tsrikris (TraceLens team).
      2. Heuristic from kernel name (gemm / attn / norm / activation / …) for
         any candidate that pre-dates the orchestrator's category tagging
         (e.g. raw-trace fallback path).
      3. ``unknown``
    """
    cat = (candidate.get("tracelens_category") or "").strip()
    if cat:
        return normalize_upstream_category(cat)
    if candidate.get("source_type") == "flydsl":
        return "FlyDSL"
    name = str(candidate.get("name") or "").lower()
    if any(t in name for t in ("gemm", "matmul", "rocblas", "hipblas",
                                "cijk", "sgemm", "hgemm",
                                # PyTorch op-name variants the priority-1
                                # csv lookup misses when unified_perf_summary
                                # is absent (raw-trace path or empty op
                                # category column).
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
    """Read ``{name: raw TraceLens op category}`` from unified_perf_summary.csv.

    TraceLens emits one row per (name, shape); we keep the first
    non-empty op category per name (stable across shapes for one op).
    Returns ``{}`` when the csv is unavailable so callers degrade
    gracefully to the name-heuristic fallback in ``derive_kernel_category``.
    """
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
    """Apply source resolution / pybind upgrade / backend recommend / notes.

    Shared post-processing for parsed candidate rows. Mutates ``top`` in place.

    When ``perf_report_csv_dir`` is provided, populate each item's
    ``tracelens_category`` from TraceLens's unified_perf_summary.csv so
    ``derive_kernel_category`` takes its priority-1 (upstream) branch
    instead of falling through to the name heuristic.
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
        # Shapes here are always extracted from trace events; mark the
        # provenance so the dispatch-time validator can distinguish a
        # trace-anchored shape from any future non-trace source.
        if item.get("shapes"):
            item.setdefault("shape_provenance", "torch_trace")
        item["kernel_id"] = f"k{idx:03d}"
        # Honour pre-computed gpu_pct when present, else compute now.
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
        # PR-K: aiter @compile_ops launcher → device source promotion.
        # Capture the wrapper path BEFORE the upgrade so the LLM prompt
        # builder can render BOTH (device source as the rewrite target +
        # python launcher as call-site context). Only set
        # ``launcher_source_file`` when promotion actually changed the
        # path — otherwise the field would carry the same value as
        # ``source_file`` and add nothing to the prompt. ``tracelens_
        # launcher_path`` (the verbatim TraceLens kernel-path string) is
        # set elsewhere in tracelens_skill_runner._row_to_candidate and
        # is preserved through this pass for AST-grouping consumers.
        wrapper_before_promotion = item.get("source_file", "")
        item["source_file"] = upgrade_aiter_compile_ops_launcher(
            wrapper_before_promotion, item["name"], item.get("kernel_repo", "")
        )
        if item["source_file"] != wrapper_before_promotion:
            item["launcher_source_file"] = wrapper_before_promotion
            item["source_promoted_from_launcher"] = True
        # Re-resolve repo in case the upgraded path lives in a different repo
        # (rare, but defensive).
        item["kernel_repo"] = find_repo_root(item.get("source_file", "")) or item["kernel_repo"]
        item["source_type"] = source_type_for(item["name"], item.get("source_file", ""))
        # PR #668 FlyDSL pseudo-ops (pseudo_op::moe_flydsl_stage{1,2}) are
        # synthetic and carry no source_file, so classify_patchability below
        # would reject them ("source file not resolved") and kernel-opt would
        # see zero reusable-native candidates. Inject the real FlyDSL MoE
        # kernel source BEFORE the patchability gate so FlyDSL routes to GEAK.
        # FlyDSL pseudo-ops either carry no source_file or a profiler frame
        # label like ``aiter/fused_moe.py(986): fused_moe_2stages`` that is
        # not a real rewritable path under a reusable framework root. In both
        # cases inject the real FlyDSL MoE kernel source.
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
        # Single classify_patchability call covers both the boolean
        # routing gate AND the human-readable rejection reason consumed
        # by the summary.json audit sidecar (PR-A §3 of issue tbd).
        reusable, skip_reason = classify_patchability(item)
        item["reusable_native_kernel"] = reusable
        item["skip_reason"] = skip_reason
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
        # TraceLens csv lookup activates derive_kernel_category's
        # priority-1 (upstream) path. Only overrides when there is no
        # pre-set tracelens_category (analysis.md parsing path may set
        # it directly).
        if op_cat_map and not str(item.get("tracelens_category") or "").strip():
            csv_cat = op_cat_map.get(str(item.get("name") or ""))
            if csv_cat:
                item["tracelens_category"] = csv_cat
        # Surface a stable kernel_category for GEAK to dispatch on, plus a
        # source_path mirror for downstream prompt/report consumers. Shape is
        # already populated in `shapes`.
        item["kernel_category"] = derive_kernel_category(item)
        item.setdefault("source_path", item.get("source_file", ""))
    return top


def recommend_backends(candidate: dict[str, Any]) -> list[str]:
    """Recommend a backend ladder for a reusable native kernel.

    Policy (#144 last comment Layer 1, broadened): every kernel that
    Claude/Codex can rewrite, GEAK can rewrite too. Include GEAK in
    every default ladder so high-priority kernels reach GEAK FIRST;
    Claude/Codex stay on as fallbacks if GEAK times out or rejects.

    The kernel-agent's :func:`kernel_optimization.choose_backends` still
    sets ``geak_without_benchmark=True`` when no harness is present so
    operators / downstream KEEP gates can audit verification confidence
    — but GEAK is no longer pre-filtered from the ladder upstream.

    Only ``[]`` returns: unresolved source, non-reusable native (e.g.
    Inductor cache), vendor binaries, and runtime-generated kernels
    (where there's no stable source to rewrite).
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
    # Unified ladder per #144 last comment Layer 1 (broadened): every kernel
    # Claude/Codex/Cursor can rewrite, GEAK can rewrite too. GEAK is FIRST
    # (high-priority handoff). Cursor backend needs CURSOR_API_KEY (separate
    # Cursor gateway); skip from recommendations when the operator has not
    # provisioned a key so we don't advertise a backend the run will spend
    # time 401-ing on.
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
    }


def build_kernel_roofline_payload(
    *,
    trace_input: str,
    trace_input_type: str,
    analysis_md_path: str,
    kernel_candidates_path: str,
    roofline_json_path: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the structured per-kernel roofline sidecar.

    This sidecar is a view over TraceLens candidates plus optional
    ``--roofline-json`` enrichment. It does not invent missing
    utilization counters: absent compute/bandwidth utilization stays
    ``null`` in JSON.
    """
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
    """Return the session-level kernel roofline report path (latest pointer).

    PR-C: ``run_dir`` is now ``.../runs/<session_id>/<ts>_<run_id>/`` so
    we walk one extra level up to reach the kernel-agent root.
    Pre-PR-C layout (``.../runs/<session_id>/``) is still supported via
    the fallback branch in case a caller invokes this with an old-shape
    path.

    The session-level path remains the dashboard's stable entry point
    (one path, always the latest). Per-snapshot analysis.md /
    kernel_candidates.json etc. live under the per-invocation run_dir
    and are stamped into ``roofline_snapshots[i].analysis_md_path``
    so historical snapshots survive intact.
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
    "mi325x": "gfx942",
    "mi355x": "gfx950",
}
_FLYDSL_SMEM_MARKERS = ("SmemAllocator", "SmemPtr", "smem_alloc")
_FLYDSL_BUFFER_LOAD_MARKERS = (
    "make_buffer_tensor", "BufferCopy", "rocdl", "buffer_load",
)


def _resolve_flydsl_source_fallback() -> str:
    """Resolve the real FlyDSL MoE kernel source for pseudo-op candidates.

    PR #668 pseudo-ops (``pseudo_op::moe_flydsl_stage{1,2}``) are synthetic
    and carry no ``source_file``, so :func:`classify_patchability` would
    reject them ("source file not resolved") and the kernel-opt routing
    would see zero reusable-native candidates. The real rewritable source
    is FlyDSL's 2-stage MoE GEMM kernel under ``$DSL2_ROOT/kernels/``.
    Returns the first existing path, else "".
    """
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
    """Return FlyDSL-specific kernel_params for prompt construction.

    Mirrors the metadata GEAK's ``skills/flydsl/`` cheatsheet keys off:
    JIT cache state, MLIR target arch, and whether the source uses
    SmemAllocator / buffer-load intrinsics. Best-effort — missing fields
    just omit; never raise.
    """
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
            # PR #668 pseudo-ops carry no source_file (or a non-file frame
            # label). Inject the real FlyDSL MoE kernel source so GEAK has a
            # file to rewrite.
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
    """Aggregate finalized candidates by AST-resolved source function.

    Thin wrapper over
    :func:`tracelens_skill_runner.aggregate_by_source_function` that:

    * only considers candidates marked ``reusable_native_kernel`` so
      vendor / aten:: / runtime-generated kernels are never grouped
      (they were rejected upstream by ``classify_patchability``);
    * filters each group's ``kernel_ids`` and ``primary_kernel_id`` to
      the same reusable subset, so a group that mixes a reusable
      Triton function with a non-reusable aten:: launcher in its row
      list still dispatches only the reusable kernel_id.

    Returns ``[]`` when no candidate carries a parseable launcher path
    (the LLama70B fixture case — all rows have empty Kernel Path), so
    callers fall through to per-kernel dispatch.
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
    """Build the ``tracelens/summary.json`` payload from finalized candidates.

    Splits ``candidates`` into ``tasks`` (those that pass
    :func:`classify_patchability` and are routable to a kernel-opt backend)
    and ``skipped`` (those rejected, each carrying ``skip_reason`` so an
    operator can see exactly why a TraceLens hot kernel was dropped from
    routing). Both halves preserve the priority order of the input list.

    The function is pure — it reads ``candidates`` and returns a dict — so
    it is straightforward to test against fixture data without touching
    the filesystem.
    """
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
    # PR-B §1: summarize task_groups (source-function aggregation)
    # alongside the per-kernel ``tasks`` view. Each group entry is a
    # compact projection: the function identity + member kernel_ids +
    # aggregate cost. The full per-row data lives inside the
    # ``task_groups[]`` list on kernel_candidates.json — summary.json
    # is the audit view, not the dispatch payload.
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
        # T3: trace-quality findings (e.g. high GPU idle, future:
        # exposed-comm spikes, allocator contention). An empty list is
        # the steady-state signal; non-empty entries explain to the
        # operator why ``tasks`` may be empty even though the trace
        # parsed cleanly.
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
    """Write Hyperloom-owned sidecar JSONs and surface the upstream Markdown.

    The canonical Markdown final report (``analysis.md``) is owned by the
    TraceLens v0.3 SDK orchestrator per ``sub_agent_spec.md``: it is the
    single, contracted exit point for the analysis pipeline. Hyperloom
    no longer copies or aliases it under other names — the v0.2-era
    ``standalone_analysis.md`` / ``tracelens_report.md`` /
    ``--compat-report-path`` outputs were removed in #203 because they
    (1) wrote multiple byte-identical copies of the same file under
    different names, (2) bypassed the upstream contract, and
    (3) silently fabricated a Markdown when the SDK orchestrator failed,
    masking the upstream error from operators.

    When the SDK orchestrator does not produce ``analysis.md`` (e.g.
    Claude SDK login expired, orchestrator crashed mid-run), this
    function raises ``RuntimeError`` rather than synthesizing a
    replacement. The caller surfaces the underlying TraceLens error to
    the operator (and to the TraceLens team if reproducible).
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
    # PR-B §1: aggregate reusable candidates into source-function task
    # groups so downstream dispatch (run_optimization_handler) can submit
    # one GEAK task per (path, line, fn) instead of one per kernel_id.
    # Aggregation is additive: ``hot_kernels[]`` keeps its full per-row
    # priority order, and groups reference candidates by kernel_id.
    # Candidates whose ``tracelens_launcher_path`` cannot be parsed
    # (LLama70B fixture, raw-trace path, csv fallback) produce zero
    # groups and the caller falls through to the legacy per-kernel
    # dispatch.
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
    # Per AMD-AGI/Hyperloom#314: ``kernel_candidates.json::hot_kernels``
    # is the dispatch payload consumed by the kernel-opt path
    # (``kernel_optimization.load_candidates``, ``parallel_e2e_runner``,
    # ``kernel_request_handlers``). Filter it down to candidates that
    # ``classify_patchability`` actually marked routable so downstream
    # batch dispatchers no longer have to re-apply the same filter and
    # so ``num_hot_kernels`` accounting reflects what we will really
    # send to a backend. The full unfiltered list stays available in
    # ``tracelens/tracelens_report.json`` for audit, and the routable
    # vs. skipped split is also surfaced in ``tracelens/summary.json``
    # (``tasks[]`` / ``skipped[]``). ``skipped_kernels[]`` carries the
    # FULL candidate dicts (not just an audit projection) so direct
    # lookup paths (``kernel_optimization.load_candidates``, the CLI's
    # ``find_candidate``) can still resolve a non-routable kernel by
    # id; the dispatcher's ``_validate_reusable_native_kernel`` guard
    # is what blocks them from reaching a backend.
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

    # PR-A §3: per-run audit sidecar listing which TraceLens hot kernels
    # were routed to a kernel-opt backend (``tasks``) and which were
    # dropped (``skipped``, each with ``skip_reason``). Mirrors the
    # feature branch's ``tracelens_geak_task_parser.summary.json`` and is
    # the primary debug surface when GEAK comes back with surprising
    # results — operator can answer "did TraceLens see kernel X? did we
    # send it to GEAK? if not, why not?" in one read. PR-B adds
    # ``task_groups[]`` so an operator can also see which kernels
    # collapsed into the same source function.
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
                # No TraceLens SDK report exists because we intentionally
                # refused to run it on a raw/non-steady trace. Keep the
                # structured JSON sidecars and leave trace_report_path empty
                # rather than fabricating a misleading analysis.md.
                existing_report_path = None
            else:
                raise RuntimeError(
                    "TraceLens SDK orchestrator did not produce analysis.md "
                    f"(expected at {existing_report_path}); refusing to "
                    "fabricate a Markdown report. Inspect the TraceLens skill "
                    "log and report upstream if this is reproducible."
                )
        else:
            # ``--dry-run``: synthesize a tiny stub so test wiring that
            # checks ``trace_report_path`` existence still passes. Never
            # taken on a real (non-test) invocation.
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

    artifact_paths = {
        "trace_input_manifest": str(run_dir / "trace_input_manifest.json"),
        "kernel_candidates": str(kernel_candidates_path),
        "kernel_roofline": str(kernel_roofline_path),
        "tracelens_report_json": str(tracelens_dir / "tracelens_report.json"),
        # Post-#203 (PR #217): the canonical Markdown exit IS the
        # upstream SDK orchestrator's analysis.md surfaced via
        # ``existing_report_path``; Hyperloom no longer aliases or
        # copies it. PR-A §3 adds the ``summary.json`` audit sidecar
        # alongside (separate file, not a Markdown alias).
        "trace_report_path": str(existing_report_path) if existing_report_path else "",
        "tracelens_summary": str(summary_path),
    }
    return artifact_paths


def _default_workspace_path() -> str:
    """Resolve the default workspace root for ``--workspace-path``.

    Fallback order (highest precedence first):

    1. ``$USER_DATA_PATH`` — the main session-dir env introduced in
       ``b0f977c`` and unified by ``e56a0e5`` / ``1e46c14``. This is
       the user-facing root that ``inference_optimizer/paths.py``
       reads for every other Hyperloom artifact (orchestrator state,
       agents, runs, kernel-agent-workspace). Picking it up here keeps
       TraceLens artifacts co-located with the rest of the session.
    2. ``$WORKSPACE_PATH`` — the legacy kernel-agent env (default
       ``/workspace``). Kept as a second-tier fallback so existing
       launchers / CI / parallel_e2e_runner that already export it
       continue to work without changes.
    3. ``/workspace/hyperloom`` — the same hard-coded default as
       ``inference_optimizer/paths.py::DEFAULT_SESSION_DIR``, so a
       bare-image run without either env variable lands in the same
       place the orchestrator does.

    Note: GEAK / OOB tooling (``kernel_optimization.py``, the auth
    proxy, ray_runtime, install.sh) still defaults to the legacy
    ``$WORKSPACE_PATH`` (``/workspace``) until that side of the
    rollout is taken on. RPC-invoked TraceLens (via
    ``kernel_request_handlers.py``) and explicitly-parameterised
    callers (``parallel_e2e_runner.py``, the unit tests in
    ``test_tracelens_csv.py``) are unaffected — they pass
    ``--workspace-path`` explicitly.
    """
    return (
        os.environ.get("USER_DATA_PATH")
        or os.environ.get("WORKSPACE_PATH")
        or "/workspace/hyperloom"
    )


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
    parser.add_argument("--tracelens-root", default=os.environ.get("TRACELENS_ROOT", DEFAULT_TRACELENS_ROOT))
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
    # PR-C: per-tracelens-invocation sub-directory so PRELUDE + every
    # watermark refresh keeps its own analysis.md / kernel_candidates /
    # trace_split / etc. instead of overwriting the previous run's
    # files. Format ``<compact_timestamp>_<run_id>`` — sorts
    # chronologically, run_id keeps uniqueness across same-second
    # invocations. Pre-PR-C readers walking ``.../runs/<session_id>/``
    # directly will need to descend one more level; ``kernel_roofline_
    # path_for_run`` has been updated to handle both layouts.
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
    # T3: structured trace-health findings that the handler surfaces to
    # the Coordinator / GEAK. Populated from the Executive Summary's
    # ``Idle %`` row (extracted via ``extract_idle_pct_from_analysis_md``)
    # and any future trace-quality gates we add here. Stays empty in the
    # inline / non-SDK paths because raw-trace mode never produces an
    # ``analysis.md`` Executive Summary to interrogate.
    trace_health_warnings: list[dict[str, Any]] = []

    try:
        update_status(status_path, state="running", current_step="discover_trace_input",
                      log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                      started_at=started_at)
        trace_input = Path(args.trace_input).expanduser().resolve()
        trace_input_type, trace_files = discover_trace_inputs(trace_input)
        append_log(log_path, f"trace_input_type={trace_input_type}")
        append_log(log_path, f"trace_files={len(trace_files)}")

        # Fail-fast on CPU-only traces. Without GPU kernel events nothing
        # downstream — splitter, TraceLens perf-report CLI, the standalone
        # SDK orchestrator — can produce useful output. Surfacing the
        # missing-GPU-events condition here lets the caller (and the
        # operator) trigger a fresh profile instead of silently producing
        # an ABORTED report after a long detour.
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
            tl_root = Path(args.tracelens_root)
            # Internal extension is opt-in: used only when a non-empty
            # --tracelens-internal-root / TRACELENS_INTERNAL_ROOT is provided.
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
            # TraceLens v0.3 (#148): the standalone analysis skill lives
            # under TraceLens/Agent/Analysis/ with the shorter file name
            # `analysis-orchestrator.md` (renamed from
            # `standalone-analysis-orchestrator.md` and moved out of the old
            # `AgenticMode/Standalone/` tree). Override by setting
            # TRACELENS_ROOT / --tracelens-root for older release branches.
            skill = tl_root / "TraceLens/Agent/Analysis/.cursor/skills/analysis-orchestrator.md"
            if not skill.exists():
                skill = tl_root / "TraceLens/AgenticMode/Standalone/.cursor/skills/standalone-analysis-orchestrator.md"
            if not skill.exists():
                raise FileNotFoundError(f"TraceLens standalone skill not found (tried Agent/Analysis and AgenticMode/Standalone paths): {skill}")
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
            trace_split_blocked = False
            if not args.skip_split:
                update_status(status_path, state="running", current_step="split_trace",
                              log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                              started_at=started_at)
                split_dir = tracelens_dir / "trace_split"
                split_dir.mkdir(parents=True, exist_ok=True)
                # TraceLens splitter CLI (real interface):
                #   python -m TraceLens.TraceUtils.split_inference_trace_annotation
                #     <trace_path> -o <output_dir> --find-steady-state
                #     [--num-steps N] [--CONC C] [--OSL O] [--R r]
                # `--platform` does not exist; --find-steady-state writes
                # mixed_steady_state_* / decode_only_steady_state_* /
                # prefilldecode_steady_state_* into output_dir.
                # --R (#194 §3) feeds mixed-window selection's analytic
                # PD-ratio computation: per-request OSL is sampled from
                # [R*OSL, OSL], so without --R the splitter has to guess
                # the workload's prefill/decode mix from heuristics and
                # may pick a sub-optimal steady-state window.
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
                # --R is a float (splitter declares `type=float`); we
                # only pass it through when the user / env provided one
                # so the splitter's built-in default keeps working for
                # legacy trace files that pre-date the #194 alignment.
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
                # TraceLens splitter writes three parallel views of the same
                # steady-state region:
                #   - mixed_steady_state_*        (representative DO:PD mix)
                #   - decode_only_steady_state_*  (longest pure-decode run)
                #   - prefilldecode_steady_state_*(longest pure-PD run)
                # Per TraceLens docs/Inference_analysis.md these are three
                # parallel view-of-the-same-trace, NOT a fallback ladder
                # (TraceLens itself has no internal preference between them).
                # The consumer (us) picks ONE per the configured intent.
                #
                # Pre-N25 behaviour was an implicit `mixed or decode_only or
                # prefilldecode` chain that auto-fell-through silently. This
                # broke for the SOLAR-10.7B TP=1 case where the mixed window
                # degenerated to gpu_busy=0.13% (all forward in CUDA graph
                # + rocprofiler-sdk emits no Dispatch Task aggregate without
                # TP-multi-stream sync) while the prefilldecode chunk carried
                # 60% busy + 480 GEMM + 240 paged_attention. Implicit
                # fall-through hid the issue. N25 makes the mode explicit
                # (--steady-state-mode flag, default 'mixed') and hard-fails
                # when the selected chunk doesn't exist or is empty so the
                # operator can re-issue roofline with a different mode.
                def _collect(prefix: str) -> list[Path]:
                    out: list[Path] = []
                    for ext in ("trace.json.gz", "json.gz", "trace.json", "json"):
                        out.extend(sorted(split_dir.rglob(f"{prefix}_steady_state_*.{ext}")))
                    return out

                mixed_chunks = _collect("mixed")
                decode_chunks = _collect("decode_only")
                prefill_chunks = _collect("prefilldecode")
                # Splitter produced nothing at all -> existing
                # trace_split_no_steady_state failure (treated as unrecoverable
                # at the action layer; operator must re-profile).
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
                    # Requested mode produced no chunk; the OTHER two may have
                    # produced chunks but per TraceLens design we don't pick
                    # them as a silent fallback -- emit a structured warning
                    # so the coordinator can re-issue roofline with a
                    # different --steady-state-mode (N25 contract).
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

                # Sanity check: the selected chunk must have observable GPU
                # work. Splitter's execution_details.csv carries
                # `num_gpu_events` per chunk -- when it's 0 (or
                # gpu_busy_duration == 0) the chunk is structurally empty
                # (e.g. SOLAR-10.7B TP=1 mixed = pure-decode region with all
                # forward inside CUDA graph + no Dispatch Task aggregate).
                # Running TraceLens on such a chunk yields the misleading
                # "Compute %=0.18%, Idle %=99.77%, kernel_count=0" report.
                # This is NOT a heuristic chunk-reordering rule; it's a
                # data-validity gate: TraceLens analysis only makes sense
                # when the chunk has GPU events to analyze. The remediation
                # is the same as `_chunk_missing`: re-issue with a different
                # --steady-state-mode (the other two chunks may carry the
                # real workload).
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

                # N36 (May 2026) — quality gate on busy_ratio. N25
                # only catches structurally empty chunks (events==0
                # OR busy==0); a chunk like the DSR1-0528 10k/1k case
                # (160 events / 2ms busy / 3.26s duration = 0.06%
                # busy) passes N25 but is substantively garbage. The
                # quality gate looks for an alternate mode with
                # materially higher busy_ratio and emits a
                # ``steady_state_chunk_low_quality`` warning the
                # coordinator's N26 retry path consumes -- same
                # remediation flow as the empty-chunk case, no
                # additional wiring required. See N36 module-level
                # comment + test_n36_chunk_quality_gate.py.
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

                    # Per TraceLens_Report_Interfacing.docx §2, the final
                    # ``analysis.md`` report is the contracted exit point.
                    # Intermediate sidecars / CSVs are not a production
                    # fallback because they can hide malformed TraceLens
                    # reports or incorrect profiling.
                    #
                    # T3 (this PR): before consuming any candidates, gate on
                    # the Executive Summary's ``Idle %``. When idle time
                    # dominates wall-clock, kernel rewrites cannot improve
                    # end-to-end latency (Report_Interfacing.docx §2 idle-gate
                    # sanity check), so we
                    # short-circuit to empty hot_kernels[] and surface a
                    # ``trace_health_warnings`` entry that the handler (T4)
                    # uses to route to parameter optimization.
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
                        # idle_pct_value is known to be a float here because
                        # high_idle_detected required it to be not None.
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
        # Production candidate extraction is analysis.md-only. Intermediate
        # sidecars / CSVs are not parsed as fallbacks.
        candidates = agent_candidates
        if candidates:
            append_log(
                log_path,
                f"hot kernels from TraceLens SDK orchestrator ({len(candidates)})",
            )
        if not candidates:
            if allow_empty_candidates:
                # Hyperloom routing signal (high idle from docx §2 idle-gate
                # sanity check / TraceLens permanent failure per Hyperloom
                # T4 design — docx does not define this fallback):
                # keep ``candidates`` empty and let the Coordinator pivot to
                # ``params`` / ``backends`` based on the
                # ``trace_health_warnings`` we already populated. NEVER
                # fall through to ``analyze_trace_files`` here — that would
                # re-populate hot_kernels from the raw trace and silently
                # undo the idle-gate / TraceLens-failure suppression.
                candidates = []
                append_log(
                    log_path,
                    "TraceLens produced no kernel candidates; returning "
                    "empty hot_kernels[] without fallback so params/backends "
                    "optimization can continue.",
                )
            elif args.dry_run:
                # ``--dry-run`` is the test-only path that bypasses
                # TraceLens install / CLI / SDK orchestrator entirely. It
                # still parses the raw trace so unit tests can exercise
                # hot-kernel extraction and downstream wiring without a
                # real TraceLens run. Production code never sets
                # ``--dry-run``.
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

        # Per PR #155 review (TraceLens team @tsrikris) the final ``analysis.md``
        # is the contracted "exit interface" of the orchestrator. Surface its
        # path explicitly so downstream consumers (GEAK, Coordinator) can read
        # the report alongside the structured ``hot_kernels`` payload, and so
        # that an incoming kernel-optimization sub-agent has the full
        # stakeholder report to ground its actions on.
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
            # T3: structured trace-quality findings (high GPU idle, …) that
            # the handler (``trace_analyze_handler``, T4) surfaces upward
            # so the Coordinator can decide between kernel-rewrite and
            # parameter-optimization routes. Empty list is the steady-state
            # ("nothing wrong") signal.
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
        # N26: include any trace_health_warnings accumulated before the
        # exception fired so the handler / Coordinator can auto-recover
        # (e.g. re-issue with a different --steady-state-mode when a
        # steady_state_chunk_empty warning carries non_empty_modes).
        # Pre-N26 the failure JSON dropped warnings on the floor, so
        # the RooflineExecutor saw `status=failed` without the structured
        # hint it needed to decide between hard-fail and auto-retry.
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
