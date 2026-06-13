#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""TraceLens analysis tool for the resident Kernel Agent skill.

Conservative: records every step, writes a stable artifact set, supports TraceLens
capture directories, and has a dry-run path that works without TraceLens installed.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import functools
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    from inference_optimizer.orchestrator.framework_paths import (
        resolve_patch_target_roots as _resolve_patch_target_roots,
    )
except ImportError:
    _resolve_patch_target_roots = None

try:
    from apply_kernel_patch import known_target_roots as _known_target_roots
except ImportError:
    _known_target_roots = None

try:
    import aiter.jit.core as _aiter_jit_core  # type: ignore[import-untyped]
except Exception:
    _aiter_jit_core = None

from tracelens_arch_benchmark import normalize_platform, populate_gpu_arch_json
from tracelens_skill_runner import (
    _parse_launcher_path,
    _resolve_launcher_to_abs_source,
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

ARCH_BENCHMARK_TIMEOUT_ENV = "TRACELENS_ARCH_BENCHMARK_TIMEOUT_SEC"
ARCH_BENCHMARK_TIMEOUT_FLOOR_S = 600

ANALYSIS_ROUTE_ENV = "HYPERLOOM_TRACE_ANALYSIS_ROUTE"
ANALYSIS_ROUTE_DETERMINISTIC = "deterministic"
ANALYSIS_ROUTE_AGENT = "agent"
_VALID_ANALYSIS_ROUTES = {ANALYSIS_ROUTE_DETERMINISTIC, ANALYSIS_ROUTE_AGENT}


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


def _resolve_arch_benchmark_timeout_s() -> int:
    """Return the GPU arch microbenchmark timeout in seconds (floor 600s).

    Configured via ``TRACELENS_ARCH_BENCHMARK_TIMEOUT_SEC``. Empty, non-numeric,
    or out-of-range values fall back to the 600s floor rather than crashing the
    pipeline with a ``ValueError`` before the microbenchmark runs.
    """
    raw = os.environ.get(ARCH_BENCHMARK_TIMEOUT_ENV, "").strip()
    if not raw:
        return ARCH_BENCHMARK_TIMEOUT_FLOOR_S
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return ARCH_BENCHMARK_TIMEOUT_FLOOR_S
    return max(ARCH_BENCHMARK_TIMEOUT_FLOOR_S, value)


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
    """Build the ``trace_split_no_steady_state`` trace-health warning.

    Emitted when the TraceLens splitter produces no steady-state chunks,
    so analyzing the raw trace would risk misleading high-idle results.

    Args:
        trace_input (Path): The raw trace that was handed to the splitter.
        split_dir (Path): Directory the splitter wrote its outputs into.
        split_rc (int): Return code from the splitter subprocess.
        mixed_count (int): Number of ``mixed`` steady-state chunks produced.
        decode_count (int): Number of ``decode_only`` chunks produced.
        prefilldecode_count (int): Number of ``prefilldecode`` chunks produced.

    Returns:
        dict[str, Any]: A structured warning entry with code
            ``trace_split_no_steady_state`` and the supporting counts/message.
    """
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
        """Read a numeric field from the selected splitter CSV row.

        Args:
            name (str): Column name to read from ``selected_row``.

        Returns:
            float: The parsed value, or ``0.0`` when missing/unparseable.
        """
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
    """Return the minimum chunk busy-ratio threshold for the N36 quality gate.

    Reads ``INFERENCE_OPTIMIZER_CHUNK_QUALITY_MIN_BUSY_RATIO`` and falls back
    to :data:`_DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO` when unset, out of the
    ``[0.0, 1.0]`` range, or unparseable.

    Returns:
        float: The busy-ratio threshold in the inclusive range ``[0.0, 1.0]``.
    """
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
    details_path = split_dir / "execution_details.csv"
    if not details_path.is_file():
        return None
    try:
        with details_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error):
        return None

    def _row_for(chunk_path: "Path") -> "dict[str, str] | None":
        """Find the splitter CSV row whose ``output_path`` is the chunk.

        Args:
            chunk_path (Path): Chunk file to match against ``output_path``.

        Returns:
            dict[str, str] | None: The matching CSV row, or ``None`` when no
                row resolves to the same path.
        """
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
        """Extract ``(num_gpu_events, gpu_busy_duration, gpu_duration)``.

        Args:
            row (dict[str, str] | None): A splitter CSV row, or ``None``.

        Returns:
            tuple[int, float, float]: The event count, busy duration (us),
                and total duration (us); all zero when ``row`` is ``None``.
        """
        if row is None:
            return 0, 0.0, 0.0
        def _f(k: str) -> float:
            """Read a numeric field from the CSV row.

            Args:
                k (str): Column name to read.

            Returns:
                float: The parsed value, or ``0.0`` when missing/unparseable.
            """
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
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        str: The current UTC timestamp in ISO-8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write ``data`` as pretty-printed JSON to ``path``.

    Writes to a temporary file in the same directory then replaces the
    target so readers never observe a partially-written file.

    Args:
        path (Path): Destination JSON file; parent dirs are created.
        data (dict[str, Any]): JSON-serializable payload to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as tmp:
        json.dump(data, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def append_log(log_path: Path, message: str) -> None:
    """Append a single line to a log file, creating parent dirs as needed.

    Args:
        log_path (Path): Log file to append to.
        message (str): Text to append; trailing whitespace is stripped and a
            newline is added.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def read_last_lines(log_path: Path, limit: int = 20) -> list[str]:
    """Return the last ``limit`` lines of a log file.

    Args:
        log_path (Path): Log file to read.
        limit (int): Maximum number of trailing lines to return.

    Returns:
        list[str]: The trailing lines, or an empty list when the file does
            not exist.
    """
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
    """Atomically write a tracelens_analysis run-status JSON file.

    Captures the current run state, recent log tail, and (on terminal
    states) the wall-clock duration so downstream collectors can build a
    timeline event.

    Args:
        status_path (Path): Destination status JSON file.
        state (str): Current run state (e.g. ``running``, ``succeeded``).
        current_step (str): Human-readable label of the active step.
        log_path (Path): Log file whose size/tail are recorded.
        artifact_paths (dict[str, str]): Map of artifact names to paths.
        run_id (str): Unique identifier for this run.
        started_at (str): ISO-8601 start time used to compute duration.
        error (str | None): Error message recorded when the run failed.
    """
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
    """Load a JSON file, transparently handling ``.gz`` compression.

    Args:
        path (Path): JSON or gzipped-JSON file to read.

    Returns:
        dict[str, Any]: The parsed JSON payload.
    """
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
    """Resolve a trace input path into a list of trace files.

    Accepts either a single trace file or a capture directory; directories
    are searched recursively for known trace extensions, deduplicated, and
    ordered via :func:`_trace_input_sort_key`.

    Args:
        trace_input (Path): A trace file or a capture directory.

    Returns:
        tuple[str, list[Path]]: ``("file", [path])`` for a single file or
            ``("capture_dir", paths)`` for a directory.

    Raises:
        FileNotFoundError: When the path does not exist or no trace files are
            found under a supplied directory.
    """
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
    """Extract a shape annotation from a trace event when present.

    Checks both ``event['args']`` and the event top level for the first of
    several known shape keys.

    Args:
        event (dict[str, Any]): A single trace event.

    Returns:
        dict[str, Any] | None: A single-key dict ``{shape_key: value}`` for
            the first shape field found, or ``None`` when none is present.
    """
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    for key in ("shape", "shapes", "input_shape", "trace_shapes"):
        if key in args:
            return {key: args[key]}
        if key in event:
            return {key: event[key]}
    return None


def extract_source_file(event: dict[str, Any]) -> str:
    """Extract a source-file path from a trace event when present.

    Checks both ``event['args']`` and the event top level for the first of
    several known path keys.

    Args:
        event (dict[str, Any]): A single trace event.

    Returns:
        str: The first non-empty source path found, or ``""`` when none.
    """
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
    """Classify a kernel's source type from its name and source path.

    Recognizes FlyDSL pseudo-ops, runtime-generated kernels, HIP/C++,
    Triton, FlyDSL, plain Python, and vendor-binary backends.

    Args:
        name (str): Kernel symbol/name.
        source_file (str): Resolved source-file path (may be empty).

    Returns:
        str: One of ``flydsl``, ``runtime_generated``, ``hip_cpp``,
            ``triton``, ``python``, ``vendor_binary``, or ``unknown``.
    """
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
        if _resolve_patch_target_roots is None:
            raise ImportError
        roots = _resolve_patch_target_roots()
    except ImportError:
        if _known_target_roots is None:
            roots = []
        else:
            roots = _known_target_roots()
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
    if _aiter_jit_core is None:
        return ""
    raw = (getattr(_aiter_jit_core, "AITER_CSRC_DIR", "") or "").replace(os.sep, "/")
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
    """Combine all reusable-source roots used by the patchability gate.

    Returns:
        tuple[str, ...]: Discovered framework roots plus the aiter csrc root
            and FlyDSL checkout roots, deduplicated.
    """
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

    Args:
        candidate (dict[str, Any]): A hot-kernel candidate row.

    Returns:
        bool: ``True`` when the candidate is routable to a kernel-opt backend.
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
    """Remove C++ template argument blocks (``<...>``) from a symbol.

    Args:
        symbol (str): A possibly templated C++ symbol name.

    Returns:
        str: The symbol with all balanced ``<...>`` sections removed.
    """
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


def _normalize_profiler_op_name(name: str) -> str:
    """Strip graph-capture / synthetic wrappers from a TraceLens op symbol.

    Graph-captured (HIP/CUDA graph) traces record op names with a launch
    wrapper and display annotations that are not part of the kernel symbol,
    e.g. ``hipGraphLaunch->void ck::foo_kernel<...>`` or
    ``hipGraphLaunch->triton_poi_fused_2.kd (Synthetic Op)``. Left intact,
    these pollute keyword extraction (the keyword keeps the
    ``hipGraphLaunch->void `` prefix and greps to nothing) and hide an embedded
    Itanium-mangled symbol from the ``_Z`` demangling path.

    This peels off, in order: a leading ``<launcher>->`` capture wrapper, a
    leading C++ return-type token, a trailing ``(... Op)`` annotation, and a
    trailing ``.kd`` HSA code-object suffix. Already-clean names (e.g.
    ``sglang_profiler::...`` or ``aten::mm``) pass through unchanged.
    """
    s = (name or "").strip()
    if not s:
        return ""
    # Leading graph-launch capture wrapper: ``hipGraphLaunch->`` / ``cudaGraphLaunch->``.
    s = re.sub(r"^[A-Za-z][A-Za-z0-9_]*->", "", s).strip()
    # Leading C++ return-type token before the symbol (only relevant when there
    # is no ``::`` namespace to slice on later).
    s = re.sub(
        r"^(?:void|bool|int|unsigned|long|short|char|float|double|size_t)\s+",
        "", s,
    ).strip()
    # Trailing display annotation such as ``(Synthetic Op)``.
    s = re.sub(r"\s*\([^()]*\bOp\)\s*$", "", s).strip()
    # Trailing ``.kd`` HSA code-object suffix.
    if s.endswith(".kd"):
        s = s[:-3].strip()
    return s or (name or "").strip()


def _candidate_keywords(name: str) -> list[str]:
    """Pick stable search keywords from a kernel symbol.

    Prefers descriptive identifiers (e.g. cross_device_reduce_2stage, gemm_a16w16)
    over namespace/type tokens (aiter, vllm, RankData) that match too widely.

    Args:
        name (str): Kernel symbol/name (possibly Itanium-mangled).

    Returns:
        list[str]: Up to three descriptive search keywords, most-specific
            first; empty when nothing usable can be extracted.
    """
    cleaned = _normalize_profiler_op_name(name)
    if cleaned.startswith("_Z"):
        # Itanium ABI uses <len><name>; walk through and slice manually so
        # consecutive segments (e.g. 5aiter26cross_device_reduce_2stage...) are
        # parsed as separate identifiers.
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
    """Recursively grep ``root`` for source files containing ``keyword``.

    Results are cached per ``(keyword, root)`` and restricted to known
    source extensions. Failures (missing grep, timeout) yield ``[]``.

    Args:
        keyword (str): Literal string to search for.
        root (Path): Directory to search recursively.

    Returns:
        list[Path]: Existing source files that match, possibly empty.
    """
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


def _rank_paths(paths: list[Path], keyword: str = "") -> list[Path]:
    """Sort candidate source paths by likely relevance.

    Prefers real source repos over installed wheels and over optimized /
    build variants, then by file extension and path depth.

    Args:
        paths (list[Path]): Candidate source paths to rank.
        keyword (str): The search keyword; files whose stem contains it rank higher.

    Returns:
        list[Path]: ``paths`` sorted best-first.
    """
    kw_lower = keyword.lower()

    def score(path: Path) -> tuple[int, int, int, int]:
        s = str(path)
        depth_penalty = s.count("/")
        kind_score = 0
        if "/csrc/" in s:
            kind_score -= 3
        if "/optimized_versions/" in s or "/build/" in s:
            kind_score += 5
        if "/site-packages/" in s:
            kind_score += 2
        ext_score = {".cuh": 0, ".cu": 0, ".hip": 0, ".cpp": 1, ".h": 2, ".hpp": 2, ".py": 3}.get(path.suffix, 4)
        # Prefer files whose stem directly matches the keyword over incidental mentions.
        name_match = 0 if (kw_lower and kw_lower in path.stem.lower()) else 1
        # Penalize include headers and pybind wrappers (less likely to be the kernel impl).
        if "/include/" in s:
            kind_score += 1
        if "/pybind/" in s:
            kind_score += 2
        return (name_match, kind_score, ext_score, depth_penalty)

    return sorted(paths, key=score)


def _compound_subwindow_keywords(name: str) -> list[str]:
    """Trailing snake_case sub-windows of a compound/profiler-wrapped symbol.

    A profiler-wrapped op name like
    ``sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel_427``
    never appears verbatim in source: the recorded identifier is
    ``<file_stem>_<func>_<line>``. :func:`_candidate_keywords` returns only the
    full compound token, which greps to nothing. This yields progressively
    shorter trailing windows (longest first, most specific) so the embedded
    function symbol (e.g. ``invoke_fused_moe_kernel``) still resolves. The
    namespace/profiler prefix and a trailing numeric id are stripped first.
    """
    cleaned = _strip_template_args(_normalize_profiler_op_name(name))
    if "::" in cleaned:
        cleaned = cleaned.split("::")[-1]
    cleaned = re.sub(r"_\d+$", "", cleaned)  # drop a trailing launcher line number
    segs = [s for s in cleaned.split("_") if s]
    if len(segs) < 3:
        return []
    out: list[str] = []
    seen: set[str] = set()
    # Windows anchored at the end, dropping leading segments one at a time, down
    # to the last two segments. Longest (most specific) first.
    for start in range(0, len(segs) - 1):
        window = "_".join(segs[start:])
        if len(window) >= 6 and not window.isdigit() and window not in seen:
            seen.add(window)
            out.append(window)
    return out[:6]


def _file_defines_symbol(path: Path, keyword: str) -> bool:
    """True when ``path`` *defines* ``keyword`` (vs merely mentioning it).

    Distinguishes a kernel's definition site (``def invoke_fused_moe_kernel``,
    a Triton ``@triton.jit`` function, or a C/HIP ``__global__``) from a file
    that only calls/wraps it (e.g. sglang's ``kernel_shape_profiler.py`` dispatch
    shim). Best-effort: returns False on read errors.
    """
    kw = re.escape(keyword)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    patterns = (
        r"\bdef\s+" + kw + r"\b",            # Python (incl. @triton.jit) def
        r"\b" + kw + r"\s*=",                # kernel bound to a module-level name
        r"__global__[^\n;{]*\b" + kw + r"\b",  # CUDA/HIP global kernel
    )
    return any(re.search(p, text) for p in patterns)


def _prefer_symbol_definition(keyword: str, hits: list[Path]) -> list[Path]:
    """Rank definition sites of ``keyword`` ahead of mere mentions."""
    definers = [h for h in hits if _file_defines_symbol(h, keyword)]
    return _rank_paths(definers) if definers else _rank_paths(hits)


def locate_source_via_grep(name: str) -> str:
    """Locate a kernel source file by grepping known repos.

    Returns "" when no confident match exists. Never fabricates a path.

    Args:
        name (str): Kernel symbol/name to locate.

    Returns:
        str: The best-ranked matching source path, or ``""`` when none.
    """
    tried: set[str] = set()
    # Primary pass: established keyword extraction + ranking (unchanged).
    for keyword in _candidate_keywords(name):
        if not keyword or keyword in tried:
            continue
        tried.add(keyword)
        hits: list[Path] = []
        for root in KNOWN_SEARCH_ROOTS:
            hits.extend(_grep_for_keyword(keyword, Path(root)))
        if hits:
            ranked = _rank_paths(hits, keyword=keyword)
            return str(ranked[0])
    # Fallback pass: trailing sub-windows of a compound/profiler-wrapped symbol
    # (e.g. sglang_profiler::..._invoke_fused_moe_kernel_427) whose full
    # identifier never appears verbatim in source — needed so recovered
    # ops_summary.csv candidates (#515 fallback) resolve to their kernel source.
    # Prefer the file that *defines* the embedded function over dispatch shims.
    for keyword in _compound_subwindow_keywords(name):
        if not keyword or keyword in tried:
            continue
        tried.add(keyword)
        hits = []
        for root in KNOWN_SEARCH_ROOTS:
            hits.extend(_grep_for_keyword(keyword, Path(root)))
        if hits:
            return str(_prefer_symbol_definition(keyword, hits)[0])
    return ""


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


_KNOWN_HARNESS_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    # --- Normalization ---
    (
        ("rmsnorm_quant", "add_rmsnorm_quant", "rmsnorm", "add_rmsnorm"),
        (
            "/sgl-workspace/aiter/op_tests/test_rmsnorm2dFusedAddQuant.py",
            "/sgl-workspace/aiter/op_tests/test_rmsnorm2d.py",
            "/sgl-workspace/aiter/op_tests/op_benchmarks/triton/bench_rmsnorm.py",
            "/sgl-workspace/sglang/sgl-kernel/benchmark/bench_rmsnorm.py",
            "/sgl-workspace/aiter/op_tests/triton_tests/normalization/test_rmsnorm.py",
            "/sgl-workspace/aiter/op_tests/triton_tests/normalization/test_fused_add_rmsnorm_pad.py",
        ),
    ),
    # --- Activation ---
    (
        ("activation", "act_and_mul", "silu"),
        (
            "/sgl-workspace/aiter/op_tests/test_activation.py",
            "/sgl-workspace/aiter/op_tests/op_benchmarks/triton/bench_ff_a16w16_fused.py",
            "/sgl-workspace/sglang/sgl-kernel/tests/test_activation.py",
            "/sgl-workspace/sglang/sgl-kernel/benchmark/bench_activation.py",
            "/sgl-workspace/sglang/python/sglang/jit_kernel/tests/test_activation.py",
            "/sgl-workspace/sglang/python/sglang/jit_kernel/benchmark/bench_activation.py",
        ),
    ),
    # --- Attention ---
    (
        ("paged_attention", "fmha", "attention"),
        (
            "/sgl-workspace/aiter/op_tests/test_pa.py",
            "/sgl-workspace/aiter/op_tests/op_benchmarks/triton/bench_pa_decode.py",
            "/sgl-workspace/aiter/op_tests/op_benchmarks/triton/bench_pa_prefill.py",
        ),
    ),
    # --- MLA decode ---
    (
        ("mla_decode", "pseudo_mla", "mla_persistent"),
        (
            "/sgl-workspace/aiter/op_tests/test_mla.py",
            "/sgl-workspace/aiter/op_tests/test_mla_persistent.py",
            "/sgl-workspace/aiter/op_tests/op_benchmarks/triton/bench_mla_decode.py",
        ),
    ),
    # --- MoE CK two-stage ---
    (
        ("ck_moe_stage", "moe_2stage", "moe_stage1", "moe_stage2"),
        (
            "/sgl-workspace/aiter/op_tests/test_moe_2stage.py",
            "/sgl-workspace/aiter/op_tests/op_benchmarks/triton/bench_moe.py",
        ),
    ),
    # --- MoE FP8 blockscale (ASM) ---
    (
        ("fmoe_fp8_blockscale", "moe_blockscale"),
        (
            "/sgl-workspace/aiter/op_tests/test_moe_blockscale.py",
            "/sgl-workspace/aiter/op_tests/triton_tests/moe/test_moe_gemm_a8w8_blockscale.py",
        ),
    ),
    # --- GEMM A8W8 blockscale ---
    (
        ("gemm_a8w8_blockscale",),
        (
            "/sgl-workspace/aiter/op_tests/test_gemm_a8w8_blockscale.py",
            "/sgl-workspace/aiter/op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py",
        ),
    ),
    # --- Quantization ---
    (
        ("dynamic_per_token_scaled_quant", "per_token_quant"),
        (
            "/sgl-workspace/aiter/op_tests/test_quant.py",
            "/sgl-workspace/aiter/op_tests/triton_tests/quant/test_quant.py",
        ),
    ),
    # --- Batch-invariant addmm (Triton) ---
    (
        ("batch_invariant", "addmm"),
        (
            "/sgl-workspace/sglang/test/registered/unit/batch_invariant_ops/test_batch_invariant_ops.py",
        ),
    ),
)


def _known_harness_files(name: str, source_file: str) -> list[Path]:
    """Return curated benchmark/test harnesses matching a kernel.

    Looks up :data:`_KNOWN_HARNESS_HINTS` by marker substrings found in the
    kernel name / source path and returns the hinted harnesses that exist.

    Args:
        name (str): Kernel symbol/name.
        source_file (str): Resolved source-file path (may be empty).

    Returns:
        list[Path]: Existing curated harness files, possibly empty.
    """
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
        """Return whether a harness path looks multi-GPU / distributed.

        Args:
            path_str (str): Candidate harness file path.

        Returns:
            bool: ``True`` when the path contains a multi-GPU/distributed tag.
        """
        low = path_str.lower()
        return any(tag in low for tag in ("multigpu", "multi_gpu", "multinode", "/dist/", "_dist_"))
    unique.sort(key=_is_multigpu)
    return unique[:10]


_PYBIND_PARENT_DIRS = ("csrc/pybind", "csrc/python", "python_bindings")
# A pybind11 registration shim (<2KB, just PYBIND11_MODULE) has no device code; detect so callers promote it to the real .cu/.cuh.
def _is_pybind_shim(source_file: str) -> bool:
    """Detect a tiny pybind11 registration TU (no rewritable device code).

    A shim is a small (<2 KB) ``.cu``/``.cpp``/``.cc`` file under a pybind
    directory that only contains ``PYBIND11_MODULE`` glue.

    Args:
        source_file (str): Candidate source-file path.

    Returns:
        bool: ``True`` when the file is a pybind11 registration shim.
    """
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
    # rmsnorm — fused add + rmsnorm + quantization kernel (dense / MoE shared).
    ("rmsnorm", (
        "csrc/kernels/rmsnorm_quant_kernels.cu",
        "csrc/py_itfs_ck/rmsnorm_ck_kernels.cu",
    )),
)

# Fallback aiter editable-checkout root when find_repo_root can't resolve from a wheel-install wrapper (no csrc/).
_AITER_FALLBACK_REPO = "/sgl-workspace/aiter"

# Framework package-inner roots for resolving relative launcher paths.
# TraceLens emits paths like "ops/rmsnorm.py" relative to the package dir,
# not the repo root. These are tried when _resolve_launcher_to_abs_source fails.
_PACKAGE_INNER_ROOTS = (
    "/sgl-workspace/aiter/aiter",
    "/sgl-workspace/sglang/python/sglang",
)


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


_SGL_KERNEL_DEVICE_PROMOTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "silu_and_mul",
        (
            "/sgl-workspace/sglang/sgl-kernel/include/hip/hip_act_and_mul.cuh",
        ),
    ),
)


def upgrade_sgl_kernel_launcher(source_file: str, kernel_name: str) -> str:
    """Promote known sgl-kernel Python launchers to reusable HIP source."""
    if not source_file or not kernel_name:
        return source_file
    source_posix = source_file.replace(os.sep, "/")
    if "sgl-kernel/python/sgl_kernel/" not in source_posix:
        return source_file
    name_lower = kernel_name.lower()
    for marker, candidates in _SGL_KERNEL_DEVICE_PROMOTIONS:
        if marker not in name_lower:
            continue
        for candidate in candidates:
            if Path(candidate).is_file():
                return candidate
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
    """Coerce a loosely-typed call-count value into a positive int.

    Handles ``None``/empty, numpy-repr strings (``np.float64(...)``), and
    float-like strings.

    Args:
        value (Any): The raw count value to coerce.

    Returns:
        int | None: A positive integer, or ``None`` when absent, unparseable,
            or non-positive.
    """
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
    """Merge a shape/call-count pair into an accumulator list in place.

    If an entry with the same shape already exists, its ``call_num`` is
    incremented; otherwise a new entry is appended.

    Args:
        target (list[Any]): Accumulator list of ``{"shape", "call_num"}`` dicts.
        shape (Any): The shape value to merge.
        call_num (int): Call count to add for this shape.
    """
    for entry in target:
        if isinstance(entry, dict) and entry.get("shape") == shape:
            entry["call_num"] = int(entry.get("call_num") or 0) + call_num
            return
    target.append({"call_num": call_num, "shape": shape})


def _shape_call_entries(shapes: Any, call_num: Any = None) -> list[dict[str, Any]]:
    """Normalize a shapes payload into merged ``{shape, call_num}`` entries.

    Args:
        shapes (Any): A list of shape values or ``{"shape", "call_num"}`` dicts.
        call_num (Any): Default call count applied when an entry lacks one.

    Returns:
        list[dict[str, Any]]: Deduplicated shape entries with summed call
            counts; empty when ``shapes`` is not a list.
    """
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
    """Heuristic: kernel is a multi-GPU collective if name/source hints it.

    Args:
        name (str): Kernel symbol/name.
        source_file (str): Resolved source-file path (may be empty).

    Returns:
        bool: ``True`` when the name/source contains a collective /
            distributed marker.
    """
    blob = f"{name} {source_file}".lower()
    return any(tag in blob for tag in (
        "all_reduce", "allreduce", "all_gather", "allgather",
        "reduce_scatter", "broadcast", "p2p", "send_recv",
        "cross_device", "rank_signal", "ranksignals",
        "/dist/", "dist/", "communicator",
    ))


def analyze_trace_files(trace_files: list[Path], top_k: int) -> list[dict[str, Any]]:
    """Aggregate GPU kernels across raw trace files into top-K candidates.

    Sums per-kernel duration and call counts across all events, takes the
    top ``top_k`` by duration, then runs :func:`_finalize_candidates`. This
    is the dry-run / test-only raw-trace path (production uses analysis.md).

    Args:
        trace_files (list[Path]): Trace files (optionally gzipped) to scan.
        top_k (int): Number of hottest kernels to keep.

    Returns:
        list[dict[str, Any]]: Finalized hot-kernel candidate dicts.
    """
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
    out: dict[str, str] = {}
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = str(row.get("name") or "").strip()
                cat = str(row.get("op category") or "").strip()
                if name and cat and name not in out:
                    out[name] = cat
    except (OSError, csv.Error):
        return {}
    return out


# ── #727 companion: trace-anchored shape capture for the fused-MoE expert kernel ──
#
# The Triton fused-MoE expert kernel surfaces in the torch trace as a pybind
# built-in (``sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel_427``)
# whose top-level kernel event carries NO resolvable ``Input Dims`` — so the
# moe_fused category metric (and the LLM-rendered analysis.md ``Args`` column)
# emit the candidate with ``shapes: []``. Hyperloom's dispatch gate
# (``_validate_kernel_shape_and_paths`` → ``empty_kernel_shape``) then rejects the
# whole geak→claude→codex ladder before any harness is built.
#
# TraceLens DOES still capture the operands for this kernel: the per-shape rows in
# ``perf_report_csvs/ops_unique_args.csv`` carry the wrapped invocation's
# ``Input Dims`` / ``Input type`` (the gate/up and down grouped-GEMM operands).
# This recovers those operand shapes from that deterministic sidecar so the
# emitted candidate carries non-empty, trace-anchored ``shapes`` with
# ``shape_provenance="torch_trace"``. Scoped to the fused-MoE invoke kernel so
# other ops are untouched.
_FUSED_MOE_KERNEL_MARKER = "invoke_fused_moe_kernel"

# torch ``Input type`` token → compact dtype suffix used by TraceLens shape
# strings (e.g. ``(15360,2048) bf16``). Unmapped/empty types emit a bare shape.
_TRACE_DTYPE_SUFFIX = {
    "c10::bfloat16": "bf16",
    "bfloat16": "bf16",
    "c10::half": "f16",
    "half": "f16",
    "float16": "f16",
    "float": "f32",
    "float32": "f32",
    "double": "f64",
    "float64": "f64",
    "int": "i32",
    "int32": "i32",
    "long": "i64",
    "int64": "i64",
    "short": "i16",
    "int16": "i16",
    "char": "i8",
    "int8": "i8",
    "uint8": "u8",
    "bool": "bool",
}


def _format_trace_shape(dims: Any, dtype: Any) -> str | None:
    """Render one operand as ``(d0,d1,...) <dtype>`` (matching TraceLens shape strings); ``None`` for scalar/empty operands."""
    if not isinstance(dims, (list, tuple)) or not dims:
        return None
    try:
        body = ",".join(str(int(d)) for d in dims)
    except (TypeError, ValueError):
        return None
    shape = f"({body},)" if len(dims) == 1 else f"({body})"
    suffix = _TRACE_DTYPE_SUFFIX.get(str(dtype or "").strip().lower())
    return f"{shape} {suffix}" if suffix else shape


def resolve_fused_moe_shapes_from_csv(
    perf_report_csv_dir: Path | str | None,
) -> list[str]:
    """Recover the fused-MoE expert kernel operand shapes from ``ops_unique_args.csv``.

    Reads each per-shape row whose op name embeds ``invoke_fused_moe_kernel`` and
    parses its ``Input Dims`` / ``Input type`` tuple-of-tuples into TraceLens-style
    shape strings (e.g. ``(15360,2048) bf16``), deduped across rows in first-seen
    order. Returns ``[]`` when the sidecar is absent or carries no fused-MoE rows
    (so callers leave the candidate's empty ``shapes`` untouched).
    """
    if not perf_report_csv_dir:
        return []
    csv_path = Path(perf_report_csv_dir) / "ops_unique_args.csv"
    if not csv_path.is_file():
        return []
    shapes: list[str] = []
    seen: set[str] = set()
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = str(row.get("name") or "").strip().lower()
                if _FUSED_MOE_KERNEL_MARKER not in name:
                    continue
                try:
                    dims = ast.literal_eval(str(row.get("Input Dims") or "").strip() or "()")
                    dtypes = ast.literal_eval(str(row.get("Input type") or "").strip() or "()")
                except (ValueError, SyntaxError):
                    continue
                if not isinstance(dims, (list, tuple)):
                    continue
                if not isinstance(dtypes, (list, tuple)):
                    dtypes = ()
                for i, operand in enumerate(dims):
                    dtype = dtypes[i] if i < len(dtypes) else ""
                    s = _format_trace_shape(operand, dtype)
                    if s and s not in seen:
                        seen.add(s)
                        shapes.append(s)
    except (OSError, csv.Error):
        return []
    return shapes


def _is_fused_moe_candidate(item: dict[str, Any]) -> bool:
    """True for the Triton fused-MoE expert-kernel candidate (by op name or moe_fused category)."""
    name = str(item.get("name") or "").lower()
    if _FUSED_MOE_KERNEL_MARKER in name:
        return True
    cat = str(item.get("tracelens_category") or "").strip().lower()
    return cat == "moe_fused" and "moe" in name


# ── #514: high-GPU-time "other"-bucket candidate recovery (defense-in-depth) ──
#
# Hyperloom builds candidates EXCLUSIVELY from analysis.md reasoning-candidate
# (P-item) blocks (parse_analysis_md), and the driver refuses any
# priority_data/category_data/CSV fallback ("analysis.md is the single source
# of truth"). TraceLens never emits a reasoning-candidate block for a kernel it
# files under the un-roofline'd "other" category, so a dominant editable kernel
# (e.g. the Triton fused-MoE GEMM at ~67% GPU time) lands in neither
# hot_kernels nor skipped_kernels and GEAK never sees it.
#
# This is the Hyperloom-side defense-in-depth net: it recovers a HIGH-GPU-time
# "other"-bucket op that is MISSING from the analysis.md candidates from the
# per-op ranking TraceLens already writes (ops_summary.csv /
# unified_perf_summary.csv / priority_data.json), so the op flows through the
# normal _finalize_candidates -> classify_patchability path and is routed to
# GEAK iff it is a reusable native kernel. analysis.md stays PRIMARY; this only
# fires for high-time ops with no reasoning-candidate block. The TraceLens-side
# categorization gap is fixed separately upstream.

_OTHER_BUCKET_MIN_GPU_PCT_ENV = "HYPERLOOM_OTHER_BUCKET_MIN_GPU_PCT"
_DEFAULT_OTHER_BUCKET_MIN_GPU_PCT = 10.0

# Raw/normalized category labels that mean "TraceLens did not roofline this op"
# (no P-item block). Compared case-insensitively against both the raw category
# and normalize_upstream_category() output.
_OTHER_LIKE_CATEGORIES = frozenset({
    "", "other", "others", "misc", "miscellaneous", "uncategorized",
    "uncategorised", "unknown", "n/a", "na", "none", "null",
})

# Per-op ranking sidecars in preference order, relative to the skill output dir.
_OPS_RANKING_CSV_RELPATHS = (
    "ops_summary.csv",
    "perf_report_csvs/ops_summary.csv",
    "perf_report_csvs/unified_perf_summary.csv",
    "unified_perf_summary.csv",
)
_OPS_RANKING_JSON_RELPATHS = (
    "priority_data.json",
    "perf_report_csvs/priority_data.json",
)

_RANK_NAME_KEYS = ("name", "operation", "op", "kernel", "kernel_name", "op_name")
_RANK_CATEGORY_KEYS = (
    # ``categories`` is the real ops_summary.csv column (stored as a list-repr
    # string like ``['MoE_fused']``; see _clean_category_label).
    "op category", "op_category", "category", "categories", "tracelens_category",
)
# GPU-time column/field names → multiplier to microseconds (most specific first).
# ``total_direct_kernel_time_ms`` is the real ops_summary.csv per-op GPU-time
# column (its ``_sum`` sibling holds the same value in microseconds, handled by
# _RANK_TIME_US_KEYS below).
_RANK_TIME_MS_KEYS = (
    "gpu time total (ms)", "total gpu time (ms)", "gpu time (ms)",
    "self gpu time (ms)", "total_direct_kernel_time_ms",
    "total_direct_kernel_time_ms_sum", "gpu_time_ms", "gpu_time_total_ms",
    "duration (ms)", "dur (ms)", "time (ms)", "time_ms", "duration_ms",
)
_RANK_TIME_US_KEYS = (
    "gpu time (us)", "self gpu time (us)", "total_direct_kernel_time_sum",
    "total_direct_kernel_time_us", "gpu_time_us", "gpu_time_total_us",
    "duration_us", "dur (us)", "time (us)", "time_us", "dur",
)
_RANK_PCT_KEYS = (
    # ``percentage (%)`` is the real ops_summary.csv % column.
    "% gpu time", "gpu time %", "gpu %", "gpu_pct", "gpu_time_pct",
    "% of compute time", "% of compute", "%e2e", "% e2e",
    "percentage (%)", "percentage", "percent", "pct",
)


def _coerce_float(value: Any) -> float | None:
    """Parse a CSV/JSON cell to float (strips ``%`` / commas); ``None`` if not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().rstrip("%").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _lower_keyed(row: dict) -> dict[str, Any]:
    """Return a copy of ``row`` with keys trimmed and lower-cased.

    Args:
        row: Source mapping (e.g. a CSV/JSON record).

    Returns:
        A new dict keyed by the normalized column names.
    """
    return {str(k).strip().lower(): v for k, v in row.items()}


def _first_present(low: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-empty value among candidate keys.

    Args:
        low: Lower-keyed record to read from.
        keys: Candidate keys, in priority order.

    Returns:
        The first present, non-empty value as a stripped string, or ``""``.
    """
    for k in keys:
        v = low.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _clean_category_label(raw: str) -> str:
    """Normalize a category cell to a bare label.

    The real ``ops_summary.csv`` stores the category as a Python list-repr
    string, e.g. ``['MoE_fused']`` or ``['GEMM', 'Reduce']`` — return the first
    element (``MoE_fused`` / ``GEMM``). Plain labels pass through unchanged.
    """
    s = str(raw or "").strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)) and val:
                return str(val[0]).strip()
            if isinstance(val, (list, tuple)):
                return ""
        except (ValueError, SyntaxError):
            s = s.strip("[]")
    return s.strip().strip("'\"").strip()


def _record_gpu_us(low: dict[str, Any]) -> float | None:
    """Best-effort GPU time in microseconds from a per-op ranking record."""
    for k in _RANK_TIME_MS_KEYS:
        if k in low:
            val = _coerce_float(low[k])
            if val is not None:
                return val * 1000.0
    for k in _RANK_TIME_US_KEYS:
        if k in low:
            val = _coerce_float(low[k])
            if val is not None:
                return val
    return None


def _record_gpu_pct(low: dict[str, Any]) -> float | None:
    """Extract the GPU-time percentage from a per-op ranking record.

    Args:
        low: Lower-keyed ranking record.

    Returns:
        The GPU-time percentage, or ``None`` when no recognized key parses.
    """
    for k in _RANK_PCT_KEYS:
        if k in low:
            val = _coerce_float(low[k])
            if val is not None:
                return val
    return None


def _ranking_record(raw: dict) -> dict[str, Any] | None:
    """Normalize a raw ranking row into a standard candidate record.

    Args:
        raw: Raw CSV/JSON ranking row.

    Returns:
        A dict with ``name``, ``category``, ``gpu_us``, and ``gpu_pct``, or
        ``None`` when the row has no usable op name.
    """
    low = _lower_keyed(raw)
    name = _first_present(low, _RANK_NAME_KEYS)
    if not name:
        return None
    return {
        "name": name,
        "category": _clean_category_label(_first_present(low, _RANK_CATEGORY_KEYS)),
        "gpu_us": _record_gpu_us(low),
        "gpu_pct": _record_gpu_pct(low),
    }


def _load_ops_ranking_csv(path: Path) -> list[dict[str, Any]]:
    """Load per-op ranking records from a CSV sidecar.

    Args:
        path: Path to the CSV file.

    Returns:
        Normalized ranking records, or ``[]`` when the file is missing or
        cannot be read.
    """
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rec = _ranking_record(row)
                if rec is not None:
                    out.append(rec)
    except (OSError, csv.Error):
        return []
    return out


def _iter_json_ranking_records(data: Any) -> list[dict]:
    """Extract the list of ranking record dicts from parsed JSON.

    Accepts either a top-level list or a dict containing the records under one
    of several known keys.

    Args:
        data: Parsed JSON value.

    Returns:
        The list of dict records, or ``[]`` when none are found.
    """
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for key in (
            "findings", "operations", "ops", "priorities", "candidates",
            "kernels", "items", "rows",
        ):
            v = data.get(key)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
    return []


def _load_ops_ranking_json(path: Path) -> list[dict[str, Any]]:
    """Load per-op ranking records from a JSON sidecar.

    Args:
        path: Path to the JSON file.

    Returns:
        Normalized ranking records, or ``[]`` when the file is missing or
        cannot be parsed.
    """
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[dict[str, Any]] = []
    for rec_raw in _iter_json_ranking_records(data):
        rec = _ranking_record(rec_raw)
        if rec is not None:
            out.append(rec)
    return out


def load_ops_ranking(
    skill_output_dir: Path | str | None,
) -> list[dict[str, Any]]:
    """Load a per-op GPU-time ranking from TraceLens sidecars (schema-tolerant).

    Returns ``[{name, category, gpu_us, gpu_pct}]`` from the first sidecar that
    yields rows (ops_summary.csv / unified_perf_summary.csv /
    priority_data.json, under the output dir or its ``perf_report_csvs/``);
    ``[]`` when none is present/parseable. Used only by the ``other``-bucket
    recovery fallback — the contracted candidate source remains analysis.md.
    """
    if not skill_output_dir:
        return []
    root = Path(skill_output_dir)
    for rel in _OPS_RANKING_CSV_RELPATHS:
        rows = _load_ops_ranking_csv(root / rel)
        if rows:
            return rows
    for rel in _OPS_RANKING_JSON_RELPATHS:
        rows = _load_ops_ranking_json(root / rel)
        if rows:
            return rows
    return []


def _resolve_other_bucket_min_gpu_pct() -> float:
    """Return the minimum GPU-time percentage for other-bucket recovery.

    Returns:
        The value from ``HYPERLOOM_OTHER_BUCKET_MIN_GPU_PCT`` when set to a
        valid non-negative number, otherwise the default threshold.
    """
    raw = os.environ.get(_OTHER_BUCKET_MIN_GPU_PCT_ENV, "").strip()
    if raw:
        val = _coerce_float(raw)
        if val is not None and val >= 0:
            return val
    return _DEFAULT_OTHER_BUCKET_MIN_GPU_PCT


def _is_other_like_category(category: str) -> bool:
    """Return whether a category counts as the ``other`` bucket.

    Args:
        category: Raw or upstream category label.

    Returns:
        ``True`` if the label (or its normalized upstream form) is an
        other-like category.
    """
    raw_l = str(category or "").strip().lower()
    if raw_l in _OTHER_LIKE_CATEGORIES:
        return True
    return str(
        normalize_upstream_category(category or "")
    ).strip().lower() in _OTHER_LIKE_CATEGORIES


def recover_other_bucket_candidates(
    skill_output_dir: Path | str | None,
    existing_candidates: list[dict[str, Any]],
    *,
    top_k: int = 10,
    min_gpu_pct: float | None = None,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Recover HIGH-GPU-time ops missing from analysis.md (any category).

    Defense-in-depth fallback for the candidate-extraction gap: surfaces ops
    that have no analysis.md reasoning-candidate block (so ``parse_analysis_md``
    dropped them) as raw candidates, ranked by per-op sidecar GPU time, so each
    flows through ``_finalize_candidates`` -> ``classify_patchability``.

    Originally scoped to the un-roofline'd ``other`` bucket; broadened so **any**
    high-GPU-time op missing from ``existing_candidates`` is eligible. This is
    required for categories like ``MoE_fused`` whose dominant fused-MoE kernel is
    correctly categorized but whose roofline is null (so TraceLens emits no
    reasoning-candidate block and the kernel is otherwise dropped). The
    patchability gate downstream still rejects vendor / native ops, so widening
    the recovery net does not route non-patchable kernels to GEAK.

    Fires for ops that are (a) absent from ``existing_candidates`` by name and
    (b) at or above ``min_gpu_pct`` of total GPU time (env
    ``HYPERLOOM_OTHER_BUCKET_MIN_GPU_PCT``, default 10%). Returns ``[]`` when no
    sidecar is available or nothing qualifies, so analysis.md stays the primary
    source.
    """
    ranking = load_ops_ranking(skill_output_dir)
    if not ranking:
        return []
    if min_gpu_pct is None:
        min_gpu_pct = _resolve_other_bucket_min_gpu_pct()

    have = {
        str(c.get("name") or "").strip().lower()
        for c in existing_candidates
        if isinstance(c, dict)
    }
    total_us = sum(
        r["gpu_us"] for r in ranking if r.get("gpu_us") is not None
    ) or 0.0

    def _pct(rec: dict[str, Any]) -> float | None:
        """Return a record's GPU-time percentage.

        Args:
            rec: A ranking record.

        Returns:
            The explicit ``gpu_pct`` when present, else the share of
            ``total_us`` derived from ``gpu_us``, else ``None``.
        """
        if rec.get("gpu_pct") is not None:
            return float(rec["gpu_pct"])
        if total_us > 0 and rec.get("gpu_us") is not None:
            return rec["gpu_us"] / total_us * 100.0
        return None

    qualifying: list[tuple[float, dict[str, Any]]] = []
    for rec in ranking:
        name_l = str(rec.get("name") or "").strip().lower()
        if not name_l or name_l in have:
            continue
        # Gate broadened from "other-like category only" to "any high-GPU-time op
        # missing from existing candidates" so MoE_fused (and any other modeled-
        # but-unsurfaced category) is eligible.
        pct = _pct(rec)
        if pct is None or pct < min_gpu_pct:
            continue
        qualifying.append((pct, rec))

    if not qualifying:
        return []
    qualifying.sort(key=lambda t: t[0], reverse=True)

    out: list[dict[str, Any]] = []
    for pct, rec in qualifying[: max(1, top_k)]:
        gpu_us = rec.get("gpu_us")
        out.append({
            "name": rec["name"],
            "duration_us": float(gpu_us) if gpu_us is not None else 0.0,
            "call_count": 0,
            "source_file": "",
            "source_type": "unknown",
            "shapes": [],
            "tracelens_category": rec.get("category") or "other",
            "gpu_pct": round(pct, 3),
            "candidate_source": "other_bucket_fallback",
        })
        if log is not None:
            log(
                f"candidate recovery fallback (#514/#515): recovered "
                f"high-GPU-time op {rec['name']!r} (~{pct:.1f}% GPU time, "
                f"category={rec.get('category') or 'other'!r}) that has no "
                f"analysis.md reasoning-candidate block; routing through "
                f"classify_patchability so a reusable native kernel still "
                f"reaches GEAK"
            )
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
    # #727 companion: the fused-MoE expert kernel's top-level trace event carries
    # no Input Dims, so its candidate would emit empty ``shapes`` and trip the
    # dispatch gate. Recover its operand shapes once from the ops_unique_args
    # sidecar and graft them onto any fused-MoE candidate that lacks shapes.
    _fused_moe_shapes: list[str] | None = None
    for idx, item in enumerate(top, 1):
        item.pop("_extracted_source_checked", None)
        item.setdefault("source_file", "")
        item.setdefault("source_type", "unknown")
        item.setdefault("shapes", [])
        if not item.get("shapes") and _is_fused_moe_candidate(item):
            if _fused_moe_shapes is None:
                _fused_moe_shapes = resolve_fused_moe_shapes_from_csv(perf_report_csv_dir)
            if _fused_moe_shapes:
                item["shapes"] = list(_fused_moe_shapes)
                item["shape_provenance"] = "torch_trace"
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
        wrapper_before_sgl_promotion = item.get("source_file", "")
        item["source_file"] = upgrade_sgl_kernel_launcher(
            wrapper_before_sgl_promotion, item["name"],
        )
        if item["source_file"] != wrapper_before_sgl_promotion:
            item["launcher_source_file"] = wrapper_before_sgl_promotion
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
    """Build a short human-readable optimization note for a candidate.

    Args:
        candidate (dict[str, Any]): A finalized hot-kernel candidate row.

    Returns:
        str: A note describing whether/why the candidate is routable
            (resolved source vs. an explanation of why it was skipped).
    """
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


# ---------------------------------------------------------------------------
# Deterministic analysis route — runs TraceLens Python toolchain without LLM
# ---------------------------------------------------------------------------

_CATEGORY_ANALYSIS_ROUTES: dict[str, tuple[str, str | None]] = {
    "convolution": ("convolution", None),
    "conv_fwd": ("convolution", "conv_fwd"),
    "conv_bwd": ("convolution", "conv_bwd"),
    "customcollective": ("other", "customcollective"),
    "elementwise": ("elementwise", None),
    "gemm": ("gemm", None),
    "groupedgemm_fwd": ("gemm", "groupedgemm_fwd"),
    "groupedgemm_bwd": ("gemm", "groupedgemm_bwd"),
    "inferenceattention": ("sdpa", "inferenceattention"),
    "moe_fused": ("moe", "moe_fused"),
    "moe_unfused": ("moe", "moe_unfused"),
    "norm": ("norm", None),
    "norm_fwd": ("norm", "norm_fwd"),
    "norm_bwd": ("norm", "norm_bwd"),
    "other": ("other", "other"),
    "reduce": ("reduce", None),
    "rmsnorm": ("norm", "rmsnorm"),
    "sdpa": ("sdpa", "sdpa_fwd"),
    "sdpa_fwd": ("sdpa", "sdpa_fwd"),
    "sdpa_bwd": ("sdpa", "sdpa_bwd"),
    "triton": ("triton", None),
}
_SKIP_DETERMINISTIC_CATEGORIES: set[str] = {
    "cpu_idle",
    "kernel_fusion",
    "multi_kernel",
}


def _category_analysis_command(
    cat_name: str,
    tier: str,
    output_dir: Path,
) -> list[str] | None:
    """Return the TraceLens category-analysis command for one manifest category."""
    if tier != "compute_kernel":
        return None
    if cat_name in _SKIP_DETERMINISTIC_CATEGORIES:
        return None
    route = _CATEGORY_ANALYSIS_ROUTES.get(cat_name)
    if route is None:
        return None
    script_base, category_arg = route
    if script_base == "gemm" and category_arg is not None:
        # TraceLens gemm_analysis.py currently hard-codes category="gemm" and
        # has no --category flag. Reuse its helpers while passing the manifest
        # category so groupedgemm_* CSVs produce their own *_metrics.json.
        snippet = (
            "from TraceLens.Agent.Analysis.category_analyses.gemm_analysis "
            "import classify_gemm_operation, extract_category_specific; "
            "from TraceLens.Agent.Analysis.category_analyses.analysis_utils "
            "import run_category_analysis; "
            "run_category_analysis("
            f"category={cat_name!r}, "
            f"output_dir={str(output_dir)!r}, "
            "config={"
            "'extra_fields': ['Input Dims', 'Input type', 'has_perf_model'], "
            "'operation_classifier': classify_gemm_operation"
            "}, "
            "extract_fn=extract_category_specific"
            ")"
        )
        return [sys.executable, "-c", snippet]
    cmd = [
        sys.executable, "-m",
        f"TraceLens.Agent.Analysis.category_analyses.{script_base}_analysis",
        "--output-dir", str(output_dir),
    ]
    if category_arg is not None:
        cmd += ["--category", category_arg]
    return cmd


def _raise_on_failed_deterministic_pipeline(
    det_rc: int,
) -> None:
    """Fail deterministic route on any TraceLens deterministic toolchain error."""
    if det_rc == 0:
        return
    raise RuntimeError(
        "Deterministic TraceLens pipeline failed "
        f"(rc={det_rc}); refusing to return partial hot_kernels[]. "
        "Inspect the tracelens/ artifacts and logs."
    )


def _run_deterministic_tracelens_steps(
    trace_path: Path,
    output_dir: Path,
    tl_root: Path,
    *,
    platform: str,
    analysis_mode: str,
    framework: str,
    capture_folder: Path | None,
    log_path: Path,
    budget_minutes: float,
) -> int:
    """Run the TraceLens deterministic pipeline (Steps 1 + 2-5 + 7 scripts + 7.5).

    Invokes the CLI tools as subprocesses so they run in the TraceLens
    package environment. Returns 0 on success.
    """
    timeout_s = max(120, int(budget_minutes * 60))

    csv_dir = output_dir / "perf_report_csvs"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: perf report
    report_cmd = [
        sys.executable, "-m",
        "TraceLens.Reporting.generate_perf_report_pytorch_inference",
        "--profile_json_path", str(trace_path),
        "--output_csvs_dir", str(csv_dir),
        "--gpu_arch_platform", platform,
        "--include_call_stack",
        "--enable_pseudo_ops",
    ]
    if capture_folder and capture_folder.exists():
        report_cmd += ["--capture_folder", str(capture_folder)]
    rc = run_command(report_cmd, cwd=tl_root, log_path=log_path, timeout_s=timeout_s)
    if rc != 0:
        return rc

    # Steps 2-5: orchestrator_prepare
    prepare_cmd = [
        sys.executable, "-m",
        "TraceLens.Agent.Analysis.utils.orchestrator_prepare",
        "--trace-path", str(trace_path),
        "--output-dir", str(output_dir),
        "--platform", platform,
    ]
    rc = run_command(prepare_cmd, cwd=tl_root, log_path=log_path, timeout_s=timeout_s)
    if rc != 0:
        return rc

    # Step 7: category analysis scripts. The TraceLens manifest uses analyzer
    # category names (e.g. sdpa_fwd / norm_bwd), while the Python modules are
    # shared by families (sdpa_analysis / norm_analysis).
    manifest_path = output_dir / "category_data" / "category_manifest.json"
    category_failures: list[int] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            append_log(
                log_path,
                f"deterministic: failed to parse {manifest_path}: {exc}",
            )
            return 1
        for cat_entry in manifest.get("categories", []):
            cat_name = cat_entry.get("name", "")
            tier = cat_entry.get("tier", "")
            analysis_cmd = _category_analysis_command(cat_name, tier, output_dir)
            if analysis_cmd is None:
                append_log(
                    log_path,
                    f"deterministic: no category analysis script for "
                    f"category={cat_name!r} tier={tier!r}; skipping",
                )
                continue
            rc_cat = run_command(
                analysis_cmd, cwd=tl_root, log_path=log_path, timeout_s=timeout_s,
            )
            if rc_cat != 0:
                category_failures.append(rc_cat)
                append_log(
                    log_path,
                    f"deterministic: category script for {cat_name} exited "
                    f"with rc={rc_cat}; continuing with remaining categories",
                )

    # Step 7.5: generate_priority_data
    priority_cmd = [
        sys.executable, "-c",
        f"from TraceLens.Agent.Analysis.utils.report_utils import "
        f"generate_priority_data; "
        f"generate_priority_data({str(output_dir)!r})",
    ]
    rc = run_command(priority_cmd, cwd=tl_root, log_path=log_path, timeout_s=timeout_s)
    if rc != 0:
        return rc
    if category_failures:
        return category_failures[0]
    return rc


def _extract_idle_pct_from_gpu_timeline(output_dir: Path) -> float | None:
    """Read GPU idle percentage directly from gpu_timeline.csv."""
    csv_path = output_dir / "perf_report_csvs" / "gpu_timeline.csv"
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row_type = (row.get("type") or "").strip().lower()
                if row_type == "idle_time":
                    return float(row.get("percent", 0))
    except (OSError, csv.Error, ValueError):
        pass
    return None


def _extract_total_time_us_from_gpu_timeline(output_dir: Path) -> float | None:
    """Read the trace total_time from gpu_timeline.csv (ms -> us)."""
    csv_path = output_dir / "perf_report_csvs" / "gpu_timeline.csv"
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (row.get("type") or "").strip().lower() == "total_time":
                    return float(row.get("time ms", 0)) * 1000.0
    except (OSError, csv.Error, ValueError):
        pass
    return None


_MATCH_OP_MAX_DELTA_MS = 5.0


def _match_op_by_time(
    ops: list[dict[str, Any]], name: str, time_ms: float,
) -> dict[str, Any]:
    """Find the operation in *_metrics.json matching by name and time_ms.

    Multiple operations can share the same name (e.g. ``aten::mm`` with
    different shapes). We match by ``time_ms`` with a small tolerance to
    account for floating-point rounding in JSON serialization.

    Returns an empty dict when no candidate is within
    ``_MATCH_OP_MAX_DELTA_MS`` milliseconds, preventing silent
    mis-association of launcher paths and shapes.
    """
    best: dict[str, Any] = {}
    best_delta = float("inf")
    for op in ops:
        if op.get("name") != name:
            continue
        op_time = op.get("time_ms", 0)
        delta = abs(op_time - time_ms)
        if delta < best_delta:
            best_delta = delta
            best = op
            if delta < 0.01:
                break
    if best_delta > _MATCH_OP_MAX_DELTA_MS:
        return {}
    return best


def _resolve_source_file_from_kernel_path(kernel_path: str) -> str:
    """Resolve a TraceLens launcher path to an existing absolute source file."""
    raw_path, _, _ = _parse_launcher_path(kernel_path)
    if not raw_path:
        return ""
    if os.path.isabs(raw_path) and os.path.isfile(raw_path):
        return raw_path
    if raw_path.startswith("sgl_kernel/"):
        sgl_kernel_source = Path("/sgl-workspace/sglang/sgl-kernel/python") / raw_path
        if sgl_kernel_source.is_file():
            return str(sgl_kernel_source)
    resolved = _resolve_launcher_to_abs_source(kernel_path)
    if resolved is not None:
        return resolved[0]
    # Fallback: TraceLens launcher paths for aiter ops are relative to the
    # aiter package dir (e.g. "ops/rmsnorm.py" → /sgl-workspace/aiter/aiter/ops/rmsnorm.py).
    # Try known framework package roots when the head segment isn't a top-level package.
    if not os.path.isabs(raw_path):
        for pkg_root in _PACKAGE_INNER_ROOTS:
            candidate = os.path.join(pkg_root, raw_path)
            if os.path.isfile(candidate):
                return candidate
    return ""


def deterministic_extract_hot_kernels(
    output_dir: Path,
    top_k: int = 10,
    *,
    log_path: Path | None = None,
    fail_on_corrupt_priority: bool = False,
) -> list[dict[str, Any]]:
    """Extract hot kernels directly from TraceLens deterministic outputs.

    Reads ``*_metrics.json`` and ``priority_data.json`` to produce the same
    candidate list that ``parse_analysis_md()`` would extract from
    ``analysis.md``, without any LLM involvement.

    Each candidate maps to the same schema that ``_finalize_candidates``
    expects downstream (name, duration_us, efficiency_percent, etc.).
    """
    priority_path = output_dir / "priority_data.json"
    if not priority_path.exists():
        return []

    try:
        priority_data = json.loads(priority_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if log_path is not None:
            append_log(
                log_path,
                f"deterministic: failed to parse {priority_path}: {exc}",
            )
        if fail_on_corrupt_priority:
            raise RuntimeError(
                f"Deterministic TraceLens pipeline failed to parse "
                f"{priority_path}: {exc}"
            ) from exc
        return []
    findings = priority_data.get("findings", [])

    cat_data_dir = output_dir / "category_data"
    ops_by_category: dict[str, list[dict[str, Any]]] = {}
    if cat_data_dir.is_dir():
        for fname in sorted(cat_data_dir.iterdir()):
            if not fname.name.endswith("_metrics.json"):
                continue
            try:
                metrics = json.loads(fname.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metrics.get("status") in ("ERROR", "NO_DATA"):
                continue
            cat = metrics.get("category", fname.stem.replace("_metrics", ""))
            ops_by_category[cat] = metrics.get("operations", [])

    candidates: list[dict[str, Any]] = []

    # --- Phase 1: Collect candidates from priority_data findings ---
    for finding in findings:
        global_rank = finding.get("global_rank", 0)
        category = finding.get("category", "")
        impact_score = finding.get("impact_score", 0.0)
        members = finding.get("members", [])

        cat_ops = ops_by_category.get(category, [])

        sorted_members = sorted(
            members, key=lambda m: m.get("efficiency_pct", 100),
        )

        for member in sorted_members:
            op_name = member.get("operation", "")
            member_time_ms = member.get("time_ms", 0)

            # Match by (name, time_ms) to avoid collisions when multiple
            # ops share the same name (e.g. many aten::mm instances).
            full_op = _match_op_by_time(cat_ops, op_name, member_time_ms)
            if not full_op:
                if log_path is not None:
                    append_log(
                        log_path,
                        "deterministic: skipping priority member with no "
                        "matching metrics row "
                        f"(category={category!r}, operation={op_name!r}, "
                        f"time_ms={member_time_ms!r}, "
                        f"max_delta_ms={_MATCH_OP_MAX_DELTA_MS})",
                    )
                continue

            duration_us = member_time_ms * 1000
            eff_pct = member.get("efficiency_pct", 0)

            launcher_path = full_op.get("launcher_path", "")
            if launcher_path in ("\u2014", "-", ""):
                launcher_path = ""
            source_file = _resolve_source_file_from_kernel_path(launcher_path)

            shapes_raw = full_op.get("args", "")
            shapes = [shapes_raw] if shapes_raw else []
            op_count = full_op.get("count", 1)

            # Build non-synthetic input_shapes directly from TraceLens
            # metrics so _structured_benchmark_shape_cases sees real data
            # instead of the synthetic conversion in enrich_candidates.
            input_shapes: list[dict[str, Any]] = []
            if shapes_raw:
                input_shapes.append({
                    "call_num": op_count,
                    "shape": shapes_raw,
                })

            candidate = {
                "name": op_name,
                "duration_us": round(duration_us, 3),
                "call_count": op_count,
                "efficiency_percent": round(eff_pct, 2),
                "impact_score": member.get("impact_score", impact_score),
                "bound_type": member.get("bound_type", ""),
                "tracelens_category": category,
                "tracelens_pitem_rank": global_rank,
                "kernel_path": launcher_path,
                "tracelens_launcher_path": launcher_path,
                "source_file": source_file,
                "shapes": shapes,
                "input_shapes": input_shapes,
                "library": member.get("library", full_op.get("library", "")),
            }
            candidates.append(candidate)

    # --- Phase 2: Include "other" category ops with actionable source files ---
    # These are often the largest GPU-time consumers (e.g. Triton fused_moe,
    # 75% of GPU time) but don't appear in priority_data because TraceLens
    # categorizes them as "other" with no efficiency model.
    other_ops = ops_by_category.get("other", [])
    for op in other_ops:
        time_ms = op.get("time_ms", 0)
        if time_ms < 1.0:
            continue

        # TraceLens "other" metrics rows carry the profiler op name under
        # ``name`` (e.g. ``sglang_profiler::fused_moe_triton_kernels_invoke_
        # fused_moe_kernel_427``), which embeds the kernel *definition* file
        # stem + function. ``launcher_path`` only points at the Python wrapper
        # that *calls* the kernel (e.g. ``fused_moe.py(391)``), so it must not
        # be used as the editable source.
        op_name = op.get("name", "") or op.get("operation", "")
        launcher_path = op.get("launcher_path", "")
        if launcher_path in ("\u2014", "-"):
            launcher_path = ""

        # Resolve the symbol to its definition site (handles the launcher-vs-
        # definition split, e.g. the ``fused_moe.py`` wrapper vs the actual
        # ``fused_moe_triton_kernels.py`` @triton.jit kernel). We deliberately
        # do NOT fall back to ``launcher_path`` as the source: a launcher path
        # points at the calling wrapper, not an editable kernel body, which is
        # exactly the regression this guards against. If the definition cannot
        # be located, skip — classify_patchability would drop a wrapper anyway.
        if not op_name:
            if log_path is not None:
                append_log(
                    log_path,
                    "deterministic: other-bucket op skipped (no op name) "
                    f"time_ms={time_ms} launcher={launcher_path!r}",
                )
            continue
        source_file = locate_source_via_grep(op_name)
        if not source_file:
            # Never silently drop a hot op: surface unresolved high-GPU-time
            # kernels (e.g. vendor CK/Tensile templates that exist only as
            # compiled .so, or graph-captured names) so "missing hot_kernels"
            # is observable instead of vanishing without a trace.
            if log_path is not None:
                append_log(
                    log_path,
                    "deterministic: other-bucket op skipped (no editable "
                    f"source resolved) time_ms={time_ms:.3f} "
                    f"name={op_name!r} launcher={launcher_path!r}",
                )
            continue

        duration_us = time_ms * 1000
        op_count = op.get("count", 1)
        shapes_raw = op.get("args", "")
        shapes = [shapes_raw] if shapes_raw else []
        input_shapes: list[dict[str, Any]] = []
        if shapes_raw:
            input_shapes.append({
                "call_num": op_count,
                "shape": shapes_raw,
            })

        candidate = {
            "name": op_name,
            "duration_us": round(duration_us, 3),
            "call_count": op_count,
            "efficiency_percent": 0.0,
            "impact_score": 0.0,
            "bound_type": "unknown",
            "tracelens_category": "other",
            "tracelens_pitem_rank": 0,
            "kernel_path": launcher_path,
            "tracelens_launcher_path": launcher_path,
            "source_file": source_file,
            "shapes": shapes,
            "input_shapes": input_shapes,
            "library": op.get("library", ""),
            "candidate_source": "other_bucket_fallback",
        }
        candidates.append(candidate)

    # --- Phase 3: Sort all candidates by GPU time (duration) descending ---
    candidates.sort(key=lambda c: c.get("duration_us", 0), reverse=True)

    return candidates[:top_k]


def generate_minimal_analysis_md(
    output_dir: Path,
    candidates: list[dict[str, Any]],
    idle_pct: float | None = None,
) -> Path:
    """Generate a minimal deterministic analysis.md for human-readable output.

    Deterministic hot-kernel extraction uses structured ``*_metrics.json`` and
    ``priority_data.json`` directly. This Markdown report is intentionally not
    the parser contract used by the LLM-agent route.
    """
    report_path = output_dir / "analysis.md"
    lines: list[str] = []

    gpu_timeline_path = output_dir / "perf_report_csvs" / "gpu_timeline.csv"
    gpu_rows: list[dict[str, str]] = []
    if gpu_timeline_path.exists():
        try:
            with open(gpu_timeline_path, encoding="utf-8") as fh:
                gpu_rows = list(csv.DictReader(fh))
        except (OSError, csv.Error):
            pass

    lines.append("# TraceLens Performance Analysis Report")
    lines.append("")
    lines.append("> Generated via deterministic analysis route "
                 "(HYPERLOOM_TRACE_ANALYSIS_ROUTE=deterministic)")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    if gpu_rows:
        lines.append("| Metric | Time (ms) | % |")
        lines.append("|--------|-----------|---|")
        for row in gpu_rows:
            rtype = row.get("type", "")
            time_ms = row.get("time ms", "0")
            pct = row.get("percent", "0")
            lines.append(f"| {rtype} | {time_ms} | {pct}% |")
        lines.append("")

    if idle_pct is not None:
        lines.append(f"**Idle %**: {idle_pct:.2f}%")
        lines.append("")

    # System-Level Signals — deterministic GPU-timeline shares (idle / exposed
    # communication / exposed device copy). Read straight from gpu_timeline.csv
    # with no LLM interpretation; idle is flagged against the same threshold
    # that gates hot-kernel extraction. The agent route surfaces these as
    # LLM-written recommendations — here we expose only the underlying numbers.
    if gpu_rows:
        def _timeline_pct(row_type: str) -> float | None:
            for row in gpu_rows:
                if (row.get("type") or "").strip().lower() == row_type:
                    try:
                        return float(row.get("percent", 0))
                    except (TypeError, ValueError):
                        return None
            return None

        idle_share = _timeline_pct("idle_time")
        comm_share = _timeline_pct("exposed_comm_time")
        memcpy_share = _timeline_pct("exposed_memcpy_time")
        if any(v is not None for v in (idle_share, comm_share, memcpy_share)):
            idle_threshold = _resolve_idle_pct_threshold()
            lines.append("## System-Level Signals")
            lines.append("")
            lines.append("| Signal | % of total GPU time | Note |")
            lines.append("|--------|---------------------|------|")
            if idle_share is not None:
                note = (
                    f"above {idle_threshold:.0f}% idle gate"
                    if idle_share > idle_threshold
                    else f"within {idle_threshold:.0f}% idle gate"
                )
                lines.append(f"| GPU idle | {idle_share:.2f}% | {note} |")
            if comm_share is not None:
                lines.append(
                    f"| Exposed communication | {comm_share:.2f}% | - |"
                )
            if memcpy_share is not None:
                lines.append(
                    f"| Exposed memcpy (device copy) | {memcpy_share:.2f}% | - |"
                )
            lines.append("")

    # Top Hot Kernels table
    lines.append("## Top Hot Kernels")
    lines.append("")
    if candidates:
        lines.append(
            "| Rank | Operation | Time (us) | Efficiency | Impact | "
            "Category | Bound |"
        )
        lines.append(
            "|------|-----------|-----------|------------|--------|"
            "----------|-------|"
        )
        for i, c in enumerate(candidates, 1):
            lines.append(
                f"| {i} | {c.get('name', '')} "
                f"| {c.get('duration_us', 0):.1f} "
                f"| {c.get('efficiency_percent', 0):.1f}% "
                f"| {c.get('impact_score', 0):.2f} "
                f"| {c.get('tracelens_category', '')} "
                f"| {c.get('bound_type', '')} |"
            )
        lines.append("")

    # Per-P-item details for humans/downstream display; deterministic route
    # consumers should use the structured JSON artifacts instead of parsing this.
    seen_ranks: set[int] = set()
    for c in candidates:
        rank = c.get("tracelens_pitem_rank", 0)
        if rank in seen_ranks:
            continue
        seen_ranks.add(rank)
        rank_cands = [
            x for x in candidates
            if x.get("tracelens_pitem_rank") == rank
        ]
        cat = rank_cands[0].get("tracelens_category", "") if rank_cands else ""

        lines.append(f"### P{rank}: {cat} kernels")
        lines.append("")
        lines.append(
            f"<!-- reasoning-candidate tier=compute rank={rank} -->"
        )
        lines.append("")
        lines.append(
            "**Data:**\n\n"
            "| Operation | Time (us) | %E2E | Count | FLOPS/Byte | "
            "Efficiency | Bound | Args | Kernel Path |"
        )
        lines.append(
            "|-----------|-----------|------|-------|------------|"
            "------------|-------|------|-------------|"
        )
        for rc in rank_cands:
            lines.append(
                f"| {rc.get('name', '')} "
                f"| {rc.get('duration_us', 0):.1f} "
                f"| {rc.get('impact_score', 0):.2f} "
                f"| {rc.get('call_count', 1)} "
                f"| - "
                f"| {rc.get('efficiency_percent', 0):.1f}% "
                f"| {rc.get('bound_type', '')} "
                f"| {' '.join(rc.get('shapes', []))} "
                f"| {rc.get('kernel_path', '')} |"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None,
    log_path: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
) -> int:
    """Run a subprocess, tee its output to a log, and return the exit code.

    Args:
        cmd: Command and arguments to execute.
        cwd: Working directory, or ``None`` to inherit the current one.
        log_path: Log file the command line and output are appended to.
        timeout_s: Subprocess timeout in seconds.
        env: Optional environment for the child process.

    Returns:
        The process return code.
    """
    append_log(log_path, f"$ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    append_log(log_path, proc.stdout or "")
    append_log(log_path, f"[exit_code] {proc.returncode}")
    return proc.returncode


def roofline_match_key(name: str) -> str:
    """Normalize trace and rocprof names enough to join roofline data.

    Args:
        name (str): A kernel name from a trace or rocprof report.

    Returns:
        str: A canonical match key (e.g. ``hipblaslt_gemm``, ``attention``),
            falling back to the first 80 lower-cased chars.
    """
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
    """Load roofline results JSON keyed by normalized kernel match key.

    Args:
        path (str | None): Path to a roofline results JSON file (a list of
            rows or a dict with a ``results`` list); may be empty/``None``.

    Returns:
        dict[str, dict[str, Any]]: Map of :func:`roofline_match_key` to row;
            empty when the path is missing or unparseable.
    """
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
    """Merge roofline metrics into candidate rows in place.

    For each candidate, looks up roofline data by normalized name and copies
    bottleneck / utilization / suggestion fields; candidates without a match
    get conservative ``None``/default placeholders.

    Args:
        candidates (list[dict[str, Any]]): Hot-kernel candidate rows to enrich.
        roofline_by_name (dict[str, dict[str, Any]]): Roofline rows keyed by
            :func:`roofline_match_key`.
    """
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
    """Return the first argument that is neither ``None`` nor empty string.

    Args:
        *values (Any): Candidate values in priority order.

    Returns:
        Any: The first value that is not ``None`` or ``""``, else ``None``.
    """
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _kernel_roofline_row(candidate: dict[str, Any]) -> dict[str, Any]:
    """Project one hot-kernel candidate into the kernel-roofline view.

    Args:
        candidate (dict[str, Any]): A finalized hot-kernel candidate row.

    Returns:
        dict[str, Any]: A flattened roofline row with identity, cost, and
            (possibly ``None``) utilization/bottleneck fields.
    """
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
    """Enumerate candidate ``config.json`` paths for a model name/path.

    Considers the value as a direct JSON path, a directory containing
    ``config.json``, and locations under ``$HYPERLOOM_MODELS_ROOT``.

    Args:
        model_name (str): A model name or filesystem path.

    Returns:
        list[Path]: Deduplicated candidate config paths in priority order;
            empty when ``model_name`` is blank.
    """
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
    """Read HF config.json and return attention parameters relevant to GEAK.

    Args:
        model_name (str): A model name or path used to locate ``config.json``.

    Returns:
        dict[str, Any]: Attention/MLA params (e.g. ``HEAD_SIZE``,
            ``NUM_ATTENTION_HEADS``) plus ``MODEL_CONFIG_PATH``; empty when no
            readable config is found.
    """
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
    """Attach stable runtime metadata fields before GEAK prompt generation.

    Mutates each candidate in place, filling framework, shapes/dtypes,
    model and FlyDSL kernel params, and runtime flags so the downstream
    prompt builder sees a consistent schema.

    Args:
        candidates (list[dict[str, Any]]): Hot-kernel candidate rows to enrich.
        args (argparse.Namespace): Parsed CLI args carrying framework, model
            name, target platform, and runtime flags.
    """
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
            item["_input_shapes_synthetic"] = True
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
    """CLI entry point for the TraceLens analysis tool.

    Parses arguments, optionally splits the trace into a steady-state chunk,
    runs the TraceLens SDK orchestrator, extracts hot-kernel candidates,
    merges roofline data, and writes the run's report sidecars and status.

    Returns:
        int: ``0`` on success, ``1`` when the run failed (the error is also
            written to status and printed as JSON).
    """
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
    default_analysis_route = (
        os.environ.get(ANALYSIS_ROUTE_ENV, "").strip().lower()
        or ANALYSIS_ROUTE_AGENT
    )
    if default_analysis_route not in _VALID_ANALYSIS_ROUTES:
        default_analysis_route = ANALYSIS_ROUTE_AGENT
    parser.add_argument(
        "--analysis-route",
        choices=sorted(_VALID_ANALYSIS_ROUTES),
        default=default_analysis_route,
        help=(
            "Trace analysis pipeline route. 'agent' (default) runs the "
            "TraceLens analysis-orchestrator LLM skill via Claude SDK to "
            "produce analysis.md, then parses hot kernels from it. "
            "'deterministic' bypasses all LLM calls and instead runs the "
            "TraceLens deterministic Python toolchain (perf report, "
            "orchestrator_prepare, category analysis scripts, "
            "generate_priority_data) to extract hot kernels directly from "
            "*_metrics.json. A minimal analysis.md is generated from "
            "templates for downstream prompt injection. "
            "Env: HYPERLOOM_TRACE_ANALYSIS_ROUTE."
        ),
    )
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

    args.target_platform = normalize_platform(args.target_platform)

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

            # ---- #390: backfill MAF for the open-source TraceLens path ----
            # Public TraceLens carries no MAF values, so on MI355+ roofline
            # (and the whole kernel-optimization loop) fails. The benchmark is
            # gated on whether the TraceLens-internal extension is enabled:
            # when it is, the extension backfills MAF itself and we skip the
            # microbenchmark; when it is not (open-source path) we measure the
            # missing arch/MAF spec on an idle GPU before report generation.
            update_status(
                status_path,
                state="running",
                current_step="populate_gpu_arch_json",
                log_path=log_path,
                artifact_paths=artifacts,
                run_id=run_id,
                started_at=started_at,
            )
            arch_benchmark_timeout_s = _resolve_arch_benchmark_timeout_s()
            gpu_arch_path = populate_gpu_arch_json(
                tracelens_root=tl_root,
                platform=args.target_platform,
                internal_extension_enabled=tl_internal_root is not None,
                log=lambda msg: append_log(log_path, msg),
                run_command=lambda cmd, *, cwd, timeout_s, env=None: run_command(
                    cmd,
                    cwd=cwd,
                    log_path=log_path,
                    timeout_s=timeout_s,
                    env=env,
                ),
                timeout_s=arch_benchmark_timeout_s,
            )
            if gpu_arch_path is not None:
                artifacts["tracelens_gpu_arch_json"] = str(gpu_arch_path)

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
                    """Collect splitter chunk files for a steady-state prefix.

                    Args:
                        prefix (str): Chunk prefix (``mixed``, ``decode_only``,
                            or ``prefilldecode``).

                    Returns:
                        list[Path]: Sorted chunk files matching the prefix
                            across known trace extensions.
                    """
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

            # ------ Discover capture_folder (shared by both routes) ------
            trace_input_path = Path(args.trace_input).expanduser().resolve()
            capture_folder: Path | None = (
                Path(args.capture_folder).expanduser().resolve()
                if args.capture_folder else
                discover_capture_folder(trace_input_path, trace_files)
            )
            if capture_folder:
                append_log(
                    log_path,
                    f"capture_folder resolved: {capture_folder} "
                    f"(exists={capture_folder.is_dir()})",
                )

            # ------ Route: deterministic vs agent ------
            use_deterministic = (
                args.analysis_route == ANALYSIS_ROUTE_DETERMINISTIC
            )

            if use_deterministic and not trace_split_blocked:
                update_status(
                    status_path, state="running",
                    current_step="deterministic_pipeline",
                    log_path=log_path, artifact_paths=artifacts,
                    run_id=run_id, started_at=started_at,
                )
                append_log(
                    log_path,
                    "analysis-route=deterministic: running TraceLens "
                    "deterministic Python toolchain (no LLM calls)",
                )

                det_rc = _run_deterministic_tracelens_steps(
                    trace_path=cli_trace_path,
                    output_dir=tracelens_dir,
                    tl_root=tl_root,
                    platform=args.target_platform,
                    analysis_mode=args.analysis_mode,
                    framework=args.framework,
                    capture_folder=capture_folder,
                    log_path=log_path,
                    budget_minutes=args.budget_minutes,
                )
                orchestrator_mode = "deterministic"
                if det_rc != 0:
                    orchestrator_error = (
                        "Deterministic TraceLens pipeline returned "
                        f"rc={det_rc}"
                    )
                _raise_on_failed_deterministic_pipeline(det_rc)

                idle_pct_value = _extract_idle_pct_from_gpu_timeline(
                    tracelens_dir,
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
                            report_path=tracelens_dir / "analysis.md",
                        )
                    )
                    append_log(
                        log_path,
                        f"deterministic: GPU Idle % = {idle_pct_value:.2f}% "
                        f"(threshold {idle_pct_threshold:.2f}%); "
                        "suppressing hot_kernels[]",
                    )
                else:
                    if idle_pct_value is not None:
                        append_log(
                            log_path,
                            f"deterministic: GPU Idle % = "
                            f"{idle_pct_value:.2f}% "
                            f"(threshold {idle_pct_threshold:.2f}%) "
                            "-- below gate, extracting candidates",
                        )
                    raw_det_candidates = deterministic_extract_hot_kernels(
                        tracelens_dir,
                        args.top_k,
                        log_path=log_path,
                        fail_on_corrupt_priority=True,
                    )
                    if raw_det_candidates:
                        total_dur = (
                            _extract_total_time_us_from_gpu_timeline(
                                tracelens_dir
                            )
                            or sum(
                                float(c.get("duration_us") or 0)
                                for c in raw_det_candidates
                            )
                        )
                        agent_candidates = _finalize_candidates(
                            raw_det_candidates,
                            total_dur=total_dur or None,
                            perf_report_csv_dir=(
                                tracelens_dir / "perf_report_csvs"
                            ),
                        )
                        append_log(
                            log_path,
                            f"deterministic pipeline produced "
                            f"{len(agent_candidates)} hot kernels",
                        )
                    else:
                        agent_candidates = []
                        allow_empty_candidates = True
                        append_log(
                            log_path,
                            "deterministic pipeline: no candidates "
                            "extracted from *_metrics.json / "
                            "priority_data.json; returning empty "
                            "hot_kernels[]",
                        )

                agent_report_path = generate_minimal_analysis_md(
                    tracelens_dir,
                    agent_candidates or [],
                    idle_pct=idle_pct_value,
                )
                artifacts["tracelens_agent_report"] = str(agent_report_path)

            elif args.use_llm_orchestrator and not trace_split_blocked:
                update_status(status_path, state="running",
                              current_step="run_tracelens_sdk_orchestrator",
                              log_path=log_path, artifact_paths=artifacts,
                              run_id=run_id, started_at=started_at)
                try:
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
                        # #514 defense-in-depth: recover any HIGH-GPU-time
                        # "other"-bucket op that TraceLens filed without a
                        # reasoning-candidate block (so parse_analysis_md
                        # dropped it) from the per-op ranking sidecar, so the
                        # dominant editable kernel still reaches GEAK. No-op
                        # when the sidecars are absent or nothing qualifies;
                        # analysis.md stays the primary source.
                        fallback_cands = recover_other_bucket_candidates(
                            skill_result.output_dir,
                            report_cands,
                            top_k=args.top_k,
                            log=lambda msg: append_log(log_path, msg),
                        )
                        if fallback_cands:
                            report_cands = report_cands + fallback_cands
                        if report_cands:
                            raw_agent_candidates = report_cands
                            report_source = (
                                "analysis.md+other_bucket_fallback"
                                if fallback_cands else "analysis.md"
                            )
                        else:
                            agent_candidates = []
                            allow_empty_candidates = True
                            append_log(
                                log_path,
                                "TraceLens analysis.md had no Detailed "
                                "Analysis compute candidate blocks "
                                "(v0.3 contract: analysis.md is the single "
                                "source of truth) and the other-bucket "
                                "fallback found no high-GPU-time op to "
                                "recover. Producing empty hot_kernels[] — "
                                "downstream Coordinator will route to "
                                "params/backends.",
                            )

                    if raw_agent_candidates:
                        # Use whole-trace GPU time as the gpu_pct denominator
                        # (same as the deterministic route) so a kernel's
                        # gpu_pct means its share of total GPU time, not its
                        # share of the top-k candidate sum. Falls back to the
                        # candidate sum only when gpu_timeline.csv is missing.
                        total_dur = (
                            _extract_total_time_us_from_gpu_timeline(
                                skill_result.output_dir
                            )
                            or sum(
                                float(c.get("duration_us") or 0)
                                for c in raw_agent_candidates
                            )
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
                if use_deterministic:
                    raise RuntimeError(
                        "Deterministic analysis route failed to produce "
                        "any candidates; check the TraceLens toolchain "
                        "outputs under the tracelens/ directory."
                    )
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
