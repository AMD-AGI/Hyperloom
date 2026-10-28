# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Trace-health gates: whether a trace is idle, compute-starved, or a chunk worth analysing."""

from __future__ import annotations

import csv
import os
import zlib
from pathlib import Path
from typing import Any

# Trace-health gate thresholds + warnings (high idle, low compute): shared single
# source of truth so the TraceLens and bypass routes gate on identical semantics.
from ._idle_gate import (
    build_graph_under_recorded_warning as _build_graph_under_recorded_warning,
    build_high_idle_warning as _build_high_idle_warning,
    build_low_compute_warning as _build_low_compute_warning,
    resolve_idle_pct_threshold,
    resolve_min_compute_pct_threshold,
)


def _evaluate_high_idle_gate(idle_pct: float | None, report_path: Path) -> tuple[float, dict[str, Any] | None]:
    """Return the idle threshold plus a warning when the gate is exceeded.

    ``_build_high_idle_warning`` and ``resolve_idle_pct_threshold`` are the
    shared ``_idle_gate`` helpers imported at module top, so the threshold, gate
    semantics, and warning shape stay unified across routes.
    """
    threshold = resolve_idle_pct_threshold()
    if idle_pct is None or idle_pct <= threshold:
        return threshold, None
    return threshold, _build_high_idle_warning(
        idle_pct=idle_pct,
        threshold_pct=threshold,
        report_path=report_path,
    )


def _graph_coverage_from_raw_trace(trace_path: str | Path | None) -> dict[str, Any]:
    """Return the bypass reader's ``graph_coverage`` for the raw trace, or ``{}``.

    Reuses the tested ``_bypass_trace_reader.analyze_trace`` graph-launch
    detection so the TraceLens route can tell a cuda/HIP-graph under-recorded
    capture (profiler activity-buffer overflow → only ~1 of N replays recorded →
    inflated idle%) apart from a genuinely idle/launch-bound workload. An
    unreadable trace returns ``{}`` so the caller falls back to the plain idle gate.

    ``graph_coverage`` is derived from the launch/kernel correlation scan and is
    independent of the returned aggregation lists, so we pass ``emit_launches=
    False`` and ``top_k=1`` to avoid materializing the per-launch rows and full
    top-N lists on large raw traces.
    """
    if not trace_path:
        return {}
    try:
        from . import _bypass_trace_reader as _reader

        analyze = _reader.analyze_trace(str(trace_path), _reader.TraceAnalysisOptions(top_k=1, emit_launches=False))
        if analyze.get("truncated"):
            return {}
        cov = analyze.get("graph_coverage") if isinstance(analyze, dict) else None
        return cov if isinstance(cov, dict) else {}
    except (OSError, zlib.error):
        return {}


def _evaluate_idle_gate_with_graph_guard(
    idle_pct: float | None,
    report_path: Path,
    trace_path: str | Path | None,
) -> tuple[float, dict[str, Any] | None, dict[str, Any] | None]:
    """Idle gate that first honors graph under-recording.

    A cuda/HIP-graph trace that under-records replays reports an unreliable
    (inflated) idle%, so gating candidates on it wrongly suppresses the whole
    hot-kernel list on a workload that is actually compute-bound (the exact
    failure the bypass route already guards against via
    ``bypass_graph_under_recorded``). When under-recording is detected we skip
    the high-idle suppression and surface the graph-under-recorded warning
    instead; otherwise the plain idle gate applies unchanged.

    Returns:
        ``(threshold, high_idle_warning, graph_under_recorded_warning)`` where at
        most one of the two warnings is non-``None``.
    """
    threshold = resolve_idle_pct_threshold()
    if idle_pct is None or idle_pct <= threshold:
        return threshold, None, None
    cov = _graph_coverage_from_raw_trace(trace_path)
    if cov.get("graph_under_recorded"):
        return (
            threshold,
            None,
            _build_graph_under_recorded_warning(
                graph_launch_count=int(cov.get("graph_launch_count", 0) or 0),
                idle_pct=float(idle_pct),
            ),
        )
    _, high_idle_warning = _evaluate_high_idle_gate(idle_pct, report_path)
    return threshold, high_idle_warning, None


def _evaluate_low_compute_gate(
    compute_pct: float | None,
    exposed_comm_pct: float | None,
    report_path: Path,
) -> tuple[float, dict[str, Any] | None]:
    """Return the compute threshold plus a warning when the compute share is too low.

    Complements :func:`_evaluate_idle_gate_with_graph_guard`. The idle gate
    cannot see a window whose wall time is consumed by a spin-waiting
    collective, because that wait is charged as GPU-busy time -- such a trace
    reports ~0% idle alongside a single-digit compute share. Since a kernel
    rewrite is bounded by the compute share, both regimes warrant the same
    response: suppress the hot-kernel list and let the Coordinator route
    elsewhere.

    Args:
        compute_pct: Compute share of trace wall time, or ``None`` when the
            source report did not expose it (gate is skipped).
        exposed_comm_pct: Exposed-communication share, for warning context.
        report_path: Path to the source report, recorded in the warning.

    Returns:
        The resolved threshold and the ``low_gpu_compute_pct`` warning, or
        ``None`` when the gate does not fire.
    """
    threshold = resolve_min_compute_pct_threshold()
    if compute_pct is None or compute_pct >= threshold:
        return threshold, None
    return threshold, _build_low_compute_warning(
        compute_pct=compute_pct,
        threshold_pct=threshold,
        report_path=report_path,
        exposed_comm_pct=exposed_comm_pct,
    )


_DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO = 0.05  # 5%

# Alternate must beat the requested mode by this margin to avoid thrashing.
CHUNK_QUALITY_ALTERNATE_MARGIN = 0.10  # 10 ppt


def resolve_min_busy_ratio() -> float:
    """Return the minimum chunk busy-ratio threshold for the N36 quality gate.

    Reads ``INFERENCE_OPTIMIZER_CHUNK_QUALITY_MIN_BUSY_RATIO`` and falls back
    to :data:`_DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO` when unset, out of the
    ``[0.0, 1.0]`` range, or unparseable.

    Returns:
        float: The busy-ratio threshold in the inclusive range ``[0.0, 1.0]``.
    """
    raw = os.environ.get(
        "INFERENCE_OPTIMIZER_CHUNK_QUALITY_MIN_BUSY_RATIO",
        "",
    ).strip()
    if not raw:
        return _DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO
    try:
        v = float(raw)
        return v if 0.0 <= v <= 1.0 else _DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO
    except ValueError:
        return _DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO


def _read_busy_ratio(row: "dict[str, str] | None") -> "float | None":
    """Read ``busy_ratio`` from a splitter CSV row, falling back to
    ``gpu_busy_duration / gpu_duration`` when the column is absent."""
    if row is None:
        return None
    raw = row.get("busy_ratio", "")
    if raw not in (None, ""):
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            pass  # Unparseable; fall through to gpu_busy_duration/gpu_duration fallback.
    try:
        dur = float(row.get("gpu_duration") or "0")
        busy = float(row.get("gpu_busy_duration") or "0")
        if dur > 0:
            return max(0.0, min(1.0, busy / dur))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def _num_gpu_events(row: "dict[str, str] | None") -> int:
    """Read ``num_gpu_events`` from a splitter CSV row."""
    if row is None:
        return 0
    try:
        return int(float(row.get("num_gpu_events") or "0"))
    except (TypeError, ValueError):
        return 0


def _row_for_chunk(rows: "list[dict[str, str]]", chunk_path: "Path") -> "dict[str, str] | None":
    """Find the splitter CSV row whose ``output_path`` matches ``chunk_path``."""
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


def _check_chunk_quality(
    *,
    split_dir: "Path",
    selected_chunk: "Path",
    mode: str,
    available_modes: "dict[str, tuple[str, list[Path]]]",
) -> "dict[str, Any] | None":
    """Quality gate on the selected steady-state chunk.

    A chunk with zero GPU events is refused unconditionally.  A chunk whose
    ``busy_ratio`` is below threshold is refused when a materially better
    alternative exists.
    """
    details_path = split_dir / "execution_details.csv"
    if not details_path.is_file():
        return None
    try:
        with details_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error):
        return None

    selected_row = _row_for_chunk(rows, selected_chunk)
    if selected_row is None:
        return None
    sel_events = _num_gpu_events(selected_row)
    sel_ratio = _read_busy_ratio(selected_row)
    if sel_ratio is None and sel_events > 0:
        return None
    if sel_ratio is None:
        sel_ratio = 0.0
    threshold = resolve_min_busy_ratio()
    if sel_ratio >= threshold:
        return None

    # Below threshold: look for an alternate mode with better busy_ratio.
    # When the selected chunk has no GPU work, any alternate carrying GPU events
    # is strictly better — skip the anti-thrash margin so the retry path has a target.
    alternates: list[tuple[str, float]] = []
    for other_mode, (_label, chunks) in available_modes.items():
        if other_mode == mode or not chunks:
            continue
        other_row = _row_for_chunk(rows, chunks[0])
        oth_ratio = _read_busy_ratio(other_row)
        if oth_ratio is None:
            continue
        if sel_events == 0:
            if oth_ratio > 0.0:
                alternates.append((other_mode, oth_ratio))
        elif oth_ratio >= threshold and (oth_ratio - sel_ratio) >= CHUNK_QUALITY_ALTERNATE_MARGIN:
            alternates.append((other_mode, oth_ratio))
    if not alternates and sel_events > 0:
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
        "busy_ratio": sel_ratio,
        "threshold": threshold,
        "non_empty_modes": non_empty_modes,
        "alternate_busy_ratios": dict(alternates),
        "remediation": (
            "Re-issue roofline with env "
            "INFERENCE_OPTIMIZER_STEADY_STATE_MODE set to one of "
            f"{non_empty_modes}. The TraceLens splitter chunk for the "
            f"requested mode '{mode}' has busy_ratio={sel_ratio * 100:.2f}% "
            f"(threshold {threshold * 100:.0f}%) -- "
            f"{'zero GPU events' if sel_events == 0 else 'non-empty but substantively garbage'}. "
            "Most common cause for prefill-heavy workloads: profile window "
            "misalignment (_workload_envs.delay_iters formula only considers "
            "OSL, so high-ISL workloads land in pure-decode windows)."
        ),
        "message": (
            f"TraceLens splitter selected chunk ({mode}) busy_ratio="
            f"{sel_ratio * 100:.3f}% (events={sel_events}) -- below "
            f"the {threshold * 100:.0f}% threshold and alternate "
            f"modes have higher busy_ratio. Refusing to feed it into "
            "TraceLens analysis (would produce a misleading analysis.md "
            "with reusable_native_kernel_ids=[] and stall the "
            "optimization loop, per DSR1-0528 10k/1k case)."
        ),
    }
