"""Shared GPU work-share gates: threshold resolution + trace-health warnings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

HIGH_IDLE_PCT_THRESHOLD_DEFAULT = 80.0
HIGH_IDLE_PCT_THRESHOLD_ENV = "HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD"

LOW_COMPUTE_PCT_THRESHOLD_DEFAULT = 10.0
LOW_COMPUTE_PCT_THRESHOLD_ENV = "HYPERLOOM_TRACELENS_MIN_COMPUTE_PCT_THRESHOLD"


def _resolve_pct_threshold(env_name: str, default: float) -> float:
    """Return a percentage threshold from ``env_name``, falling back to ``default``."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value < 0.0:
        return default
    return value


def resolve_idle_pct_threshold() -> float:
    """Return the idle-percent gate threshold (default 80.0%)."""
    return _resolve_pct_threshold(
        HIGH_IDLE_PCT_THRESHOLD_ENV,
        HIGH_IDLE_PCT_THRESHOLD_DEFAULT,
    )


def resolve_min_compute_pct_threshold() -> float:
    """Return the minimum-compute-percent gate threshold (default 10.0%)."""
    return _resolve_pct_threshold(
        LOW_COMPUTE_PCT_THRESHOLD_ENV,
        LOW_COMPUTE_PCT_THRESHOLD_DEFAULT,
    )


def build_high_idle_warning(
    *,
    idle_pct: float,
    threshold_pct: float,
    report_path: Path,
) -> dict[str, Any]:
    """Build the ``trace_health_warnings[]`` entry for a high-idle trace."""
    return {
        "code": "high_gpu_idle_pct",
        "severity": "warning",
        "idle_pct": round(idle_pct, 2),
        "threshold_pct": round(threshold_pct, 2),
        "source": str(report_path),
        "message": (
            f"GPU was idle {idle_pct:.2f}% of trace wall time (threshold "
            f"{threshold_pct:.2f}%). Most of the wall time is spent outside "
            "kernel execution, so the bottleneck is scheduling/host-side and "
            "kernel-level rewriting is unlikely to improve end-to-end "
            "latency in this regime — recommend parameter optimization "
            "(batch size, KV-cache shape, prefill/decode split) over "
            "per-kernel rewrites. Hyperloom is suppressing the hot-kernel "
            "candidate list and surfacing this warning so the Coordinator "
            "can route to params/backends."
        ),
    }


def build_low_compute_warning(
    *,
    compute_pct: float,
    threshold_pct: float,
    report_path: Path,
    exposed_comm_pct: float | None = None,
) -> dict[str, Any]:
    """Build the ``trace_health_warnings[]`` entry for a low-compute-share trace."""
    entry: dict[str, Any] = {
        "code": "low_gpu_compute_pct",
        "severity": "warning",
        "compute_pct": round(compute_pct, 2),
        "threshold_pct": round(threshold_pct, 2),
        "source": str(report_path),
    }
    comm_note = ""
    if isinstance(exposed_comm_pct, (int, float)) and not isinstance(exposed_comm_pct, bool):
        entry["exposed_comm_pct"] = round(float(exposed_comm_pct), 2)
        comm_note = f" Exposed communication accounts for {float(exposed_comm_pct):.2f}% of the window."
    entry["message"] = (
        f"Only {compute_pct:.2f}% of trace wall time is compute (threshold "
        f"{threshold_pct:.2f}%).{comm_note} A kernel rewrite is bounded by the "
        "compute share, so it cannot move end-to-end latency in this regime "
        "even at infinite speedup. Note that a spin-waiting collective is "
        "charged as GPU-busy time, so this window can report near-zero idle "
        "while carrying almost no usable work — check for cross-rank arrival "
        "skew (one collective invocation absorbing the window) before reading "
        "it as a genuine communication bottleneck. Hyperloom is suppressing "
        "the hot-kernel candidate list and surfacing this warning so the "
        "Coordinator can route to comm/params instead."
    )
    return entry


def build_graph_under_recorded_warning(
    *,
    graph_launch_count: int,
    idle_pct: float | None = None,
) -> dict[str, Any]:
    """Build the ``trace_health_warnings[]`` entry for a graph under-recorded trace."""
    idle_note = f" (computed idle% {idle_pct:.2f}% is unreliable here)" if isinstance(idle_pct, (int, float)) else ""
    return {
        "code": "bypass_graph_under_recorded",
        "severity": "warning",
        "graph_launch_count": graph_launch_count,
        "message": (
            f"graph-mode trace under-recorded: only ~1 of {graph_launch_count} graph "
            f"replays captured (profiler activity-buffer overflow under continuous GPU "
            f"saturation){idle_note}; idle% is unreliable and the idle gate is skipped. "
            "Hot-kernel candidates are still ranked by recorded-kernel GPU share, which "
            "is a representative sample of one replay."
        ),
    }
