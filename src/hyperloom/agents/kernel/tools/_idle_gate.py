"""Shared GPU-idle gate: threshold resolution + high-idle trace-health warning.

Single source of truth for the idle-percent gate applied by BOTH trace-analysis
routes (the TraceLens agent/deterministic pipeline and the standalone bypass
reader), so the threshold, gate semantics, and warning shape stay identical
regardless of which backend produced the trace analysis.

Idle metric convention (unified across routes): ``idle_pct`` is the GPU idle
fraction of the analyzed trace's wall span -- ``idle_time / total_time * 100``.
TraceLens reads it from ``gpu_timeline.csv`` (or the Executive Summary); bypass
computes ``idle_ms / total_ms * 100`` from the profiler timeline. Both express
the same quantity, so the same threshold gates them consistently.

Kept dependency-free (stdlib only) so the bypass reader can consume it without
importing or shelling out to TraceLens.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

HIGH_IDLE_PCT_THRESHOLD_DEFAULT = 80.0
HIGH_IDLE_PCT_THRESHOLD_ENV = "HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD"


def resolve_idle_pct_threshold() -> float:
    """Return the idle-percent gate threshold (default 80.0%).

    Relaxed from the docx §2 ~20% target to 80% because real SGLang inference
    traces sit structurally at ~50-60% idle (host scheduling + JIT/launch
    overhead), which the 20% gate over-suppressed. Pin via
    ``HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD``.

    Returns:
        The idle-percent gate threshold.
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


def build_high_idle_warning(
    *,
    idle_pct: float,
    threshold_pct: float,
    report_path: Path,
) -> dict[str, Any]:
    """Build the ``trace_health_warnings[]`` entry for a high-idle trace.

    Consumed by the Coordinator to route to parameter optimization instead of
    per-kernel rewriting when the GPU is mostly idle.

    Args:
        idle_pct: The measured GPU idle percentage.
        threshold_pct: The idle-gate threshold that was exceeded.
        report_path: Path to the source report, recorded in the entry.

    Returns:
        The structured ``high_gpu_idle_pct`` warning entry.
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
