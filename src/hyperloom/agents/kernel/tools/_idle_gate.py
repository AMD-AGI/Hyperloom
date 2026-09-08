"""Shared GPU work-share gates: threshold resolution + trace-health warnings.

Single source of truth for the trace-health gates applied by BOTH trace-analysis
routes (the TraceLens agent pipeline and the standalone bypass reader), so the
thresholds, gate semantics, and warning shapes stay identical regardless of
which backend produced the trace analysis.

Two complementary gates live here, both answering "can rewriting a kernel move
end-to-end latency on this trace?":

* **High idle** -- ``idle_pct`` is the GPU idle fraction of the analyzed trace's
  wall span (``idle_time / total_time * 100``). TraceLens reads it from
  ``gpu_timeline.csv`` (or the Executive Summary); bypass computes
  ``idle_ms / total_ms * 100`` from the profiler timeline.
* **Low compute** -- ``compute_pct`` is the fraction of that same span spent in
  compute kernels. Idle alone cannot catch a window dominated by *exposed
  communication*, because a spin-waiting collective (e.g. a custom all-reduce
  that polls peer-rank signals from inside the kernel) is charged as GPU-busy
  time. Such a window reports a near-zero idle share while carrying almost no
  usable work, so it slips through the idle gate. Gating on the compute share
  restores the original intent for both regimes.

The high-idle gate runs on both routes. The low-compute gate currently runs on
the TraceLens route only, because the bypass reader does not model exposed
communication (``_bypass_report`` reports ``exposed_comm_pct: None``) and so
cannot separate compute from collective time. Wiring bypass in requires teaching
it to classify collectives first; until then it keeps the idle gate alone rather
than gating on a compute share it cannot measure.

Kept dependency-free (stdlib only) so the bypass reader can consume it without
importing or shelling out to TraceLens.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

HIGH_IDLE_PCT_THRESHOLD_DEFAULT = 80.0
HIGH_IDLE_PCT_THRESHOLD_ENV = "HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD"

LOW_COMPUTE_PCT_THRESHOLD_DEFAULT = 10.0
LOW_COMPUTE_PCT_THRESHOLD_ENV = "HYPERLOOM_TRACELENS_MIN_COMPUTE_PCT_THRESHOLD"


def _resolve_pct_threshold(env_name: str, default: float) -> float:
    """Return a percentage threshold from ``env_name``, falling back to ``default``.

    Args:
        env_name: Environment variable holding the override.
        default: Value used when the override is unset, unparseable, or negative.

    Returns:
        The resolved percentage threshold.
    """
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
    """Return the idle-percent gate threshold (default 80.0%).

    Pin via ``HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD``.

    Returns:
        The idle-percent gate threshold.
    """
    return _resolve_pct_threshold(
        HIGH_IDLE_PCT_THRESHOLD_ENV,
        HIGH_IDLE_PCT_THRESHOLD_DEFAULT,
    )


def resolve_min_compute_pct_threshold() -> float:
    """Return the minimum-compute-percent gate threshold (default 10.0%).

    Deliberately well below the ``100 - HIGH_IDLE_PCT_THRESHOLD_DEFAULT`` (20%)
    that would make this gate the exact mirror of the idle gate, so it only
    fires on unambiguously degenerate windows. Pin via
    ``HYPERLOOM_TRACELENS_MIN_COMPUTE_PCT_THRESHOLD``.

    Returns:
        The minimum-compute-percent gate threshold.
    """
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
    """Build the ``trace_health_warnings[]`` entry for a low-compute-share trace.

    Consumed by the Coordinator the same way ``high_gpu_idle_pct`` is: the
    hot-kernel list is suppressed and the session is routed to comm/parameter
    optimization instead of per-kernel rewriting.

    Args:
        compute_pct: The measured compute share of trace wall time.
        threshold_pct: The minimum-compute threshold that was not met.
        report_path: Path to the source report, recorded in the entry.
        exposed_comm_pct: Exposed-communication share, when known, so the
            Coordinator can tell a comm-bound window from a host-bound one.

    Returns:
        The structured ``low_gpu_compute_pct`` warning entry.
    """
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
    """Build the ``trace_health_warnings[]`` entry for a graph under-recorded trace.

    Under continuous CUDA/HIP graph replay the profiler activity buffer overflows
    and captures only ~1 of ``graph_launch_count`` replays, so idle% is unreliable
    and must not gate candidates; ranking by recorded-kernel GPU share stays valid.

    Args:
        graph_launch_count: Number of graph-launch runtime events in the trace.
        idle_pct: The (unreliable) measured GPU idle percentage, for context.

    Returns:
        The structured ``bypass_graph_under_recorded`` warning entry.
    """
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
