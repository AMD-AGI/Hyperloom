# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Projection of a trace-analyze result envelope into the ``last_trace_analyze`` record."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom.common.timeutil import now_iso

log = logging.getLogger(__name__)

__all__ = [
    "ArtifactPaths",
    "RecoveredEnvelope",
    "artifact_paths",
    "build_trace_analyze_record",
    "next_roofline_snapshot_id",
    "recover_from_session_report",
]

# How many hot / skipped kernels ``record_trace_analyze`` keeps in the trace summary (matches the ``*_top15`` field
# names).
_TOP_N = 15

# Fields whose presence means a hot-kernel row has something the roofline view can plot.
_ROOFLINE_SIGNAL_KEYS: tuple[str, ...] = (
    "bound_type",
    "arithmetic_intensity",
    "flops_per_byte",
    "efficiency_percent",
    "compute_utilization_pct",
    "bandwidth_utilization_pct",
    "suggestion",
    "roofline_name",
    "rocprof_roofline",
)


class ArtifactPaths(NamedTuple):
    """The artifacts a trace-analyze envelope names, directly or under ``artifact_paths``."""

    candidates: Any
    kernel_roofline: Any
    steady_state_trace: Any


class RecoveredEnvelope(NamedTuple):
    """A trace-analyze envelope after the session roofline report filled in what it lost in transit."""

    result: Mapping[str, Any]
    candidates_path: Any
    kernel_roofline_path: Any


class _HotKernelSummaries(NamedTuple):
    """The three projections of a trace's hot kernels that ``last_trace_analyze`` records."""

    summary: list[dict[str, Any]]
    kernel_roofline: list[dict[str, Any]]
    reusable_native_kernel_ids: list[str]


def artifact_paths(result: Mapping[str, Any]) -> ArtifactPaths:
    """Resolve the artifacts an envelope names, its top-level keys winning over ``artifact_paths``."""
    artifacts = result.get("artifact_paths")
    if not isinstance(artifacts, dict):
        artifacts = {}
    return ArtifactPaths(
        result.get("candidates_path") or artifacts.get("kernel_candidates", "") or "",
        result.get("kernel_roofline_path") or artifacts.get("kernel_roofline", "") or "",
        result.get("steady_state_trace") or artifacts.get("tracelens_steady_state_trace") or "",
    )


def recover_from_session_report(
    result: Mapping[str, Any],
    paths: ArtifactPaths,
    disk: Mapping[str, Any] | None,
) -> RecoveredEnvelope:
    """Refill a thinned trace-analyze envelope from the session-level roofline report TraceLens wrote."""
    if not disk:
        return RecoveredEnvelope(result, paths.candidates, paths.kernel_roofline)
    kernel_roofline_path = paths.kernel_roofline or str(disk["path"])
    candidates_path = paths.candidates or str(disk["payload"].get("kernel_candidates_path") or "")
    if not result.get("trace_report_path"):
        result = dict(result)
        result["trace_report_path"] = str(disk["payload"].get("analysis_md_path") or "")
    if not (result.get("hot_kernels") or result.get("hot_kernels_top15")):
        result = dict(result)
        result["hot_kernels"] = disk["kernels"]
        log.warning(
            "record_trace_analyze: envelope carried no hot kernels; "
            "recovered %d from %s. The analysis succeeded — the result "
            "envelope lost its payload in transit.",
            len(disk["kernels"]),
            disk["path"],
        )
    return RecoveredEnvelope(result, candidates_path, kernel_roofline_path)


def next_roofline_snapshot_id(previous: Any) -> int:
    """Advance the monotonic roofline snapshot counter, restarting from 1 when the cache was cleared."""
    recorded = previous.get("roofline_snapshot_id") if isinstance(previous, dict) else None
    return recorded + 1 if isinstance(recorded, int) else 1


def _rocprof_roofline_sidecar(kernel_roofline_path: Any) -> dict[str, Any]:
    """Index the per-kernel rocprof roofline sidecar by kernel id; empty when it cannot be read."""
    if not kernel_roofline_path:
        return {}
    by_kernel_id: dict[str, Any] = {}
    try:
        payload = json.loads(Path(kernel_roofline_path).read_text(encoding="utf-8"))
        for row in payload.get("kernels") or []:
            if not isinstance(row, dict) or not row.get("kernel_id"):
                continue
            by_kernel_id[str(row["kernel_id"])] = row.get("rocprof_roofline")
    except (OSError, ValueError, AttributeError):  # sidecar merge is best-effort
        return {}
    return by_kernel_id


def _hot_kernel_row(entry: Mapping[str, Any], rocprof_by_kernel_id: Mapping[str, Any]) -> dict[str, Any]:
    """Project one hot kernel onto the key whitelist the prompt and the kernel lanes read."""
    kid = entry.get("kernel_id")
    rocprof_roofline = entry.get("rocprof_roofline")
    if rocprof_roofline is None and kid is not None:
        rocprof_roofline = rocprof_by_kernel_id.get(str(kid))
    arithmetic_intensity = entry.get("arithmetic_intensity")
    if arithmetic_intensity is None:
        arithmetic_intensity = entry.get("flops_per_byte")
    efficiency_percent = entry.get("efficiency_percent")
    if efficiency_percent is None:
        efficiency_percent = entry.get("efficiency_pct")
    return {
        "kernel_id": kid,
        "name": entry.get("name"),
        # TraceLens kernel_category bucket ("" when absent).
        "kernel_category": entry.get("kernel_category") or "",
        "gpu_pct": entry.get("gpu_pct"),
        "tracelens_pitem_rank": entry.get("tracelens_pitem_rank"),
        "impact_score": entry.get("impact_score"),
        "bottleneck": entry.get("bottleneck"),
        "bound_type": entry.get("bound_type"),
        "arithmetic_intensity": arithmetic_intensity,
        "flops_per_byte": entry.get("flops_per_byte"),
        "efficiency_percent": efficiency_percent,
        "compute_utilization_pct": entry.get("compute_utilization_pct"),
        "bandwidth_utilization_pct": entry.get("bandwidth_utilization_pct"),
        "suggestion": entry.get("suggestion") or "",
        "roofline_name": entry.get("roofline_name"),
        "rocprof_roofline": rocprof_roofline,
        "source_file": entry.get("source_file"),
        "reusable_native_kernel": bool(entry.get("reusable_native_kernel")),
        "kernel_contract": entry.get("kernel_contract"),
        "is_multigpu": entry.get("is_multigpu") is True,
        # Names the deterministic extractor a row came from, so a reader need not re-derive provenance from
        # the kernel name alone.
        "candidate_source": entry.get("candidate_source") or "",
        "recommended_backends": entry.get("recommended_backends") or [],
        "recommended_actions": entry.get("recommended_actions") or [],
        # Vendor-playbook fields (mori's dispatch+combine is the first case -- see
        # _vendor_operator_playbooks.py).
        "patch_strategy": entry.get("patch_strategy") or "",
        "vendor_playbook_group_id": entry.get("vendor_playbook_group_id") or "",
        "vendor_playbook_aggregate_gpu_pct": entry.get("vendor_playbook_aggregate_gpu_pct"),
        "vendor_playbook_min_gpu_pct_floor": entry.get("vendor_playbook_min_gpu_pct_floor"),
    }


def _carries_roofline_signal(row: Mapping[str, Any]) -> bool:
    """Whether a projected hot-kernel row has anything the roofline view can plot."""
    return any(row.get(key) not in (None, "", []) for key in _ROOFLINE_SIGNAL_KEYS)


def _summarize_hot_kernels(result: Mapping[str, Any], kernel_roofline_path: Any) -> _HotKernelSummaries:
    """Project the top-N hot kernels, merging the optional per-kernel rocprof roofline sidecar."""
    hot = result.get("hot_kernels") or []
    rocprof_by_kernel_id = _rocprof_roofline_sidecar(kernel_roofline_path)
    summaries = _HotKernelSummaries([], [], [])
    for entry in hot[:_TOP_N] if isinstance(hot, list) else []:
        if not isinstance(entry, dict):
            continue
        row = _hot_kernel_row(entry, rocprof_by_kernel_id)
        summaries.summary.append(row)
        if _carries_roofline_signal(row):
            summaries.kernel_roofline.append(dict(row))
        if row["reusable_native_kernel"] and row["kernel_id"]:
            summaries.reusable_native_kernel_ids.append(str(row["kernel_id"]))
    return summaries


def _kernel_gpu_pct(row: Mapping[str, Any]) -> float:
    """Share of GPU time a kernel row accounts for, zero when it reports none."""
    return float(row.get("gpu_pct") or 0.0)


def _summarize_skipped_kernels(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project the non-routable candidates the analysis skipped, so the LLM sees unoptimizable operators."""
    skipped = result.get("skipped_kernels") or []
    if not isinstance(skipped, list):
        return []
    ranked = sorted((e for e in skipped if isinstance(e, dict)), key=_kernel_gpu_pct, reverse=True)
    return [
        {
            "kernel_id": entry.get("kernel_id"),
            "name": entry.get("name"),
            "skip_reason": entry.get("skip_reason") or "",
            "gpu_pct": entry.get("gpu_pct"),
        }
        for entry in ranked[:_TOP_N]
    ]


def _coded_health_warnings(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep the trace-health warnings that name a code, dropping anything shapeless."""
    raw = result.get("trace_health_warnings") or []
    if not isinstance(raw, list):
        return []
    return [dict(entry) for entry in raw if isinstance(entry, dict) and entry.get("code")]


def _analysis_markdown(analysis_md_path: str) -> str:
    """Read the analysis report verbatim; the prompt path strips its base64 data-URLs."""
    if not analysis_md_path:
        return ""
    try:
        return Path(analysis_md_path).read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def build_trace_analyze_record(
    recovered: RecoveredEnvelope,
    *,
    trace_input: Any,
    steady_state_trace: Any,
    snapshot_id: int,
    baseline_gain: float,
) -> dict[str, Any]:
    """Build the ``last_trace_analyze`` record a completed trace analysis leaves behind."""
    result = recovered.result
    kernels = _summarize_hot_kernels(result, recovered.kernel_roofline_path)
    task_groups = result.get("task_groups") or []
    analysis_md_path = str(result.get("trace_report_path") or "")
    return {
        "trace_input": str(trace_input),
        "steady_state_trace": str(steady_state_trace),
        "candidates_path": str(recovered.candidates_path),
        "kernel_roofline_path": str(recovered.kernel_roofline_path),
        "hot_kernels_top15": kernels.summary,
        "kernel_roofline_top15": kernels.kernel_roofline,
        "skipped_kernels_top": _summarize_skipped_kernels(result),
        "task_groups": task_groups if isinstance(task_groups, list) else [],
        "reusable_native_kernel_ids": kernels.reusable_native_kernel_ids,
        "trace_health_warnings": _coded_health_warnings(result),
        "analysis_md_path": analysis_md_path,
        "analysis_md_text": _analysis_markdown(analysis_md_path),
        "roofline_snapshot_id": snapshot_id,
        "roofline_baseline_gain_at_snapshot": float(baseline_gain),
        "ts": now_iso(),
    }
