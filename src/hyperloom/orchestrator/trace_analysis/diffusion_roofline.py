#!/usr/bin/env python3
"""Workload-level roofline for diffusion/scriptable traces."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_float

UNIFIED_CSV = "unified_perf_summary.csv"
GPU_TIMELINE_CSV = "gpu_timeline.csv"

# Lift the csv field-size limit to the platform max for very long fused-kernel names.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# Column names as emitted by generate_perf_report_pytorch's unified summary.
COL_KERNEL_TIME_SUM = "Kernel Time (\u00b5s)_sum"
COL_ROOFLINE_TIME = "Roofline Time (\u00b5s)_first"
COL_OP_COUNT = "operation_count"
COL_BOUND = "Roofline Bound"
COL_CATEGORY = "op category"
COL_NAME = "name"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of dict rows (empty list when missing)."""
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def aggregate_unified(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Aggregate the unified per-kernel summary into workload totals."""
    sigma_actual_us = 0.0
    sigma_ideal_us = 0.0
    compute_us = 0.0
    memory_us = 0.0
    no_model_us = 0.0
    for r in rows:
        actual = to_float(r.get(COL_KERNEL_TIME_SUM), 0.0)
        count = to_float(r.get(COL_OP_COUNT)) or 1.0
        # Roofline Time is the per-instance ideal; scale by the aggregated count.
        ideal = to_float(r.get(COL_ROOFLINE_TIME), 0.0) * count
        sigma_actual_us += actual
        sigma_ideal_us += ideal
        bound = (r.get(COL_BOUND) or "").upper()
        if "COMPUTE" in bound:
            compute_us += actual
        elif "MEMORY" in bound:
            memory_us += actual
        else:
            no_model_us += actual
    kernel_eff = (sigma_ideal_us / sigma_actual_us) if sigma_actual_us > 0 else 0.0
    return {
        "sigma_actual_kernel_us": sigma_actual_us,
        "sigma_ideal_roofline_us": sigma_ideal_us,
        "kernel_roofline_efficiency": kernel_eff,
        "compute_bound_us": compute_us,
        "memory_bound_us": memory_us,
        "no_perf_model_us": no_model_us,
    }


def parse_gpu_timeline(rows: list[dict[str, str]]) -> dict[str, float]:
    """Extract busy / computation / exposed percentages from gpu_timeline.csv."""
    out: dict[str, float] = {}
    for r in rows:
        kind = (r.get("type") or "").strip()
        if kind:
            out[kind] = to_float(r.get("percent"), 0.0)
    return out


def top_kernels(rows: list[dict[str, str]], k: int) -> list[dict[str, Any]]:
    """Return the top-k kernels by actual kernel time for the summary block."""
    ranked = sorted(rows, key=lambda r: to_float(r.get(COL_KERNEL_TIME_SUM), 0.0), reverse=True)
    out: list[dict[str, Any]] = []
    for r in ranked[:k]:
        out.append(
            {
                "name": (r.get(COL_NAME) or "")[:48],
                "category": r.get(COL_CATEGORY) or "",
                "bound": r.get(COL_BOUND) or "",
                "kernel_time_us": to_float(r.get(COL_KERNEL_TIME_SUM), 0.0),
            }
        )
    return out


def build_report(
    csv_dir: Path,
    num_denoise_steps: int | None,
    top_k: int,
) -> dict[str, Any]:
    """Assemble the workload-level roofline report from a TraceLens CSV dir."""
    unified_path = csv_dir / UNIFIED_CSV
    unified_rows = _read_csv_rows(unified_path)
    if not unified_rows:
        raise FileNotFoundError(f"missing or empty {unified_path} (run generate_perf_report_pytorch first)")

    totals = aggregate_unified(unified_rows)
    timeline = parse_gpu_timeline(_read_csv_rows(csv_dir / GPU_TIMELINE_CSV))
    report = assemble_report(
        totals,
        timeline,
        num_denoise_steps,
        top_kernels(unified_rows, top_k),
        source="tracelens_csv",
    )
    report["source_csv_dir"] = str(csv_dir)
    return report


def assemble_report(
    totals: dict[str, Any],
    timeline: dict[str, float],
    num_denoise_steps: int | None,
    top_kernels_list: list[dict[str, Any]],
    *,
    source: str = "tracelens_csv",
) -> dict[str, Any]:
    """Assemble the workload roofline report from pre-aggregated inputs."""
    busy_pct = timeline.get("busy_time")
    gpu_busy_ratio = (busy_pct / 100.0) if busy_pct is not None else None
    kernel_eff = totals["kernel_roofline_efficiency"]
    end_to_end_eff = (kernel_eff * gpu_busy_ratio) if gpu_busy_ratio is not None else None

    report: dict[str, Any] = {
        "source": source,
        "totals": totals,
        "gpu_timeline_pct": timeline,
        "gpu_busy_ratio": gpu_busy_ratio,
        "end_to_end_efficiency_estimate": end_to_end_eff,
        "top_kernels": top_kernels_list,
    }

    if num_denoise_steps and num_denoise_steps > 0:
        report["num_denoise_steps"] = num_denoise_steps
        report["per_step"] = {
            "actual_kernel_us": totals["sigma_actual_kernel_us"] / num_denoise_steps,
            "ideal_roofline_us": totals["sigma_ideal_roofline_us"] / num_denoise_steps,
        }

    return report


def aggregate_bypass_candidates(hot_kernels: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the bypass analytical candidate set into workload totals."""
    sigma_actual = 0.0
    sigma_ideal = 0.0
    compute_us = 0.0
    memory_us = 0.0
    no_model_us = 0.0
    for c in hot_kernels:
        actual = to_float(c.get("duration_us"), 0.0)
        sigma_actual += actual
        src = str(c.get("roofline_source") or "")
        # Binding-side attainment (cross-route comparable).
        attain = to_float(c.get("roofline_attainment_pct"), 0.0)
        if src not in ("", "placeholder") and attain > 0:
            sigma_ideal += actual * (attain / 100.0)
            bound = str(c.get("bound_type") or "").upper()
            if "COMPUTE" in bound:
                compute_us += actual
            elif "MEMORY" in bound:
                memory_us += actual
            else:
                no_model_us += actual
        else:
            no_model_us += actual
    kernel_eff = (sigma_ideal / sigma_actual) if sigma_actual > 0 else 0.0
    return {
        "sigma_actual_kernel_us": sigma_actual,
        "sigma_ideal_roofline_us": sigma_ideal,
        "kernel_roofline_efficiency": kernel_eff,
        "compute_bound_us": compute_us,
        "memory_bound_us": memory_us,
        "no_perf_model_us": no_model_us,
    }


def _top_bypass_kernels(hot_kernels: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Top-k bypass candidates by GPU time for the summary block."""
    ranked = sorted(hot_kernels, key=lambda c: to_float(c.get("duration_us"), 0.0), reverse=True)
    return [
        {
            "name": (c.get("name") or "")[:48],
            "category": c.get("kernel_category") or "",
            "bound": c.get("bound_type") or "",
            "kernel_time_us": to_float(c.get("duration_us"), 0.0),
        }
        for c in ranked[:k]
    ]


@dataclass(frozen=True)
class WorkloadTotals:
    """Roofline totals aggregated over every analyzed device kernel, and how many kernels they cover."""

    totals: dict[str, Any]
    kernel_count: int


def build_report_from_bypass(
    hot_kernels: list[dict[str, Any]],
    timeline: dict[str, Any],
    num_denoise_steps: int | None,
    top_k: int,
    *,
    workload: WorkloadTotals | None = None,
) -> dict[str, Any]:
    """Build the workload roofline report from the bypass candidate set."""
    full_scope = workload is not None
    totals = workload.totals if workload is not None else aggregate_bypass_candidates(hot_kernels)
    timeline_pct: dict[str, float] = {}
    if isinstance(timeline, dict):
        if timeline.get("busy_pct") is not None:
            timeline_pct["busy_time"] = to_float(timeline.get("busy_pct"), 0.0)
        if timeline.get("idle_pct") is not None:
            timeline_pct["idle_time"] = to_float(timeline.get("idle_pct"), 0.0)
    report = assemble_report(
        totals,
        timeline_pct,
        num_denoise_steps,
        _top_bypass_kernels(hot_kernels, top_k),
        source="bypass_analytical",
    )
    # Scope + count reflect all kernels under full scope, else the top-k subset.
    report["kernel_scope"] = "all_device_kernels" if full_scope else "analyzed_candidates"
    report["kernels_aggregated"] = int(workload.kernel_count) if workload is not None else len(hot_kernels)
    return report
