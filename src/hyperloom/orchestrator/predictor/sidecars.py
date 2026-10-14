# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Evidence read from the JSON files a trace analysis writes beside ``analysis.md``.

:mod:`~hyperloom.orchestrator.predictor.evidence` parses the report layout the
bypass route renders. On the TraceLens route a model writes the report in its
own layout, so the window and the operator split come from here instead: the
window from TraceLens's ``analysis.json``, the per-kernel rows from the
``summary.json`` both routes write. Values are given in the bypass report's
units, the ones the predictor was trained on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_float, to_int
from hyperloom.common.jsonio import read_json
from hyperloom.orchestrator.trace_analysis._kernel_category import canonical_category

ANALYSIS_JSON = "analysis.json"
SUMMARY_JSON = "summary.json"


def _window(analysis: dict[str, Any]) -> dict[str, Any] | None:
    """The four window timings, or ``None`` unless all are present."""
    summary = analysis.get("executive_summary")
    metrics = summary.get("metrics") if isinstance(summary, dict) else None
    metrics = metrics if isinstance(metrics, dict) else {}
    total = to_float(metrics.get("total_time_ms"))
    idle = to_float(metrics.get("idle_pct"))
    comm = to_float(metrics.get("exposed_communication_pct"))
    if total is None or idle is None or comm is None:
        return None
    return {
        "total_gpu_time_ms": round(total, 3),
        # The bypass report's busy share is everything but idle: compute plus exposed communication and copies.
        "gpu_busy_pct": round(100.0 - idle, 2),
        "gpu_idle_pct": round(idle, 2),
        "exposed_comm_pct": round(comm, 2),
    }


def _operators(tasks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """GPU share per kernel category, summed the way the bypass report's P-item tables group it."""
    category_pct: dict[str, float] = {}
    for task in tasks:
        share = to_float(task.get("gpu_pct"))
        category = canonical_category(task.get("kernel_category"))
        if share is not None and category:
            category_pct[category] = round(category_pct.get(category, 0.0) + share, 2)
    if not category_pct:
        return None
    return {
        "top_bottleneck_category": max(category_pct, key=category_pct.__getitem__),
        "attribution_pct": None,
        "category_pct": category_pct,
        "top3_cumulative_pct": round(sum(sorted(category_pct.values(), reverse=True)[:3]), 2),
    }


def _kernels(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Call count and device time keyed by kernel id and by name, the id winning as in ``load_source_sites``."""
    by_name: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for task in tasks:
        stats = {"call_count": to_int(task.get("call_count")), "time_us": to_float(task.get("duration_us"))}
        name = str(task.get("name") or "").strip()
        if name:
            by_name.setdefault(name, stats)
        kernel_id = str(task.get("kernel_id") or "").strip()
        if kernel_id:
            by_id[kernel_id] = stats
    return {**by_name, **by_id}


def load_sidecars(analysis_md_path: Any) -> dict[str, Any]:
    """``window``, ``operators`` and per-kernel ``kernels`` stats, each present only when its file yields it.

    Empty when the files are missing or unreadable: like a source location, a
    sidecar enriches the request and never gates it.
    """
    raw = str(analysis_md_path or "").strip()
    if not raw:
        return {}
    directory = Path(raw).parent
    analysis = read_json(directory / ANALYSIS_JSON, {}, require_dict=True)
    summary = read_json(directory / SUMMARY_JSON, {}, require_dict=True)
    tasks = summary.get("tasks")
    tasks = [task for task in tasks if isinstance(task, dict)] if isinstance(tasks, list) else []
    found = {"window": _window(analysis), "operators": _operators(tasks), "kernels": _kernels(tasks)}
    return {key: value for key, value in found.items() if value}
