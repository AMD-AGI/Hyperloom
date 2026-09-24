###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Read TraceLens ``analysis.json`` into hot-kernel candidate rows.

TraceLens owns the parse/group/prose/impact/order/cap and writes a typed
``compute_optimizations[]`` (``candidate_schema.ComputeTask``). This reader
expands each task's ``members[]`` into one candidate per member, lifts the task
scalars and TraceLens metrics verbatim (``gpu_pct := pct_e2e``), then resolves
each device symbol through TraceLens' single ``resolve_kernel_source`` and stamps
the resolution onto the row. Host-only classification (patchability, backends,
category, shapes) is applied afterwards by ``tracelens_analysis`` over these rows.

Ordering is TraceLens' already: tasks by descending summed member impact, members
within a task by descending ``impact_score``. This reader preserves that order and
does not regroup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _kernel_source import ResolveResult, resolve_source_verdict


def load_report_tasks(analysis_json: str | Path, *, framework: str = "") -> list[dict[str, Any]]:
    """Read ``analysis.json`` into ordered hot-kernel candidate rows.

    Expands every ``compute_optimizations[]`` task's ``members[]`` into one
    candidate per member, carrying the task's operation/prose/impact and the
    member's TraceLens metrics, then resolves each device symbol to its source
    via TraceLens' ``resolve_kernel_source``. The returned rows are ready
    for the host-only finalize pass; order matches TraceLens' emitted order.

    Args:
        analysis_json: Path to the ``analysis.json`` TraceLens wrote.
        framework: Optional serving framework. Accepted but not yet forwarded to
            the resolver; reserved for a future TraceLens per-call search-path filter.

    Returns:
        Candidate rows in TraceLens' task/member order. Empty when the file is
        absent, unreadable, or carries no ``compute_optimizations[]``.
    """
    path = Path(analysis_json)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(report, dict):
        return []
    tasks = report.get("compute_optimizations")
    if not isinstance(tasks, list):
        return []

    candidates: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        for member in task.get("members") or []:
            if not isinstance(member, dict):
                continue
            candidate = _member_to_candidate(task, member)
            _resolve_member_source(candidate, framework=framework)
            candidates.append(candidate)
    return candidates


def _member_to_candidate(task: dict[str, Any], member: dict[str, Any]) -> dict[str, Any]:
    """Build one candidate row from a task's scalars and a member's fields.

    ``gpu_pct := pct_e2e``: the per-kernel %E2E is TraceLens' number, read
    everywhere but never recomputed here. ``args_shapes`` and ``args_datatypes``
    are folded back into the pipeline's ``shapes`` strings as ``"(dims) dtype"``,
    the shape the downstream consumers and the cross-package GEMM extraction read
    the dtype out of; ``input_dtypes`` is derived from those strings downstream.
    """
    kernel_names = [str(n) for n in (member.get("kernel_name") or []) if str(n).strip()]
    operation = str(task.get("operation") or "").strip()
    # A graph-collapsed member can carry no Operation; its device symbol is the
    # identity, matching TraceLens' own deterministic-fallback grouping.
    name = operation or (kernel_names[0] if kernel_names else "")

    pct_e2e = _as_float(member.get("pct_e2e"))
    time_ms = _as_float(member.get("time_ms"))
    impact = task.get("impact") if isinstance(task.get("impact"), dict) else {}

    candidate: dict[str, Any] = {
        "name": name,
        "duration_us": (time_ms or 0.0) * 1000.0,
        "call_count": _as_int(member.get("count")),
        "gpu_pct": pct_e2e,
        "device_kernel_name": kernel_names[0] if kernel_names else "",
        "device_kernel_names": kernel_names,
        "shapes": _shapes_with_dtypes(member.get("args_shapes"), member.get("args_datatypes")),
        "library": str(member.get("library") or ""),
        "tracelens_category": str(member.get("category") or ""),
        "bound_type": str(member.get("bound") or ""),
        "flops_per_byte": _as_float(member.get("flops_per_byte")) or 0.0,
        "efficiency_percent": _as_float(member.get("efficiency_percent")) or 0.0,
        "efficiency_peak_value": _as_float(member.get("efficiency_peak_value")) or 0.0,
        "efficiency_peak_unit": str(member.get("efficiency_peak_unit") or ""),
        "kernel_launcher_path": str(member.get("kernel_launcher_path") or ""),
        "impact_score": _as_float(member.get("impact_score")) or 0.0,
        "tracelens_pitem_rank": _as_int(task.get("priority")),
        "tracelens_pitem_title": operation,
        "identification": str(task.get("identification") or ""),
        "reasoning_for_slowdown": str(task.get("reasoning") or ""),
        "resolution": str(task.get("resolution") or ""),
        "impact_low_e2e_pct": _as_float(impact.get("low")),
        "impact_high_e2e_pct": _as_float(impact.get("high")),
    }
    if candidate["shapes"]:
        candidate["shape_provenance"] = "torch_trace"
    return candidate


def _resolve_member_source(candidate: dict[str, Any], *, framework: str = "") -> None:
    """Resolve the candidate's primary device symbol through TraceLens.

    One ``resolve_kernel_source`` call per symbol; the returned ``ResolveResult``
    fields are read straight onto the candidate. No fallback: an unresolvable
    symbol comes back with ``method="unresolved"`` and an empty ``source_file``,
    which the host-only patchability gate then drops to ``skipped_kernels``.
    """
    symbol = str(candidate.get("device_kernel_name") or candidate.get("name") or "")
    result: ResolveResult = resolve_source_verdict(
        symbol,
        kernel_file=str(candidate.get("kernel_launcher_path") or ""),
        op_name=str(candidate.get("name") or ""),
        library=str(candidate.get("library") or ""),
    )
    location = result.location
    candidate["source_file"] = location.source_file if location else ""
    candidate["source_line"] = location.line if location else None
    candidate["op_to_source_patchable"] = result.patchable
    candidate["op_to_source_reason"] = result.reason
    candidate["op_to_source_status"] = "resolved" if result.patchable else "non_rewritable"
    candidate["kernel_kind"] = result.kind
    candidate["source_resolution_method"] = result.method
    candidate["source_resolution_reason"] = result.reason


def _shapes_with_dtypes(shapes: Any, dtypes: Any) -> list[str]:
    """Fold ``args_shapes`` and ``args_datatypes`` into ``"(dims) dtype"`` strings.

    TraceLens splits the operand dims and dtype into two parallel lists; the HL
    pipeline reads the dtype back out of the shape string, so they are rejoined
    here in the ``"(640,4096) bf16"`` form the shape consumers expect.
    """
    shape_list = list(shapes) if isinstance(shapes, list) else []
    dtype_list = list(dtypes) if isinstance(dtypes, list) else []
    out: list[str] = []
    for i, shape in enumerate(shape_list):
        dtype = str(dtype_list[i]).strip() if i < len(dtype_list) and dtype_list[i] is not None else ""
        text = str(shape).strip()
        out.append(f"{text} {dtype}".strip() if dtype else text)
    return out


def _as_float(value: Any) -> float | None:
    """Coerce a JSON scalar to ``float``, or ``None`` when it is not numeric."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    """Coerce a JSON scalar to a non-negative ``int`` count, defaulting to 0."""
    number = _as_float(value)
    return int(number) if number is not None else 0
