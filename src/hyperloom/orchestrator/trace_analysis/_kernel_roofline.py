# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The per-kernel roofline payload and its fields, and the framework and input shapes a candidate row carries."""

from __future__ import annotations

import os
from typing import Any

# Canonical roofline_source provenance enum (see _roofline_source for the value ladder).
from ._roofline_source import ANALYTICAL as _RL_ANALYTICAL, PLACEHOLDER as _RL_PLACEHOLDER
from ._trace_shapes import _dtypes_from_shapes, _shape_call_entries


def default_roofline_fields(candidates: list[dict[str, Any]]) -> None:
    """Give every candidate row the roofline fields it does not already carry, as unknown/None placeholders."""
    for item in candidates:
        if not isinstance(item, dict):
            continue
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
        "bottleneck": candidate.get("bottleneck"),
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
        # Contract alignment (F6): emit the shared roofline_source provenance enum
        # so both routes carry it. TraceLens rows come from its per-op perf model
        # (analytical) when that model produced numbers, else placeholder.
        "roofline_source": _RL_ANALYTICAL
        if any(
            candidate.get(k) is not None
            for k in (
                "arithmetic_intensity",
                "flops_per_byte",
                "efficiency_percent",
                "compute_utilization_pct",
                "bandwidth_utilization_pct",
            )
        )
        else _RL_PLACEHOLDER,
    }


def build_kernel_roofline_payload(
    *,
    trace_input: str,
    trace_input_type: str,
    analysis_md_path: str,
    kernel_candidates_path: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the per-kernel roofline sidecar payload.

    A view over the candidates; missing counters stay null.

    Args:
        trace_input: The trace input path recorded in the payload.
        trace_input_type: Whether ``trace_input`` is a file or capture dir.
        analysis_md_path: Path to the source analysis report.
        kernel_candidates_path: Path to the candidates JSON.
        candidates: The finalized hot-kernel candidate rows.

    Returns:
        The kernel-roofline sidecar payload dict.
    """
    rows = [_kernel_roofline_row(candidate) for candidate in candidates if isinstance(candidate, dict)]
    return {
        "schema_version": 1,
        "source": "tracelens_analysis",
        "trace_input": trace_input,
        "trace_input_type": trace_input_type,
        "analysis_md_path": analysis_md_path,
        "kernel_candidates_path": kernel_candidates_path,
        "kernels": rows,
    }


def _resolve_flydsl_source_fallback() -> str:
    """Resolve the real FlyDSL MoE kernel source for synthetic pseudo-ops.

    Used for FlyDSL pseudo-ops, looking for
    ``$DSL2_ROOT/kernels/moe_gemm_2stage.py`` and known fallback roots.

    Returns:
        The first existing FlyDSL MoE kernel source path, or ``""`` when none
        exists.
    """
    roots = [
        os.environ.get("DSL2_ROOT", "").strip(),
        os.environ.get("FLYDSL_ROOT", "").strip(),
        "/opt/FlyDSL",
        "/sgl-workspace/flydsl",
    ]
    for root in roots:
        if not root:
            continue
        cand = os.path.join(root, "kernels", "moe_gemm_2stage.py")
        if os.path.isfile(cand):
            return cand
    return ""


def add_framework_and_input_shapes(candidates: list[dict[str, Any]], framework: str) -> None:
    """Give every candidate its framework and the per-call input shapes and dtypes its readers expect.

    Mutates each candidate in place.

    Args:
        candidates (list[dict[str, Any]]): Hot-kernel candidate rows to enrich.
        framework (str): The serving framework the trace was profiled on; blank leaves the rows unnamed.
    """
    framework = framework.strip()
    for item in candidates:
        if framework:
            item.setdefault("framework", framework)
            item.setdefault("backend", framework)
        if "input_shapes" not in item:
            item["input_shapes"] = _shape_call_entries(
                item.get("shapes"),
                item.get("call_count"),
            )
            item["_input_shapes_synthetic"] = True
        item.setdefault("output_shapes", [])
        # Per-arg dtypes: TraceLens records them INLINE in each ``shapes`` string
        # ("(64,5120) bf16"); surface them as a clean, positionally-aligned list.
        # Falls back to any explicit ``dtypes`` field, then ``[]``.
        existing_dtypes = item.get("input_dtypes") or item.get("dtypes")
        if not existing_dtypes:
            existing_dtypes = _dtypes_from_shapes(item.get("shapes"))
        item["input_dtypes"] = existing_dtypes
        item.setdefault("output_dtypes", [])
