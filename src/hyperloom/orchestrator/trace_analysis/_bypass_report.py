###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Build downstream artifacts (candidates / summary / kernel_roofline) for the bypass analysis backend from the classified device-kernel aggregates."""

from __future__ import annotations

import csv
import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from ._bypass_classify import classify_kernel
from ._bypass_fusion import analyze_fusion
from ._analysis_md import AnalysisReport, render_report
from ._bypass_roofline import KernelAggregate, compute_roofline
from ._kernel_category import canonical_category
from ._kernel_partition import build_kernel_candidates_document

# Source path mapping delegates to TraceLens' independent kernel_source
# path-identifier; this path uses only TraceLens' path-mapping features,
# not its analysis layer.
from ._kernel_source import resolve_source_verdict
from ._idle_gate import resolve_idle_pct_threshold
from ._roofline_source import PLACEHOLDER as _RL_PLACEHOLDER
from ._task_group_contract import (
    build_operator_identity,
    build_task_group_shape_cases,
    legacy_operator_identity_keys,
    operator_identity_key,
)
from hyperloom.common.kernel_shape_contract import (
    DISPATCHABLE_SHAPE_PROVENANCE as _DISPATCHABLE_SHAPE_PROVENANCE,
)

# Category-appropriate optimization guidance (structured, not LLM prose).
_ACTION_BY_CATEGORY: dict[str, str] = {
    "SDPA": "Profile the attention kernel for tile/occupancy; consider a fused/flash "
    "attention backend; increase decode batch to amortize KV reads.",
    "GEMM": "Tune GEMM tile size / precision and fuse the epilogue where possible; "
    "vendor-library GEMMs (Tensile/rocBLAS) are not rewritable — tune via library config.",
    "Quantization": "Fuse quantization into the adjacent GEMM epilogue and drop redundant per-tensor scaling passes.",
    "KVCacheStore": "Fuse the KV-cache write into attention to remove the separate reshape pass.",
    "Normalization": "Use a fused RMSNorm/LayerNorm and fold the residual/quant into the norm kernel.",
    "Convolution": "Pick an NHWC/implicit-GEMM conv algorithm for the shape; fuse "
    "bias/activation into the conv epilogue (VAE encode/decode).",
    "Elementwise": "Fuse elementwise chains to cut intermediate memory traffic.",
    "MoE": "Optimize expert GEMM and routing; fuse gate/up projections.",
    "MemCpy": "Reduce host/device copies; keep tensors resident on device.",
    "Others": "Profile the kernel for tile size and wave occupancy.",
}

_UNKNOWN_BOUND = "\u2014"  # em dash "unknown bound" marker.
_REUSABLE_BACKENDS = ["forge", "geak"]

# Bound-type display prefixes for the (deterministic) per-kernel suggestion.
_BOUND_PREFIX: dict[str, str] = {"compute_bound": "Compute-bound", "memory_bound": "Memory-bound"}


def _build_suggestion(category: str, bound_type: str) -> str:
    """Deterministic optimization hint from ``category`` + ``bound_type``."""
    action = _ACTION_BY_CATEGORY.get(category, _ACTION_BY_CATEGORY["Others"])
    prefix = _BOUND_PREFIX.get(bound_type, "")
    return f"{prefix}: {action}" if prefix else action


# torch ``Input type`` token -> compact dtype suffix for the shape-string contract (e.g. ``(15360,2048) bf16``);
# unmapped/empty types emit a bare shape.
_DTYPE_SUFFIX: dict[str, str] = {
    # Suffixes MUST match the shared harness dtype_map + roofline peak table; a compact "f16"/"f32" makes the harness
    # emit an invalid ``torch.f16``.
    "c10::bfloat16": "bf16",
    "bfloat16": "bf16",
    "c10::half": "fp16",
    "half": "fp16",
    "float16": "fp16",
    "float": "fp32",
    "float32": "fp32",
    "double": "fp64",
    "float64": "fp64",
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


def _format_operand_shape(dims: Any, dtype: Any) -> str | None:
    """Render one operand as a ``(d0,d1,...) <dtype>`` string (or ``None``)."""
    if not isinstance(dims, (list, tuple)) or not dims:
        return None
    try:
        body = ",".join(str(int(d)) for d in dims)
    except (TypeError, ValueError):
        return None
    shape = f"({body},)" if len(dims) == 1 else f"({body})"
    suffix = _DTYPE_SUFFIX.get(str(dtype or "").strip().lower())
    return f"{shape} {suffix}" if suffix else shape


def _trace_shape_entries(op_shapes: Any, op_dtypes: Any, call_count: int) -> list[dict[str, Any]]:
    """Build the downstream ``input_shapes`` contract from Kineto Input Dims/type."""
    dtypes = op_dtypes if isinstance(op_dtypes, (list, tuple)) else []
    operands: list[str] = []
    for i, dims in enumerate(op_shapes if isinstance(op_shapes, (list, tuple)) else []):
        rendered = _format_operand_shape(dims, dtypes[i] if i < len(dtypes) else "")
        if rendered:
            operands.append(rendered)
    if not operands:
        return []
    return [{"call_num": int(call_count) if call_count else 1, "shape": "<br>".join(operands)}]


# Triton autotune names embed the tile shape, e.g.
# ``_gemm_a16_w16_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_...``.
_TILE_NAME_RE = re.compile(r"BLOCK_SIZE_([A-Za-z]+)_(\d+)")


def _launch_grid_shape_entries(grid: Any, block: Any, call_count: int) -> list[dict[str, Any]]:
    """Build a shape entry from a kernel's launch geometry (grid/block)."""

    def _dims(v: Any) -> list[int]:
        out: list[int] = []
        for d in v if isinstance(v, (list, tuple)) else []:
            try:
                n = int(d)
            except (TypeError, ValueError):
                return []
            out.append(n)
        return out

    g = _dims(grid)
    b = _dims(block)
    parts: list[str] = []
    if any(x > 0 for x in g):
        parts.append("grid=(" + ",".join(str(x) for x in g) + ")")
    if any(x > 0 for x in b):
        parts.append("block=(" + ",".join(str(x) for x in b) + ")")
    if not parts:
        return []
    return [{"call_num": int(call_count) if call_count else 1, "shape": "<br>".join(parts)}]


def _tile_name_shape_entries(kernel_name: str, call_count: int) -> list[dict[str, Any]]:
    """Extract a tile shape from a Triton autotune kernel name (``BLOCK_SIZE_*``)."""
    matches = _TILE_NAME_RE.findall(kernel_name or "")
    if not matches:
        return []
    body = "<br>".join(f"{axis}{val}" for axis, val in matches)
    return [{"call_num": int(call_count) if call_count else 1, "shape": body}]


def _shape_fields(k: dict[str, Any]) -> dict[str, Any]:
    """A kernel row's candidate shape fields, from the first evidence that yields a shape: its own cpu_op dims, the
    unambiguous same-name capture-time backfill, its launch geometry, then the tile in its Triton autotune name."""
    count = k.get("count")
    dtypes = k.get("op_dtypes", [])
    entries = _trace_shape_entries(k.get("op_shapes"), dtypes, count)
    provenance = "torch_trace"
    if not entries and not k.get("backfill_ambiguous"):
        backfill_dtypes = k.get("backfill_dtypes", [])
        entries = _trace_shape_entries(k.get("backfill_shapes"), backfill_dtypes, count)
        if entries:
            dtypes, provenance = backfill_dtypes, "capture_backfill"
    if not entries:
        entries = _launch_grid_shape_entries(k.get("launch_grid"), k.get("launch_block"), count)
        provenance = "launch_grid"
    if not entries:
        entries = _tile_name_shape_entries(k.get("name", ""), count)
        provenance = "tile_name"
    if not entries:
        provenance = "unresolved"
    return {
        "shapes": entries,
        "input_shapes": entries,
        "input_dtypes": dtypes,
        "shape_provenance": provenance,
        # launch_grid / tile_name shapes are geometry, so the kernel-opt gate rejects them.
        "shape_dispatchable": provenance in _DISPATCHABLE_SHAPE_PROVENANCE,
    }


def _source_type_for_op(op_name: str) -> str:
    """Best-effort source-type guess from a launching op name."""
    n = (op_name or "").lower()
    if not n:
        return "unknown"
    if n.startswith(("aten::", "vllm::", "vllm_aiter::", "_c_cache_ops::", "_rocm_c::")):
        return "python"
    if "aiter" in n or "ck" in n:
        return "hip_cpp"
    return "unknown"


# Native device-code extensions.
_NATIVE_SOURCE_EXTS = (
    ".cu",
    ".cuh",
    ".hip",
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    ".hh",
    ".h",
    ".c",
)


def _build_task_groups(hot_kernels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group routable candidates that share an editable source into task groups."""
    buckets: dict[str, dict[str, Any]] = {}
    for c in hot_kernels:
        src = c["source_file"].strip()
        if not c["reusable_native_kernel"] or not src:
            continue
        operation = c["name"]
        identity_args = {
            "source_kind": "native" if src.lower().endswith(_NATIVE_SOURCE_EXTS) else "py",
            "source_path": src,
            "operation": operation,
        }
        identity = build_operator_identity(**identity_args, function_name=operation)
        key = operator_identity_key(**identity_args, function_name=operation)
        legacy_keys = legacy_operator_identity_keys(
            **identity_args, function_name=c.get("device_kernel_name") or operation
        )
        shapes = c.get("input_shapes", [])
        row = {
            "kernel_id": c["kernel_id"],
            "name": operation,
            "device_kernel_name": c.get("device_kernel_name", ""),
            "shapes": shapes,
            "input_shapes": shapes,
            "input_dtypes": c.get("input_dtypes", []),
            "output_shapes": [],
            "output_dtypes": [],
            "raw_arg_spec": {},
            "call_count": c["call_count"],
            "duration_us": c["duration_us"],
            "percent_of_total": c.get("percent_of_total", 0.0),
            "gpu_pct": c["gpu_pct"],
            "bound_type": c.get("bound_type", ""),
        }
        bucket = buckets.setdefault(
            key,
            {
                "task_group_id": "",
                "task_group_key": key,
                "operator_identity": identity,
                "identity_route": "bypass",
                "legacy_task_group_keys": [],
                "operation": operation,
                "operation_key": identity["operation"],
                "source_path": identity["source_path"],
                "kernel_ids": [],
                "primary_kernel_id": "",
                "rows": [],
                "aggregate_duration_us": 0.0,
                "aggregate_call_count": 0,
                "aggregate_gpu_pct": 0.0,
                "source": "bypass",
            },
        )
        bucket["legacy_task_group_keys"] = list(dict.fromkeys([*bucket["legacy_task_group_keys"], *legacy_keys]))
        bucket["kernel_ids"].append(row["kernel_id"])
        bucket["rows"].append(row)
        bucket["aggregate_duration_us"] += row["duration_us"]
        bucket["aggregate_call_count"] += row["call_count"]
        bucket["aggregate_gpu_pct"] += row["gpu_pct"]

    ordered = sorted(buckets.values(), key=lambda g: g["aggregate_duration_us"], reverse=True)
    for idx, group in enumerate(ordered, start=1):
        group["task_group_id"] = f"tg{idx:03d}"
        group["rows"].sort(key=lambda r: r["duration_us"], reverse=True)
        group["primary_kernel_id"] = group["rows"][0]["kernel_id"]
        group["aggregate_duration_us"] = round(group["aggregate_duration_us"], 3)
        group["aggregate_gpu_pct"] = round(group["aggregate_gpu_pct"], 3)
        group["shape_cases"] = build_task_group_shape_cases(group)
    return ordered


def _source_type_from_path(path: str) -> str:
    """Derive source type from a resolved source file's extension."""
    low = (path or "").lower()
    if low.endswith(_NATIVE_SOURCE_EXTS):
        return "hip_cpp"
    if low.endswith(".py"):
        return "python"
    return ""


def _implementation_contract(
    *,
    op_name: str,
    source_file: str,
    trace_kernel_kind: str,
) -> tuple[str, str]:
    """Derive source type and kind only from explicit trace/source evidence."""
    traced_kind = str(trace_kernel_kind or "").strip().lower().replace("-", "_")
    normalized_op = str(op_name or "").strip().lower()
    if "flydsl" in traced_kind or (normalized_op.startswith("pseudo_op::") and "flydsl" in normalized_op):
        return "flydsl", "flydsl"
    if "triton" in traced_kind:
        return "triton", "triton"
    source_type = _source_type_from_path(source_file) or _source_type_for_op(op_name)
    if source_type in {"triton", "flydsl", "ck"}:
        return source_type, source_type
    return source_type, ""


def _short_name(kernel_name: str) -> str:
    """Shorten a mangled device-kernel name for candidate identity display."""
    n = kernel_name or ""
    # Strip C++ template/mangling tails.
    n = re.sub(r"<.*$", "", n)
    n = n.strip()
    return n[:80] if n else "unknown_kernel"


def _candidate(idx: int, k: dict[str, Any], *, framework: str, target_platform: str) -> dict[str, Any]:
    """One top device kernel's candidate row: classification, source verdict, shape evidence, roofline and ROI."""
    kname, op_name = k.get("name", ""), k.get("op_name", "")
    kc = classify_kernel(kname, op_name=op_name)
    # Native HIP-graph kernels arrive with op_name=""; op_to_source_patchable, not source presence, gates routing.
    resolution = resolve_source_verdict(kname, kernel_file=k.get("op_kernel_file", ""), op_name=op_name)
    location = resolution.location
    source_file = location.source_file if location else ""
    shape = _shape_fields(k)
    if not kc.reusable:
        skip_reason = kc.skip_reason
    elif not source_file:
        skip_reason = "source file not resolved"
    elif not resolution.patchable:
        skip_reason = f"non-patchable: {resolution.reason}" if resolution.reason else "non-patchable kernel"
    elif not shape["shape_dispatchable"]:
        skip_reason = f"shape not dispatchable (provenance={shape['shape_provenance']}); need operand dims"
    else:
        skip_reason = ""
    source_type, kernel_kind = _implementation_contract(
        op_name=op_name, source_file=source_file, trace_kernel_kind=k.get("op_kernel_backend", "")
    )
    cand: dict[str, Any] = {
        "kernel_id": f"k{idx:03d}",
        "name": op_name or _short_name(kname),
        "kernel_category": kc.category,
        "device_kernel_name": kname[:120],
        "device_kernel_names": [kname[:120]],
        "duration_us": k.get("gpu_time_us", 0.0),
        "gpu_pct": k.get("gpu_pct", 0.0),
        "percent_of_total": k.get("gpu_pct", 0.0),
        "call_count": k.get("count", 0),
        "bound_type": _UNKNOWN_BOUND,
        "efficiency_percent": 0.0,
        "roofline_attainment_pct": None,
        "flops_per_byte": None,
        "arithmetic_intensity": None,
        "compute_utilization_pct": None,
        "bandwidth_utilization_pct": None,
        # Placeholder roofline: bound_type/AI/util above are structural defaults, NOT measured.
        "roofline_source": _RL_PLACEHOLDER,
        "library": "",
        "backend": framework,
        "framework": framework,
        "source_file": source_file,
        # None for native kernels resolved by symbol index.
        "source_line": location.line if location else None,
        "source_resolution_method": resolution.method,
        "source_type": source_type,
        "kernel_kind": kernel_kind,
        "reusable_native_kernel": kc.reusable,
        "skip_reason": skip_reason,
        "identity_route": "bypass",
        "op_to_source_patchable": resolution.patchable,
        "op_to_source_reason": resolution.reason,
        "op_to_source_status": "resolved" if resolution.patchable else "non_rewritable",
        "recommended_backends": list(_REUSABLE_BACKENDS) if kc.reusable else [],
        "benchmark_files": [],
        "kernel_repo": "",
        **shape,
    }
    if shape["shape_dispatchable"]:
        # Analytical roofline from the operand shape and measured time.
        aggregate = KernelAggregate(kc.category, shape["shapes"][0]["shape"], cand["duration_us"], cand["call_count"])
        cand.update(compute_roofline(aggregate, gpu_type=target_platform) or {})
    # Optimization ROI = GPU-time share x headroom (1 - efficiency); with no analytical efficiency it is gpu_pct.
    cand["optimization_priority"] = round(cand["gpu_pct"] * (1.0 - cand["efficiency_percent"] / 100.0), 4)
    # Deterministic per-kernel hint for the specialist prompt's action slot.
    cand["suggestion"] = _build_suggestion(kc.category, cand["bound_type"])
    cand["recommended_actions"] = [cand["suggestion"]]
    return cand


def build_candidates(
    analyze_out: dict[str, Any],
    *,
    framework: str,
    target_platform: str,
    top_k: int = 15,
) -> dict[str, Any]:
    """Turn classified top device kernels into the candidate payload."""
    hot_kernels = [
        _candidate(idx, k, framework=framework, target_platform=target_platform)
        for idx, k in enumerate((analyze_out.get("kernels") or [])[:top_k], start=1)
    ]

    # Group repeated shapes of one operator and editable source into one task; stamp that shared contract onto each
    # member.
    task_groups = _build_task_groups(hot_kernels)
    kid_to_group = {kid: g for g in task_groups for kid in g["kernel_ids"]}
    for c in hot_kernels:
        g = kid_to_group.get(c["kernel_id"])
        if g is not None:
            c["task_group"] = g

    # 1-based rank by optimization ROI, stamped WITHOUT reordering hot_kernels (that list stays gpu_pct-sorted).
    for rank, c in enumerate(sorted(hot_kernels, key=lambda x: x["optimization_priority"], reverse=True), start=1):
        c["priority_rank"] = rank

    # A candidate routes to kernel-opt exactly when nothing gave it a skip reason: reusable, a source TraceLens
    # confirms patchable, and a dispatch-grade operand shape. The four-key document shape and the split are owned by
    # the shared builder both routes call.
    header = {
        "source": "bypass",
        "framework": framework,
        "target_platform": target_platform,
        "aggregation_scope": analyze_out.get("aggregation_scope", "full_trace"),
    }
    return build_kernel_candidates_document(
        header, hot_kernels, task_groups, is_routable=lambda c: not c["skip_reason"]
    )


def build_summary(
    candidates: dict[str, Any],
    *,
    framework: str,
    target_platform: str,
    generated_at: str,
    trace_health_warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the routed-vs-skipped audit ``summary.json`` payload."""

    # ``tasks`` / ``skipped`` reuse the SAME split build_candidates computed so summary.json never disagrees with
    # kernel_candidates.json.
    def _audit_row(c: dict[str, Any]) -> dict[str, Any]:
        return {
            "kernel_id": c["kernel_id"],
            "name": c["name"],
            "kernel_category": c["kernel_category"],
            "duration_us": c["duration_us"],
            "gpu_pct": c["gpu_pct"],
            "call_count": c["call_count"],
            "source_type": c["source_type"],
            "reusable_native_kernel": c["reusable_native_kernel"],
        }

    tasks = []
    for c in candidates.get("routable_kernels") or []:
        row = _audit_row(c)
        row["recommended_backends"] = c["recommended_backends"]
        tasks.append(row)
    skipped = []
    for c in candidates.get("skipped_kernels") or []:
        row = _audit_row(c)
        row["skip_reason"] = c["skip_reason"]
        skipped.append(row)
    # Reusable kernels that resolved a source but were held out of ``tasks`` because their shape is geometry-only
    # (launch_grid/tile_name).
    geometry_only_skipped_count = sum(
        1
        for c in candidates.get("skipped_kernels") or []
        if c.get("reusable_native_kernel") and c.get("source_file") and not c.get("shape_dispatchable")
    )
    # Compact task-group projection for the audit view.
    group_entries = [
        {
            "task_group_id": g.get("task_group_id", ""),
            "operation": g.get("operation", ""),
            "source_path": g.get("source_path", ""),
            "primary_kernel_id": g.get("primary_kernel_id", ""),
            "kernel_ids": g.get("kernel_ids", []),
            "row_count": len(g.get("rows") or []),
            "aggregate_duration_us": g.get("aggregate_duration_us", 0.0),
            "aggregate_gpu_pct": g.get("aggregate_gpu_pct", 0.0),
        }
        for g in (candidates.get("task_groups") or [])
    ]
    return {
        "source": "bypass",
        "framework": framework,
        "target_platform": target_platform,
        "generated_at": generated_at,
        "aggregation_scope": candidates.get("aggregation_scope", "full_trace"),
        "tasks": tasks,
        "skipped": skipped,
        "task_groups": group_entries,
        "task_count": len(tasks),
        # ``tasks`` already excludes geometry-only kernels, so dispatchable_count == task_count; kept explicit so the
        # audit view states dispatchability.
        "dispatchable_count": len(tasks),
        "geometry_only_skipped_count": geometry_only_skipped_count,
        "skipped_count": len(skipped),
        "task_group_count": len(group_entries),
        "trace_health_warnings": list(trace_health_warnings or []),
    }


def _category_rollup(analyze_out: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate GPU time by category over *all* device kernels."""
    kernels = analyze_out.get("kernels") or []
    total_us = sum(float(k.get("gpu_time_us") or 0.0) for k in kernels) or 1.0
    cat_us: dict[str, float] = defaultdict(float)
    cat_cnt: dict[str, int] = defaultdict(int)
    for k in kernels:
        kc = classify_kernel(k.get("name", "") or "", op_name=k.get("op_name", "") or "")
        us = float(k.get("gpu_time_us") or 0.0)
        cat_us[kc.category] += us
        cat_cnt[kc.category] += int(k.get("count") or 0)
    rows = [
        {
            "category": cat,
            "gpu_ms": round(us / 1000.0, 3),
            "gpu_pct": round(us / total_us * 100.0, 2),
            "kernel_count": cat_cnt[cat],
        }
        for cat, us in cat_us.items()
    ]
    rows.sort(key=lambda r: r["gpu_ms"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Structured CSV export (deterministic, code-generated).
# ---------------------------------------------------------------------------
#: Stable column order for the per-kernel metrics CSV.
_METRICS_COLUMNS: list[str] = [
    "priority_rank",
    "optimization_priority",
    "kernel_id",
    "name",
    "kernel_category",
    "device_kernel_name",
    "duration_us",
    "gpu_pct",
    "call_count",
    "bound_type",
    "arithmetic_intensity",
    "efficiency_percent",
    "compute_utilization_pct",
    "bandwidth_utilization_pct",
    "roofline_source",
    "reusable_native_kernel",
    "source_file",
    "source_type",
    "recommended_backends",
    "benchmark_files_count",
    "skip_reason",
    "representative_shape",
    "input_dtypes",
    "shape_provenance",
    "suggestion",
]

_SUMMARY_COLUMNS: list[str] = [
    "kernel_category",
    "kernel_count",
    "total_gpu_pct",
    "total_duration_us",
    "mean_efficiency_percent",
    "dominant_bound_type",
    "routable_count",
]


def _join_list(value: Any) -> str:
    """Render a list cell as a ``;``-joined string (CSV-cell friendly)."""
    return ";".join(str(x) for x in value) if isinstance(value, list) else ""


def build_metrics_rows(candidates: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ALL hot kernels (routable + skipped) into per-kernel metric rows."""
    rows: list[dict[str, Any]] = []
    for c in candidates.get("hot_kernels") or []:
        shapes = c.get("shapes") or []
        rep = shapes[0].get("shape", "") if shapes and isinstance(shapes[0], dict) else ""
        rows.append(
            {
                "priority_rank": c.get("priority_rank", ""),
                "optimization_priority": c.get("optimization_priority", ""),
                "kernel_id": c.get("kernel_id", ""),
                "name": c.get("name", ""),
                "kernel_category": c.get("kernel_category", ""),
                "device_kernel_name": c.get("device_kernel_name", ""),
                "duration_us": c.get("duration_us", ""),
                "gpu_pct": c.get("gpu_pct", ""),
                "call_count": c.get("call_count", ""),
                "bound_type": c.get("bound_type", ""),
                "arithmetic_intensity": c.get("arithmetic_intensity", ""),
                "efficiency_percent": c.get("efficiency_percent", ""),
                "compute_utilization_pct": c.get("compute_utilization_pct", ""),
                "bandwidth_utilization_pct": c.get("bandwidth_utilization_pct", ""),
                "roofline_source": c.get("roofline_source", ""),
                "reusable_native_kernel": c.get("reusable_native_kernel", ""),
                "source_file": c.get("source_file", ""),
                "source_type": c.get("source_type", ""),
                "recommended_backends": _join_list(c.get("recommended_backends")),
                "benchmark_files_count": len(c.get("benchmark_files") or []),
                "skip_reason": c.get("skip_reason", ""),
                "representative_shape": rep,
                "input_dtypes": _join_list(c.get("input_dtypes")),
                "shape_provenance": c.get("shape_provenance", ""),
                "suggestion": c.get("suggestion", ""),
            }
        )
    return rows


def build_category_summary(candidates: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate hot kernels by category (the CSV 'summary' view)."""
    agg: dict[str, dict[str, Any]] = {}
    for c in candidates.get("hot_kernels") or []:
        cat = c.get("kernel_category") or "Others"
        a = agg.setdefault(
            cat,
            {
                "kernel_count": 0,
                "total_gpu_pct": 0.0,
                "total_duration_us": 0.0,
                "eff_sum": 0.0,
                "eff_n": 0,
                "bounds": Counter(),
                "routable": 0,
            },
        )
        a["kernel_count"] += 1
        a["total_gpu_pct"] += float(c.get("gpu_pct") or 0.0)
        a["total_duration_us"] += float(c.get("duration_us") or 0.0)
        eff = c.get("efficiency_percent")
        if isinstance(eff, (int, float)) and eff > 0:
            a["eff_sum"] += float(eff)
            a["eff_n"] += 1
        bt = c.get("bound_type") or ""
        if bt in ("compute_bound", "memory_bound"):
            a["bounds"][bt] += 1
        if c.get("reusable_native_kernel"):
            a["routable"] += 1
    rows = [
        {
            "kernel_category": cat,
            "kernel_count": a["kernel_count"],
            "total_gpu_pct": round(a["total_gpu_pct"], 4),
            "total_duration_us": round(a["total_duration_us"], 3),
            "mean_efficiency_percent": round(a["eff_sum"] / a["eff_n"], 3) if a["eff_n"] else "",
            "dominant_bound_type": a["bounds"].most_common(1)[0][0] if a["bounds"] else "",
            "routable_count": a["routable"],
        }
        for cat, a in agg.items()
    ]
    rows.sort(key=lambda r: r["total_gpu_pct"], reverse=True)
    return rows


def _rows_to_csv(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """Serialize rows to CSV text (stdlib ``csv``; empty cell for None/missing)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})
    return buf.getvalue()


def build_metrics_csv(candidates: dict[str, Any]) -> str:
    """Per-kernel metrics as CSV text (all hot kernels; see :data:`_METRICS_COLUMNS`)."""
    return _rows_to_csv(_METRICS_COLUMNS, build_metrics_rows(candidates))


def build_category_summary_csv(candidates: dict[str, Any]) -> str:
    """Category-aggregated summary as CSV text (see :data:`_SUMMARY_COLUMNS`)."""
    return _rows_to_csv(_SUMMARY_COLUMNS, build_category_summary(candidates))


@dataclass(frozen=True)
class BypassReportContext:
    """The run identity and CSV links a bypass ``analysis.md`` names."""

    model_name: str
    framework: str
    target_platform: str
    throughput_unit: str = "tok/s"
    metrics_csv_path: str = ""
    summary_csv_path: str = ""


def render_analysis_md(
    candidates: dict[str, Any],
    analyze_out: dict[str, Any],
    context: BypassReportContext,
) -> str:
    """Render the human/downstream ``analysis.md`` report (bypass route)."""
    timeline = analyze_out.get("timeline") or {}
    attribution = analyze_out.get("attribution") or {}
    hot = candidates["hot_kernels"]
    rollup = _category_rollup(analyze_out)
    scope = candidates["aggregation_scope"]
    total_ms = timeline.get("total_time_ms")
    memcpy_ms = timeline.get("gpu_memcpy_ms")
    idle_pct = timeline.get("idle_pct")
    routable = [c for c in hot if c["reusable_native_kernel"]]
    provenance = (
        f"framework={context.framework or 'unknown'}, platform={context.target_platform or 'unknown'}, "
        f"throughput_unit={context.throughput_unit}, aggregation_scope={scope}. "
        f"Per-kernel roofline (bound/AI/efficiency) is computed analytically from captured "
        f"operand shapes + measured kernel time (roofline_source=analytical)."
    )
    extra = [
        *_ranking_sections(rollup, hot),
        *_optimization_sections(candidates, routable),
        *_detailed_analysis(hot),
        *_appendix(context, scope, analyze_out.get("event_total", 0), attribution),
    ]
    return render_report(
        AnalysisReport(
            route="bypass",
            model_name=context.model_name or "Workload",
            provenance_detail=provenance,
            exec_summary={
                "total_gpu_time_ms": total_ms,
                "gpu_busy_pct": timeline.get("busy_pct"),
                "gpu_idle_pct": idle_pct,
                "gpu_memcpy_ms": memcpy_ms,
                "top_bottleneck_category": rollup[0]["category"] if rollup else "n/a",
                "attribution_pct": attribution.get("attributed_pct"),
            },
            system_signals={
                "idle_pct": idle_pct,
                "exposed_comm_pct": None,  # bypass does not model exposed communication
                "exposed_memcpy_pct": memcpy_ms / total_ms * 100.0 if total_ms else None,
            },
            idle_threshold=resolve_idle_pct_threshold(),
            hot_kernels=[_spine_row(c) for c in hot],
            # One P-item per routable candidate, ranked by optimization ROI.
            p_items=[
                {"rank": c["priority_rank"], "category": c["kernel_category"], "rows": [_spine_row(c)]}
                for c in routable
            ],
            extra_sections="\n".join(extra),
        )
    )


def _spine_row(c: dict[str, Any]) -> dict[str, Any]:
    """A candidate as a row of the shared spine's Top Hot Kernels and P-item Data tables; bypass models neither %E2E
    nor the launcher Kernel Path, so those columns stay empty."""
    return {
        "name": c["name"],
        "time_us": c["duration_us"],
        "gpu_pct": c["gpu_pct"],
        "call_count": c["call_count"],
        "flops_per_byte": c["flops_per_byte"],
        "efficiency_percent": c["roofline_attainment_pct"],
        "arithmetic_intensity": c["arithmetic_intensity"],
        "bound_type": c["bound_type"],
        "category": c["kernel_category"],
        "args": [s["shape"] for s in c["input_shapes"]],
        "source_file": c["source_file"],
    }


def _attainment(c: dict[str, Any]) -> str:
    """A candidate's roofline attainment, or the unknown-bound dash when the roofline did not model it."""
    value = c["roofline_attainment_pct"]
    return _UNKNOWN_BOUND if value is None else f"{value:.1f}%"


def _intensity(c: dict[str, Any]) -> str:
    """A candidate's arithmetic intensity, or the unknown-bound dash when the roofline did not model it."""
    value = c["arithmetic_intensity"]
    return _UNKNOWN_BOUND if value is None else f"{value:.3g}"


def _ranking_sections(rollup: list[dict[str, Any]], hot: list[dict[str, Any]]) -> list[str]:
    """Top Operations by category with the analytical-bound split, then the Top 10 kernels by optimization priority."""
    lines = ["## Top Operations", ""]
    if rollup:
        lines += [
            "| Rank | Category | GPU % | Time (ms) | Kernels |",
            "|------|----------|-------|-----------|---------|",
        ]
        lines += [
            f"| {i} | {canonical_category(r['category'])} | {r['gpu_pct']} | {r['gpu_ms']} | {r['kernel_count']} |"
            for i, r in enumerate(rollup, start=1)
        ]
    else:
        lines.append("_No GPU kernels found in trace._")
    lines.append("")
    bounds = Counter(c["bound_type"] for c in hot)
    if bounds["compute_bound"] or bounds["memory_bound"]:
        lines += [
            f"_Analytical roofline bound: {bounds['compute_bound']} compute-bound, "
            f"{bounds['memory_bound']} memory-bound hot kernel(s)._",
            "",
        ]
    lines += ["## Top 10 Kernels by Optimization Priority", ""]
    ranked = sorted(hot, key=lambda c: c["optimization_priority"], reverse=True)[:10]
    if ranked:
        lines += [
            "_Priority = GPU% x (1 - efficiency): high-impact, low-efficiency kernels "
            "first. Full per-kernel metrics in the CSV linked below._",
            "",
            "| # | kernel_id | Name | Category | GPU% | Bound | AI | Eff% | Priority | Suggestion |",
            "|---|-----------|------|----------|------|-------|----|----|---------|------------|",
        ]
        lines += [
            f"| {i} | `{c['kernel_id']}` | {c['name']} | {canonical_category(c['kernel_category'])} "
            f"| {c['gpu_pct']:.2f}% | {c['bound_type']} | {_intensity(c)} | {_attainment(c)} "
            f"| {c['optimization_priority']:.2f} | {c['suggestion']} |"
            for i, c in enumerate(ranked, start=1)
        ]
    else:
        lines.append("_No GPU kernels found in trace._")
    lines.append("")
    return lines


def _optimization_sections(candidates: dict[str, Any], routable: list[dict[str, Any]]) -> list[str]:
    """Compute Kernel Optimizations (insight, action, source and impact per rewritable candidate), then the Task Groups
    table and the count of non-rewritable hot kernels."""
    lines = ["## Compute Kernel Optimizations", ""]
    if not routable:
        lines += ["_No rewritable compute-kernel candidates identified._", ""]
    else:
        with_source = [c for c in routable if c["source_file"]]
        dispatchable = [c for c in with_source if c["shape_dispatchable"]]
        lines += [
            f"_{len(dispatchable)} of {len(routable)} rewritable candidate(s) are auto-dispatchable "
            f"(resolved editable source AND a dispatch-grade operand shape); "
            f"{len(with_source) - len(dispatchable)} have a source but only a geometry shape, "
            f"and the rest need a source first._",
            "",
        ]
    for i, c in enumerate(routable, start=1):
        category = canonical_category(c["kernel_category"])
        if c["source_file"]:
            group = c.get("task_group")
            dispatch_note = (
                "dispatch-grade shape"
                if c["shape_dispatchable"]
                else f"geometry-only shape ({c['shape_provenance']}) — NOT auto-dispatchable, needs operand dims"
            )
            source = (
                f"**Source**: `{c['source_file']}` (via {c['source_resolution_method'] or 'unknown'}); "
                f"{dispatch_note}" + (f"; task group `{group['task_group_id']}`" if group else "") + "."
            )
        else:
            source = (
                "**Source**: unresolved — not auto-dispatchable (rewritable by classification, "
                "but no editable source was located for its launching op)."
            )
        lines += [
            f"### P{i}: {c['name']} ({category})",
            "",
            f"**Insight**: {category} kernel consuming "
            f"{c['gpu_pct']:.2f}% of GPU time across {c['call_count']} launches.",
            "",
            f"**Action**: {_ACTION_BY_CATEGORY.get(c['kernel_category'], _ACTION_BY_CATEGORY['Others'])}",
            "",
            source,
            "",
            f"**Impact**: {c['gpu_pct']:.2f}% of GPU time; bound={c['bound_type']}, attainment={_attainment(c)}, "
            f"priority={c['optimization_priority']:.2f} (roofline_source={c['roofline_source']}).",
            "",
        ]
    if candidates["task_groups"]:
        lines += [
            "## Task Groups",
            "",
            "_Repeated shapes of one rewritable operator collapse into a single dispatch with all observed cases._",
            "",
            "| Group | Source | Kernels | GPU % | Time (ms) |",
            "|-------|--------|---------|-------|-----------|",
        ]
        lines += [
            f"| {g['task_group_id']} | {g['source_path'].split('/')[-1] or '?'} | {len(g['kernel_ids'])} "
            f"| {g['aggregate_gpu_pct']} | {round(g['aggregate_duration_us'] / 1000.0, 3)} |"
            for g in candidates["task_groups"]
        ]
        lines.append("")
    if candidates["skipped_kernels"]:
        lines += [
            f"_{len(candidates['skipped_kernels'])} hot kernel(s) are non-rewritable "
            f"(vendor library / unresolved source) — see Detailed Analysis._",
            "",
        ]
    return lines


def _detailed_analysis(hot: list[dict[str, Any]]) -> list[str]:
    """The Detailed Analysis section: identification, data, source, roofline and suggested action per hot kernel."""
    lines = ["## Detailed Analysis", ""]
    for c in hot:
        skip_note = "" if c["reusable_native_kernel"] else f", skip_reason={c['skip_reason']}"
        source = f"`{c['source_file']}`" if c["source_file"] else "unresolved"
        lines += [
            f"### {c['kernel_id']}: {c['name']} ({canonical_category(c['kernel_category'])})",
            "",
            f"**Identification:** {c['gpu_pct']:.2f}% GPU time, {c['call_count']} launches, "
            f"reusable={c['reusable_native_kernel']}{skip_note}.",
            "",
            f"**Data:** device kernel `{c['device_kernel_name']}`; duration {c['duration_us'] / 1000.0:.2f} ms.",
            "",
            f"**Source:** {source} (shape provenance: {c['shape_provenance']}).",
            "",
            f"**Roofline:** bound={c['bound_type']}, AI={_intensity(c)}, attainment={_attainment(c)}, "
            f"priority={c['optimization_priority']:.2f} (roofline_source={c['roofline_source']}).",
            "",
            f"**Suggested action:** {c['suggestion']}",
            "",
        ]
    return lines


def _appendix(context: BypassReportContext, scope: str, event_total: int, attribution: dict[str, Any]) -> list[str]:
    """The Appendix (run identity, scan size, op attribution) and the links to the code-generated metric CSVs."""
    lines = [
        "## Appendix",
        "",
        f"- Framework: {context.framework or 'unknown'}",
        f"- Platform: {context.target_platform or 'unknown'}",
        f"- Throughput unit: {context.throughput_unit}",
        f"- Aggregation scope: {scope}",
        f"- Events scanned: {event_total}",
        f"- Attribution: {attribution.get('attributed_kernels', 0)}/"
        f"{attribution.get('kernel_count', 0)} kernels linked to an op "
        f"({attribution.get('attributed_pct', 0)}% of GPU time)",
        "",
    ]
    if context.metrics_csv_path or context.summary_csv_path:
        lines += [
            "## Structured Metrics (CSV)",
            "",
            "_Code-generated (no LLM). The Top-10 table above is a preview; these CSVs carry the full data._",
            "",
        ]
        if context.metrics_csv_path:
            lines.append(f"- Per-kernel metrics (all hot kernels): `{context.metrics_csv_path}`")
        if context.summary_csv_path:
            lines.append(f"- Category summary: `{context.summary_csv_path}`")
        lines.append("")
    return lines


def build_workload_roofline_totals(
    analyze_out: dict[str, Any],
    *,
    target_platform: str,
) -> dict[str, Any]:
    """Aggregate the analytical roofline over ALL analyzed device kernels."""
    sigma_actual = 0.0
    sigma_ideal = 0.0
    compute_us = 0.0
    memory_us = 0.0
    no_model_us = 0.0
    for k in analyze_out.get("kernels") or []:
        dur = float(k.get("gpu_time_us") or 0.0)
        if dur <= 0:
            continue
        sigma_actual += dur
        kc = classify_kernel(k.get("name", "") or "", op_name=k.get("op_name", "") or "")
        _count = k.get("count") or 0
        # Match the candidate-side waterfall for dispatch-grade shapes: own cpu_op dims, then the same-name
        # capture-time backfill.
        shape_entries = _trace_shape_entries(k.get("op_shapes") or [], k.get("op_dtypes") or [], _count)
        if not shape_entries:
            shape_entries = _trace_shape_entries(k.get("backfill_shapes") or [], k.get("backfill_dtypes") or [], _count)
        rl = (
            compute_roofline(
                KernelAggregate(
                    category=kc.category,
                    shape_str=shape_entries[0]["shape"] if shape_entries else "",
                    gpu_time_us=dur,
                    call_count=int(k.get("count") or 1),
                ),
                gpu_type=target_platform,
            )
            if shape_entries
            else None
        )
        attain = rl.get("roofline_attainment_pct") if rl else None
        if rl and isinstance(attain, (int, float)) and attain > 0:
            sigma_ideal += dur * (float(attain) / 100.0)
            bound = str(rl.get("bound_type") or "").upper()
            if "COMPUTE" in bound:
                compute_us += dur
            elif "MEMORY" in bound:
                memory_us += dur
            else:
                no_model_us += dur
        else:
            no_model_us += dur
    kernel_eff = (sigma_ideal / sigma_actual) if sigma_actual > 0 else 0.0
    return {
        "sigma_actual_kernel_us": round(sigma_actual, 3),
        "sigma_ideal_roofline_us": round(sigma_ideal, 3),
        "kernel_roofline_efficiency": kernel_eff,
        "compute_bound_us": round(compute_us, 3),
        "memory_bound_us": round(memory_us, 3),
        "no_perf_model_us": round(no_model_us, 3),
    }


def build_kernel_roofline(
    candidates: dict[str, Any],
    *,
    analysis_md_path: str,
    kernel_candidates_path: str,
) -> dict[str, Any]:
    """Build the per-kernel roofline sidecar payload."""
    rows = []
    for c in candidates.get("hot_kernels") or []:
        rows.append(
            {
                "kernel_id": c["kernel_id"],
                "name": c["name"],
                "kernel_category": c["kernel_category"],
                "duration_us": c["duration_us"],
                "gpu_pct": c["gpu_pct"],
                "call_count": c["call_count"],
                "bound_type": c["bound_type"],
                # bottleneck falls back to bound_type; roofline_name has no analytical analogue so it stays null.
                "bottleneck": c.get("bottleneck") or c.get("bound_type"),
                "roofline_name": c.get("roofline_name"),
                "suggestion": c.get("suggestion") or "",
                "recommended_actions": list(c.get("recommended_actions") or []),
                "efficiency_percent": c["efficiency_percent"],
                # Binding-side attainment (compute util if compute-bound, else bw).
                "roofline_attainment_pct": c.get("roofline_attainment_pct"),
                "arithmetic_intensity": c["arithmetic_intensity"],
                "compute_utilization_pct": c["compute_utilization_pct"],
                "bandwidth_utilization_pct": c["bandwidth_utilization_pct"],
                "reusable_native_kernel": c["reusable_native_kernel"],
                "source_file": c["source_file"],
                "flops_per_byte": c.get("flops_per_byte"),
                # roofline_source is how the bound was derived.
                "roofline_source": c.get("roofline_source", _RL_PLACEHOLDER),
            }
        )
    return {
        "source": "bypass",
        "analysis_md_path": analysis_md_path,
        "kernel_candidates_path": kernel_candidates_path,
        "kernels": rows,
    }


def build_fusion(analyze_out: dict[str, Any]) -> dict[str, Any]:
    """Build the kernel-fusion opportunity payload from the launch sequence."""
    launches = analyze_out.get("kernel_launches") or []
    # Memoize classification by (device name, op name): few distinct kernels but many launches, so caching keeps this
    # O(distinct) not O(launches x rules).
    _cat_cache: dict[tuple[str, str], str] = {}

    def _category(name: str, op_name: str) -> str:
        key = (name, op_name)
        cat = _cat_cache.get(key)
        if cat is None:
            cat = classify_kernel(name, op_name=op_name).category
            _cat_cache[key] = cat
        return cat

    categorized = [
        {
            "name": lc.get("name", ""),
            "op_name": lc.get("op_name", ""),
            "category": _category(lc.get("name", "") or "", lc.get("op_name", "") or ""),
            "ts": lc.get("ts", 0.0),
            "dur": lc.get("dur", 0.0),
        }
        for lc in launches
    ]
    payload = analyze_fusion(categorized)
    payload["source"] = "bypass"
    payload["aggregation_scope"] = analyze_out.get("aggregation_scope", "full_trace")
    return payload
