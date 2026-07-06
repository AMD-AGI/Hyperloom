###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build downstream artifacts (candidates / summary / kernel_roofline) for the
bypass analysis backend from the classified device-kernel aggregates.

Primary ranking unit is the device kernel (full coverage; robust to cudagraph),
classified by :mod:`_bypass_classify`, enriched with a best-effort launching op
name resolved via Kineto correlation (may be empty under cudagraph replay).

Schema mirrors the fields the Coordinator / kernel-agent consume in the
TraceLens ``kernel_candidates.json`` / ``summary.json`` / ``kernel_roofline.json``
contract. Roofline hardware fields (bound_type / efficiency / arithmetic
intensity) are left null here and filled by the rocprof-compute enrichment
stage (M6).
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter, defaultdict
from typing import Any

from _bypass_benchmark_resolver import find_benchmark_files, repo_root_from_source
from _bypass_classify import classify_kernel
from _bypass_fusion import analyze_fusion
from _bypass_roofline import compute_roofline
from _bypass_source_resolver import editable_trace_source, resolve_source

# Category-appropriate optimization guidance (structured, not LLM prose).
_ACTION_BY_CATEGORY: dict[str, str] = {
    "SDPA": "Profile the attention kernel for tile/occupancy; consider a fused/flash "
    "attention backend; increase decode batch to amortize KV reads.",
    "GEMM": "Tune GEMM tile size / precision and fuse the epilogue where possible; "
    "vendor-library GEMMs (Tensile/rocBLAS) are not rewritable — tune via library config.",
    "Quantization": "Fuse quantization into the adjacent GEMM epilogue and drop redundant "
    "per-tensor scaling passes.",
    "KVCacheStore": "Fuse the KV-cache write into attention to remove the separate reshape pass.",
    "Normalization": "Use a fused RMSNorm/LayerNorm and fold the residual/quant into the norm kernel.",
    "Convolution": "Pick an NHWC/implicit-GEMM conv algorithm for the shape; fuse "
    "bias/activation into the conv epilogue (VAE encode/decode).",
    "Elementwise": "Fuse elementwise chains to cut intermediate memory traffic.",
    "MoE": "Optimize expert GEMM and routing; fuse gate/up projections.",
    "MemCpy": "Reduce host/device copies; keep tensors resident on device.",
    "Others": "Profile the kernel for tile size and wave occupancy.",
}

_UNKNOWN_BOUND = "\u2014"  # em dash, matching the golden "unknown bound" marker.
_REUSABLE_BACKENDS = ["forge", "geak", "claude", "codex"]

# Bound-type display prefixes for the (deterministic) per-kernel suggestion.
_BOUND_PREFIX: dict[str, str] = {"compute_bound": "Compute-bound", "memory_bound": "Memory-bound"}


def _build_suggestion(category: str, bound_type: str) -> str:
    """Deterministic optimization hint from ``category`` + ``bound_type``.

    Pure category->text lookup (``_ACTION_BY_CATEGORY``) optionally prefixed with
    the analytical bound; NOT LLM-generated, so the ``suggestion`` column is
    reproducible and attributable. Feeds the specialist prompt's ``action`` slot
    (reads ``suggestion``/``recommended_actions``), which is otherwise empty for
    bypass candidates.
    """
    action = _ACTION_BY_CATEGORY.get(category, _ACTION_BY_CATEGORY["Others"])
    prefix = _BOUND_PREFIX.get(bound_type, "")
    return f"{prefix}: {action}" if prefix else action

# torch ``Input type`` token -> compact dtype suffix for the shape-string contract
# (e.g. ``(15360,2048) bf16``). Independent reimplementation of the TraceLens map
# (this module never imports TraceLens); unmapped/empty types emit a bare shape.
_DTYPE_SUFFIX: dict[str, str] = {
    # Suffixes MUST match the shared harness dtype_map (bf16/fp16/fp32) + the
    # roofline peak table; a compact "f16"/"f32" makes the harness emit an invalid
    # ``torch.f16`` (crashes at runtime, e.g. under rocprof/GEAK). See
    # test_bypass_downstream_contract::...valid_torch_dtype_in_harness.
    "c10::bfloat16": "bf16", "bfloat16": "bf16",
    "c10::half": "fp16", "half": "fp16", "float16": "fp16",
    "float": "fp32", "float32": "fp32",
    "double": "fp64", "float64": "fp64",
    "int": "i32", "int32": "i32",
    "long": "i64", "int64": "i64",
    "short": "i16", "int16": "i16",
    "char": "i8", "int8": "i8",
    "uint8": "u8", "bool": "bool",
}


def _format_operand_shape(dims: Any, dtype: Any) -> str | None:
    """Render one operand as a ``(d0,d1,...) <dtype>`` string (or ``None``).

    Scalar / empty / non-integer operands return ``None`` (dropped from the
    shape string), matching the downstream harness contract. A 1-D operand keeps
    the trailing comma (``(d,)``) so it round-trips as a tuple.
    """
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
    """Build the downstream ``input_shapes`` contract from Kineto Input Dims/type.

    Converts a call's per-arg dims (``op_shapes``) + dtypes (``op_dtypes``) into
    ``[{"call_num", "shape"}]`` where ``shape`` is the ``<br>``-joined operand
    strings the GEAK harness (``_build_configs`` / ``_parse_shape_string``) and
    TraceLens candidates consume. Returns ``[]`` when no operand is renderable.

    Args:
        op_shapes: List of per-arg dimension lists (Kineto ``Input Dims``).
        op_dtypes: List of per-arg dtype tokens (Kineto ``Input type``), aligned
            by argument index with ``op_shapes``.
        call_count: Number of launches (stamped as ``call_num``).

    Returns:
        A one-entry ``[{"call_num", "shape"}]`` list, or ``[]``.
    """
    dtypes = op_dtypes if isinstance(op_dtypes, (list, tuple)) else []
    operands: list[str] = []
    for i, dims in enumerate(op_shapes if isinstance(op_shapes, (list, tuple)) else []):
        rendered = _format_operand_shape(dims, dtypes[i] if i < len(dtypes) else "")
        if rendered:
            operands.append(rendered)
    if not operands:
        return []
    return [{"call_num": int(call_count) if call_count else 1, "shape": "<br>".join(operands)}]


def _source_type_for_op(op_name: str) -> str:
    """Best-effort source-type guess from a launching op name.

    Args:
        op_name: Resolved launching op name (may be empty).

    Returns:
        ``"python"`` / ``"hip_cpp"`` / ``"unknown"``.
    """
    n = (op_name or "").lower()
    if not n:
        return "unknown"
    if n.startswith(("aten::", "vllm::", "vllm_aiter::", "_c_cache_ops::", "_rocm_c::")):
        return "python"
    if "aiter" in n or "ck" in n:
        return "hip_cpp"
    return "unknown"


# Native device-code extensions (grouped by source file only; a python launcher
# frame can host distinct kernels so it also keys on the operation — see below).
_NATIVE_SOURCE_EXTS = (".cu", ".cuh", ".hip", ".h")


def _build_task_groups(hot_kernels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group routable candidates that share an editable source into task groups.

    Only candidates that are both ``reusable_native_kernel`` and carry a resolved
    ``source_file`` participate: an unresolved (empty) source is not a shared
    function, so those stay standalone (per-kernel dispatch, unchanged). Native
    ``.cu`` sources key on the source file alone (collapsing template/shape
    variants of one ``__global__`` into one job); python/Triton sources also key
    on the operation, since one launcher frame can host distinct kernels.

    Each group carries compact ``rows`` (not full candidates, to avoid nesting):
    ``kernel_id`` / ``name`` / ``device_kernel_name`` / ``shapes`` / ``call_count``
    / ``duration_us`` / ``percent_of_total`` / ``gpu_pct`` / ``bound_type``. Groups
    are ranked by aggregate GPU time; the heaviest row is the primary.

    Args:
        hot_kernels: The candidate rows from :func:`build_candidates`.

    Returns:
        Ordered task-group dicts (``tg001`` first = heaviest), or ``[]`` when no
        candidate is routable-with-source.
    """
    buckets: dict[tuple, dict[str, Any]] = {}
    for c in hot_kernels:
        if not c.get("reusable_native_kernel"):
            continue
        src = str(c.get("source_file") or "").strip()
        if not src:
            continue
        operation = str(c.get("name") or "")
        if src.lower().endswith(_NATIVE_SOURCE_EXTS):
            key: tuple = ("native", src)
        else:
            key = ("py", src, operation)
        shapes = c.get("input_shapes") or []
        row = {
            "kernel_id": c.get("kernel_id", ""),
            "name": c.get("name", ""),
            "device_kernel_name": c.get("device_kernel_name", ""),
            # A single representative call's per-arg dims (harness-consumable).
            "shapes": shapes[0] if shapes else [],
            "call_count": c.get("call_count", 0),
            "duration_us": c.get("duration_us", 0.0),
            "percent_of_total": c.get("percent_of_total", 0.0),
            "gpu_pct": c.get("gpu_pct", 0.0),
            "bound_type": c.get("bound_type", ""),
        }
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = {
                "task_group_id": "",
                "operation": operation,
                "source_path": src,
                "kernel_ids": [],
                "primary_kernel_id": "",
                "rows": [],
                "aggregate_duration_us": 0.0,
                "aggregate_call_count": 0,
                "aggregate_gpu_pct": 0.0,
                "source": "bypass",
            }
        if row["kernel_id"] and row["kernel_id"] not in bucket["kernel_ids"]:
            bucket["kernel_ids"].append(row["kernel_id"])
        bucket["rows"].append(row)
        bucket["aggregate_duration_us"] += float(row["duration_us"] or 0.0)
        bucket["aggregate_call_count"] += int(row["call_count"] or 0)
        bucket["aggregate_gpu_pct"] += float(row["gpu_pct"] or 0.0)

    ordered = sorted(buckets.values(), key=lambda g: g["aggregate_duration_us"], reverse=True)
    for idx, group in enumerate(ordered, start=1):
        group["task_group_id"] = f"tg{idx:03d}"
        group["rows"].sort(key=lambda r: float(r.get("duration_us") or 0.0), reverse=True)
        if group["rows"]:
            group["primary_kernel_id"] = group["rows"][0]["kernel_id"]
        group["aggregate_duration_us"] = round(group["aggregate_duration_us"], 3)
        group["aggregate_gpu_pct"] = round(group["aggregate_gpu_pct"], 3)
    return ordered


def _source_type_from_path(path: str) -> str:
    """Derive source type from a resolved source file's extension.

    Args:
        path: Resolved editable source path.

    Returns:
        ``"hip_cpp"`` for native device code, ``"python"`` for a ``.py`` source,
        or ``""`` when the extension is unrecognized (caller falls back to the
        op-name heuristic).
    """
    low = (path or "").lower()
    if low.endswith(_NATIVE_SOURCE_EXTS):
        return "hip_cpp"
    if low.endswith(".py"):
        return "python"
    return ""


def _short_name(kernel_name: str) -> str:
    """Shorten a mangled device-kernel name for candidate identity display."""
    n = kernel_name or ""
    # Strip common C++ template/mangling tails for readability.
    n = re.sub(r"<.*$", "", n)
    n = n.strip()
    return n[:80] if n else "unknown_kernel"


def build_candidates(
    analyze_out: dict[str, Any],
    *,
    framework: str,
    target_platform: str,
    top_k: int = 15,
    discover_benchmarks: bool = False,
) -> dict[str, Any]:
    """Turn classified top device kernels into the candidate payload.

    Args:
        analyze_out: Result of :func:`_bypass_trace_reader.analyze_trace`.
        framework: Serving framework tag.
        target_platform: GPU platform tag.
        top_k: Max number of hot-kernel candidates to emit.

    Returns:
        A dict with ``hot_kernels`` (routable + non-routable, ranked), plus
        ``skipped_kernels`` and ``task_groups`` mirrors, and aggregate counts.
    """
    kernels = analyze_out.get("kernels") or []
    hot_kernels: list[dict[str, Any]] = []
    skipped_kernels: list[dict[str, Any]] = []
    for idx, k in enumerate(kernels[: top_k if top_k and top_k > 0 else len(kernels)], start=1):
        kname = k.get("name", "") or ""
        op_name = k.get("op_name", "") or ""
        kc = classify_kernel(kname, op_name=op_name)
        kernel_id = f"k{idx:03d}"
        display = op_name or _short_name(kname)

        # Source resolution (unlocks downstream kernel-opt dispatch, which
        # requires a readable source_file). Priority: (1) a Triton kernel_file
        # straight from the trace's cpu_op args; (2) the op_to_source.json
        # dictionary lookup; else unresolved (candidate stays non-routable).
        source_file = editable_trace_source(k.get("op_kernel_file", "") or "", k.get("op_kernel_backend", "") or "")
        source_method = "trace_kernel_file" if source_file else "unresolved"
        if not source_file and op_name:
            source_file, method = resolve_source(op_name, framework=framework, device_kernel_name=kname)
            if source_file:
                source_method = method

        # Real per-arg dims/dtypes from the trace, rendered into the downstream
        # shape-string contract ([{"call_num","shape":"(dims) dtype<br>..."}]) the
        # GEAK harness (_build_configs) + TraceLens candidates consume;
        # ``shape_provenance="torch_trace"`` marks the dims real.
        op_shapes = k.get("op_shapes") or []
        op_dtypes = k.get("op_dtypes") or []
        shape_entries = _trace_shape_entries(op_shapes, op_dtypes, k.get("count") or 0)

        # Benchmark discovery (opt-in; gated by the caller because only the
        # rocprof-compute roofline enrichment consumes it). A routable kernel's
        # on-disk test/benchmark seeds the shared GEAK harness that rocprof
        # profiles for real bound/AI — without it the enrichment skips the row.
        bench_files: list[str] = []
        kernel_repo = ""
        if discover_benchmarks and kc.reusable and source_file:
            kernel_repo = repo_root_from_source(source_file)
            bench_files = find_benchmark_files(op_name, source_file)

        cand: dict[str, Any] = {
            "kernel_id": kernel_id,
            "name": display,
            "kernel_category": kc.category,
            "device_kernel_name": kname[:120],
            "device_kernel_names": [kname[:120]],
            "duration_us": k.get("gpu_time_us", 0.0),
            "gpu_pct": k.get("gpu_pct", 0.0),
            "percent_of_total": k.get("gpu_pct", 0.0),
            "call_count": k.get("count", 0),
            "bound_type": _UNKNOWN_BOUND,
            "efficiency_percent": 0.0,
            "flops_per_byte": None,
            "arithmetic_intensity": None,
            "compute_utilization_pct": None,
            "bandwidth_utilization_pct": None,
            "rocprof_roofline": None,
            # Placeholder roofline: bound_type/AI/util above are structural
            # defaults, NOT measured. ``roofline_source`` tracks how the bound was
            # derived: "placeholder" (unestimable) -> "analytical" (from shapes +
            # measured time, below) -> "rocprof" (opt-in measured, sets
            # roofline_measured=True). Lets downstream/LLM distinguish them.
            "roofline_measured": False,
            "roofline_source": "placeholder",
            "library": "",
            "backend": framework,
            "framework": framework,
            "source_file": source_file,
            "source_resolution_method": source_method,
            # Prefer the resolved source's extension; fall back to the op-name heuristic.
            "source_type": _source_type_from_path(source_file) or _source_type_for_op(op_name),
            "reusable_native_kernel": kc.reusable,
            "skip_reason": "" if kc.reusable else kc.skip_reason,
            "recommended_backends": list(_REUSABLE_BACKENDS) if kc.reusable else [],
            # Seeds for the shared GEAK harness + rocprof roofline enrichment
            # (populated only when discover_benchmarks is set; else empty).
            "benchmark_files": bench_files,
            "kernel_repo": kernel_repo,
            # ``shapes`` / ``input_shapes`` use the same downstream contract form.
            # The orchestrator kernel-opt gate (_validate_kernel_shape_and_paths)
            # reads ``shapes`` (rejects ``empty_kernel_shape`` when absent) and the
            # GEAK harness parses ``shape`` strings, so a candidate with captured
            # dims MUST expose them in this format or it can never reach/run GEAK.
            "shapes": shape_entries,
            "input_shapes": shape_entries,
            "input_dtypes": op_dtypes,
            "shape_provenance": "torch_trace" if shape_entries else "unresolved",
        }
        # Analytical roofline: derive bound_type / AI / efficiency from the
        # captured shapes + measured time for EVERY estimable kernel — including
        # vendor kernels (hipBLASLt GEMM / MIOpen conv) that the reusable-gated
        # rocprof enrichment skips (this is what gives xDiT a real bound). The
        # opt-in rocprof enrichment later refines it to a measured roofline.
        rl = compute_roofline(
            category=kc.category,
            shape_str=shape_entries[0]["shape"] if shape_entries else "",
            gpu_time_us=float(cand["duration_us"] or 0.0),
            call_count=int(cand["call_count"] or 1),
            gpu_type=target_platform,
        )
        if rl:
            cand.update(rl)
        # Optimization ROI = share of GPU time x headroom (1 - efficiency).
        # High-impact + low-efficiency kernels rank first. With no analytical
        # efficiency (placeholder), headroom=1 so it degrades to gpu_pct (pure
        # cost) -- a sensible fallback. Deterministic + reproducible.
        eff = cand.get("efficiency_percent")
        eff = float(eff) if isinstance(eff, (int, float)) else 0.0
        headroom = 1.0 - min(max(eff, 0.0), 100.0) / 100.0
        cand["optimization_priority"] = round(float(cand.get("gpu_pct") or 0.0) * headroom, 4)
        # Deterministic per-kernel hint (fills the specialist prompt's action slot).
        suggestion = _build_suggestion(kc.category, str(cand.get("bound_type") or ""))
        cand["suggestion"] = suggestion
        cand["recommended_actions"] = [suggestion]
        hot_kernels.append(cand)
        if not kc.reusable:
            skipped_kernels.append(cand)

    # Group routable candidates that share an editable source so the optimizer
    # dispatches one job per source function (with all observed shapes) instead
    # of a redundant run per device-kernel variant. The compact group is stamped
    # onto each member candidate (downstream reads ``candidate["task_group"]``).
    task_groups = _build_task_groups(hot_kernels)
    kid_to_group = {kid: g for g in task_groups for kid in g["kernel_ids"]}
    for c in hot_kernels:
        g = kid_to_group.get(c["kernel_id"])
        if g is not None:
            c["task_group"] = g

    # 1-based rank by optimization ROI. Stamped as a field WITHOUT reordering
    # hot_kernels (that list stays gpu_pct-sorted for downstream/top15 stability);
    # the CSV / md Top-N views sort by this rank themselves.
    for rank, c in enumerate(sorted(hot_kernels, key=lambda x: x.get("optimization_priority") or 0.0, reverse=True), start=1):
        c["priority_rank"] = rank

    return {
        "source": "bypass",
        "framework": framework,
        "target_platform": target_platform,
        "aggregation_scope": analyze_out.get("aggregation_scope", "full_trace"),
        "hot_kernels": hot_kernels,
        "skipped_kernels": skipped_kernels,
        "task_groups": task_groups,
    }


def build_summary(
    candidates: dict[str, Any],
    *,
    framework: str,
    target_platform: str,
    generated_at: str,
    trace_health_warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the routed-vs-skipped audit ``summary.json`` payload.

    Args:
        candidates: Output of :func:`build_candidates`.
        framework: Serving framework tag.
        target_platform: GPU platform tag.
        generated_at: ISO timestamp string.
        trace_health_warnings: Optional health warnings to record.

    Returns:
        The ``summary.json`` payload dict.
    """
    hot = candidates.get("hot_kernels") or []
    tasks = []
    skipped = []
    for c in hot:
        row = {
            "kernel_id": c["kernel_id"],
            "name": c["name"],
            "kernel_category": c["kernel_category"],
            "duration_us": c["duration_us"],
            "gpu_pct": c["gpu_pct"],
            "call_count": c["call_count"],
            "source_type": c["source_type"],
            "reusable_native_kernel": c["reusable_native_kernel"],
        }
        if c["reusable_native_kernel"]:
            row["recommended_backends"] = c["recommended_backends"]
            tasks.append(row)
        else:
            row["skip_reason"] = c["skip_reason"]
            skipped.append(row)
    # Compact task-group projection for the audit view (full rows live on
    # kernel_candidates.json's candidate[].task_group).
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
        "skipped_count": len(skipped),
        "task_group_count": len(group_entries),
        "trace_health_warnings": list(trace_health_warnings or []),
    }


def _category_rollup(analyze_out: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate GPU time by category over *all* device kernels.

    Uses the full kernel list (not just the top-K candidates) so category
    shares are complete. Requires the reader to have been called with
    ``top_k=0``.

    Returns:
        Category rows sorted by GPU time desc, each with gpu_ms / gpu_pct /
        kernel count / reusable_ms.
    """
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
# Structured CSV export (deterministic, code-generated; NOT LLM).
# ---------------------------------------------------------------------------
#: Stable column order for the per-kernel metrics CSV. Every numeric column is
#: reproducible from the trace (measured) or a documented formula
#: (optimization_priority = gpu_pct * (1 - efficiency/100)); roofline_source /
#: shape_provenance state each value's provenance so the CSV is fully attributable.
_METRICS_COLUMNS: list[str] = [
    "priority_rank", "optimization_priority", "kernel_id", "name", "kernel_category",
    "device_kernel_name", "duration_us", "gpu_pct", "call_count",
    "bound_type", "arithmetic_intensity", "efficiency_percent",
    "compute_utilization_pct", "bandwidth_utilization_pct", "roofline_source", "roofline_measured",
    "reusable_native_kernel", "source_file", "source_type", "recommended_backends",
    "benchmark_files_count", "skip_reason", "representative_shape", "input_dtypes",
    "shape_provenance", "suggestion",
]

_SUMMARY_COLUMNS: list[str] = [
    "kernel_category", "kernel_count", "total_gpu_pct", "total_duration_us",
    "mean_efficiency_percent", "dominant_bound_type", "routable_count",
]


def _join_list(value: Any) -> str:
    """Render a list cell as a ``;``-joined string (CSV-cell friendly)."""
    return ";".join(str(x) for x in value) if isinstance(value, list) else ""


def build_metrics_rows(candidates: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ALL hot kernels (routable + skipped) into per-kernel metric rows.

    Args:
        candidates: Output of :func:`build_candidates`.

    Returns:
        One flat dict per hot kernel keyed by :data:`_METRICS_COLUMNS`.
    """
    rows: list[dict[str, Any]] = []
    for c in candidates.get("hot_kernels") or []:
        shapes = c.get("shapes") or []
        rep = shapes[0].get("shape", "") if shapes and isinstance(shapes[0], dict) else ""
        rows.append({
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
            "roofline_measured": c.get("roofline_measured", ""),
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
        })
    return rows


def build_category_summary(candidates: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate hot kernels by category (the CSV 'summary' view).

    Args:
        candidates: Output of :func:`build_candidates`.

    Returns:
        One row per category (keyed by :data:`_SUMMARY_COLUMNS`), GPU%-descending.
    """
    agg: dict[str, dict[str, Any]] = {}
    for c in candidates.get("hot_kernels") or []:
        cat = c.get("kernel_category") or "Others"
        a = agg.setdefault(cat, {
            "kernel_count": 0, "total_gpu_pct": 0.0, "total_duration_us": 0.0,
            "eff_sum": 0.0, "eff_n": 0, "bounds": Counter(), "routable": 0,
        })
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


def render_analysis_md(
    candidates: dict[str, Any],
    analyze_out: dict[str, Any],
    *,
    model_name: str,
    framework: str,
    target_platform: str,
    throughput_unit: str = "tok/s",
    metrics_csv_path: str = "",
    summary_csv_path: str = "",
) -> str:
    """Render the human/downstream ``analysis.md`` report (bypass route).

    Structured (not LLM prose); mirrors the golden section layout but is not
    consumed by ``parse_analysis_md`` (bypass builds candidates directly).

    Args:
        candidates: Output of :func:`build_candidates`.
        analyze_out: Output of :func:`_bypass_trace_reader.analyze_trace`
            (must be produced with ``top_k=0`` for a complete category rollup).
        model_name: Model identifier for the title.
        framework: Serving framework tag.
        target_platform: GPU platform tag.
        throughput_unit: ``tok/s`` (text-gen) or ``img/s`` (xDiT).

    Returns:
        The full markdown report text.
    """
    timeline = analyze_out.get("timeline") or {}
    attribution = analyze_out.get("attribution") or {}
    hot = candidates.get("hot_kernels") or []
    rollup = _category_rollup(analyze_out)
    top_cat = rollup[0]["category"] if rollup else "n/a"
    scope = candidates.get("aggregation_scope", "full_trace")

    L: list[str] = []
    L.append(f"# Bypass Analysis Report - {model_name or 'Workload'}")
    L.append("")
    L.append(
        f"> Generated via bypass route (HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass). "
        f"framework={framework or 'unknown'}, platform={target_platform or 'unknown'}, "
        f"throughput_unit={throughput_unit}, aggregation_scope={scope}. "
        f"Not a TraceLens report; per-kernel roofline (bound/AI/efficiency) is computed "
        f"analytically from captured operand shapes + measured kernel time "
        f"(roofline_source=analytical), with optional rocprof-compute refinement."
    )
    L.append("")

    # Executive Summary
    L.append("## Executive Summary")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|--------|-------|")
    L.append(f"| Total GPU Time | {timeline.get('total_time_ms', 0)} ms |")
    L.append(f"| GPU Busy % | {timeline.get('busy_pct', 0)}% |")
    L.append(f"| Idle % | {timeline.get('idle_pct', 0)}% |")
    L.append(f"| GPU MemCpy | {timeline.get('gpu_memcpy_ms', 0)} ms |")
    L.append(f"| Top Bottleneck Category | {top_cat} |")
    L.append(f"| Op-attribution Coverage | {attribution.get('attributed_pct', 0)}% |")
    L.append("")

    # Top Operations (category rollup)
    L.append("## Top Operations")
    L.append("")
    if rollup:
        L.append("| Rank | Category | GPU % | Time (ms) | Kernels |")
        L.append("|------|----------|-------|-----------|---------|")
        for i, r in enumerate(rollup, start=1):
            L.append(f"| {i} | {r['category']} | {r['gpu_pct']} | {r['gpu_ms']} | {r['kernel_count']} |")
    else:
        L.append("_No GPU kernels found in trace._")
    L.append("")
    # Analytical roofline bound distribution (one-liner).
    n_compute = sum(1 for c in hot if c.get("bound_type") == "compute_bound")
    n_memory = sum(1 for c in hot if c.get("bound_type") == "memory_bound")
    if n_compute or n_memory:
        L.append(f"_Analytical roofline bound: {n_compute} compute-bound, {n_memory} memory-bound hot kernel(s)._")
        L.append("")

    # Top 10 kernels by optimization ROI (priority = gpu_pct x (1 - efficiency);
    # high-impact + low-efficiency first). Full per-kernel table lives in the CSV.
    L.append("## Top 10 Kernels by Optimization Priority")
    L.append("")
    ranked = sorted(hot, key=lambda c: c.get("optimization_priority") or 0.0, reverse=True)[:10]
    if ranked:
        L.append(
            "_Priority = GPU% x (1 - efficiency): high-impact, low-efficiency kernels "
            "first. Full per-kernel metrics in the CSV linked below._"
        )
        L.append("")
        L.append("| # | kernel_id | Name | Category | GPU% | Bound | AI | Eff% | Priority | Suggestion |")
        L.append("|---|-----------|------|----------|------|-------|----|----|---------|------------|")
        for i, c in enumerate(ranked, start=1):
            ai = c.get("arithmetic_intensity")
            ai_str = f"{float(ai):.3g}" if isinstance(ai, (int, float)) else "\u2014"
            eff = c.get("efficiency_percent")
            eff_str = f"{float(eff):.1f}%" if isinstance(eff, (int, float)) else "\u2014"
            L.append(
                f"| {i} | `{c.get('kernel_id', '')}` | {c.get('name', '')} | {c.get('kernel_category', '')} "
                f"| {float(c.get('gpu_pct') or 0.0):.2f}% | {c.get('bound_type', '')} | {ai_str} | {eff_str} "
                f"| {float(c.get('optimization_priority') or 0.0):.2f} | {c.get('suggestion', '')} |"
            )
    else:
        L.append("_No GPU kernels found in trace._")
    L.append("")

    # Compute Kernel Optimizations
    L.append("## Compute Kernel Optimizations")
    L.append("")
    routable = [c for c in hot if c.get("reusable_native_kernel")]
    dispatchable = [c for c in routable if c.get("source_file")]
    if not routable:
        L.append("_No rewritable compute-kernel candidates identified._")
        L.append("")
    else:
        # Downstream kernel-opt dispatches only candidates that are both
        # rewritable AND carry a resolved editable source; surface that split so
        # the report never implies an unresolved candidate is actionable.
        L.append(
            f"_{len(dispatchable)} of {len(routable)} rewritable candidate(s) have a resolved "
            f"editable source (auto-dispatchable to kernel-opt); the rest need a source first._"
        )
        L.append("")
        for i, c in enumerate(routable, start=1):
            L.append(f"### P{i}: {c['name']} ({c['kernel_category']})")
            L.append("")
            L.append(
                f"**Insight**: {c['kernel_category']} kernel consuming "
                f"{c['gpu_pct']:.2f}% of GPU time across {c['call_count']} launches."
            )
            L.append("")
            L.append(f"**Action**: {_ACTION_BY_CATEGORY.get(c['kernel_category'], _ACTION_BY_CATEGORY['Others'])}")
            L.append("")
            src = c.get("source_file") or ""
            if src:
                tg = (c.get("task_group") or {}).get("task_group_id") or ""
                L.append(
                    f"**Source**: `{src}` (via {c.get('source_resolution_method') or 'unknown'}); "
                    f"shapes captured: {'yes' if c.get('input_shapes') else 'no'}"
                    + (f"; task group `{tg}`" if tg else "")
                    + "."
                )
            else:
                L.append(
                    "**Source**: unresolved — not auto-dispatchable (rewritable by classification, "
                    "but no editable source was located for its launching op)."
                )
            L.append("")
            bound = c.get("bound_type") or "\u2014"
            eff = c.get("efficiency_percent")
            eff_str = f"{float(eff):.1f}%" if isinstance(eff, (int, float)) else "\u2014"
            L.append(
                f"**Impact**: {c['gpu_pct']:.2f}% of GPU time; bound={bound}, "
                f"efficiency={eff_str}, priority={float(c.get('optimization_priority') or 0.0):.2f} "
                f"(roofline_source={c.get('roofline_source', 'placeholder')})."
            )
            L.append("")

    # Task Groups (source-function dispatch grouping)
    task_groups = candidates.get("task_groups") or []
    if task_groups:
        L.append("## Task Groups")
        L.append("")
        L.append(
            "_Rewritable candidates sharing one editable source collapse into a single "
            "dispatch (all observed shapes)._"
        )
        L.append("")
        L.append("| Group | Source | Kernels | GPU % | Time (ms) |")
        L.append("|-------|--------|---------|-------|-----------|")
        for g in task_groups:
            src_disp = (g.get("source_path", "") or "?").split("/")[-1] or "?"
            L.append(
                f"| {g.get('task_group_id', '')} | {src_disp} | {len(g.get('kernel_ids') or [])} "
                f"| {g.get('aggregate_gpu_pct', 0)} | {round(float(g.get('aggregate_duration_us', 0) or 0) / 1000.0, 3)} |"
            )
        L.append("")

    # System-Level Signals
    L.append("## System-Level Signals")
    L.append("")
    L.append(f"- GPU idle: {timeline.get('idle_pct', 0)}% of total")
    L.append(f"- Device memcpy: {timeline.get('gpu_memcpy_ms', 0)} ms")
    skipped = candidates.get("skipped_kernels") or []
    if skipped:
        L.append(
            f"- {len(skipped)} hot kernel(s) are non-rewritable "
            f"(vendor library / unresolved source) — see Appendix."
        )
    L.append("")

    # Detailed Analysis
    L.append("## Detailed Analysis")
    L.append("")
    for c in hot:
        L.append(f"### {c['kernel_id']}: {c['name']} ({c['kernel_category']})")
        L.append("")
        L.append(
            f"**Identification:** {c['gpu_pct']:.2f}% GPU time, {c['call_count']} launches, "
            f"reusable={c['reusable_native_kernel']}"
            + (f", skip_reason={c['skip_reason']}" if not c["reusable_native_kernel"] else "")
            + "."
        )
        L.append("")
        L.append(f"**Data:** device kernel `{c['device_kernel_name']}`; duration {c['duration_us'] / 1000.0:.2f} ms.")
        L.append("")
        src = c.get("source_file") or ""
        L.append(
            f"**Source:** {('`' + src + '`') if src else 'unresolved'} "
            f"(shape provenance: {c.get('shape_provenance', 'unresolved')})."
        )
        L.append("")
        _eff = c.get("efficiency_percent")
        _eff_s = f"{float(_eff):.1f}%" if isinstance(_eff, (int, float)) else _UNKNOWN_BOUND
        _ai = c.get("arithmetic_intensity")
        _ai_s = f"{float(_ai):.3g}" if isinstance(_ai, (int, float)) else _UNKNOWN_BOUND
        _bound = c.get("bound_type") or _UNKNOWN_BOUND
        L.append(
            f"**Roofline:** bound={_bound}, AI={_ai_s}, "
            f"efficiency={_eff_s}, priority={float(c.get('optimization_priority') or 0.0):.2f} "
            f"(roofline_source={c.get('roofline_source', 'placeholder')})."
        )
        L.append("")
        L.append(f"**Suggested action:** {c.get('suggestion', '')}")
        L.append("")

    # Appendix
    L.append("## Appendix")
    L.append("")
    L.append(f"- Framework: {framework or 'unknown'}")
    L.append(f"- Platform: {target_platform or 'unknown'}")
    L.append(f"- Throughput unit: {throughput_unit}")
    L.append(f"- Aggregation scope: {scope}")
    L.append(f"- Events scanned: {analyze_out.get('event_total', 0)}")
    L.append(
        f"- Attribution: {attribution.get('attributed_kernels', 0)}/"
        f"{attribution.get('kernel_count', 0)} kernels linked to an op "
        f"({attribution.get('attributed_pct', 0)}% of GPU time)"
    )
    L.append("")

    # Structured CSV export (full data; code-generated, machine-readable).
    if metrics_csv_path or summary_csv_path:
        L.append("## Structured Metrics (CSV)")
        L.append("")
        L.append("_Code-generated (no LLM). The Top-10 table above is a preview; these CSVs carry the full data._")
        L.append("")
        if metrics_csv_path:
            L.append(f"- Per-kernel metrics (all hot kernels): `{metrics_csv_path}`")
        if summary_csv_path:
            L.append(f"- Category summary: `{summary_csv_path}`")
        L.append("")
    return "\n".join(L)


def build_kernel_roofline(
    candidates: dict[str, Any],
    *,
    analysis_md_path: str,
    kernel_candidates_path: str,
) -> dict[str, Any]:
    """Build the per-kernel roofline sidecar payload.

    Hardware roofline fields are null here (populated by the rocprof-compute
    enrichment stage).

    Args:
        candidates: Output of :func:`build_candidates`.
        analysis_md_path: Path to the written ``analysis.md``.
        kernel_candidates_path: Path to the written ``kernel_candidates.json``.

    Returns:
        The ``kernel_roofline.json`` payload dict.
    """
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
                "efficiency_percent": c["efficiency_percent"],
                "arithmetic_intensity": c["arithmetic_intensity"],
                "compute_utilization_pct": c["compute_utilization_pct"],
                "bandwidth_utilization_pct": c["bandwidth_utilization_pct"],
                "reusable_native_kernel": c["reusable_native_kernel"],
                "source_file": c["source_file"],
                "rocprof_roofline": c["rocprof_roofline"],
                "flops_per_byte": c.get("flops_per_byte"),
                # False = not hardware-measured; roofline_source is how the bound
                # was derived (placeholder / analytical / rocprof).
                "roofline_measured": c.get("roofline_measured", False),
                "roofline_source": c.get("roofline_source", "placeholder"),
            }
        )
    return {
        "source": "bypass",
        "analysis_md_path": analysis_md_path,
        "kernel_candidates_path": kernel_candidates_path,
        "kernels": rows,
    }


def build_fusion(analyze_out: dict[str, Any]) -> dict[str, Any]:
    """Build the kernel-fusion opportunity payload from the launch sequence.

    Classifies each time-ordered launch (device name + launching op) and finds
    fusable clusters + adjacent transitions (see :mod:`_bypass_fusion`). Returns
    an empty payload when the reader did not emit ``kernel_launches``.

    Args:
        analyze_out: Result of :func:`_bypass_trace_reader.analyze_trace`
            (with ``emit_launches=True``).

    Returns:
        The ``kernel_sequence`` payload dict.
    """
    launches = analyze_out.get("kernel_launches") or []
    categorized = [
        {
            "name": lc.get("name", ""),
            "op_name": lc.get("op_name", ""),
            "category": classify_kernel(lc.get("name", "") or "", op_name=lc.get("op_name", "") or "").category,
            "ts": lc.get("ts", 0.0),
            "dur": lc.get("dur", 0.0),
        }
        for lc in launches
    ]
    payload = analyze_fusion(categorized)
    payload["source"] = "bypass"
    payload["aggregation_scope"] = analyze_out.get("aggregation_scope", "full_trace")
    return payload
