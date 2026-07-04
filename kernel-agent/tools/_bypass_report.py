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

import re
from collections import defaultdict
from typing import Any

from _bypass_classify import classify_kernel
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
        kc = classify_kernel(kname)
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

        # Real per-arg dims/dtypes from the trace (values synthesized later by
        # the harness); ``shape_provenance="torch_trace"`` marks the dims real.
        op_shapes = k.get("op_shapes") or []
        op_dtypes = k.get("op_dtypes") or []

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
            "input_shapes": [op_shapes] if op_shapes else [],
            "input_dtypes": op_dtypes,
            "shape_provenance": "torch_trace" if op_shapes else "unresolved",
        }
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
        kc = classify_kernel(k.get("name", "") or "")
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


def render_analysis_md(
    candidates: dict[str, Any],
    analyze_out: dict[str, Any],
    *,
    model_name: str,
    framework: str,
    target_platform: str,
    throughput_unit: str = "tok/s",
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
        f"Not a TraceLens report; hardware roofline (bound/efficiency) is filled by "
        f"the rocprof-compute enrichment stage when available."
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
            L.append(
                f"**Impact**: {c['gpu_pct']:.2f}% of GPU time "
                f"(bound type pending rocprof-compute enrichment)."
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
        L.append("**Impact estimate:** bound type / efficiency pending rocprof-compute enrichment.")
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
            }
        )
    return {
        "source": "bypass",
        "analysis_md_path": analysis_md_path,
        "kernel_candidates_path": kernel_candidates_path,
        "kernels": rows,
    }
