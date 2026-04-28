#!/usr/bin/env python3
"""
Trace analysis for MLPerf training profiling.

Extracted from profile.md Steps 5-7 and Step 12. Provides:
  - operator_summary(): key_averages()-equivalent operator stats + NVTX phase breakdown
  - kernel_breakdown(): per-kernel GPU time aggregation
  - categorize_kernels(): map kernels to categories (gemm, attention, moe_dispatch, ...)
  - identify_geak_candidates(): find non-vendor kernels ranked by GPU time (no hard threshold)
  - compute_heuristic_adjustments(): profile-informed DFS action score multipliers
"""

import argparse
import json
import sys
from collections import defaultdict


def load_trace(trace_path: str) -> dict:
    with open(trace_path) as f:
        return json.load(f)


def operator_summary(trace: dict) -> dict:
    """Torch Profiler operator summary (key_averages equivalent)."""
    events = trace.get("traceEvents", [])

    op_stats = defaultdict(lambda: {
        "total_us": 0, "count": 0, "min_us": float("inf"), "max_us": 0
    })
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat = e.get("cat", "")
        name = e.get("name", "unknown")
        dur = e["dur"]
        if cat == "kernel":
            key = f"[CUDA] {name}"
        elif cat in ("cpu_op", "cuda_runtime"):
            key = f"[CPU] {name}"
        else:
            continue
        op_stats[key]["total_us"] += dur
        op_stats[key]["count"] += 1
        op_stats[key]["min_us"] = min(op_stats[key]["min_us"], dur)
        op_stats[key]["max_us"] = max(op_stats[key]["max_us"], dur)

    sorted_ops = sorted(op_stats.items(), key=lambda x: -x[1]["total_us"])

    phases = defaultdict(float)
    for e in events:
        if e.get("cat") in ("user_annotation", "gpu_user_annotation") and "dur" in e:
            phases[e["name"]] += e["dur"]

    cpu_total_us = sum(s["total_us"] for k, s in op_stats.items() if k.startswith("[CPU]"))
    cuda_total_us = sum(s["total_us"] for k, s in op_stats.items() if k.startswith("[CUDA]"))

    return {
        "top_cuda_ops": [
            {"name": name.removeprefix("[CUDA] "), **stats}
            for name, stats in sorted_ops if name.startswith("[CUDA]")
        ][:30],
        "top_cpu_ops": [
            {"name": name.removeprefix("[CPU] "), **stats}
            for name, stats in sorted_ops if name.startswith("[CPU]")
        ][:15],
        "phases": dict(phases),
        "cpu_total_ms": cpu_total_us / 1000,
        "cuda_total_ms": cuda_total_us / 1000,
        "total_kernel_launches": sum(
            s["count"] for k, s in op_stats.items() if k.startswith("[CUDA]")
        ),
        "unique_kernels": len([k for k in op_stats if k.startswith("[CUDA]")]),
    }


def kernel_breakdown(trace: dict) -> tuple[dict, dict, float]:
    """Per-kernel GPU time and call count. Returns (kernel_time, kernel_count, total_us)."""
    gpu_events = [
        e for e in trace.get("traceEvents", [])
        if e.get("cat") == "kernel" and "dur" in e
    ]

    kernel_time = defaultdict(float)
    kernel_count = defaultdict(int)
    for e in gpu_events:
        kernel_time[e["name"]] += e["dur"]
        kernel_count[e["name"]] += 1

    total = sum(kernel_time.values())
    return dict(kernel_time), dict(kernel_count), total


def categorize_kernels(kernel_time: dict, total: float) -> dict:
    """Classify kernels into high-level categories by GPU time percentage."""
    categories = {
        "gemm": 0, "attention": 0, "moe_dispatch": 0,
        "communication": 0, "elementwise": 0, "fp8_ops": 0, "other": 0
    }
    if total == 0:
        return categories

    for name, t in kernel_time.items():
        pct = t / total * 100
        if "Cijk_" in name or "hipblas" in name.lower():
            categories["gemm"] += pct
        elif "fmha" in name or "attn" in name.lower() or "flash" in name.lower():
            categories["attention"] += pct
        elif "permute" in name or "scatter" in name or "moe" in name.lower():
            categories["moe_dispatch"] += pct
        elif "nccl" in name.lower() or "allreduce" in name.lower() or "alltoall" in name.lower():
            categories["communication"] += pct
        elif "cast_transpose" in name or "fp8" in name.lower() or "amax" in name.lower():
            categories["fp8_ops"] += pct
        elif "elementwise" in name or "vectorized" in name:
            categories["elementwise"] += pct
        else:
            categories["other"] += pct

    return categories


MIN_CANDIDATES = 10

def identify_geak_candidates(
    kernel_time: dict, kernel_count: dict, total: float,
    tracelens_roofline: dict | None = None,
) -> list[dict]:
    """Find all non-vendor kernels ranked by GPU time. GPU% is advisory, not a hard filter."""
    candidates = []
    for name, t in sorted(kernel_time.items(), key=lambda x: -x[1])[:30]:
        pct = t / total * 100 if total > 0 else 0
        if "Cijk_" in name:
            continue
        if "aiter::" in name:
            continue
        if "nccl" in name.lower():
            continue
        bound_type = (tracelens_roofline or {}).get(name, "unknown")
        candidates.append({
            "name": name, "gpu_pct": round(pct, 2),
            "count": kernel_count.get(name, 0), "bound_type": bound_type,
            "low_gpu_pct": pct < 2.0,
        })
    if len(candidates) < MIN_CANDIDATES:
        seen = {c["name"] for c in candidates}
        for name, t in sorted(kernel_time.items(), key=lambda x: -x[1]):
            if name in seen:
                continue
            pct = t / total * 100 if total > 0 else 0
            if "Cijk_" in name or "aiter::" in name or "nccl" in name.lower():
                continue
            bound_type = (tracelens_roofline or {}).get(name, "unknown")
            candidates.append({
                "name": name, "gpu_pct": round(pct, 2),
                "count": kernel_count.get(name, 0), "bound_type": bound_type,
                "low_gpu_pct": pct < 2.0,
            })
            if len(candidates) >= MIN_CANDIDATES:
                break
    return candidates


def compute_heuristic_adjustments(categories: dict, tracelens: dict | None = None) -> dict:
    """Profile-informed DFS action score multipliers.

    Returns a dict of {action_name: multiplier} to apply to score priors.
    """
    m = {}  # multipliers: action -> cumulative factor

    def boost(action, factor):
        m[action] = m.get(action, 1.0) * factor

    if categories.get("gemm", 0) > 60:
        boost("fusion-flags", 0.7)
        boost("kernel-opt", 0.7)

    if categories.get("moe_dispatch", 0) > 5:
        boost("fusion-flags", 1.5)

    if categories.get("communication", 0) > 15:
        boost("comm-tuning", 1.3)

    if categories.get("elementwise", 0) > 5:
        boost("kernel-opt", 1.3)

    if tracelens:
        overlap = tracelens.get("comm_compute_overlap", 0)
        if overlap < 0.5:
            boost("comm-tuning", 1.5)
        elif overlap > 0.7:
            boost("comm-tuning", 0.8)

        mem_bound_pct = sum(
            v.get("gpu_pct", 0)
            for v in tracelens.get("operator_breakdown", {}).values()
            if v.get("bound_type") == "memory_bound"
        )
        if mem_bound_pct > 20:
            boost("kernel-opt", 1.5)
        if tracelens.get("mfma_utilization", 0) > 0.85:
            boost("kernel-opt", 0.7)
        if categories.get("fp8_ops", 0) > 3 or tracelens.get("fp8_cast_transpose_pct", 0) > 3:
            boost("fp8-recipe-tuning", 1.3)
        if tracelens.get("eval_overhead_pct", 0) > 5:
            boost("convergence-speed", 1.3)
        if tracelens.get("stall_pct", 0) > 5:
            boost("runtime-tunables", 1.3)

    for action in ("kernel-opt", "comm-tuning"):
        if action in m and m[action] < 0.5:
            m[action] = 0.5

    return m


def main():
    parser = argparse.ArgumentParser(description="MLPerf trace analysis")
    parser.add_argument("--trace-path", required=True, help="Path to Chrome JSON trace")
    parser.add_argument("--result-dir", default=".", help="Directory for output JSON files")
    parser.add_argument("--compute-heuristics", action="store_true",
                        help="Compute heuristic adjustments from saved categories + tracelens")
    parser.add_argument("--categories", help="Path to categories.json (for --compute-heuristics)")
    parser.add_argument("--tracelens", help="Path to tracelens_metrics.json (optional)")
    args = parser.parse_args()

    if args.compute_heuristics:
        cats = json.load(open(args.categories)) if args.categories else {}
        tl = json.load(open(args.tracelens)) if args.tracelens else None
        adjustments = compute_heuristic_adjustments(cats, tl)
        json.dump(adjustments, sys.stdout, indent=2)
        print()
        return

    trace = load_trace(args.trace_path)

    summary = operator_summary(trace)
    kt, kc, total = kernel_breakdown(trace)
    cats = categorize_kernels(kt, total)
    candidates = identify_geak_candidates(kt, kc, total)

    import os
    rd = args.result_dir
    os.makedirs(rd, exist_ok=True)

    with open(os.path.join(rd, "profiler_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(rd, "categories.json"), "w") as f:
        json.dump(cats, f, indent=2)
    with open(os.path.join(rd, "geak_candidates.json"), "w") as f:
        json.dump(candidates, f, indent=2)
    with open(os.path.join(rd, "kernel_breakdown.json"), "w") as f:
        top20 = sorted(kt.items(), key=lambda x: -x[1])[:30]
        json.dump([
            {"name": n, "time_us": t, "pct": round(t / total * 100, 2) if total else 0,
             "count": kc.get(n, 0)}
            for n, t in top20
        ], f, indent=2)

    print(f"CPU: {summary['cpu_total_ms']:.1f}ms | "
          f"CUDA: {summary['cuda_total_ms']:.1f}ms | "
          f"Kernels: {summary['total_kernel_launches']} launches, "
          f"{summary['unique_kernels']} unique")
    print(f"\nCategories: { {k: round(v, 1) for k, v in cats.items()} }")
    print(f"GEAK candidates: {len(candidates)}")
    for c in candidates:
        print(f"  {c['name'][:60]:60s}  {c['gpu_pct']:5.1f}%  {c['count']:4d}x  {c['bound_type']}")


if __name__ == "__main__":
    main()
