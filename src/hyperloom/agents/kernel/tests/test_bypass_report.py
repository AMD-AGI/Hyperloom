###############################################################################
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the bypass artifact builders (_bypass_report)."""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _bypass_report as report  # noqa: E402


def _analyze(kernels):
    total = sum(k["gpu_time_us"] for k in kernels) or 1.0
    for k in kernels:
        k.setdefault("gpu_pct", round(k["gpu_time_us"] / total * 100.0, 4))
        k.setdefault("count", 1)
        k.setdefault("op_name", "")
    return {
        "status": "ok",
        "aggregation_scope": "full_trace",
        "event_total": 42,
        "timeline": {"total_time_ms": 1.0, "busy_pct": 80.0, "idle_pct": 20.0, "gpu_memcpy_ms": 0.05},
        "attribution": {"attributed_pct": 40.0, "attributed_kernels": 1, "kernel_count": 2},
        "ops": [],
        "kernels": kernels,
    }


_KERNELS = [
    {"name": "paged_attention_v1", "op_name": "aten::paged_attn", "gpu_time_us": 300.0, "count": 3},
    {"name": "Cijk_Alik_Bljk_HHS", "op_name": "aten::mm", "gpu_time_us": 200.0, "count": 2},
]


def test_build_candidates_routing():
    cands = report.build_candidates(_analyze([dict(k) for k in _KERNELS]), framework="vllm", target_platform="MI300X")
    hot = cands["hot_kernels"]
    assert len(hot) == 2
    by_name = {c["name"]: c for c in hot}
    # display name prefers the op name
    assert "aten::paged_attn" in by_name
    attn = by_name["aten::paged_attn"]
    assert attn["kernel_category"] == "SDPA"
    assert attn["reusable_native_kernel"] is True
    assert attn["kernel_id"] == "k001"

    gemm = by_name["aten::mm"]
    assert gemm["kernel_category"] == "GEMM"
    assert gemm["reusable_native_kernel"] is False
    # Non-reusable vendor GEMM is always skipped.
    assert any(c["name"] == "aten::mm" for c in cands["skipped_kernels"])


def test_build_workload_roofline_totals_covers_all_kernels():
    # 20 kernels > any top-k cap: the workload totals must sum every device
    # kernel, not just the top-k candidate list.
    kernels = [
        {"name": "aten::mm", "op_name": "aten::mm", "gpu_time_us": float(100 - i), "count": 1}
        for i in range(20)
    ]
    analyze = _analyze([dict(k) for k in kernels])
    totals = report.build_workload_roofline_totals(analyze, target_platform="MI300X")
    assert totals["sigma_actual_kernel_us"] == round(sum(k["gpu_time_us"] for k in kernels), 3)
    for key in (
        "sigma_ideal_roofline_us", "kernel_roofline_efficiency",
        "compute_bound_us", "memory_bound_us", "no_perf_model_us",
    ):
        assert key in totals


def test_build_workload_roofline_totals_splits_compute_and_memory():
    # With real shapes the workload totals must populate the attainment-weighted
    # sigma_ideal AND the compute/memory split (not all no_perf_model).
    kernels = [
        {"name": "Cijk_Alik_Bljk_HHS", "op_name": "aten::mm", "gpu_time_us": 500.0, "count": 1,
         "op_shapes": [[4096, 4096], [4096, 4096]], "op_dtypes": ["c10::BFloat16", "c10::BFloat16"]},
        {"name": "vectorized_elementwise_kernel", "op_name": "aten::add", "gpu_time_us": 200.0, "count": 1,
         "op_shapes": [[4096, 4096], [4096, 4096]], "op_dtypes": ["c10::BFloat16", "c10::BFloat16"]},
    ]
    analyze = _analyze([dict(k) for k in kernels])
    t = report.build_workload_roofline_totals(analyze, target_platform="mi300x")
    assert t["sigma_actual_kernel_us"] == 700.0
    assert t["compute_bound_us"] > 0.0   # the GEMM
    assert t["memory_bound_us"] > 0.0    # the elementwise
    assert t["sigma_ideal_roofline_us"] > 0.0
    assert 0.0 < t["kernel_roofline_efficiency"] <= 1.0


def test_build_candidates_exposes_routable_subset():
    # hot_kernels stays the full ranked set; the reusable dispatch subset is
    # exposed separately as routable_kernels.
    cands = report.build_candidates(_analyze([dict(k) for k in _KERNELS]), framework="vllm", target_platform="MI300X")
    hot = cands["hot_kernels"]
    routable = cands["routable_kernels"]
    assert len(hot) == 2  # full (routable + non-routable)
    assert len(routable) <= len(hot)
    assert all(c.get("reusable_native_kernel") and c.get("source_file") for c in routable)
    assert not any(c["name"] == "aten::mm" for c in routable)  # non-reusable never routable


def test_build_candidates_partition_covers_reusable_without_source(monkeypatch):
    # A reusable kernel whose source is unresolved is not dispatchable, so it must
    # land on the ``skipped_kernels`` side of the partition, never in neither
    # bucket. Force source resolution to fail so the reusable SDPA kernel is
    # guaranteed source-less regardless of the ambient op_to_source table.
    monkeypatch.setattr(report, "editable_trace_source", lambda *a, **k: "")
    monkeypatch.setattr(report, "resolve_source", lambda *a, **k: ("", "unresolved"))
    cands = report.build_candidates(
        _analyze([dict(k) for k in _KERNELS]), framework="vllm", target_platform="MI300X"
    )
    hot_ids = {c["kernel_id"] for c in cands["hot_kernels"]}
    routable_ids = {c["kernel_id"] for c in cands["routable_kernels"]}
    skipped_ids = {c["kernel_id"] for c in cands["skipped_kernels"]}

    # Guard: the fixture really did produce a reusable-but-source-less kernel.
    gap = [
        c
        for c in cands["hot_kernels"]
        if c.get("reusable_native_kernel") and not c.get("source_file")
    ]
    assert gap, "fixture must contain a reusable kernel with unresolved source"
    # Such a kernel is dispatch-blocked -> skipped with an explicit reason.
    assert all(c["kernel_id"] in skipped_ids for c in gap)
    assert all(c["skip_reason"] == "source file not resolved" for c in gap)

    # Partition invariant: hot == routable + skipped, no overlap, no leakage.
    assert routable_ids | skipped_ids == hot_ids
    assert routable_ids & skipped_ids == set()
    assert len(cands["hot_kernels"]) == len(cands["routable_kernels"]) + len(
        cands["skipped_kernels"]
    )


def test_build_summary_counts(monkeypatch):
    # summary.json mirrors kernel_candidates.json's routable/skipped partition:
    # tasks == routable, skipped == the rest. Resolve the reusable SDPA kernel's
    # source so it is a task; the non-reusable GEMM stays skipped.
    monkeypatch.setattr(report, "resolve_source", lambda *a, **k: ("/src/paged_attn.py", "op_to_source"))
    cands = report.build_candidates(_analyze([dict(k) for k in _KERNELS]), framework="vllm", target_platform="MI300X")
    summ = report.build_summary(cands, framework="vllm", target_platform="MI300X", generated_at="2026-01-01T00:00:00")
    assert summ["task_count"] == 1
    assert summ["skipped_count"] == 1
    assert summ["tasks"][0]["recommended_backends"]
    assert "skip_reason" in summ["skipped"][0]
    # summary partition stays coherent with the kernel_candidates.json partition.
    assert summ["task_count"] == len(cands["routable_kernels"])
    assert summ["skipped_count"] == len(cands["skipped_kernels"])


def test_build_kernel_roofline_shape():
    cands = report.build_candidates(_analyze([dict(k) for k in _KERNELS]), framework="vllm", target_platform="MI300X")
    kr = report.build_kernel_roofline(cands, analysis_md_path="/x/analysis.md", kernel_candidates_path="/x/kc.json")
    assert kr["source"] == "bypass"
    rows = kr["kernels"]
    assert len(rows) == 2
    assert all("kernel_id" in r for r in rows)
    assert all(r["rocprof_roofline"] is None for r in rows)
    # rows are a superset of the TraceLens kernel_roofline row schema.
    tracelens_row_keys = {
        "kernel_id", "name", "gpu_pct", "duration_us", "call_count", "kernel_category",
        "source_file", "bottleneck", "bound_type", "arithmetic_intensity", "flops_per_byte",
        "efficiency_percent", "compute_utilization_pct", "bandwidth_utilization_pct",
        "suggestion", "roofline_name", "recommended_actions", "reusable_native_kernel",
        "rocprof_roofline",
    }
    for r in rows:
        assert tracelens_row_keys.issubset(r.keys())
        # bypass superset also exposes the binding-side roofline attainment.
        assert "roofline_attainment_pct" in r
        # bottleneck falls back to bound_type; recommended_actions is always a list.
        assert r["bottleneck"] == (r.get("bottleneck") or r["bound_type"])
        assert isinstance(r["recommended_actions"], list)
        assert "roofline_name" in r  # present even when null (no analytical analogue)


def test_build_candidates_fills_analytical_roofline_incl_vendor():
    # A vendor GEMM (Cijk_, non-reusable) with captured shapes gets an analytical
    # bound purely from shapes + measured time, not the "—" placeholder.
    kernels = [{
        "name": "Cijk_Alik_Bljk_HHS", "op_name": "aten::mm",
        "gpu_time_us": 500.0, "count": 1,
        "op_shapes": [[4096, 4096], [4096, 4096]], "op_dtypes": ["c10::BFloat16", "c10::BFloat16"],
    }]
    cand = report.build_candidates(_analyze(kernels), framework="xdit", target_platform="mi300x")["hot_kernels"][0]
    assert cand["reusable_native_kernel"] is False  # vendor, non-rewritable
    assert cand["bound_type"] == "compute_bound"
    assert cand["arithmetic_intensity"] > 1000
    assert cand["roofline_source"] == "analytical"
    assert cand["roofline_measured"] is False  # analytical, not hardware-measured
    assert 0.0 < cand["efficiency_percent"] <= 100.0


def test_build_candidates_no_shapes_stays_placeholder_roofline():
    kernels = [{"name": "mystery_kernel", "op_name": "aten::mystery", "gpu_time_us": 10.0, "count": 1}]
    cand = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="mi300x")["hot_kernels"][0]
    assert cand["bound_type"] == "\u2014"
    assert cand["roofline_source"] == "placeholder"
    assert cand["arithmetic_intensity"] is None


def test_optimization_priority_and_suggestion_are_attributable():
    kernels = [{
        "name": "Cijk_Alik_Bljk_HHS", "op_name": "aten::mm",
        "gpu_time_us": 500.0, "count": 1,
        "op_shapes": [[4096, 4096], [4096, 4096]], "op_dtypes": ["c10::BFloat16", "c10::BFloat16"],
    }]
    cand = report.build_candidates(_analyze(kernels), framework="xdit", target_platform="mi300x")["hot_kernels"][0]
    # priority = gpu_pct * (1 - eff/100): reproducible from two columns in the row.
    expected = round(cand["gpu_pct"] * (1.0 - cand["efficiency_percent"] / 100.0), 4)
    assert cand["optimization_priority"] == expected
    assert cand["priority_rank"] == 1
    # 4096^3 GEMM is compute-bound -> suggestion carries the bound prefix + GEMM action.
    assert cand["suggestion"].startswith("Compute-bound:")
    assert cand["recommended_actions"] == [cand["suggestion"]]


def test_priority_falls_back_to_gpu_pct_without_efficiency():
    kernels = [{"name": "mystery_kernel", "op_name": "aten::mystery", "gpu_time_us": 10.0, "count": 1}]
    cand = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="mi300x")["hot_kernels"][0]
    # No analytical efficiency -> headroom 1 -> priority == gpu_pct; no bound prefix.
    assert cand["optimization_priority"] == round(cand["gpu_pct"], 4)
    assert not cand["suggestion"].startswith(("Compute-bound", "Memory-bound"))


def test_priority_rank_orders_by_roi():
    kernels = [
        {"name": "big_elementwise_kernel", "op_name": "aten::add", "gpu_time_us": 900.0, "count": 1,
         "op_shapes": [[8192, 8192], [8192, 8192]], "op_dtypes": ["c10::BFloat16", "c10::BFloat16"]},
        {"name": "small_mm_kernel", "op_name": "aten::mm", "gpu_time_us": 100.0, "count": 1,
         "op_shapes": [[4096, 4096], [4096, 4096]], "op_dtypes": ["c10::BFloat16", "c10::BFloat16"]},
    ]
    hot = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="mi300x")["hot_kernels"]
    # The big high-share kernel out-ranks the small one by ROI.
    big = max(hot, key=lambda c: c["duration_us"])
    assert big["priority_rank"] == 1


def test_metrics_csv_has_all_kernels_and_columns():
    kernels = [
        {"name": "Cijk_x", "op_name": "aten::mm", "gpu_time_us": 500.0, "count": 1,
         "op_shapes": [[4096, 4096], [4096, 4096]], "op_dtypes": ["c10::BFloat16", "c10::BFloat16"]},
        {"name": "mystery", "op_name": "aten::mystery", "gpu_time_us": 100.0, "count": 2},
    ]
    cands = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="mi300x")
    rows = list(csv.DictReader(io.StringIO(report.build_metrics_csv(cands))))
    assert len(rows) == 2  # all hot kernels, incl. the non-routable mystery one
    assert set(report._METRICS_COLUMNS).issubset(rows[0].keys())
    assert all(r["suggestion"] for r in rows)  # deterministic hint present
    assert all(r["optimization_priority"] for r in rows)


def test_category_summary_aggregates_by_category():
    kernels = [
        {"name": "Cijk_a", "op_name": "aten::mm", "gpu_time_us": 300.0, "count": 1,
         "op_shapes": [[4096, 4096], [4096, 4096]], "op_dtypes": ["c10::BFloat16", "c10::BFloat16"]},
        {"name": "Cijk_b", "op_name": "aten::mm", "gpu_time_us": 300.0, "count": 1,
         "op_shapes": [[4096, 4096], [4096, 4096]], "op_dtypes": ["c10::BFloat16", "c10::BFloat16"]},
        {"name": "add_elementwise", "op_name": "aten::add", "gpu_time_us": 100.0, "count": 1,
         "op_shapes": [[1024, 1024], [1024, 1024]], "op_dtypes": ["c10::BFloat16", "c10::BFloat16"]},
    ]
    cands = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="mi300x")
    summ = report.build_category_summary(cands)
    by_cat = {r["kernel_category"]: r for r in summ}
    assert by_cat["GEMM"]["kernel_count"] == 2
    assert by_cat["GEMM"]["total_gpu_pct"] > by_cat["Elementwise"]["total_gpu_pct"]
    assert summ[0]["total_gpu_pct"] >= summ[-1]["total_gpu_pct"]  # GPU%-descending
    assert by_cat["GEMM"]["dominant_bound_type"] in ("compute_bound", "memory_bound")


def test_summary_csv_parseable():
    kernels = [{"name": "Cijk_a", "op_name": "aten::mm", "gpu_time_us": 300.0, "count": 1,
                "op_shapes": [[4096, 4096], [4096, 4096]], "op_dtypes": ["c10::BFloat16", "c10::BFloat16"]}]
    cands = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="mi300x")
    rows = list(csv.DictReader(io.StringIO(report.build_category_summary_csv(cands))))
    assert rows and rows[0]["kernel_category"] == "GEMM"


def test_roofline_rows_flag_placeholder_not_measured():
    # The bypass roofline is analytical, not a hardware measurement. Mark it
    # ``roofline_measured=False`` so record_trace_analyze / the LLM don't
    # mistake the "—" bound for a real measured roofline.
    cands = report.build_candidates(_analyze([dict(k) for k in _KERNELS]), framework="vllm", target_platform="MI300X")
    kr = report.build_kernel_roofline(cands, analysis_md_path="/x/a.md", kernel_candidates_path="/x/kc.json")
    assert kr["kernels"]
    for r in kr["kernels"]:
        assert r.get("roofline_measured") is False
        assert r["bound_type"] == "\u2014"  # em-dash "unknown" marker
    # candidates carry the same honest marker.
    assert all(c.get("roofline_measured") is False for c in cands["hot_kernels"])


def test_render_analysis_md_sections_textgen():
    analyze = _analyze([dict(k) for k in _KERNELS])
    cands = report.build_candidates(analyze, framework="vllm", target_platform="MI300X")
    md = report.render_analysis_md(cands, analyze, model_name="Llama", framework="vllm", target_platform="MI300X")
    for header in (
        # Shared canonical spine (identical across routes).
        "# Performance Analysis Report \u2014 Llama",
        "## Executive Summary",
        "## System-Level Signals",
        "## Top Hot Kernels",
        # bypass-only richer sections appended after the divider.
        "## Top Operations",
        "## Compute Kernel Optimizations",
        "## Detailed Analysis",
        "## Appendix",
    ):
        assert header in md, header
    assert "> Generated via bypass route (HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass)." in md
    # shared System-Level Signals is now a table, not bullets
    assert "| Signal | % of total GPU time | Note |" in md
    # divider that separates the shared spine from bypass-only extras
    assert "Additional route-specific detail below" in md
    assert "throughput_unit=tok/s" in md
    assert "Throughput unit: tok/s" in md
    # routable SDPA candidate rendered as a P-item
    assert "P1:" in md


def test_render_analysis_md_top10_and_csv_and_no_stale_text():
    kernels = [{
        "name": "Cijk_x", "op_name": "aten::mm", "gpu_time_us": 500.0, "count": 1,
        "op_shapes": [[4096, 4096], [4096, 4096]], "op_dtypes": ["c10::BFloat16", "c10::BFloat16"],
    }]
    analyze = _analyze(kernels)
    cands = report.build_candidates(analyze, framework="vllm", target_platform="mi300x")
    md = report.render_analysis_md(
        cands, analyze, model_name="M", framework="vllm", target_platform="mi300x",
        metrics_csv_path="/x/kernel_metrics.csv", summary_csv_path="/x/kernel_summary.csv",
    )
    assert "## Top 10 Kernels by Optimization Priority" in md
    assert "| # | kernel_id | Name | Category | GPU% | Bound | AI | Eff% | Priority | Suggestion |" in md
    assert "Analytical roofline bound:" in md
    assert "## Structured Metrics (CSV)" in md
    assert "kernel_metrics.csv" in md and "kernel_summary.csv" in md
    # stale placeholder phrasing must be gone (bound now analytical)
    assert "pending rocprof" not in md


def test_render_analysis_md_xdit_unit():
    analyze = _analyze([dict(k) for k in _KERNELS])
    cands = report.build_candidates(analyze, framework="xdit", target_platform="MI300X")
    md = report.render_analysis_md(
        cands, analyze, model_name="FLUX", framework="xdit", target_platform="MI300X", throughput_unit="img/s"
    )
    assert "throughput_unit=img/s" in md
    assert "Throughput unit: img/s" in md


def test_render_empty_kernels_is_valid():
    analyze = _analyze([])
    cands = report.build_candidates(analyze, framework="vllm", target_platform="MI300X")
    assert cands["hot_kernels"] == []
    md = report.render_analysis_md(cands, analyze, model_name="Empty", framework="vllm", target_platform="MI300X")
    assert "_No GPU kernels found in trace._" in md
    assert "_No rewritable compute-kernel candidates identified._" in md


# ── source resolution + shape population ─────────────────────────────────────


def test_source_file_from_trace_kernel_file_wins(monkeypatch):
    # A repo Triton kernel_file from the trace resolves source_file directly and
    # must take priority over the op_to_source dictionary (never consulted here).
    def _boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("resolve_source must not run when trace kernel_file hits")

    monkeypatch.setattr(report, "resolve_source", _boom)
    kernels = [{
        "name": "triton_silu", "op_name": "aten::silu", "gpu_time_us": 100.0, "count": 1,
        "op_kernel_file": "/repo/aiter/triton/silu.py", "op_kernel_backend": "triton",
        "op_shapes": [[8, 16]], "op_dtypes": ["c10::BFloat16"],
    }]
    cand = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="MI300X")["hot_kernels"][0]
    assert cand["source_file"] == "/repo/aiter/triton/silu.py"
    assert cand["source_resolution_method"] == "trace_kernel_file"
    # shapes flow through in the downstream contract form (one call, "(dims) dtype").
    assert cand["input_shapes"] == [{"call_num": 1, "shape": "(8,16) bf16"}]
    assert cand["input_dtypes"] == ["c10::BFloat16"]
    assert cand["shape_provenance"] == "torch_trace"


def test_routable_candidate_carries_shapes_for_orchestrator_gate():
    # The orchestrator shape gate reads candidate["shapes"] and rejects dispatch
    # with "empty_kernel_shape" when empty, so a routable candidate with real
    # trace-captured dims must expose a non-empty "shapes" list.
    kernels = [{
        "name": "triton_silu", "op_name": "aten::silu", "gpu_time_us": 100.0, "count": 1,
        "op_kernel_file": "/repo/aiter/triton/silu.py", "op_kernel_backend": "triton",
        "op_shapes": [[8, 16]], "op_dtypes": ["c10::BFloat16"],
    }]
    cand = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="MI300X")["hot_kernels"][0]
    shapes = cand.get("shapes")
    assert isinstance(shapes, list) and shapes, "routable candidate must expose a non-empty 'shapes' for the orchestrator gate"
    assert cand["shape_provenance"] in {"torch_trace", "tuning_csv"}
    # shapes mirrors input_shapes in the harness-consumable contract form.
    assert cand["shapes"] == cand["input_shapes"] == [{"call_num": 1, "shape": "(8,16) bf16"}]
    # each entry is a dict the harness can parse (not a raw dim list).
    assert isinstance(cand["shapes"][0], dict) and "shape" in cand["shapes"][0]


def test_trace_shape_entries_contract_format():
    # Kineto Input Dims + Input type -> downstream contract string:
    # multi-operand <br>-joined "(dims) dtype", 1-D keeps trailing comma,
    # scalar/empty operand dropped, call_count stamped.
    out = report._trace_shape_entries([[4, 1024], [1024], []], ["c10::BFloat16", "float", "int"], 5)
    assert out == [{"call_num": 5, "shape": "(4,1024) bf16<br>(1024,) fp32"}]
    # unmapped dtype -> bare shape (no suffix).
    assert report._trace_shape_entries([[8, 8]], ["weird"], 1) == [{"call_num": 1, "shape": "(8,8)"}]
    # no renderable operand -> empty (gate will reject as empty_kernel_shape).
    assert report._trace_shape_entries([[]], ["float"], 1) == []
    assert report._trace_shape_entries([], [], 1) == []


def test_unresolved_shape_candidate_has_empty_shapes():
    # A kernel with no captured dims stays shape-less: "shapes" is an empty list
    # (present, not absent) and the gate will correctly reject it as
    # empty_kernel_shape.
    kernels = [{"name": "mystery_kernel", "op_name": "aten::mystery", "gpu_time_us": 100.0, "count": 1}]
    cand = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="MI300X")["hot_kernels"][0]
    assert cand.get("shapes") == []
    assert cand["input_shapes"] == []
    assert cand["shape_provenance"] == "unresolved"


def test_source_file_from_op_to_source_when_no_kernel_file(monkeypatch):
    monkeypatch.setattr(report, "resolve_source", lambda op, **k: ("/opt/aiter/csrc/act.cu", "op_to_source"))
    kernels = [{"name": "act_kernel", "op_name": "_C::silu_and_mul", "gpu_time_us": 100.0, "count": 1}]
    cand = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="MI300X")["hot_kernels"][0]
    assert cand["source_file"] == "/opt/aiter/csrc/act.cu"
    assert cand["source_resolution_method"] == "op_to_source"


def test_source_unresolved_when_both_miss(monkeypatch):
    monkeypatch.setattr(report, "resolve_source", lambda op, **k: ("", "unresolved"))
    kernels = [{"name": "mystery_kernel", "op_name": "aten::mystery", "gpu_time_us": 100.0, "count": 1}]
    cand = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="MI300X")["hot_kernels"][0]
    assert cand["source_file"] == ""
    assert cand["source_resolution_method"] == "unresolved"
    # no shapes -> input_shapes empty + unresolved provenance
    assert cand["input_shapes"] == []
    assert cand["shape_provenance"] == "unresolved"


def test_inductor_kernel_file_rejected_falls_through(monkeypatch):
    # A /tmp inductor kernel_file is not editable; must fall through to lookup.
    monkeypatch.setattr(report, "resolve_source", lambda op, **k: ("", "unresolved"))
    kernels = [{
        "name": "triton_poi_fused", "op_name": "aten::add", "gpu_time_us": 100.0, "count": 1,
        "op_kernel_file": "/tmp/torchinductor_root/cabc.py",
    }]
    cand = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="MI300X")["hot_kernels"][0]
    assert cand["source_file"] == ""
    assert cand["source_resolution_method"] == "unresolved"


# ── task_groups ──────────────────────────────────────────────────────────────


def _cand(kernel_id, name, source_file, *, reusable=True, dur=100.0, gpu_pct=10.0, shapes=None, count=1):
    """Build a minimal candidate row shaped like build_candidates output."""
    return {
        "kernel_id": kernel_id,
        "name": name,
        "device_kernel_name": name + "_dev",
        "source_file": source_file,
        "reusable_native_kernel": reusable,
        "duration_us": dur,
        "percent_of_total": gpu_pct,
        "gpu_pct": gpu_pct,
        "call_count": count,
        "bound_type": "\u2014",
        "input_shapes": [shapes] if shapes else [],
    }


def test_task_groups_merge_same_native_source():
    hot = [
        _cand("k001", "quant_a", "/opt/aiter/csrc/quant_kernels.cu", dur=300.0, gpu_pct=30.0, shapes=[[8, 8]]),
        _cand("k002", "quant_b", "/opt/aiter/csrc/quant_kernels.cu", dur=100.0, gpu_pct=10.0, shapes=[[4, 4]]),
    ]
    groups = report._build_task_groups(hot)
    assert len(groups) == 1
    g = groups[0]
    assert g["task_group_id"] == "tg001"
    assert set(g["kernel_ids"]) == {"k001", "k002"}
    # heaviest row is primary + first
    assert g["primary_kernel_id"] == "k001"
    assert g["rows"][0]["kernel_id"] == "k001"
    assert g["aggregate_duration_us"] == 400.0
    assert g["aggregate_gpu_pct"] == 40.0
    # rows carry harness-consumable shapes
    assert g["rows"][0]["shapes"] == [[8, 8]]


def test_task_groups_py_keys_on_operation():
    hot = [
        _cand("k001", "op_x", "/repo/triton/fused.py", dur=100.0),
        _cand("k002", "op_x", "/repo/triton/fused.py", dur=50.0),   # same file+op -> merge
        _cand("k003", "op_y", "/repo/triton/fused.py", dur=40.0),   # same file, diff op -> separate
    ]
    groups = report._build_task_groups(hot)
    by_op = {g["operation"]: g for g in groups}
    assert set(by_op) == {"op_x", "op_y"}
    assert set(by_op["op_x"]["kernel_ids"]) == {"k001", "k002"}
    assert by_op["op_y"]["kernel_ids"] == ["k003"]


def test_task_groups_skip_unresolved_and_nonreusable():
    hot = [
        _cand("k001", "no_src", "", dur=500.0),                       # unresolved source -> skip
        _cand("k002", "vendor", "/x/g.cu", reusable=False, dur=400.0),  # not reusable -> skip
        _cand("k003", "ok", "/x/act.cu", dur=100.0),                  # routable -> grouped
    ]
    groups = report._build_task_groups(hot)
    assert len(groups) == 1
    assert groups[0]["kernel_ids"] == ["k003"]


def test_task_groups_ordered_by_aggregate_time():
    hot = [
        _cand("k001", "small", "/x/a.cu", dur=50.0),
        _cand("k002", "big", "/x/b.cu", dur=900.0),
    ]
    groups = report._build_task_groups(hot)
    assert [g["source_path"] for g in groups] == ["/x/b.cu", "/x/a.cu"]
    assert groups[0]["task_group_id"] == "tg001"


def test_render_surfaces_source_dispatchability_and_task_groups(monkeypatch):
    # One candidate resolves a source, one does not -> report must reflect the
    # real dispatchable split, show the source line, and render a Task Groups table.
    monkeypatch.setattr(
        report, "resolve_source",
        lambda op, **k: ("/opt/aiter/csrc/act.cu", "op_to_source") if op == "aiter::act" else ("", "unresolved"),
    )
    kernels = [
        {"name": "aiter_act_kernel", "op_name": "aiter::act", "gpu_time_us": 300.0, "count": 3},
        {"name": "aiter_act_kernel2", "op_name": "aiter::act", "gpu_time_us": 50.0, "count": 1},  # same src -> group
        {"name": "rms_norm_kernel", "op_name": "vllm::rms_norm", "gpu_time_us": 100.0, "count": 1},  # reusable, unresolved
    ]
    analyze = _analyze(kernels)
    cands = report.build_candidates(analyze, framework="vllm", target_platform="MI300X")
    md = report.render_analysis_md(cands, analyze, model_name="M", framework="vllm", target_platform="MI300X")
    # dispatchable split line present
    assert "rewritable candidate(s) have a resolved editable source" in md
    # resolved source surfaced with method
    assert "/opt/aiter/csrc/act.cu" in md and "via op_to_source" in md
    # unresolved reusable candidate flagged as not auto-dispatchable
    assert "not auto-dispatchable" in md
    # Task Groups section rendered (act.cu group exists)
    assert "## Task Groups" in md and "act.cu" in md


def test_source_type_from_resolved_source(monkeypatch):
    monkeypatch.setattr(report, "resolve_source", lambda op, **k: ("/x/k.cu", "op_to_source"))
    kernels = [{"name": "aten::x", "op_name": "aten::x", "gpu_time_us": 100.0, "count": 1}]
    cand = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="MI300X")["hot_kernels"][0]
    assert cand["source_type"] == "hip_cpp"  # from .cu extension, not op-name heuristic


def test_source_type_python_from_trace_kernel_file():
    kernels = [{
        "name": "op", "op_name": "op", "gpu_time_us": 100.0, "count": 1,
        "op_kernel_file": "/repo/triton/k.py",
    }]
    cand = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="MI300X")["hot_kernels"][0]
    assert cand["source_file"] == "/repo/triton/k.py"
    assert cand["source_type"] == "python"


def test_build_candidates_discovers_benchmark_files_when_enabled(tmp_path, monkeypatch):
    # discover_benchmarks populates benchmark_files/kernel_repo on a routable
    # candidate (seeds the rocprof roofline enrichment); off by default.
    repo = tmp_path / "aiter"
    (repo / "op_tests").mkdir(parents=True)
    (repo / "csrc").mkdir(parents=True)
    src = repo / "csrc" / "foo_kernel.cu"
    src.write_text("// foo_op\n", encoding="utf-8")
    (repo / "op_tests" / "test_foo.py").write_text("def test():\n    foo_op(x)\n", encoding="utf-8")
    monkeypatch.setattr(report, "resolve_source", lambda op, **k: (str(src), "op_to_source"))
    base = [{"name": "triton_foo_kernel", "op_name": "aiter::foo_op", "gpu_time_us": 100.0, "count": 1}]

    on = report.build_candidates(
        _analyze([dict(k) for k in base]), framework="vllm", target_platform="MI300X", discover_benchmarks=True
    )["hot_kernels"][0]
    assert on["reusable_native_kernel"] and on["source_file"] == str(src)
    assert any(Path(f).name == "test_foo.py" for f in on["benchmark_files"])
    assert on["kernel_repo"] == str(repo.resolve())

    off = report.build_candidates(
        _analyze([dict(k) for k in base]), framework="vllm", target_platform="MI300X"
    )["hot_kernels"][0]
    assert off["benchmark_files"] == [] and off["kernel_repo"] == ""


def test_build_candidates_attaches_task_group_and_summary_counts(monkeypatch):
    # Two candidates resolve to the same .cu -> one group attached to both.
    monkeypatch.setattr(report, "resolve_source", lambda op, **k: ("/opt/aiter/csrc/quant.cu", "op_to_source"))
    kernels = [
        {"name": "aiter::quant_a", "op_name": "aiter::quant_a", "gpu_time_us": 300.0, "count": 2},
        {"name": "aiter::quant_b", "op_name": "aiter::quant_b", "gpu_time_us": 100.0, "count": 1},
    ]
    cands = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="MI300X")
    assert len(cands["task_groups"]) == 1
    for c in cands["hot_kernels"]:
        assert c.get("task_group", {}).get("task_group_id") == "tg001"
    summ = report.build_summary(cands, framework="vllm", target_platform="MI300X", generated_at="t")
    assert summ["task_group_count"] == 1
    entry = summ["task_groups"][0]
    assert entry["row_count"] == 2 and entry["primary_kernel_id"]
    # compact projection: no full rows leak into the summary entry
    assert "rows" not in entry
