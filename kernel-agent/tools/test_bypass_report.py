###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the bypass artifact builders (_bypass_report)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    assert cands["skipped_kernels"] and cands["skipped_kernels"][0]["name"] == "aten::mm"


def test_build_summary_counts():
    cands = report.build_candidates(_analyze([dict(k) for k in _KERNELS]), framework="vllm", target_platform="MI300X")
    summ = report.build_summary(cands, framework="vllm", target_platform="MI300X", generated_at="2026-01-01T00:00:00")
    assert summ["task_count"] == 1
    assert summ["skipped_count"] == 1
    assert summ["tasks"][0]["recommended_backends"]
    assert "skip_reason" in summ["skipped"][0]


def test_build_kernel_roofline_shape():
    cands = report.build_candidates(_analyze([dict(k) for k in _KERNELS]), framework="vllm", target_platform="MI300X")
    kr = report.build_kernel_roofline(cands, analysis_md_path="/x/analysis.md", kernel_candidates_path="/x/kc.json")
    assert kr["source"] == "bypass"
    rows = kr["kernels"]
    assert len(rows) == 2
    assert all("kernel_id" in r for r in rows)
    assert all(r["rocprof_roofline"] is None for r in rows)


def test_roofline_rows_flag_placeholder_not_measured():
    # The bypass roofline is a structural placeholder (bound_type "—", AI/util
    # null/0) until the opt-in rocprof-compute enrichment runs. Mark it
    # ``roofline_measured=False`` so record_trace_analyze / the LLM don't mistake
    # the "—" bound for a real measured roofline.
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
        "# Bypass Analysis Report - Llama",
        "## Executive Summary",
        "## Top Operations",
        "## Compute Kernel Optimizations",
        "## System-Level Signals",
        "## Detailed Analysis",
        "## Appendix",
    ):
        assert header in md, header
    assert "throughput_unit=tok/s" in md
    assert "Throughput unit: tok/s" in md
    # routable SDPA candidate rendered as a P-item
    assert "P1:" in md


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


# ── source resolution + shape population (B/C) ───────────────────────────────


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
    # Regression: the orchestrator kernel-opt shape gate
    # (_validate_kernel_shape_and_paths in kernel_request_handlers.py) reads
    # candidate["shapes"] and rejects dispatch with error_class
    # "empty_kernel_shape" when it is missing/empty — even if input_shapes was
    # captured. A routable candidate with real trace-captured dims MUST expose a
    # non-empty "shapes" list (trusted provenance), or bypass candidates can
    # never reach GEAK optimization.
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
    assert out == [{"call_num": 5, "shape": "(4,1024) bf16<br>(1024,) f32"}]
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


# ── task_groups (M3) ─────────────────────────────────────────────────────────


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
