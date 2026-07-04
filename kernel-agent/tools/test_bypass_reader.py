###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the bypass streaming Kineto reader (_bypass_trace_reader).

Builds a tiny hand-authored Kineto trace so the streaming parser, correlation
attribution, timeline union math, and annotation-window extraction are all
covered deterministically without any large fixture.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bypass_trace_reader as reader  # noqa: E402

# A minimal but representative trace:
#  - one attributed GEMM kernel (Cijk, corr 5 -> aten::mm)
#  - one cudagraph-replay-style unlinked SDPA kernel (corr 999, no runtime)
#  - one device memcpy
#  - one ProfilerStep annotation window
_TRACE_EVENTS = [
    {"cat": "cpu_op", "name": "aten::mm", "args": {"External id": 100}},
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 5, "External id": 100}},
    {"cat": "kernel", "ph": "X", "name": "Cijk_Alik_Bljk_HHS", "ts": 1000, "dur": 200, "args": {"correlation": 5}},
    {"cat": "kernel", "ph": "X", "name": "paged_attention_v1", "ts": 1300, "dur": 300, "args": {"correlation": 999}},
    {"cat": "gpu_memcpy", "ph": "X", "name": "Memcpy DtoH", "ts": 1700, "dur": 50},
    {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#1", "ts": 1000, "dur": 700},
]


def _write_trace(path: Path, gz: bool = False) -> Path:
    payload = json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8")
    if gz:
        with gzip.open(path, "wb") as f:
            f.write(payload)
    else:
        path.write_bytes(payload)
    return path


def test_analyze_basic_aggregates(tmp_path):
    tf = _write_trace(tmp_path / "t.trace.json")
    out = reader.analyze_trace(tf, top_k=0)
    assert out["status"] == "ok"
    assert out["event_total"] == 6

    kernels = {k["name"]: k for k in out["kernels"]}
    assert set(kernels) == {"Cijk_Alik_Bljk_HHS", "paged_attention_v1"}
    # denom = total kernel us (500); pct 60/40.
    assert kernels["paged_attention_v1"]["gpu_pct"] == 60.0
    assert kernels["Cijk_Alik_Bljk_HHS"]["gpu_pct"] == 40.0
    assert kernels["paged_attention_v1"]["count"] == 1


def test_correlation_attribution(tmp_path):
    tf = _write_trace(tmp_path / "t.trace.json")
    out = reader.analyze_trace(tf, top_k=0)

    attr = out["attribution"]
    assert attr["kernel_count"] == 2
    assert attr["attributed_kernels"] == 1
    assert attr["unlinked_kernels"] == 1
    assert attr["attributed_gpu_ms"] == 0.2

    kernels = {k["name"]: k for k in out["kernels"]}
    # Cijk resolved to its launching op; replay kernel stays unlinked ("").
    assert kernels["Cijk_Alik_Bljk_HHS"]["op_name"] == "aten::mm"
    assert kernels["paged_attention_v1"]["op_name"] == ""


def test_timeline_union_math(tmp_path):
    tf = _write_trace(tmp_path / "t.trace.json")
    tl = reader.analyze_trace(tf, top_k=0)["timeline"]
    assert tl["total_time_ms"] == 0.75  # (1750 - 1000) us
    assert tl["busy_time_ms"] == 0.55   # 200 + 300 + 50 us (disjoint)
    assert tl["idle_time_ms"] == 0.2
    assert tl["kernel_union_ms"] == 0.5
    assert tl["gpu_memcpy_ms"] == 0.05


def test_annotation_windows(tmp_path):
    tf = _write_trace(tmp_path / "t.trace.json")
    out = reader.analyze_trace(tf, top_k=0)
    windows = out["annotation_windows"]
    assert len(windows) == 1
    assert windows[0]["name"] == "ProfilerStep#1"
    assert windows[0]["dur"] == 700.0


def test_gzip_trace_is_transparent(tmp_path):
    tf = _write_trace(tmp_path / "t.trace.json.gz", gz=True)
    out = reader.analyze_trace(tf, top_k=0)
    assert out["status"] == "ok"
    assert len(out["kernels"]) == 2


def test_directory_input_resolves_largest_trace(tmp_path):
    d = tmp_path / "torch_trace"
    d.mkdir()
    _write_trace(d / "t.trace.json")
    resolved = reader.resolve_trace_file(d)
    assert resolved is not None and resolved.name == "t.trace.json"
    out = reader.analyze_trace(d, top_k=0)
    assert out["status"] == "ok"


def test_missing_trace_returns_failed(tmp_path):
    out = reader.analyze_trace(tmp_path / "nope.trace.json")
    assert out["status"] == "failed"
    assert "error" in out


# ── multi-rank trace selection (P0-1) ────────────────────────────────────────


def _write_ranked(d: Path, rank: int, extra_events: int = 0) -> Path:
    d.mkdir(exist_ok=True)
    events = list(_TRACE_EVENTS) + [
        {"cat": "kernel", "ph": "X", "name": f"pad_{i}", "ts": 5000 + i, "dur": 1, "args": {"correlation": 900 + i}}
        for i in range(extra_events)
    ]
    p = d / f"rank_{rank}.trace.json.gz"
    with gzip.open(p, "wb") as f:
        f.write(json.dumps({"traceEvents": events}).encode("utf-8"))
    return p


def test_resolve_trace_file_prefers_rank0_deterministically(tmp_path):
    d = tmp_path / "torch_trace"
    # rank_2 is the LARGEST; deterministic rank policy must still pick rank_0.
    _write_ranked(d, 0)
    _write_ranked(d, 1, extra_events=5)
    _write_ranked(d, 2, extra_events=50)
    resolved = reader.resolve_trace_file(d)
    assert resolved is not None and resolved.name == "rank_0.trace.json.gz"


def test_resolve_trace_file_prefers_merged(tmp_path):
    d = tmp_path / "torch_trace"
    _write_ranked(d, 0)
    _write_ranked(d, 1)
    merged = d / "merged-all.trace.json.gz"
    with gzip.open(merged, "wb") as f:
        f.write(json.dumps({"traceEvents": list(_TRACE_EVENTS)}).encode("utf-8"))
    resolved = reader.resolve_trace_file(d)
    assert resolved is not None and resolved.name == "merged-all.trace.json.gz"


def test_analyze_reports_rank_provenance(tmp_path):
    d = tmp_path / "torch_trace"
    _write_ranked(d, 0)
    _write_ranked(d, 1)
    _write_ranked(d, 2)
    out = reader.analyze_trace(d, top_k=0)
    assert out["status"] == "ok"
    assert out["analyzed_rank"] == 0
    assert out["rank_count"] == 3


def test_single_file_rank_provenance_is_none(tmp_path):
    tf = _write_trace(tmp_path / "t.trace.json")
    out = reader.analyze_trace(tf, top_k=0)
    assert out["analyzed_rank"] is None
    assert out["rank_count"] == 1


def test_top_k_limits_returned_rows(tmp_path):
    tf = _write_trace(tmp_path / "t.trace.json")
    out = reader.analyze_trace(tf, top_k=1)
    assert len(out["kernels"]) == 1
    # highest share first
    assert out["kernels"][0]["name"] == "paged_attention_v1"


def test_full_trace_scope_is_default(tmp_path):
    tf = _write_trace(tmp_path / "t.trace.json")
    out = reader.analyze_trace(tf, top_k=0)
    assert out["aggregation_scope"] == "full_trace"
    assert "steady_window" not in out


# ── steady-state windowing (M2.5) ────────────────────────────────────────────

# Three ProfilerStep windows: #1 is warm-up (dropped); a warm-up GEMM sits in
# #1, the steady attention kernel sits in #3 (the selected representative step).
_STEADY_EVENTS = [
    {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#1", "ts": 0, "dur": 100},
    {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#2", "ts": 100, "dur": 100},
    {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#3", "ts": 200, "dur": 100},
    {"cat": "kernel", "ph": "X", "name": "warmup_gemm", "ts": 50, "dur": 40, "args": {"correlation": 1}},
    {"cat": "kernel", "ph": "X", "name": "paged_attention_v1", "ts": 250, "dur": 30, "args": {"correlation": 2}},
]


def _write_steady(path: Path) -> Path:
    path.write_bytes(json.dumps({"traceEvents": _STEADY_EVENTS}).encode("utf-8"))
    return path


def test_select_steady_window_drops_warmup_and_picks_representative():
    windows = [
        {"name": "ProfilerStep#1", "ts": 0.0, "dur": 100.0},
        {"name": "ProfilerStep#2", "ts": 100.0, "dur": 100.0},
        {"name": "ProfilerStep#3", "ts": 200.0, "dur": 100.0},
    ]
    win = reader.select_steady_window(windows)
    assert win is not None
    assert win["step_name"] == "ProfilerStep"
    assert win["step_count"] == 3
    assert win["method"] == "annotation_step"
    # first (warm-up) dropped; representative window is inside the later steps.
    assert win["start_us"] >= 100.0


def test_select_steady_window_high_count_loop_not_masked_by_spurious_step():
    # A single spurious "step"-named annotation must NOT mask a real high-count
    # loop (regression: rank picked the 1-count step then rejected on threshold).
    windows = [{"name": "optimizer_step", "ts": 0.0, "dur": 5.0}] + [
        {"name": "graph_call", "ts": float(100 + i * 100), "dur": 100.0} for i in range(5)
    ]
    win = reader.select_steady_window(windows)
    assert win is not None
    assert win["step_name"] == "graph_call"
    assert win["step_count"] == 5


def test_select_steady_window_prefers_step_marker_over_higher_count():
    windows = [
        {"name": f"ProfilerStep#{i}", "ts": float(i * 100), "dur": 100.0} for i in range(1, 5)
    ] + [{"name": "elementwise_region", "ts": float(500 + i * 10), "dur": 10.0} for i in range(10)]
    win = reader.select_steady_window(windows)
    assert win is not None
    # Step marker wins over the higher-count non-step group.
    assert win["step_name"] == "ProfilerStep"
    assert win["step_count"] == 4


def test_select_steady_window_needs_enough_repeats():
    # Two generic (non-step) markers is below the default min_repeats.
    windows = [
        {"name": "region_a", "ts": 0.0, "dur": 10.0},
        {"name": "region_a", "ts": 10.0, "dur": 10.0},
    ]
    assert reader.select_steady_window(windows) is None
    # xDiT diffusion steps are homogeneous -> two repeats are enough.
    steps = [
        {"name": "denoise_step_0", "ts": 0.0, "dur": 10.0},
        {"name": "denoise_step_1", "ts": 10.0, "dur": 10.0},
    ]
    assert reader.select_steady_window(steps, framework="xdit") is not None


def test_analyze_steady_state_filters_to_window(tmp_path):
    tf = _write_steady(tmp_path / "s.trace.json")
    full = reader.analyze_trace(tf, top_k=0)
    assert full["aggregation_scope"] == "full_trace"
    assert {k["name"] for k in full["kernels"]} == {"warmup_gemm", "paged_attention_v1"}

    steady = reader.analyze_trace(tf, top_k=0, steady_state=True)
    assert steady["aggregation_scope"] == "steady_state"
    assert steady["steady_window"]["step_name"] == "ProfilerStep"
    # only the in-window kernel survives the steady filter.
    assert {k["name"] for k in steady["kernels"]} == {"paged_attention_v1"}
    assert steady["attribution"]["kernel_count"] == 1
    # Timeline is scoped to the representative step's wall span (ts 200..300us).
    tl = steady["timeline"]
    assert tl["total_time_ms"] == 0.1   # 100us window span
    assert tl["busy_time_ms"] == 0.03   # single 30us kernel, fully inside
    assert tl["idle_time_ms"] == 0.07
    assert tl["busy_pct"] == 30.0


# A single step whose kernel overhangs the window end: busy must clip to the
# window (no busy_pct > 100%), while GPU-time share uses the full kernel cost.
_OVERHANG_EVENTS = [
    {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#1", "ts": 0, "dur": 100},
    {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#2", "ts": 100, "dur": 100},
    {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#3", "ts": 200, "dur": 100},
    {"cat": "kernel", "ph": "X", "name": "long_kernel", "ts": 210, "dur": 200, "args": {"correlation": 1}},
]


def test_steady_window_clips_overhanging_kernel(tmp_path):
    tf = tmp_path / "o.trace.json"
    tf.write_bytes(json.dumps({"traceEvents": _OVERHANG_EVENTS}).encode("utf-8"))
    out = reader.analyze_trace(tf, top_k=0, steady_state=True)
    assert out["aggregation_scope"] == "steady_state"
    tl = out["timeline"]
    # window [200,300); kernel [210,410) clips to [210,300) = 90us busy.
    assert tl["total_time_ms"] == 0.1
    assert tl["busy_time_ms"] == 0.09
    assert tl["busy_pct"] == 90.0
    assert tl["busy_pct"] <= 100.0
    # GPU-time share ranks by the FULL (unclipped) kernel duration (200us).
    assert out["kernels"][0]["gpu_time_ms"] == 0.2


def test_steady_state_falls_back_to_full_when_no_windows(tmp_path):
    # The basic trace has a single ProfilerStep (no repeating loop) -> no window.
    tf = _write_trace(tmp_path / "t.trace.json")
    out = reader.analyze_trace(tf, top_k=0, steady_state=True)
    assert out["aggregation_scope"] == "full_trace"
    assert out.get("steady_window_status")
    assert len(out["kernels"]) == 2


# ── boundary / malformed inputs (P0-3) ───────────────────────────────────────


def test_non_kineto_json_yields_no_kernels(tmp_path):
    # Valid JSON without a traceEvents array: the reader must not crash and
    # must report an ok, empty result (the tool turns this into a warning).
    tf = tmp_path / "notrace.json"
    tf.write_bytes(json.dumps({"foo": "bar", "schemaVersion": 1}).encode("utf-8"))
    out = reader.analyze_trace(tf, top_k=0)
    assert out["status"] == "ok"
    assert out["kernels"] == []
    assert out["event_total"] == 0


def test_empty_trace_events_array(tmp_path):
    tf = tmp_path / "empty.trace.json"
    tf.write_bytes(json.dumps({"traceEvents": []}).encode("utf-8"))
    out = reader.analyze_trace(tf, top_k=0)
    assert out["status"] == "ok"
    assert out["kernels"] == []
    assert out["timeline"]["total_time_ms"] == 0.0


def test_truncated_trace_recovers_complete_events(tmp_path):
    # Emulate a profiler that died mid-write: a complete first kernel followed
    # by a cut-off second object. The streaming reader must yield the complete
    # event(s) and stop cleanly at the truncation instead of raising.
    good = {"cat": "kernel", "ph": "X", "name": "paged_attention_v1", "ts": 1000, "dur": 300, "args": {"correlation": 1}}
    text = '{"traceEvents": [' + json.dumps(good) + ', {"cat": "kernel", "ph": "X", "name": "Cij'
    tf = tmp_path / "truncated.trace.json"
    tf.write_bytes(text.encode("utf-8"))
    out = reader.analyze_trace(tf, top_k=0)
    assert out["status"] == "ok"
    assert {k["name"] for k in out["kernels"]} == {"paged_attention_v1"}


def test_single_event_trace(tmp_path):
    tf = tmp_path / "single.trace.json"
    one = [{"cat": "kernel", "ph": "X", "name": "rms_norm_kernel", "ts": 5, "dur": 10, "args": {"correlation": 1}}]
    tf.write_bytes(json.dumps({"traceEvents": one}).encode("utf-8"))
    out = reader.analyze_trace(tf, top_k=0)
    assert out["status"] == "ok"
    assert len(out["kernels"]) == 1
    assert out["kernels"][0]["name"] == "rms_norm_kernel"


def test_malformed_events_missing_fields_do_not_crash(tmp_path):
    # Kernel events missing ts/dur/name/args must be tolerated (fields default).
    events = [
        {"cat": "kernel", "ph": "X"},  # no name/ts/dur/args
        {"cat": "kernel", "ph": "X", "name": "silu_kernel", "ts": 100},  # no dur
        {"cat": "cuda_runtime"},  # no args
        {"cat": "gpu_user_annotation", "ph": "X"},  # no name/ts/dur
    ]
    tf = tmp_path / "malformed.trace.json"
    tf.write_bytes(json.dumps({"traceEvents": events}).encode("utf-8"))
    out = reader.analyze_trace(tf, top_k=0)
    assert out["status"] == "ok"
    # Two kernel events parsed (one unnamed, one silu); no exception raised.
    assert out["attribution"]["kernel_count"] == 2


def test_reader_extracts_shapes_and_triton_kernel_file(tmp_path):
    # cpu_op carries Input Dims / Input type + Triton kernel_file; the reader
    # must attach the majority op's meta onto the hot-kernel row.
    events = [
        {
            "cat": "cpu_op",
            "name": "aten::silu_and_mul",
            "args": {
                "External id": 100,
                "Input Dims": [[1024, 256], []],
                "Input type": ["c10::BFloat16", "Scalar"],
                "kernel_file": "/repo/aiter/triton/silu.py",
                "kernel_backend": "triton",
            },
        },
        {"cat": "cuda_runtime", "name": "launch", "args": {"correlation": 5, "External id": 100}},
        {"cat": "kernel", "ph": "X", "name": "triton_silu_kernel", "ts": 0, "dur": 100, "args": {"correlation": 5}},
    ]
    tf = tmp_path / "meta.trace.json"
    tf.write_bytes(json.dumps({"traceEvents": events}).encode("utf-8"))
    out = reader.analyze_trace(tf, top_k=0)
    k = next(k for k in out["kernels"] if k["name"] == "triton_silu_kernel")
    assert k["op_name"] == "aten::silu_and_mul"
    assert k["op_shapes"] == [[1024, 256], []]
    assert k["op_dtypes"] == ["c10::BFloat16", "Scalar"]
    assert k["op_kernel_file"] == "/repo/aiter/triton/silu.py"
    assert k["op_kernel_backend"] == "triton"


def test_reader_meta_absent_defaults_empty(tmp_path):
    # Unlinked kernel (no cpu_op) -> meta fields present but empty, never missing.
    tf = _write_trace(tmp_path / "t.trace.json")
    out = reader.analyze_trace(tf, top_k=0)
    replay = next(k for k in out["kernels"] if k["name"] == "paged_attention_v1")
    assert replay["op_shapes"] == [] and replay["op_kernel_file"] == ""


def test_steady_state_with_zero_annotations_falls_back(tmp_path):
    events = [
        {"cat": "kernel", "ph": "X", "name": "k_a", "ts": 0, "dur": 10, "args": {"correlation": 1}},
        {"cat": "kernel", "ph": "X", "name": "k_b", "ts": 20, "dur": 10, "args": {"correlation": 2}},
    ]
    tf = tmp_path / "noann.trace.json"
    tf.write_bytes(json.dumps({"traceEvents": events}).encode("utf-8"))
    out = reader.analyze_trace(tf, top_k=0, steady_state=True, framework="xdit")
    assert out["aggregation_scope"] == "full_trace"
    assert out.get("steady_window_status")
    assert len(out["kernels"]) == 2
