###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the bypass streaming Kineto reader (_bypass_trace_reader).

Builds a tiny hand-authored Kineto trace so the streaming parser, correlation
attribution, timeline union math, and annotation-window extraction are all
covered deterministically.
"""

from __future__ import annotations

import gzip
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

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


_FALLBACK_TRACE_EVENTS = [
    # Capture-time launch of add_rmsnorm: cpu_op carries Input Dims (shape source).
    {
        "cat": "cpu_op",
        "name": "aiter::add_rmsnorm",
        "args": {"External id": 100, "Input Dims": [[17, 7168]], "Input type": ["c10::BFloat16"]},
    },
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 5, "External id": 100}},
    {
        "cat": "kernel",
        "ph": "X",
        "name": "add_rmsnorm_kernel",
        "ts": 1000,
        "dur": 100,
        "args": {"correlation": 5, "grid": [17, 1, 1], "block": [256, 1, 1]},
    },
    # Graph-replay of the SAME kernel: no cpu_op link, grid-less Dispatch Task.
    {"cat": "cuda_runtime", "name": "hipGraphLaunch", "args": {"correlation": 6}},
    {"cat": "kernel", "ph": "X", "name": "add_rmsnorm_kernel", "ts": 2000, "dur": 100, "args": {"correlation": 6}},
    # Pure-Triton kernel: no cpu_op, but launch carries grid/block geometry.
    {"cat": "cuda_runtime", "name": "hipModuleLaunchKernel", "args": {"correlation": 7}},
    {
        "cat": "kernel",
        "ph": "X",
        "name": "_score_kernel",
        "ts": 3000,
        "dur": 100,
        "args": {"correlation": 7, "grid": [17, 2, 1], "block": [512, 1, 1]},
    },
]


def test_launch_geom_and_backfill_threaded_to_rows(tmp_path):
    payload = json.dumps({"traceEvents": _FALLBACK_TRACE_EVENTS}).encode("utf-8")
    tf = tmp_path / "fb.trace.json"
    tf.write_bytes(payload)
    out = reader.analyze_trace(tf, top_k=0)
    rows = {k["name"]: k for k in out["kernels"]}

    # Same-name kernel resolved a shape at capture time -> backfill index seeded.
    assert rows["add_rmsnorm_kernel"]["backfill_shapes"] == [[17, 7168]]
    # Pure-Triton kernel: no cpu_op shape, but launch geometry is retained.
    score = rows["_score_kernel"]
    assert score["op_shapes"] == []
    assert score["launch_grid"] == [17, 2, 1]
    assert score["launch_block"] == [512, 1, 1]


_MULTI_SHAPE_BACKFILL_EVENTS = [
    {
        "cat": "cpu_op",
        "name": "aten::small",
        "args": {"External id": 1, "Input Dims": [[8, 8]], "Input type": ["c10::BFloat16"]},
    },
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 1, "External id": 1}},
    {"cat": "kernel", "ph": "X", "name": "dyn_kernel", "ts": 1000, "dur": 10, "args": {"correlation": 1}},
    {
        "cat": "cpu_op",
        "name": "aten::large",
        "args": {"External id": 2, "Input Dims": [[64, 64]], "Input type": ["c10::BFloat16"]},
    },
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 2, "External id": 2}},
    {"cat": "kernel", "ph": "X", "name": "dyn_kernel", "ts": 2000, "dur": 100, "args": {"correlation": 2}},
]


def test_backfill_majority_shape_and_ambiguous_flag(tmp_path):
    payload = json.dumps({"traceEvents": _MULTI_SHAPE_BACKFILL_EVENTS}).encode("utf-8")
    tf = tmp_path / "multi.trace.json"
    tf.write_bytes(payload)
    row = reader.analyze_trace(tf, top_k=0)["kernels"][0]
    assert row["name"] == "dyn_kernel"
    assert row["backfill_shapes"] == [[64, 64]]
    assert row["backfill_ambiguous"] is True


def test_timeline_union_math(tmp_path):
    tf = _write_trace(tmp_path / "t.trace.json")
    tl = reader.analyze_trace(tf, top_k=0)["timeline"]
    assert tl["total_time_ms"] == 0.75  # (1750 - 1000) us
    assert tl["busy_time_ms"] == 0.55  # 200 + 300 + 50 us (disjoint)
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


# ── multi-rank trace selection ───────────────────────────────────────────────


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


# ── sglang CUDA-graph capture fragments ──────────────────────────────────────


def _write_capture_fragment(capture_dir: Path, batch_size: int, rank: int = 0) -> Path:
    # A sparse sglang CUDA-graph capture shard: rank-tagged filename but only a
    # couple of device kernels (the real workload is not captured here).
    capture_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {"cat": "kernel", "ph": "X", "name": "graph_capture_marker", "ts": 0, "dur": 1, "args": {"correlation": 1}},
    ]
    p = capture_dir / f"bs_{batch_size}_rank{rank}.json.gz"
    with gzip.open(p, "wb") as f:
        f.write(json.dumps({"traceEvents": events}).encode("utf-8"))
    return p


def _write_main_tp_trace(d: Path, name: str = "1783387979.6664605-TP-0.trace.json.gz") -> Path:
    # The content-rich main sglang profiler trace; not rank-tagged (``-TP-0``
    # does not match the rank regex).
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    with gzip.open(p, "wb") as f:
        f.write(json.dumps({"traceEvents": list(_TRACE_EVENTS)}).encode("utf-8"))
    return p


def test_resolve_trace_file_ignores_sglang_capture_fragments(tmp_path):
    # The rank-tagged capture shards must not hijack selection away from the
    # non-rank-tagged content-rich main trace.
    d = tmp_path / "torch_trace"
    main = _write_main_tp_trace(d)
    cap = d / "capture_traces"
    for bs in (512, 496, 480):
        _write_capture_fragment(cap, bs)
    resolved = reader.resolve_trace_file(d)
    assert resolved is not None and resolved.name == main.name


def test_capture_fragment_dir_selects_main_trace_content(tmp_path):
    # End-to-end via analyze_trace: the selected trace yields the main trace's
    # real kernels, not the 1-kernel capture shard.
    d = tmp_path / "torch_trace"
    _write_main_tp_trace(d)
    cap = d / "capture_traces"
    for bs in (512, 496):
        _write_capture_fragment(cap, bs)
    out = reader.analyze_trace(d, top_k=0)
    assert out["status"] == "ok"
    assert {k["name"] for k in out["kernels"]} == {"Cijk_Alik_Bljk_HHS", "paged_attention_v1"}
    assert out["rank_count"] == 1
    assert out["analyzed_rank"] is None


def test_resolve_trace_file_falls_back_when_only_capture_fragments(tmp_path):
    # If only capture shards exist, still resolve one (never None).
    d = tmp_path / "torch_trace"
    cap = d / "capture_traces"
    _write_capture_fragment(cap, 512)
    resolved = reader.resolve_trace_file(d)
    assert resolved is not None and resolved.name.startswith("bs_512_rank0")


def test_bs_named_fragment_without_subdir_is_deprioritized(tmp_path):
    # Even without the capture_traces/ subdir, the ``bs_<n>_rank<n>`` filename
    # marks a capture shard; a top-level main trace still wins.
    d = tmp_path / "torch_trace"
    main = _write_main_tp_trace(d)
    _write_capture_fragment(d, 256)  # writes bs_256_rank0.json.gz at top level
    resolved = reader.resolve_trace_file(d)
    assert resolved is not None and resolved.name == main.name


def test_unpatched_sglang_capture_dir_is_deprioritized(tmp_path):
    # Both routes now share one classifier, so the bypass reader also knows the
    # SGLang-without-profiler-patch layout: a ``graph_capture_profile/`` holding
    # ``cuda_graph_capture-*``. The reader's own copy of the rule only knew
    # ``bs_<n>_rank<n>`` / ``capture_traces/`` and took the sidecar as a trace.
    d = tmp_path / "torch_trace"
    main = _write_main_tp_trace(d)
    cap_dir = d / "graph_capture_profile"
    cap_dir.mkdir(parents=True, exist_ok=True)
    _write_capture_fragment(cap_dir, 512)
    (cap_dir / "cuda_graph_capture-DecodeCudaGraphRunner-TP-3.json.gz").write_bytes(
        (cap_dir / "bs_512_rank0.json.gz").read_bytes()
    )
    resolved = reader.resolve_trace_file(d)
    assert resolved is not None and resolved.name == main.name


def test_capture_traces_detection_is_relative_to_trace_root(tmp_path):
    # An unrelated ancestor dir named ``capture_traces`` above the trace root must
    # not flag the main trace; only a genuine subdir within the root marks shards.
    root = tmp_path / "capture_traces" / "torch_trace"
    main = _write_main_tp_trace(root, name="rank_0.trace.json.gz")
    _write_capture_fragment(root / "capture_traces", 512)  # genuine sub-shard
    resolved = reader.resolve_trace_file(root)
    assert resolved is not None and resolved.name == main.name


def test_uppercase_capture_dir_with_generic_shard_name(tmp_path):
    # A generic-named larger shard under an uppercase ``Capture_Traces/`` dir is
    # excluded, so a smaller top-level main trace wins.
    d = tmp_path / "torch_trace"
    main = _write_main_tp_trace(d)
    cap = d / "Capture_Traces"
    cap.mkdir(parents=True, exist_ok=True)
    big = cap / "shard.trace.json.gz"
    with gzip.open(big, "wb") as f:
        f.write(json.dumps({"traceEvents": list(_TRACE_EVENTS) * 50}).encode("utf-8"))
    resolved = reader.resolve_trace_file(d)
    assert resolved is not None and resolved.name == main.name


def test_rank_count_ignores_multi_rank_capture_shards(tmp_path):
    # Multi-GPU capture emits bs_*_rank0 and bs_*_rank1 shards; their rank tags
    # must not be counted as real per-rank workload traces.
    d = tmp_path / "torch_trace"
    _write_main_tp_trace(d)
    cap = d / "capture_traces"
    _write_capture_fragment(cap, 512, rank=0)
    _write_capture_fragment(cap, 512, rank=1)
    out = reader.analyze_trace(d, top_k=0)
    assert out["rank_count"] == 1
    assert {k["name"] for k in out["kernels"]} == {"Cijk_Alik_Bljk_HHS", "paged_attention_v1"}


def test_selected_capture_fragment_flag(tmp_path):
    # Only capture shards -> analyze marks selected_capture_fragment; a normal
    # main trace does not.
    only_shards = tmp_path / "torch_trace_shards"
    _write_capture_fragment(only_shards / "capture_traces", 512)
    out = reader.analyze_trace(only_shards, top_k=0)
    assert out["selected_capture_fragment"] is True

    main_dir = tmp_path / "torch_trace_main"
    _write_main_tp_trace(main_dir)
    out2 = reader.analyze_trace(main_dir, top_k=0)
    assert out2["selected_capture_fragment"] is False


def test_multi_rank_main_traces_survive_capture_shard_filter(tmp_path):
    # Genuine top-level per-rank main traces coexisting with sglang capture shards
    # must still resolve to rank_0 (shards filtered out, then lowest-rank policy).
    d = tmp_path / "torch_trace"
    _write_ranked(d, 0)
    _write_ranked(d, 1, extra_events=20)
    _write_ranked(d, 2, extra_events=40)
    cap = d / "capture_traces"
    for bs in (512, 496):
        _write_capture_fragment(cap, bs)  # bs_*_rank0.json.gz shards
    resolved = reader.resolve_trace_file(d)
    assert resolved is not None and resolved.name == "rank_0.trace.json.gz"
    # rank_count reflects the 3 real per-rank traces, not the rank0 shards.
    assert reader._trace_rank_count(d) == 3


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


# ── steady-state windowing ───────────────────────────────────────────────────

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
    # A single spurious "step"-named annotation must not mask a real high-count
    # loop.
    windows = [{"name": "optimizer_step", "ts": 0.0, "dur": 5.0}] + [
        {"name": "graph_call", "ts": float(100 + i * 100), "dur": 100.0} for i in range(5)
    ]
    win = reader.select_steady_window(windows)
    assert win is not None
    assert win["step_name"] == "graph_call"
    assert win["step_count"] == 5


def test_select_steady_window_prefers_step_marker_over_higher_count():
    windows = [{"name": f"ProfilerStep#{i}", "ts": float(i * 100), "dur": 100.0} for i in range(1, 5)] + [
        {"name": "elementwise_region", "ts": float(500 + i * 10), "dur": 10.0} for i in range(10)
    ]
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
    assert tl["total_time_ms"] == 0.1  # 100us window span
    assert tl["busy_time_ms"] == 0.03  # single 30us kernel, fully inside
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


# ── boundary / malformed inputs ──────────────────────────────────────────────


def test_non_kineto_json_yields_no_kernels(tmp_path):
    # Valid JSON without a traceEvents array: the reader must not crash and
    # must report an ok, empty result (the tool turns this into a warning).
    tf = tmp_path / "notrace.json"
    tf.write_bytes(json.dumps({"foo": "bar", "schemaVersion": 1}).encode("utf-8"))
    out = reader.analyze_trace(tf, top_k=0)
    assert out["status"] == "ok"
    assert out["kernels"] == []
    assert out["event_total"] == 0
    assert out["stream_errors"] == ["traceEvents array not found"]


def test_empty_trace_events_array(tmp_path):
    tf = tmp_path / "empty.trace.json"
    tf.write_bytes(json.dumps({"traceEvents": []}).encode("utf-8"))
    out = reader.analyze_trace(tf, top_k=0)
    assert out["status"] == "ok"
    assert out["kernels"] == []
    assert out["timeline"]["total_time_ms"] == 0.0
    assert out["stream_errors"] == []


def test_truncated_trace_recovers_complete_events(tmp_path):
    # Emulate a profiler that died mid-write: a complete first kernel followed
    # by a cut-off second object. The streaming reader must yield the complete
    # event(s) and stop cleanly at the truncation instead of raising.
    good = {
        "cat": "kernel",
        "ph": "X",
        "name": "paged_attention_v1",
        "ts": 1000,
        "dur": 300,
        "args": {"correlation": 1},
    }
    text = '{"traceEvents": [' + json.dumps(good) + ', {"cat": "kernel", "ph": "X", "name": "Cij'
    tf = tmp_path / "truncated.trace.json"
    tf.write_bytes(text.encode("utf-8"))
    out = reader.analyze_trace(tf, top_k=0)
    assert out["status"] == "ok"
    assert {k["name"] for k in out["kernels"]} == {"paged_attention_v1"}
    assert len(out["stream_errors"]) == 1
    assert "truncated after 1 event(s)" in out["stream_errors"][0]


def test_truncated_gzip_records_eof_error_without_raising():
    """A gzip EOF must be reported through the stream error channel."""
    payload = json.dumps(
        {
            "traceEvents": [
                {"cat": "kernel", "name": "first"},
                {"cat": "kernel", "name": "second-" + "x" * 4096},
            ]
        }
    ).encode("utf-8")
    compressed = gzip.compress(payload)
    errors: list[str] = []
    with gzip.GzipFile(fileobj=io.BytesIO(compressed[:-32]), mode="rb") as fh:
        list(reader.stream_events(fh, bufsize=32, errors=errors))
    assert any("EOFError" in error for error in errors)


def test_bad_gzip_records_error_without_raising():
    """An invalid gzip header must fail soft like a truncated stream."""
    errors: list[str] = []
    with gzip.GzipFile(fileobj=io.BytesIO(b"not-a-gzip-stream"), mode="rb") as fh:
        assert list(reader.stream_events(fh, bufsize=8, errors=errors)) == []
    assert any("BadGzipFile" in error for error in errors)


def test_trace_events_null_does_not_capture_a_later_array():
    """A non-array traceEvents value must not redirect parsing elsewhere."""
    payload = b'{"traceEvents": null, "other": [{"cat": "kernel", "name": "wrong-array"}]}'
    errors: list[str] = []
    assert list(reader.stream_events(io.BytesIO(payload), bufsize=16, errors=errors)) == []
    assert errors == ["traceEvents value is not an array"]


def test_trace_events_text_value_before_key_is_ignored():
    """A string value named traceEvents must not shadow the real member."""
    good = {"cat": "kernel", "name": "real-array"}
    payload = json.dumps({"label": "traceEvents", "traceEvents": [good]}).encode("utf-8")
    errors: list[str] = []
    assert list(reader.stream_events(io.BytesIO(payload), bufsize=9, errors=errors)) == [good]
    assert errors == []


def test_utf8_character_split_across_chunks_is_preserved():
    """Incremental decoding must preserve a split multibyte code point."""
    marker = chr(0x20AC)
    expected = "kernel-" + marker + "-suffix"
    payload = json.dumps(
        {"traceEvents": [{"cat": "kernel", "name": expected}]},
        ensure_ascii=False,
    ).encode("utf-8")
    marker_start = payload.index(marker.encode("utf-8"))
    errors: list[str] = []
    events = list(
        reader.stream_events(
            io.BytesIO(payload),
            bufsize=marker_start + 1,
            errors=errors,
        )
    )
    assert events[0]["name"] == expected
    assert errors == []


def test_complete_malformed_object_resyncs_to_later_event():
    """A balanced bad object must not hide a later valid event."""
    good = {"cat": "kernel", "name": "recovered"}
    payload = b'{"traceEvents": [{"cat": "kernel", "name": invalid},' + json.dumps(good).encode("utf-8") + b"]}"
    errors: list[str] = []
    events = list(reader.stream_events(io.BytesIO(payload), bufsize=11, errors=errors))
    assert events == [good]
    assert len(errors) == 1
    assert "malformed after 0 event(s)" in errors[0]


def test_brace_inside_a_string_split_across_chunks_is_not_an_object_end():
    """A refill must resume mid-string rather than close on a quoted brace.

    The decoder reports this partial object as an error positioned at the
    opening quote, well before the buffer end, so "the failure is not at the
    end" cannot be used to conclude the input is corrupt.
    """
    good = {"cat": "kernel", "name": "a}b", "ts": 1}
    payload = json.dumps({"traceEvents": [good]}).encode("utf-8")
    errors: list[str] = []
    assert list(reader.stream_events(io.BytesIO(payload), bufsize=payload.index(b"a}b") + 1, errors=errors)) == [good]
    assert errors == []


def test_unterminated_string_at_eof_reports_truncation_not_corruption():
    """An object cut off inside a string is truncated, not malformed."""
    payload = b'{"traceEvents": [{"cat": "kernel", "name": "abc'
    errors: list[str] = []
    assert list(reader.stream_events(io.BytesIO(payload), bufsize=8, errors=errors)) == []
    assert len(errors) == 1
    assert "truncated after 0 event(s)" in errors[0]


def test_trace_prefix_growth_is_bounded(monkeypatch):
    """A missing late traceEvents key must not grow the prefix indefinitely."""
    monkeypatch.setattr(reader, "_MAX_TRACE_PREFIX_CHARS", 32)
    payload = b'{"padding":"' + b"x" * 64 + b'","traceEvents":[]}'
    errors: list[str] = []
    assert list(reader.stream_events(io.BytesIO(payload), bufsize=8, errors=errors)) == []
    assert len(errors) == 1
    assert "prefix exceeds" in errors[0]


def test_single_event_growth_is_bounded(monkeypatch):
    """An unclosed oversized object must stop at the per-event limit."""
    monkeypatch.setattr(reader, "_MAX_EVENT_CHARS", 64)
    payload = b'{"traceEvents":[{"cat":"kernel","name":"' + b"x" * 128
    errors: list[str] = []
    assert list(reader.stream_events(io.BytesIO(payload), bufsize=8, errors=errors)) == []
    assert len(errors) == 1
    assert "object exceeds" in errors[0]


def test_trace_events_opener_may_cross_a_read_boundary():
    """A chunk ending after the key must refill instead of raising ValueError."""
    payload = b'{"traceEvents"  : [{"cat": "kernel"}]}'
    errors: list[str] = []
    events = list(
        reader.stream_events(
            io.BytesIO(payload),
            bufsize=len('{"traceEvents"'),
            errors=errors,
        )
    )
    assert events == [{"cat": "kernel"}]
    assert errors == []


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


# ---- device-stream duration sanity ---------------------------------------
def _stream_trace(path: Path, events: list[dict]) -> Path:
    path.write_bytes(json.dumps({"traceEvents": events}).encode("utf-8"))
    return path


def _gpu(name: str, ts: float, dur: float, pid: int = 1, tid: int = 1) -> dict:
    return {"cat": "kernel", "ph": "X", "name": name, "ts": ts, "dur": dur, "pid": pid, "tid": tid, "args": {}}


def test_stream_overlap_absent_on_consistent_trace(tmp_path):
    """Durations that fit between their neighbours must not be flagged."""
    tf = _write_trace(tmp_path / "ok.trace.json")
    out = reader.analyze_trace(tf, top_k=0)
    assert "stream_overlap" not in (out.get("timeline") or {})


def test_stream_overlap_detects_impossible_duration(tmp_path):
    """An event overrunning its successor on a serial stream is corrupt.

    The corrupt event also extends the stream's span, so a sum-vs-span ratio
    would read ~1.0 here; the pairwise check still catches it.
    """
    events = [
        _gpu("k_a", 1000, 100),
        _gpu("corrupt_kernel", 1100, 20_000),  # ends 21100, next starts 1200
        _gpu("k_b", 1200, 100),
        _gpu("k_c", 1300, 100),
    ]
    tf = _stream_trace(tmp_path / "bad.trace.json", events)
    out = reader.analyze_trace(tf, top_k=0)
    so = (out.get("timeline") or {}).get("stream_overlap")
    assert so, "corrupt duration should be reported"
    assert (so["pid"], so["tid"]) == (1, 1)
    assert so["worst_event"] == "corrupt_kernel"
    assert so["overlapping_events"] == 1
    # Overruns k_b's start (1200) by 19900us.
    assert so["worst_event_excess_ms"] == 19.9

    # A naive sum-vs-span ratio would have looked innocent.
    total_dur = sum(e["dur"] for e in events)
    span = max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events)
    assert total_dur / span < 1.02


def test_stream_overlap_not_triggered_by_concurrent_streams(tmp_path):
    """Two genuinely concurrent streams must not be mistaken for corruption.

    Pooling every device event would make the summed duration twice the wall
    span here, which is exactly the false positive the per-stream check avoids.
    """
    events = []
    for i in range(5):
        events.append(_gpu(f"s1_{i}", 1000 + i * 100, 100, pid=1, tid=1))
        events.append(_gpu(f"s2_{i}", 1000 + i * 100, 100, pid=1, tid=2))
    tf = _stream_trace(tmp_path / "concurrent.trace.json", events)
    out = reader.analyze_trace(tf, top_k=0)
    assert "stream_overlap" not in (out.get("timeline") or {})


def test_stream_overlap_reports_worst_stream():
    # Overruns must clear the absolute floor (1 ms) to be reported at all.
    mild = [(0.0, 5_000.0, "mild"), (2_000.0, 4_000.0, "b"), (10_000.0, 10_000.0, "c")]
    severe = [(0.0, 30_000.0, "severe"), (1_000.0, 2_000.0, "b"), (10_000.0, 10_000.0, "c")]
    worst = reader._stream_overlap_health({(1, 1): list(mild), (1, 2): list(severe)})
    assert worst["tid"] == 2
    assert worst["worst_event"] == "severe"


def test_stream_overlap_ignores_rounding_and_short_streams():
    # Sub-microsecond overrun is timestamp rounding, not corruption.
    tiny = [(0.0, 100.5, "a"), (100.0, 200.0, "b"), (1000.0, 1000.0, "c")]
    assert reader._stream_overlap_health({(1, 1): tiny}) == {}
    # Real but immaterial overlap stays below the reporting share.
    small = [(0.0, 110.0, "a"), (100.0, 200.0, "b"), (100_000.0, 100_000.0, "c")]
    assert reader._stream_overlap_health({(1, 1): small}) == {}
    # Degenerate inputs must not divide by zero.
    assert reader._stream_overlap_health({(1, 1): [(0.0, 10.0, "only")]}) == {}
    assert reader._stream_overlap_health({(1, 1): [(5.0, 5.0, "a"), (5.0, 5.0, "b")]}) == {}
    assert reader._stream_overlap_health({}) == {}


def test_stream_overlap_final_event_is_not_detectable():
    """Documented limitation: nothing follows the last event to contradict it."""
    evs = [(0.0, 100.0, "a"), (100.0, 99_999.0, "corrupt_last")]
    assert reader._stream_overlap_health({(1, 1): evs}) == {}


# ---- review round 2: threshold denominator, absolute floor, severity ------
def _spread(n: int, dur: float, stride: float, prefix: str = "k") -> list[tuple[float, float, str]]:
    return [(i * stride, i * stride + dur, f"{prefix}{i}") for i in range(n)]


def test_threshold_uses_summed_device_time_not_span():
    """A lone corrupt duration must not hide behind the span it inflates.

    1000 kernels spread over 60 s with one duration written as 2 s: the corrupt
    event is 95% of the summed device time but only ~3% of the wall span, so a
    span-denominator threshold reported nothing -- the same blind spot the
    pairwise detector exists to avoid, moved into the threshold.
    """
    evs = _spread(1000, 100.0, 60_000_000 / 1000)
    evs[500] = (evs[500][0], evs[500][0] + 2_000_000.0, "corrupt_2s")
    device_us = sum(e - s for s, e, _ in evs)
    span_us = max(e for _s, e, _ in evs) - evs[0][0]

    out = reader._stream_overlap_health({(1, 1): list(evs)})
    assert out, "corrupt duration dominating summed device time must be reported"
    assert out["worst_event"] == "corrupt_2s"
    assert out["severity"] == "warning"
    # The share is meaningful against summed device time, negligible against span.
    assert out["excess_share"] > 0.9
    assert (out["excess_ms"] * 1000.0) / span_us < 0.05
    assert abs(out["device_ms"] - device_us / 1000.0) < 1e-3


def test_dense_jitter_is_not_escalated():
    """Many sub-millisecond overruns are profiler jitter, not corruption.

    Accumulates well past the 5% share while the worst single overrun is 4 us;
    telling the user to recapture the trace here is a false alarm.
    """
    evs = _spread(20_000, 10.0, 6.0)  # each overruns the next by 4 us
    assert reader._stream_overlap_health({(1, 1): list(evs)}) == {}


def test_identical_timestamps_are_not_overlap():
    """Events sharing a ts cannot contradict each other."""
    same = [(0.0, 50_000.0, "a"), (0.0, 50_000.0, "b"), (0.0, 50_000.0, "c"), (100_000.0, 100_000.0, "d")]
    assert reader._stream_overlap_health({(1, 1): same}) == {}


def test_severity_grades_with_share_of_device_time():
    """Marginal distortion is info; a materially wrong ranking is a warning."""
    # ~10% of summed device time -> info.
    mild = _spread(40, 10_000.0, 10_000.0)
    mild[0] = (0.0, 10_000.0 + 40_000.0, "mild_overrun")
    out_mild = reader._stream_overlap_health({(1, 1): list(mild)})
    assert out_mild and out_mild["severity"] == "info"
    assert 0.05 <= out_mild["excess_share"] < 0.25

    # One event swamping the stream -> warning.
    severe = _spread(40, 10_000.0, 10_000.0)
    severe[0] = (0.0, 10_000.0 + 900_000.0, "severe_overrun")
    out_severe = reader._stream_overlap_health({(1, 1): list(severe)})
    assert out_severe and out_severe["severity"] == "warning"
    assert out_severe["excess_share"] >= 0.25
