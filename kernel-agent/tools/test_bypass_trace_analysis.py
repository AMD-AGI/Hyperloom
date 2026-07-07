###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""End-to-end tests for the bypass CLI (bypass_trace_analysis.main).

Covers the failure-fallback contract (bad/missing trace still yields valid
artifacts + health warnings), the opt-in rocprof enrichment wiring, and the
stdout-is-a-single-result-JSON invariant the handler relies on.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bypass_trace_analysis as bta  # noqa: E402
import diffusion_roofline as dr  # noqa: E402

_TRACE_EVENTS = [
    {"cat": "cpu_op", "name": "aten::paged_attn", "args": {"External id": 100}},
    {"cat": "cpu_op", "name": "aten::mm", "args": {"External id": 200}},
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 5, "External id": 100}},
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 7, "External id": 200}},
    {"cat": "kernel", "ph": "X", "name": "paged_attention_v1", "ts": 1000, "dur": 300, "args": {"correlation": 5}},
    {"cat": "kernel", "ph": "X", "name": "Cijk_Alik_Bljk_HHS", "ts": 1300, "dur": 200, "args": {"correlation": 7}},
]

# Same two kernels but separated by a large idle gap so the GPU busy span is a
# tiny fraction of the trace wall span (idle_pct ~ 99%), tripping the idle gate.
_HIGH_IDLE_TRACE_EVENTS = [
    {"cat": "cpu_op", "name": "aten::paged_attn", "args": {"External id": 100}},
    {"cat": "cpu_op", "name": "aten::mm", "args": {"External id": 200}},
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 5, "External id": 100}},
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 7, "External id": 200}},
    {"cat": "kernel", "ph": "X", "name": "paged_attention_v1", "ts": 1000, "dur": 100, "args": {"correlation": 5}},
    {"cat": "kernel", "ph": "X", "name": "Cijk_Alik_Bljk_HHS", "ts": 100000, "dur": 100, "args": {"correlation": 7}},
]


def _run(argv, capsys):
    rc = bta.main(argv)
    out = capsys.readouterr()
    lines = [ln for ln in out.out.splitlines() if ln.strip()]
    assert lines, "no stdout produced"
    # The handler consumes stdout as a single JSON object; enrichment logs must
    # not leak here.
    assert "[rocprof_enrich]" not in out.out
    result = json.loads(lines[-1])
    return rc, result, out


def _base_argv(ws: Path, trace_input: str, extra=None):
    argv = [
        "--trace-input", trace_input,
        "--session-id", "utest",
        "--workspace-path", str(ws),
        "--framework", "vllm",
        "--target-platform", "MI300X",
        "--model-name", "utest-llm",
        "--top-k", "8",
    ]
    return argv + (extra or [])


def _assert_artifacts(result):
    for key in ("kernel_candidates", "kernel_roofline", "tracelens_summary", "trace_input_manifest", "trace_report_path"):
        p = result["artifact_paths"][key]
        assert p and Path(p).is_file(), f"missing artifact {key}: {p}"


def test_dry_run_emits_valid_artifacts(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    rc, result, _ = _run(_base_argv(tmp_path, "/tmp/whatever", extra=["--dry-run"]), capsys)
    assert rc == 0
    assert result["status"] == "ok" and result["route"] == "bypass"
    assert result["rocprof_enrich"]["status"] == "disabled"
    _assert_artifacts(result)


def test_num_denoise_steps_accepted_and_recorded(tmp_path, capsys, monkeypatch):
    # F1: the coordinator forwards --num-denoise-steps for scriptable workloads;
    # bypass must accept it (no argparse crash -> no degraded roofline) and surface
    # it in the result (effective = requested when the trace infers no steps).
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    rc, result, _ = _run(
        _base_argv(tmp_path, "/tmp/whatever", extra=["--dry-run", "--num-denoise-steps", "20"]),
        capsys,
    )
    assert rc == 0
    assert result["status"] == "ok"
    assert result["num_denoise_steps"] == 20


def test_missing_trace_falls_back_gracefully(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    rc, result, _ = _run(_base_argv(tmp_path, str(tmp_path / "does_not_exist.trace.json")), capsys)
    assert rc == 0
    assert result["status"] == "ok"
    assert result["hot_kernels"] == []
    codes = {w["code"] for w in result["trace_health_warnings"]}
    assert "bypass_trace_parse_failed" in codes
    _assert_artifacts(result)


def test_real_trace_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    trace = tmp_path / "t.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert rc == 0
    assert result["status"] == "ok"
    cats = {k["kernel_category"] for k in result["hot_kernels"]}
    # attention + gemm kernels both classified
    assert "SDPA" in cats and "GEMM" in cats
    # sidecar has one row per hot kernel
    kr = json.loads(Path(result["artifact_paths"]["kernel_roofline"]).read_text())
    assert len(kr["kernels"]) == len(result["hot_kernels"])


def test_gzip_trace_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    trace = tmp_path / "t.trace.json.gz"
    with gzip.open(trace, "wb") as f:
        f.write(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert rc == 0
    assert result["hot_kernels"], "expected kernels from gzip trace"


def test_multi_rank_provenance_and_warning(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    for rank in (0, 1):
        with gzip.open(trace_dir / f"rank_{rank}.trace.json.gz", "wb") as f:
            f.write(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    _, result, _ = _run(_base_argv(tmp_path, str(trace_dir)), capsys)
    assert result["status"] == "ok"
    assert result["rank_count"] == 2
    assert result["analyzed_rank"] == 0
    codes = {w["code"] for w in result["trace_health_warnings"]}
    assert "bypass_multi_rank_single_analyzed" in codes
    manifest = json.loads(Path(result["artifact_paths"]["trace_input_manifest"]).read_text())
    assert manifest["rank_count"] == 2 and manifest["analyzed_rank"] == 0


def test_high_gpu_idle_gate_suppresses_hot_kernels(tmp_path, capsys, monkeypatch):
    # F3: contract parity with the TraceLens route -- when the GPU is idle beyond
    # the shared threshold, bypass suppresses every candidate list and surfaces a
    # high_gpu_idle_pct warning so the Coordinator routes to parameter opt.
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    monkeypatch.delenv("HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD", raising=False)
    trace = tmp_path / "idle.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _HIGH_IDLE_TRACE_EVENTS}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert rc == 0
    assert result["status"] == "ok"
    # timeline confirms the high-idle regime that trips the default 80% gate
    assert result["timeline"]["idle_pct"] > 80.0
    # every candidate list is suppressed (parity with TraceLens agent_candidates=[])
    assert result["hot_kernels"] == []
    assert result["routable_kernels"] == []
    assert result["skipped_kernels"] == []
    warn = next(w for w in result["trace_health_warnings"] if w["code"] == "high_gpu_idle_pct")
    assert warn["threshold_pct"] == 80.0
    assert warn["idle_pct"] > 80.0
    # kernel_candidates.json on disk is suppressed too (downstream dispatch reads it)
    kc = json.loads(Path(result["artifact_paths"]["kernel_candidates"]).read_text())
    assert kc["hot_kernels"] == [] and kc.get("routable_kernels") == []


def test_high_idle_gate_respects_threshold_env(tmp_path, capsys, monkeypatch):
    # A high threshold disables the gate: the same idle trace keeps its hot kernels.
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    monkeypatch.setenv("HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD", "99.999")
    trace = tmp_path / "idle.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _HIGH_IDLE_TRACE_EVENTS}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert rc == 0
    assert result["hot_kernels"], "gate must not fire below the configured threshold"
    assert "high_gpu_idle_pct" not in {w["code"] for w in result["trace_health_warnings"]}


# ── diffusion workload roofline (F4) ─────────────────────────────────────────


def test_bypass_diffusion_aggregation_numerics():
    # aggregate_bypass_candidates: sigma_ideal = sum(actual * eff); placeholder
    # kernels (no analytical roofline) count only toward no_perf_model_us.
    hot = [
        {"duration_us": 100.0, "efficiency_percent": 50.0, "bound_type": "compute_bound", "roofline_source": "analytical", "name": "gemm_k", "kernel_category": "GEMM"},
        {"duration_us": 60.0, "efficiency_percent": 25.0, "bound_type": "memory_bound", "roofline_source": "analytical", "name": "attn_k", "kernel_category": "SDPA"},
        {"duration_us": 40.0, "efficiency_percent": 0.0, "roofline_source": "placeholder", "name": "p_k", "kernel_category": "Other"},
    ]
    r = dr.build_report_from_bypass(hot, {"busy_pct": 80.0, "idle_pct": 20.0}, 4, 10)
    t = r["totals"]
    assert t["sigma_actual_kernel_us"] == 200.0
    assert t["sigma_ideal_roofline_us"] == 65.0
    assert round(t["kernel_roofline_efficiency"], 4) == 0.325
    assert t["compute_bound_us"] == 100.0 and t["memory_bound_us"] == 60.0
    assert t["no_perf_model_us"] == 40.0
    assert r["gpu_busy_ratio"] == 0.8
    assert round(r["end_to_end_efficiency_estimate"], 4) == 0.26
    assert r["source"] == "bypass_analytical" and r["kernel_scope"] == "analyzed_candidates"
    assert r["num_denoise_steps"] == 4
    assert r["per_step"]["actual_kernel_us"] == 50.0 and r["per_step"]["ideal_roofline_us"] == 16.25
    assert r["top_kernels"][0]["name"] == "gemm_k"


def test_bypass_diffusion_report_shape_without_steps():
    # No denoise steps -> no per_step block, but the core report keys stay put
    # (identical shape to the TraceLens CSV path via the shared assembler).
    r = dr.build_report_from_bypass([], {"busy_pct": 0.0}, None, 10)
    for key in ("source", "totals", "gpu_timeline_pct", "gpu_busy_ratio", "end_to_end_efficiency_estimate", "top_kernels"):
        assert key in r
    assert "per_step" not in r


def test_xdit_emits_diffusion_roofline(tmp_path, capsys, monkeypatch):
    # F4: parity with the TraceLens route -- the xDiT/scriptable path emits a
    # workload-level diffusion_roofline.json that consumes --num-denoise-steps.
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "t.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    argv = [
        "--trace-input", str(trace), "--session-id", "utest-xdit-diff",
        "--workspace-path", str(tmp_path), "--framework", "xdit",
        "--target-platform", "MI300X", "--model-name", "utest-dit",
        "--top-k", "8", "--num-denoise-steps", "20",
    ]
    rc, result, _ = _run(argv, capsys)
    assert rc == 0
    path = result["artifact_paths"].get("diffusion_roofline")
    assert path and Path(path).is_file()
    assert result["diffusion_roofline_path"] == path
    rep = json.loads(Path(path).read_text())
    assert rep["source"] == "bypass_analytical"
    assert rep["num_denoise_steps"] == 20
    assert "per_step" in rep and "totals" in rep


def test_non_xdit_omits_diffusion_roofline(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "t.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)  # vllm route
    assert rc == 0
    assert "diffusion_roofline" not in result["artifact_paths"]
    assert "diffusion_roofline_path" not in result


# ── boundary inputs end-to-end (P0-3) ────────────────────────────────────────


def test_non_kineto_json_yields_valid_artifacts_and_warns(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    # A valid JSON that is not a Kineto trace: the pipeline must still emit the
    # full artifact set + a no-GPU-kernels warning instead of crashing.
    trace = tmp_path / "notrace.json"
    trace.write_bytes(json.dumps({"foo": "bar"}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert rc == 0
    assert result["status"] == "ok"
    assert result["hot_kernels"] == []
    assert "bypass_no_gpu_kernels" in {w["code"] for w in result["trace_health_warnings"]}
    _assert_artifacts(result)


def test_empty_trace_events_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "empty.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": []}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert rc == 0
    assert result["status"] == "ok"
    assert result["hot_kernels"] == []
    assert "bypass_no_gpu_kernels" in {w["code"] for w in result["trace_health_warnings"]}
    _assert_artifacts(result)


# ── analysis-quality health signals (P0-2, _emit_quality_warnings) ───────────


def _analyze(*, kernels=None, attributed_pct=100.0, steady_status=None, capture_fragment=False):
    """Build a minimal analyze dict for the quality-warning unit tests."""
    out = {
        "kernels": kernels if kernels is not None else [],
        "attribution": {"attributed_pct": attributed_pct},
        "selected_capture_fragment": capture_fragment,
    }
    if steady_status is not None:
        out["steady_window_status"] = steady_status
    return out


# A well-classified, well-correlated single-window analysis: no signals fire.
_HEALTHY_KERNELS = [
    {"name": "paged_attention_v1", "gpu_time_us": 900.0, "count": 3},
    {"name": "some_mystery_kernel_xyz", "gpu_time_us": 100.0, "count": 1},
]


def _codes(warnings):
    return {w["code"] for w in warnings}


def test_quality_warnings_silent_when_healthy(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BYPASS_OTHERS_WARN_PCT", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_CORR_WARN_PCT", raising=False)
    warnings: list = []
    bta._emit_quality_warnings(_analyze(kernels=_HEALTHY_KERNELS, attributed_pct=80.0), warnings)
    assert warnings == []


def test_quality_warning_high_unclassified_share(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BYPASS_OTHERS_WARN_PCT", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_CORR_WARN_PCT", raising=False)
    # 90% of GPU time in an unclassified ("Others") kernel -> taxonomy-gap signal.
    kernels = [
        {"name": "some_mystery_kernel_xyz", "gpu_time_us": 900.0, "count": 9},
        {"name": "paged_attention_v1", "gpu_time_us": 100.0, "count": 1},
    ]
    warnings: list = []
    bta._emit_quality_warnings(_analyze(kernels=kernels, attributed_pct=80.0), warnings)
    assert "bypass_high_unclassified_share" in _codes(warnings)
    w = next(w for w in warnings if w["code"] == "bypass_high_unclassified_share")
    assert w["severity"] == "warning"


def test_quality_warning_low_op_correlation(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BYPASS_OTHERS_WARN_PCT", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_CORR_WARN_PCT", raising=False)
    warnings: list = []
    bta._emit_quality_warnings(_analyze(kernels=_HEALTHY_KERNELS, attributed_pct=5.0), warnings)
    codes = _codes(warnings)
    assert "bypass_low_op_correlation" in codes
    assert "bypass_high_unclassified_share" not in codes  # Others share is only 10%


def test_quality_warning_steady_fallback(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BYPASS_OTHERS_WARN_PCT", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_CORR_WARN_PCT", raising=False)
    warnings: list = []
    bta._emit_quality_warnings(
        _analyze(kernels=_HEALTHY_KERNELS, attributed_pct=80.0, steady_status="no_repeating_window_fell_back_to_full_trace"),
        warnings,
    )
    assert "bypass_steady_fallback_full_trace" in _codes(warnings)


def test_quality_warning_only_capture_fragments(monkeypatch):
    # F2: analysis ran on a sglang CUDA-graph capture shard (no main trace) ->
    # a warning-severity signal so the sparse analysis is never silent.
    monkeypatch.delenv("HYPERLOOM_BYPASS_OTHERS_WARN_PCT", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_CORR_WARN_PCT", raising=False)
    warnings: list = []
    bta._emit_quality_warnings(
        _analyze(kernels=_HEALTHY_KERNELS, attributed_pct=80.0, capture_fragment=True),
        warnings,
    )
    w = next((w for w in warnings if w["code"] == "bypass_only_capture_fragments"), None)
    assert w is not None and w["severity"] == "warning"
    # a normal (main-trace) analysis must not raise it.
    warnings2: list = []
    bta._emit_quality_warnings(_analyze(kernels=_HEALTHY_KERNELS, attributed_pct=80.0), warnings2)
    assert "bypass_only_capture_fragments" not in _codes(warnings2)


def test_quality_warning_others_threshold_env(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BYPASS_CORR_WARN_PCT", raising=False)
    # Healthy kernels have 10% Others; a low env threshold flips the verdict.
    monkeypatch.setenv("HYPERLOOM_BYPASS_OTHERS_WARN_PCT", "5")
    warnings: list = []
    bta._emit_quality_warnings(_analyze(kernels=_HEALTHY_KERNELS, attributed_pct=80.0), warnings)
    assert "bypass_high_unclassified_share" in _codes(warnings)


def test_quality_warning_corr_threshold_env(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BYPASS_OTHERS_WARN_PCT", raising=False)
    # attributed_pct=50 is healthy by default (>=10) but trips a stricter env.
    monkeypatch.setenv("HYPERLOOM_BYPASS_CORR_WARN_PCT", "60")
    warnings: list = []
    bta._emit_quality_warnings(_analyze(kernels=_HEALTHY_KERNELS, attributed_pct=50.0), warnings)
    assert "bypass_low_op_correlation" in _codes(warnings)


def test_quality_warning_bad_env_falls_back_to_default(monkeypatch):
    # A non-numeric threshold must not crash; it falls back to the default (40).
    monkeypatch.setenv("HYPERLOOM_BYPASS_OTHERS_WARN_PCT", "not-a-number")
    monkeypatch.delenv("HYPERLOOM_BYPASS_CORR_WARN_PCT", raising=False)
    warnings: list = []
    bta._emit_quality_warnings(_analyze(kernels=_HEALTHY_KERNELS, attributed_pct=80.0), warnings)
    # 10% Others < default 40 -> no unclassified warning, and no crash.
    assert "bypass_high_unclassified_share" not in _codes(warnings)


def test_rocprof_enrich_disabled_by_default(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    trace = tmp_path / "t.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    _, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert result["rocprof_enrich"]["status"] == "disabled"
    summ = json.loads(Path(result["artifact_paths"]["tracelens_summary"]).read_text())
    assert summ["rocprof_enrich"]["status"] == "disabled"


_STEADY_EVENTS = [
    {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#1", "ts": 0, "dur": 100},
    {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#2", "ts": 100, "dur": 100},
    {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#3", "ts": 200, "dur": 100},
    {"cat": "kernel", "ph": "X", "name": "warmup_gemm", "ts": 50, "dur": 40, "args": {"correlation": 1}},
    {"cat": "kernel", "ph": "X", "name": "paged_attention_v1", "ts": 250, "dur": 30, "args": {"correlation": 2}},
]


def test_steady_state_mode_flag_enables_windowing(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "s.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _STEADY_EVENTS}).encode("utf-8"))
    argv = _base_argv(tmp_path, str(trace), extra=["--steady-state-mode", "annotation"])
    _, result, _ = _run(argv, capsys)
    assert result["aggregation_scope"] == "steady_state"
    assert result["steady_window"] and result["steady_window"]["step_name"] == "ProfilerStep"
    # only the in-window kernel is ranked.
    assert {k["device_kernel_name"] for k in result["hot_kernels"]} == {"paged_attention_v1"}
    assert result["estimated"] is False  # vllm framework


def test_xdit_steady_anchored_is_not_estimated(tmp_path, capsys, monkeypatch):
    # xDiT auto-enables steady-state; when the repeating ProfilerStep window is
    # found the per-step kernel shares are trace-anchored, so the result is NOT
    # estimated (parity with text-gen) and carries a steady-anchored info signal.
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "x.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _STEADY_EVENTS}).encode("utf-8"))
    argv = [
        "--trace-input", str(trace),
        "--session-id", "utest-xdit",
        "--workspace-path", str(tmp_path),
        "--framework", "xdit",
        "--target-platform", "MI300X",
        "--model-name", "FLUX.1-dev",
        "--top-k", "8",
    ]
    _, result, _ = _run(argv, capsys)
    assert result["aggregation_scope"] == "steady_state"
    assert result["estimated"] is False
    codes = {w["code"] for w in result["trace_health_warnings"]}
    assert "bypass_xdit_steady_anchored" in codes
    assert "bypass_xdit_estimated" not in codes
    # estimated flag flows into summary + manifest.
    summ = json.loads(Path(result["artifact_paths"]["tracelens_summary"]).read_text())
    assert summ["estimated"] is False
    manifest = json.loads(Path(result["artifact_paths"]["trace_input_manifest"]).read_text())
    assert manifest["estimated"] is False and manifest["aggregation_scope"] == "steady_state"


def test_xdit_full_trace_fallback_is_estimated(tmp_path, capsys, monkeypatch):
    # No per-step annotations -> steady-state windowing falls back to full_trace,
    # so the xDiT result is estimated and flags bypass_xdit_estimated.
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "x.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    argv = [
        "--trace-input", str(trace),
        "--session-id", "utest-xdit-full",
        "--workspace-path", str(tmp_path),
        "--framework", "xdit",
        "--target-platform", "MI300X",
        "--model-name", "FLUX.1-dev",
        "--top-k", "8",
    ]
    _, result, _ = _run(argv, capsys)
    assert result["aggregation_scope"] == "full_trace"
    assert result["estimated"] is True
    codes = {w["code"] for w in result["trace_health_warnings"]}
    assert "bypass_xdit_estimated" in codes
    summ = json.loads(Path(result["artifact_paths"]["tracelens_summary"]).read_text())
    assert summ["estimated"] is True


def test_parse_failure_flags_analysis_degraded(tmp_path, capsys, monkeypatch):
    # An unresolvable/failed trace must NOT masquerade as a successful empty
    # analysis: the pipeline still degrades gracefully (status=ok, no abort) but
    # flags analysis_degraded so record_trace_analyze / the LLM know it actually
    # failed (rather than trusting the forced-empty result).
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    missing = tmp_path / "does_not_exist.trace.json"  # resolve_trace_file -> None -> status failed
    rc, result, _ = _run(_base_argv(tmp_path, str(missing)), capsys)
    assert rc == 0
    assert result["status"] == "ok"  # graceful: never aborts the pipeline
    assert result["analysis_degraded"] is True
    codes = {w["code"] for w in result["trace_health_warnings"]}
    assert "bypass_trace_parse_failed" in codes
    summ = json.loads(Path(result["artifact_paths"]["tracelens_summary"]).read_text())
    assert summ["analysis_degraded"] is True
    manifest = json.loads(Path(result["artifact_paths"]["trace_input_manifest"]).read_text())
    assert manifest["analysis_degraded"] is True


def test_healthy_trace_is_not_degraded(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    trace = tmp_path / "ok.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    _, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert result["analysis_degraded"] is False


def test_text_gen_steady_fallback_is_estimated(tmp_path, capsys, monkeypatch):
    # Parity with xDiT: when steady-state windowing is requested for text-gen but
    # no repeating window is found, the full-trace shares are equally a mixed
    # estimate -> estimated=True (not silently False).
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "ng.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))  # no ProfilerStep
    argv = _base_argv(tmp_path, str(trace), extra=["--steady-state-mode", "auto"])
    _, result, _ = _run(argv, capsys)
    assert result["aggregation_scope"] == "full_trace"
    assert result["estimated"] is True


def test_text_gen_default_full_trace_not_estimated(tmp_path, capsys, monkeypatch):
    # Default text-gen (no steady-state requested): full-trace IS the norm, so it
    # is NOT flagged estimated.
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "d.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    _, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert result["aggregation_scope"] == "full_trace"
    assert result["estimated"] is False


_FUSION_EVENTS = [
    {"cat": "cpu_op", "name": "aten::add", "args": {"External id": 1}},
    {"cat": "cpu_op", "name": "aten::mul", "args": {"External id": 2}},
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 11, "External id": 1}},
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 12, "External id": 2}},
    {"cat": "kernel", "ph": "X", "name": "elementwise_add_kernel", "ts": 100, "dur": 10, "args": {"correlation": 11}},
    {"cat": "kernel", "ph": "X", "name": "elementwise_mul_kernel", "ts": 110, "dur": 10, "args": {"correlation": 12}},
]


def test_fusion_artifact_and_result(tmp_path, capsys, monkeypatch):
    # Two consecutive Elementwise launches -> one fusable cluster; emitted both in
    # the result summary and the kernel_sequence.json artifact.
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    trace = tmp_path / "f.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _FUSION_EVENTS}).encode("utf-8"))
    _, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert result["fusion"]["launch_count"] == 2
    assert result["fusion"]["fusable_cluster_count"] == 1
    seq_path = result["artifact_paths"]["kernel_sequence"]
    assert Path(seq_path).is_file()
    seq = json.loads(Path(seq_path).read_text())
    assert seq["fusable_clusters"][0]["launch_count"] == 2
    assert "Elementwise" in seq["fusable_clusters"][0]["categories"]


def test_csv_artifacts_written_and_paths_exposed(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)
    trace = tmp_path / "m.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _FUSION_EVENTS}).encode("utf-8"))
    _, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    mpath = result["artifact_paths"]["kernel_metrics_csv"]
    spath = result["artifact_paths"]["kernel_summary_csv"]
    assert Path(mpath).is_file() and Path(spath).is_file()
    assert result["kernel_metrics_csv_path"] == mpath
    rows = list(csv.DictReader(io.StringIO(Path(mpath).read_text())))
    assert rows  # the two elementwise launches -> rows
    assert "optimization_priority" in rows[0] and "suggestion" in rows[0]
    srows = list(csv.DictReader(io.StringIO(Path(spath).read_text())))
    assert srows and "kernel_category" in srows[0]


def test_rocprof_enrich_opt_in_runs_and_degrades(tmp_path, capsys, monkeypatch):
    # Enrichment on: with no benchmark files / no rocprof, it degrades to a
    # summary (rows skipped) but never aborts and never leaks logs to stdout.
    monkeypatch.setenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", "1")
    trace = tmp_path / "t.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    _, result, out = _run(_base_argv(tmp_path, str(trace)), capsys)
    enr = result["rocprof_enrich"]
    assert isinstance(enr, dict) and "status" in enr
    # rows equals hot-kernel count; nothing matched (no benchmark files / rocprof)
    assert enr.get("rows", 0) == len(result["hot_kernels"])
    assert enr.get("matched", 0) == 0
    # progress logged to stderr, not stdout
    assert "[rocprof_enrich]" in out.err
