###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""End-to-end tests for the bypass CLI (bypass_trace_analysis.main).

Covers the failure-fallback contract (bad/missing trace still yields valid
artifacts + health warnings) and the stdout-is-a-single-result-JSON invariant.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

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

# Two kernels separated by a large idle gap so idle_pct ~ 99%, tripping the idle gate.
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
    # The handler consumes stdout as a single JSON object.
    result = json.loads(lines[-1])
    return rc, result, out


def _base_argv(ws: Path, trace_input: str, extra=None):
    argv = [
        "--trace-input",
        trace_input,
        "--session-id",
        "utest",
        "--workspace-path",
        str(ws),
        "--framework",
        "vllm",
        "--target-platform",
        "MI300X",
        "--model-name",
        "utest-llm",
        "--top-k",
        "8",
    ]
    return argv + (extra or [])


def _assert_artifacts(result):
    for key in (
        "kernel_candidates",
        "kernel_roofline",
        "tracelens_summary",
        "trace_input_manifest",
        "trace_report_path",
    ):
        p = result["artifact_paths"][key]
        assert p and Path(p).is_file(), f"missing artifact {key}: {p}"


def test_dry_run_emits_valid_artifacts(tmp_path, capsys, monkeypatch):
    rc, result, _ = _run(_base_argv(tmp_path, "/tmp/whatever", extra=["--dry-run"]), capsys)
    assert rc == 0
    assert result["status"] == "ok" and result["route"] == "bypass"
    _assert_artifacts(result)


def test_num_denoise_steps_accepted_and_recorded(tmp_path, capsys, monkeypatch):
    # bypass must accept --num-denoise-steps and surface it in the result.
    rc, result, _ = _run(
        _base_argv(tmp_path, "/tmp/whatever", extra=["--dry-run", "--num-denoise-steps", "20"]),
        capsys,
    )
    assert rc == 0
    assert result["status"] == "ok"
    assert result["num_denoise_steps"] == 20


def test_missing_trace_falls_back_gracefully(tmp_path, capsys, monkeypatch):
    rc, result, _ = _run(_base_argv(tmp_path, str(tmp_path / "does_not_exist.trace.json")), capsys)
    assert rc == 0
    assert result["status"] == "ok"
    assert result["hot_kernels"] == []
    codes = {w["code"] for w in result["trace_health_warnings"]}
    assert "bypass_trace_parse_failed" in codes
    _assert_artifacts(result)


def test_real_trace_end_to_end(tmp_path, capsys, monkeypatch):
    trace = tmp_path / "t.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert rc == 0
    assert result["status"] == "ok"
    cats = {k["kernel_category"] for k in result["hot_kernels"]}
    assert "SDPA" in cats and "GEMM" in cats
    kr = json.loads(Path(result["artifact_paths"]["kernel_roofline"]).read_text())
    assert len(kr["kernels"]) == len(result["hot_kernels"])


def test_gzip_trace_end_to_end(tmp_path, capsys, monkeypatch):
    trace = tmp_path / "t.trace.json.gz"
    with gzip.open(trace, "wb") as f:
        f.write(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert rc == 0
    assert result["hot_kernels"], "expected kernels from gzip trace"


def test_multi_rank_provenance_and_warning(tmp_path, capsys, monkeypatch):
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
    # When the GPU is idle beyond the threshold, bypass suppresses every candidate
    # list and surfaces a high_gpu_idle_pct warning.
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    monkeypatch.delenv("HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD", raising=False)
    trace = tmp_path / "idle.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _HIGH_IDLE_TRACE_EVENTS}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert rc == 0
    assert result["status"] == "ok"
    assert result["timeline"]["idle_pct"] > 80.0
    # every candidate list is suppressed
    assert result["hot_kernels"] == []
    assert result["routable_kernels"] == []
    assert result["skipped_kernels"] == []
    warn = next(w for w in result["trace_health_warnings"] if w["code"] == "high_gpu_idle_pct")
    assert warn["threshold_pct"] == 80.0
    assert warn["idle_pct"] > 80.0
    # kernel_candidates.json on disk is suppressed too
    kc = json.loads(Path(result["artifact_paths"]["kernel_candidates"]).read_text())
    assert kc["hot_kernels"] == [] and kc.get("routable_kernels") == []


def test_high_idle_gate_respects_threshold_env(tmp_path, capsys, monkeypatch):
    # A high threshold disables the gate.
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    monkeypatch.setenv("HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD", "99.999")
    trace = tmp_path / "idle.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _HIGH_IDLE_TRACE_EVENTS}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert rc == 0
    assert result["hot_kernels"], "gate must not fire below the configured threshold"
    assert "high_gpu_idle_pct" not in {w["code"] for w in result["trace_health_warnings"]}


# ── diffusion workload roofline ──────────────────────────────────────────────


def test_bypass_diffusion_aggregation_numerics():
    # sigma_ideal = sum(actual * eff); placeholder kernels count only toward
    # no_perf_model_us.
    hot = [
        {
            "duration_us": 100.0,
            "roofline_attainment_pct": 50.0,
            "bound_type": "compute_bound",
            "roofline_source": "analytical",
            "name": "gemm_k",
            "kernel_category": "GEMM",
        },
        {
            "duration_us": 60.0,
            "roofline_attainment_pct": 25.0,
            "bound_type": "memory_bound",
            "roofline_source": "analytical",
            "name": "attn_k",
            "kernel_category": "SDPA",
        },
        {
            "duration_us": 40.0,
            "roofline_attainment_pct": 0.0,
            "roofline_source": "placeholder",
            "name": "p_k",
            "kernel_category": "Other",
        },
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


def test_diffusion_report_totals_param_marks_full_scope():
    # When workload totals are supplied, the report uses them verbatim and marks
    # kernel_scope=all_device_kernels.
    totals = {
        "sigma_actual_kernel_us": 100.0,
        "sigma_ideal_roofline_us": 30.0,
        "kernel_roofline_efficiency": 0.3,
        "compute_bound_us": 60.0,
        "memory_bound_us": 40.0,
        "no_perf_model_us": 0.0,
    }
    # kernels_aggregated reflects the all-kernel count, not len(hot_kernels).
    r = dr.build_report_from_bypass([], {"busy_pct": 90.0}, 8, 10, totals=totals, kernels_aggregated=137)
    assert r["kernel_scope"] == "all_device_kernels"
    assert r["kernels_aggregated"] == 137
    assert r["totals"]["sigma_actual_kernel_us"] == 100.0
    assert r["per_step"]["actual_kernel_us"] == 100.0 / 8


def test_bypass_diffusion_report_shape_without_steps():
    # No denoise steps -> no per_step block, but the core report keys stay put.
    r = dr.build_report_from_bypass([], {"busy_pct": 0.0}, None, 10)
    for key in (
        "source",
        "totals",
        "gpu_timeline_pct",
        "gpu_busy_ratio",
        "end_to_end_efficiency_estimate",
        "top_kernels",
    ):
        assert key in r
    assert "per_step" not in r


def test_xdit_emits_diffusion_roofline(tmp_path, capsys, monkeypatch):
    # The xDiT/scriptable path emits a workload-level diffusion_roofline.json
    # that consumes --num-denoise-steps.
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "t.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    argv = [
        "--trace-input",
        str(trace),
        "--session-id",
        "utest-xdit-diff",
        "--workspace-path",
        str(tmp_path),
        "--framework",
        "xdit",
        "--target-platform",
        "MI300X",
        "--model-name",
        "utest-dit",
        "--top-k",
        "8",
        "--num-denoise-steps",
        "20",
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


def test_bypass_cli_accepts_forwarded_diffusion_flags():
    # The bypass parser must accept --model-path/--precision (and the
    # diffusion-ceiling siblings) or strict parse_args exits 2.
    args = bta._build_arg_parser().parse_args(
        [
            "--trace-input",
            "/x",
            "--framework",
            "xdit",
            "--model-path",
            "/models/flux",
            "--precision",
            "fp8",
            "--height",
            "1024",
            "--width",
            "1024",
            "--cfg-batch",
            "2",
        ]
    )
    assert args.model_path == "/models/flux"
    assert args.precision == "fp8"
    assert args.height == 1024 and args.width == 1024
    assert args.cfg_batch == 2


def test_non_xdit_omits_diffusion_roofline(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "t.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)  # vllm route
    assert rc == 0
    assert "diffusion_roofline" not in result["artifact_paths"]
    assert "diffusion_roofline_path" not in result


# ── steady-state mode coverage ───────────────────────────────────────────────


def test_should_enable_steady_recognizes_tracelens_modes():
    # bypass must window TraceLens splitter chunk types, not silently full-trace.
    for m in ("mixed", "decode_only", "prefilldecode"):
        assert bta._should_enable_steady(steady_state_mode=m, framework="vllm", env_steady=False) is True


def test_should_enable_steady_off_values_stay_full_trace():
    for m in ("", "0", "false", "off", "none"):
        assert bta._should_enable_steady(steady_state_mode=m, framework="vllm", env_steady=False) is False


def test_should_enable_steady_legacy_and_xdit_and_env():
    for m in ("1", "true", "on", "auto", "annotation", "steady"):
        assert bta._should_enable_steady(steady_state_mode=m, framework="vllm", env_steady=False) is True
    assert bta._should_enable_steady(steady_state_mode="", framework="xdit", env_steady=False) is True
    assert bta._should_enable_steady(steady_state_mode="", framework="vllm", env_steady=True) is True


# ── boundary inputs end-to-end ───────────────────────────────────────────────


def test_non_kineto_json_yields_valid_artifacts_and_warns(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    # Valid JSON that is not a Kineto trace: still emit the full artifact set
    # plus a no-GPU-kernels warning instead of crashing.
    trace = tmp_path / "notrace.json"
    trace.write_bytes(json.dumps({"foo": "bar"}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert rc == 0
    assert result["status"] == "ok"
    assert result["hot_kernels"] == []
    assert "bypass_no_gpu_kernels" in {w["code"] for w in result["trace_health_warnings"]}
    _assert_artifacts(result)


def test_empty_trace_events_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "empty.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": []}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert rc == 0
    assert result["status"] == "ok"
    assert result["hot_kernels"] == []
    assert "bypass_no_gpu_kernels" in {w["code"] for w in result["trace_health_warnings"]}
    _assert_artifacts(result)


# ── analysis-quality health signals (_emit_quality_warnings) ─────────────────


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
    assert "bypass_high_unclassified_share" not in codes  # Others share only 10%


def test_quality_warning_steady_fallback(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BYPASS_OTHERS_WARN_PCT", raising=False)
    monkeypatch.delenv("HYPERLOOM_BYPASS_CORR_WARN_PCT", raising=False)
    warnings: list = []
    bta._emit_quality_warnings(
        _analyze(
            kernels=_HEALTHY_KERNELS, attributed_pct=80.0, steady_status="no_repeating_window_fell_back_to_full_trace"
        ),
        warnings,
    )
    assert "bypass_steady_fallback_full_trace" in _codes(warnings)


def test_quality_warning_only_capture_fragments(monkeypatch):
    # Analysis ran on a capture shard (no main trace) -> warning-severity signal.
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
    # A low env threshold flips the verdict on 10% Others.
    monkeypatch.setenv("HYPERLOOM_BYPASS_OTHERS_WARN_PCT", "5")
    warnings: list = []
    bta._emit_quality_warnings(_analyze(kernels=_HEALTHY_KERNELS, attributed_pct=80.0), warnings)
    assert "bypass_high_unclassified_share" in _codes(warnings)


def test_quality_warning_corr_threshold_env(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BYPASS_OTHERS_WARN_PCT", raising=False)
    # attributed_pct=50 is healthy by default but trips a stricter env.
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
    # 10% Others < default 40 -> no warning, no crash.
    assert "bypass_high_unclassified_share" not in _codes(warnings)


def test_steady_state_mode_flag_enables_windowing(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "s.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _STEADY_EVENTS}).encode("utf-8"))
    argv = _base_argv(tmp_path, str(trace), extra=["--steady-state-mode", "annotation"])
    _, result, _ = _run(argv, capsys)
    assert result["aggregation_scope"] == "steady_state"
    assert result["steady_window"] and result["steady_window"]["step_name"] == "ProfilerStep"
    # only the in-window kernel is ranked.
    assert {k["device_kernel_name"] for k in result["hot_kernels"]} == {"paged_attention_v1"}
    assert result["estimated"] is False


def test_xdit_steady_anchored_is_not_estimated(tmp_path, capsys, monkeypatch):
    # When the repeating ProfilerStep window is found, per-step shares are
    # trace-anchored, so the result is NOT estimated.
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "x.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _STEADY_EVENTS}).encode("utf-8"))
    argv = [
        "--trace-input",
        str(trace),
        "--session-id",
        "utest-xdit",
        "--workspace-path",
        str(tmp_path),
        "--framework",
        "xdit",
        "--target-platform",
        "MI300X",
        "--model-name",
        "FLUX.1-dev",
        "--top-k",
        "8",
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
    # No per-step annotations -> falls back to full_trace, so the result is
    # estimated and flags bypass_xdit_estimated.
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "x.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    argv = [
        "--trace-input",
        str(trace),
        "--session-id",
        "utest-xdit-full",
        "--workspace-path",
        str(tmp_path),
        "--framework",
        "xdit",
        "--target-platform",
        "MI300X",
        "--model-name",
        "FLUX.1-dev",
        "--top-k",
        "8",
    ]
    _, result, _ = _run(argv, capsys)
    assert result["aggregation_scope"] == "full_trace"
    assert result["estimated"] is True
    codes = {w["code"] for w in result["trace_health_warnings"]}
    assert "bypass_xdit_estimated" in codes
    summ = json.loads(Path(result["artifact_paths"]["tracelens_summary"]).read_text())
    assert summ["estimated"] is True


def test_parse_failure_flags_analysis_degraded(tmp_path, capsys, monkeypatch):
    # An unresolvable trace degrades gracefully (status=ok) but flags
    # analysis_degraded so consumers know it actually failed.
    missing = tmp_path / "does_not_exist.trace.json"  # resolve_trace_file -> None -> status failed
    rc, result, _ = _run(_base_argv(tmp_path, str(missing)), capsys)
    assert rc == 0
    assert result["status"] == "ok"  # graceful: never aborts
    assert result["analysis_degraded"] is True
    codes = {w["code"] for w in result["trace_health_warnings"]}
    assert "bypass_trace_parse_failed" in codes
    summ = json.loads(Path(result["artifact_paths"]["tracelens_summary"]).read_text())
    assert summ["analysis_degraded"] is True
    manifest = json.loads(Path(result["artifact_paths"]["trace_input_manifest"]).read_text())
    assert manifest["analysis_degraded"] is True


def test_truncated_stream_is_recovered_but_marked_degraded(tmp_path, capsys):
    """Partial recovery must not make a truncated trace look healthy."""
    good = {
        "cat": "kernel",
        "ph": "X",
        "name": "recovered_kernel",
        "ts": 10,
        "dur": 20,
        "args": {"correlation": 1},
    }
    trace = tmp_path / "truncated.trace.json"
    trace.write_text(
        '{"traceEvents": [' + json.dumps(good) + ', {"cat": "kernel", "name": "cut',
        encoding="utf-8",
    )
    _, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert result["analysis_degraded"] is True
    assert {row["name"] for row in result["hot_kernels"]} == {"recovered_kernel"}
    codes = {warning["code"] for warning in result["trace_health_warnings"]}
    assert "bypass_trace_stream_incomplete" in codes
    assert "bypass_trace_parse_failed" not in codes
    summary = json.loads(Path(result["artifact_paths"]["tracelens_summary"]).read_text())
    assert summary["analysis_degraded"] is True
    manifest = json.loads(Path(result["artifact_paths"]["trace_input_manifest"]).read_text())
    assert manifest["analysis_degraded"] is True


def test_healthy_trace_is_not_degraded(tmp_path, capsys, monkeypatch):
    trace = tmp_path / "ok.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    _, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert result["analysis_degraded"] is False


_STEADY_EVENTS = [
    {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#1", "ts": 0, "dur": 100},
    {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#2", "ts": 100, "dur": 100},
    {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#3", "ts": 200, "dur": 100},
    {"cat": "kernel", "ph": "X", "name": "warmup_gemm", "ts": 50, "dur": 40, "args": {"correlation": 1}},
    {"cat": "kernel", "ph": "X", "name": "paged_attention_v1", "ts": 250, "dur": 30, "args": {"correlation": 2}},
]


def test_text_gen_steady_fallback_is_estimated(tmp_path, capsys, monkeypatch):
    # When steady-state windowing is requested but no repeating window is found,
    # the full-trace shares are an estimate -> estimated=True.
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    trace = tmp_path / "ng.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))  # no ProfilerStep
    argv = _base_argv(tmp_path, str(trace), extra=["--steady-state-mode", "auto"])
    _, result, _ = _run(argv, capsys)
    assert result["aggregation_scope"] == "full_trace"
    assert result["estimated"] is True


def test_text_gen_default_full_trace_not_estimated(tmp_path, capsys, monkeypatch):
    # Default text-gen: full-trace is the norm, so it is NOT flagged estimated.
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


def test_fusion_result_summary(tmp_path, capsys, monkeypatch):
    """Two consecutive Elementwise launches -> one fusable cluster."""
    trace = tmp_path / "f.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _FUSION_EVENTS}).encode("utf-8"))
    _, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert result["fusion"]["launch_count"] == 2
    assert result["fusion"]["fusable_cluster_count"] == 1
    assert result["fusion"]["fusable_time_us"] > 0.0
    assert "kernel_sequence" not in result["artifact_paths"]


def test_csv_artifacts_written_and_paths_exposed(tmp_path, capsys, monkeypatch):
    trace = tmp_path / "m.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _FUSION_EVENTS}).encode("utf-8"))
    _, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    mpath = result["artifact_paths"]["kernel_metrics_csv"]
    spath = result["artifact_paths"]["kernel_summary_csv"]
    assert Path(mpath).is_file() and Path(spath).is_file()
    assert result["kernel_metrics_csv_path"] == mpath
    rows = list(csv.DictReader(io.StringIO(Path(mpath).read_text())))
    assert rows
    assert "optimization_priority" in rows[0] and "suggestion" in rows[0]
    srows = list(csv.DictReader(io.StringIO(Path(spath).read_text())))
    assert srows and "kernel_category" in srows[0]


# --- _maybe_build_shape_manifest: enabled-path coverage (WP-1) --------------
import argparse as _argparse  # noqa: E402


def _mk_args(**kw):
    base = dict(trace_input="t", capture_folder="", analysis_mode="decode", precision="fp8")
    base.update(kw)
    return _argparse.Namespace(**base)


def test_maybe_build_shape_manifest_enabled_caps_and_writes(tmp_path, monkeypatch):
    # enabled + numeric MAX_CAPTURES cap + main_trace hash + ok shard + write.
    monkeypatch.setenv("HYPERLOOM_TRACE_SHAPE_MANIFEST", "1")
    monkeypatch.setenv("HYPERLOOM_TRACE_SHAPE_MANIFEST_MAX_CAPTURES", "1")
    shards = [(tmp_path / "a.json", "bs_1", "decode"), (tmp_path / "b.json", "bs_2", "decode")]
    monkeypatch.setattr(bta, "_discover_capture_shards", lambda *a: shards)
    monkeypatch.setattr(bta, "_sha256_file", lambda p: "hash")
    monkeypatch.setattr(bta._reader, "analyze_trace", lambda *a, **k: {"status": "ok"})
    monkeypatch.setattr(bta, "_build_manifest_provenance", lambda args: {"src": "stub"})
    monkeypatch.setattr(bta._tsm, "build_shape_manifest", lambda **k: {"rows": [{"m": 1}], "warnings": []})
    res = bta._maybe_build_shape_manifest(_mk_args(), {"trace_file": "main.json"}, tmp_path, generated_at="2026-01-01")
    assert res["status"] == "ok"
    assert res["variant_count"] == 1  # 2 shards capped to 1
    assert res["row_count"] == 1
    assert (tmp_path / "trace_shape_manifest.json").is_file()


def test_maybe_build_shape_manifest_bad_max_caps_and_skips_bad_shard(tmp_path, monkeypatch):
    # non-numeric MAX_CAPTURES -> 0 (no cap); a non-ok shard is skipped.
    monkeypatch.setenv("HYPERLOOM_TRACE_SHAPE_MANIFEST", "on")
    monkeypatch.setenv("HYPERLOOM_TRACE_SHAPE_MANIFEST_MAX_CAPTURES", "notanumber")
    shards = [(tmp_path / "a.json", "bs_1", "decode"), (tmp_path / "b.json", "bs_2", "prefill")]
    monkeypatch.setattr(bta, "_discover_capture_shards", lambda *a: shards)
    monkeypatch.setattr(bta, "_sha256_file", lambda p: "h")
    _seq = iter([{"status": "ok"}, {"status": "error"}])
    monkeypatch.setattr(bta._reader, "analyze_trace", lambda *a, **k: next(_seq))
    monkeypatch.setattr(bta, "_build_manifest_provenance", lambda args: {})
    monkeypatch.setattr(bta._tsm, "build_shape_manifest", lambda **k: {"rows": [], "warnings": ["w"]})
    res = bta._maybe_build_shape_manifest(_mk_args(analysis_mode=None), {"trace_file": ""}, tmp_path, generated_at="t0")
    assert res["status"] == "ok"
    assert res["variant_count"] == 1  # second (non-ok) shard skipped
    assert res["warnings"] == ["w"]


def test_maybe_build_shape_manifest_degrades_on_error(tmp_path, monkeypatch):
    # any failure -> {"status": "error: ..."}; never raises.
    monkeypatch.setenv("HYPERLOOM_TRACE_SHAPE_MANIFEST", "1")

    def _boom(*a, **k):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(bta, "_discover_capture_shards", _boom)
    res = bta._maybe_build_shape_manifest(_mk_args(), {"trace_file": ""}, tmp_path, generated_at="t0")
    assert res["status"].startswith("error:")
    assert "RuntimeError" in res["status"]


def test_maybe_build_shape_manifest_enabled_by_default(tmp_path, monkeypatch):
    # The gate used to default off, so forge's preferred dense-shape source was
    # produced for nobody. On by default now.
    monkeypatch.delenv("HYPERLOOM_TRACE_SHAPE_MANIFEST", raising=False)
    res = bta._maybe_build_shape_manifest(_mk_args(), {"trace_file": ""}, tmp_path, generated_at="t0")
    assert res["status"] == "ok"
    assert (tmp_path / "trace_shape_manifest.json").is_file()


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", "", "  ", "none", "disabled"])
def test_maybe_build_shape_manifest_can_still_be_turned_off(tmp_path, monkeypatch, value):
    # The empty string and "none" belong here: a launcher disables a variable by
    # exporting it empty at least as often as by unsetting it, and a bare
    # {"0","false","no","off"} check read every one of those as ENABLED -- the
    # opposite of what the operator asked for. Same off-vocabulary this file
    # already documents for --steady-state-mode.
    monkeypatch.setenv("HYPERLOOM_TRACE_SHAPE_MANIFEST", value)
    res = bta._maybe_build_shape_manifest(_mk_args(), {"trace_file": ""}, tmp_path, generated_at="t0")
    assert res == {"status": "disabled"}
    assert not (tmp_path / "trace_shape_manifest.json").exists()


def test_the_cap_keeps_a_deterministic_spread_of_batch_sizes(tmp_path, monkeypatch):
    """Which variants survive the cap must not depend on directory order.

    Discovery walks the filesystem, so "first N" varied between machines; once
    sorted, "first N" would instead have kept only the small-batch end -- 64
    decode variants and no prefill one. Take an evenly spaced slice.
    """
    monkeypatch.delenv("HYPERLOOM_TRACE_SHAPE_MANIFEST", raising=False)
    monkeypatch.setenv("HYPERLOOM_TRACE_SHAPE_MANIFEST_MAX_CAPTURES", "4")
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    shards = [(Path(f"/t/bs_{b}.json"), f"bs_{b}", "decode") for b in batches]

    def _run(order):
        monkeypatch.setattr(bta, "_discover_capture_shards", lambda *a: list(order))
        monkeypatch.setattr(bta, "_sha256_file", lambda p: "h")
        monkeypatch.setattr(bta._reader, "analyze_trace", lambda *a, **k: {"status": "ok"})
        monkeypatch.setattr(bta, "_build_manifest_provenance", lambda args: {})
        seen: list[str] = []
        monkeypatch.setattr(
            bta._tsm,
            "build_shape_manifest",
            lambda **k: (seen.extend(label for label, _an in k["capture_variants"]), {"rows": [], "warnings": []})[1],
        )
        bta._maybe_build_shape_manifest(_mk_args(), {"trace_file": ""}, tmp_path, generated_at="t0")
        return seen

    forward = _run(shards)
    assert len(forward) == 4
    assert forward == _run(list(reversed(shards)))  # order-independent
    # ...and it spans the range rather than hugging the decode end. Both ends
    # inclusive: an even but half-open slice stops short of the tail, dropping
    # the widest batch -- the one prefill variant the spread exists to keep.
    assert forward[0] == "bs_1", forward
    assert forward[-1] == f"bs_{batches[-1]}", forward
    assert len(set(forward)) == 4, forward


def test_capture_shards_are_capped_by_default(tmp_path, monkeypatch):
    # Each shard costs an analyze_trace pass plus a sha256; with the manifest
    # now built on every run, an uncapped default would put that cost on all of
    # them. The cap is a budget, not a filter -- the drop is logged.
    monkeypatch.delenv("HYPERLOOM_TRACE_SHAPE_MANIFEST", raising=False)
    monkeypatch.delenv("HYPERLOOM_TRACE_SHAPE_MANIFEST_MAX_CAPTURES", raising=False)
    n = bta._DEFAULT_MAX_CAPTURES + 10
    monkeypatch.setattr(
        bta,
        "_discover_capture_shards",
        lambda _trace, _folder: [(Path(f"/tmp/bs_{i}.json"), f"bs_{i}", "decode") for i in range(n)],
    )
    analyzed: list[object] = []

    def _analyze(path, **kw):
        analyzed.append(path)
        return {"status": "empty"}

    monkeypatch.setattr(bta._reader, "analyze_trace", _analyze)
    bta._maybe_build_shape_manifest(_mk_args(), {"trace_file": ""}, tmp_path, generated_at="t0")

    assert len(analyzed) == bta._DEFAULT_MAX_CAPTURES


def _prov_args(**kw):
    base = dict(
        model_name="m",
        model_path="/p",
        framework="vllm",
        target_platform="mi355x",
        precision="fp8",
        trace_input="t",
        capture_folder="",
        analysis_mode="decode",
    )
    base.update(kw)
    return _argparse.Namespace(**base)


def test_build_manifest_provenance_shared_path(monkeypatch):
    monkeypatch.setattr(bta, "_shared_build_provenance", lambda args, env, probe: {"_provenance_source": "shared"})
    assert bta._build_manifest_provenance(_prov_args())["_provenance_source"] == "shared"


def test_build_manifest_provenance_stub_when_shared_absent(monkeypatch):
    monkeypatch.setattr(bta, "_shared_build_provenance", None)
    out = bta._build_manifest_provenance(_prov_args())
    assert out["_provenance_source"] == "wp1_stub"
    assert out["model_name"] == "m" and out["dtype"] == "fp8"


def test_build_manifest_provenance_stub_when_shared_raises(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("shared blew up")

    monkeypatch.setattr(bta, "_shared_build_provenance", _boom)
    assert bta._build_manifest_provenance(_prov_args())["_provenance_source"] == "wp1_stub"


def test_discover_capture_shards_dedups_tp_ranks(tmp_path):
    # TP>1 emits bs_16_rank0 / bs_16_rank1 (same shapes, different rank). They
    # must collapse to ONE bs_16 variant, not two duplicate-labeled shards that
    # overwrite each other's hash/meta and inflate variant_count.
    d = tmp_path / "caps"
    d.mkdir()
    for f in ("bs_16_rank0.json", "bs_16_rank1.json", "bs_64_rank0.json"):
        (d / f).write_text("{}", encoding="utf-8")
    shards = bta._discover_capture_shards(str(d), str(d))
    labels = sorted(lbl for _p, lbl, _m in shards)
    assert labels == ["bs_16", "bs_64"]  # bs_16 deduped 2 ranks -> 1
    assert len(shards) == 2


def test_discover_capture_shards_dedups_a_runner_prefixed_batch(tmp_path):
    # An SGLang without the profiler patch prefixes the runner name:
    # DecodeCudaGraphRunner_bs_104_rank0. The batch token is still what
    # identifies the variant, so the ranks must collapse the same way — an
    # anchored read finds nothing here and falls back to the per-rank stem,
    # which silently turns one variant into one-per-rank.
    d = tmp_path / "graph_capture_profile"
    d.mkdir()
    for f in (
        "DecodeCudaGraphRunner_bs_104_rank0.json",
        "DecodeCudaGraphRunner_bs_104_rank1.json",
        "DecodeCudaGraphRunner_bs_8_rank0.json",
    ):
        (d / f).write_text("{}", encoding="utf-8")
    shards = bta._discover_capture_shards(str(d), str(d))
    assert sorted(lbl for _p, lbl, _m in shards) == ["bs_104", "bs_8"]


def test_discover_capture_shards_skips_a_whole_capture_per_rank_file(tmp_path):
    # ``cuda_graph_capture-<runner>-TP-<n>`` is one file per rank holding every
    # batch size at once, so it carries no variant identity. Indexing it mints a
    # bogus variant per rank (TP=8 -> eight shape-identical entries) because the
    # label falls back to the stem, which differs per rank. It is a capture
    # sidecar — the shared classifier is right to demote it from analysis — but
    # that is a wider question than "does this shard name a variant".
    d = tmp_path / "graph_capture_profile"
    d.mkdir()
    for rank in range(3):
        (d / f"cuda_graph_capture-DecodeCudaGraphRunner-TP-{rank}.json").write_text("{}", encoding="utf-8")
    assert bta._discover_capture_shards(str(d), str(d)) == []


# ── CUDA/HIP graph-mode under-recording ──────────────────────────────────────


def _graph_under_recorded_events():
    """Synthetic graph-mode trace: 4 graph launches but only 1 replay's kernels
    recorded, spanning a large wall clock (busy fraction << 0.5)."""
    events = [{"cat": "cpu_op", "name": "aten::mm", "args": {"External id": 200}}]
    # Four graph-launch runtime events (no External id, only correlation).
    for i, corr in enumerate((5, 6, 7, 8)):
        events.append(
            {"cat": "cuda_runtime", "name": "hipGraphLaunch", "args": {"correlation": corr, "External id": None}}
        )
    # A normally-launched kernel (attributable) and graph-internal kernels for a
    # single recorded replay (correlation 5). Total wall span ~10s, busy ~200us.
    events += [
        {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 99, "External id": 200}},
        {"cat": "kernel", "ph": "X", "name": "Cijk_Alik_Bljk_HHS", "ts": 1000, "dur": 100, "args": {"correlation": 99}},
        {"cat": "kernel", "ph": "X", "name": "graph_fused_attn", "ts": 1100, "dur": 100, "args": {"correlation": 5}},
        {
            "cat": "kernel",
            "ph": "X",
            "name": "graph_fused_mlp",
            "ts": 10_000_000,
            "dur": 100,
            "args": {"correlation": 5},
        },
    ]
    return events


def test_reader_marks_graph_attributed_not_unlinked(tmp_path):
    # Graph-launch correlations classify replayed kernels as graph-attributed,
    # never as (unlinked); flags graph_mode / graph_under_recorded.
    trace = tmp_path / "g.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _graph_under_recorded_events()}).encode("utf-8"))
    out = bta._reader.analyze_trace(str(trace), top_k=0)
    attr = out["attribution"]
    cov = out["graph_coverage"]
    assert attr["graph_mode"] is True
    assert attr["graph_launch_count"] == 4
    assert attr["graph_attributed_kernels"] == 2
    assert attr["unlinked_kernels"] == 0
    assert cov["graph_under_recorded"] is True
    assert cov["busy_fraction"] < 0.5
    # graph-internal kernels are not surfaced as a (unlinked) op bucket
    assert "(unlinked)" not in {o["name"] for o in out["ops"]}


def test_reader_no_graph_mode_when_no_graph_launches(tmp_path):
    trace = tmp_path / "ng.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _TRACE_EVENTS}).encode("utf-8"))
    out = bta._reader.analyze_trace(str(trace), top_k=0)
    assert out["attribution"]["graph_mode"] is False
    assert out["graph_coverage"]["graph_under_recorded"] is False


def test_graph_under_recorded_skips_idle_gate_keeps_candidates(tmp_path, capsys, monkeypatch):
    # Under-recorded graph trace: idle% is ~99% but the idle gate must NOT clear
    # candidates; instead a bypass_graph_under_recorded warning is surfaced.
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    monkeypatch.delenv("HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD", raising=False)
    trace = tmp_path / "g.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _graph_under_recorded_events()}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert rc == 0
    assert result["timeline"]["idle_pct"] > 80.0
    assert result["graph_coverage"]["graph_under_recorded"] is True
    codes = {w["code"] for w in result["trace_health_warnings"]}
    assert "bypass_graph_under_recorded" in codes
    assert "high_gpu_idle_pct" not in codes
    # candidates are preserved (ranked by recorded-kernel GPU share)
    assert result["hot_kernels"], "candidates must survive the idle gate under graph under-recording"


def _graph_partial_replay_events():
    """Three graph launches but kernels from only one replay; busy_fraction ~0.78."""
    events = []
    for corr in (5, 6, 7):
        events.append(
            {"cat": "cuda_runtime", "name": "hipGraphLaunch", "args": {"correlation": corr, "External id": None}}
        )
    events += [
        {"cat": "kernel", "ph": "X", "name": "graph_k0", "ts": 1000, "dur": 2000, "args": {"correlation": 5}},
        {"cat": "kernel", "ph": "X", "name": "graph_k1", "ts": 3000, "dur": 2000, "args": {"correlation": 5}},
        {"cat": "kernel", "ph": "X", "name": "busy_fill", "ts": 1000, "dur": 7000, "args": {"correlation": 99}},
        {"cat": "kernel", "ph": "X", "name": "tail", "ts": 9000, "dur": 500, "args": {"correlation": 99}},
    ]
    return events


def test_graph_under_recorded_partial_replay_moderate_busy(tmp_path):
    trace = tmp_path / "partial.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _graph_partial_replay_events()}).encode("utf-8"))
    cov = bta._reader.analyze_trace(str(trace), top_k=0)["graph_coverage"]
    assert cov["graph_launch_count"] == 3
    assert 0.5 < cov["busy_fraction"] < 0.9
    assert cov["graph_under_recorded"] is True


def test_single_graph_launch_low_busy_not_under_recorded(tmp_path):
    events = [
        {"cat": "cuda_runtime", "name": "hipGraphLaunch", "args": {"correlation": 5}},
        {"cat": "kernel", "ph": "X", "name": "k", "ts": 1000, "dur": 50, "args": {"correlation": 5}},
        {"cat": "kernel", "ph": "X", "name": "pad", "ts": 10_000, "dur": 50, "args": {"correlation": 99}},
    ]
    trace = tmp_path / "single.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": events}).encode("utf-8"))
    cov = bta._reader.analyze_trace(str(trace), top_k=0)["graph_coverage"]
    assert cov["graph_launch_count"] == 1
    assert cov["graph_under_recorded"] is False


def _graph_fully_recorded_idle_events():
    """Four graph launches that EACH recorded a kernel (coverage 1.0) but spread
    over a long wall so busy% ~0 / idle% ~100%. This is a genuinely idle graph
    workload, NOT an under-recorded capture: recorded-launch coverage is full, so
    the idle gate must still suppress candidates (regression for the P1 review)."""
    events = []
    for corr in (5, 6, 7, 8):
        events.append(
            {"cat": "cuda_runtime", "name": "hipGraphLaunch", "args": {"correlation": corr, "External id": None}}
        )
    for i, corr in enumerate((5, 6, 7, 8)):
        events.append(
            {
                "cat": "kernel",
                "ph": "X",
                "name": f"graph_k{i}",
                "ts": 1000 + i * 3_000_000,
                "dur": 100,
                "args": {"correlation": corr},
            }
        )
    return events


def test_graph_fully_recorded_low_busy_not_under_recorded(tmp_path):
    trace = tmp_path / "idle.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _graph_fully_recorded_idle_events()}).encode("utf-8"))
    cov = bta._reader.analyze_trace(str(trace), top_k=0)["graph_coverage"]
    assert cov["graph_launch_count"] == 4
    assert cov["graph_launches_with_kernels"] == 4
    assert cov["busy_fraction"] < 0.5
    # Full recorded-launch coverage => NOT under-recorded even though busy is low.
    assert cov["graph_under_recorded"] is False


def test_fully_recorded_idle_graph_still_suppressed_by_idle_gate(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
    monkeypatch.delenv("HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD", raising=False)
    trace = tmp_path / "idle.trace.json"
    trace.write_bytes(json.dumps({"traceEvents": _graph_fully_recorded_idle_events()}).encode("utf-8"))
    rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert rc == 0
    assert result["timeline"]["idle_pct"] > 80.0
    assert result["graph_coverage"]["graph_under_recorded"] is False
    codes = {w["code"] for w in result["trace_health_warnings"]}
    # Genuinely idle (fully recorded) graph workload: idle gate MUST still fire.
    assert "high_gpu_idle_pct" in codes
    assert "bypass_graph_under_recorded" not in codes
    assert not result["hot_kernels"]


def test_finalize_graph_coverage_is_whole_trace_scoped_under_steady_window():
    """Regression: recorded-launch coverage must be computed over the FULL event
    stream, not the steady window. A fully-recorded 4-launch trace whose steady
    window happens to clip to a single replay must NOT read as 1/4 under-recorded
    (the denominator graph_launch_count is whole-trace, so the numerator must be
    too). Guards the steady-state bypass path (xDiT / HYPERLOOM_BYPASS_STEADY_STATE)."""
    # One recorded kernel per graph launch (coverage 1.0 on the full trace),
    # spread far apart in time so a narrow window contains only the first replay.
    k_events = [
        ("graph_k0", 100.0, 5, 1000.0, 1100.0),
        ("graph_k1", 100.0, 6, 3_000_000.0, 3_000_100.0),
        ("graph_k2", 100.0, 7, 6_000_000.0, 6_000_100.0),
        ("graph_k3", 100.0, 8, 9_000_000.0, 9_000_100.0),
    ]
    out = bta._reader._finalize(
        k_events,
        [],
        {},
        {},
        {},
        window=(0.0, 2_000_000.0),  # clips to only the corr-5 replay
        top_k=0,
        graph_launch_corrs=frozenset({5, 6, 7, 8}),
        graph_launch_count=4,
    )
    cov = out["graph_coverage"]
    assert cov["graph_launch_count"] == 4
    # whole-trace scope: all four launches recorded a kernel, not just the one in
    # the window -> coverage 1.0 -> NOT under-recorded.
    assert cov["graph_launches_with_kernels"] == 4
    assert cov["graph_under_recorded"] is False


def _graph_launch_stripped_events():
    """The graph-under-recorded kernels WITHOUT the hipGraphLaunch runtime events
    -- i.e. what a steady-state chunk can look like after the splitter drops the
    launch records. Detection must see this as non-graph (so probing the chunk
    instead of the raw trace would miss the artifact)."""
    return [
        {"cat": "kernel", "ph": "X", "name": "graph_fused_attn", "ts": 1100, "dur": 100, "args": {"correlation": 5}},
        {
            "cat": "kernel",
            "ph": "X",
            "name": "graph_fused_mlp",
            "ts": 10_000_000,
            "dur": 100,
            "args": {"correlation": 5},
        },
    ]


def test_graph_launch_events_required_for_detection(tmp_path):
    # Raw trace (with hipGraphLaunch events) -> detected as graph, under-recorded.
    raw = tmp_path / "raw.trace.json"
    raw.write_bytes(json.dumps({"traceEvents": _graph_under_recorded_events()}).encode("utf-8"))
    raw_cov = bta._reader.analyze_trace(str(raw), top_k=0)["graph_coverage"]
    assert raw_cov["graph_mode"] is True
    assert raw_cov["graph_under_recorded"] is True
    # Chunk with the launch runtime events stripped -> looks non-graph, so the
    # artifact would be missed. This is why the guard must probe the RAW trace.
    chunk = tmp_path / "chunk.trace.json"
    chunk.write_bytes(json.dumps({"traceEvents": _graph_launch_stripped_events()}).encode("utf-8"))
    chunk_cov = bta._reader.analyze_trace(str(chunk), top_k=0)["graph_coverage"]
    assert chunk_cov["graph_mode"] is False
    assert chunk_cov["graph_under_recorded"] is False


# ---- trace-health: impossible durations + denoise-step inference ----------
def _write_events(path: Path, events: list[dict]) -> Path:
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return path


def _result_codes(result) -> set[str]:
    return {w.get("code") for w in (result.get("trace_health_warnings") or [])}


def test_impossible_durations_raise_a_warning(tmp_path, capsys):
    """A kernel overrunning its successor must be surfaced, not ranked silently."""
    events = list(_TRACE_EVENTS) + [
        {"cat": "kernel", "ph": "X", "name": "corrupt", "ts": 1500, "dur": 900_000, "pid": 2, "tid": 3},
        {"cat": "kernel", "ph": "X", "name": "after_a", "ts": 1600, "dur": 100, "pid": 2, "tid": 3},
        {"cat": "kernel", "ph": "X", "name": "after_b", "ts": 1700, "dur": 100, "pid": 2, "tid": 3},
    ]
    trace = _write_events(tmp_path / "corrupt.trace.json", events)
    _rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert "bypass_impossible_kernel_durations" in _result_codes(result)
    msg = next(
        w["message"] for w in result["trace_health_warnings"] if w["code"] == "bypass_impossible_kernel_durations"
    )
    assert "corrupt" in msg
    assert "serial stream" in msg


def test_clean_trace_raises_no_duration_warning(tmp_path, capsys):
    trace = _write_events(tmp_path / "clean.trace.json", list(_TRACE_EVENTS))
    _rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    assert "bypass_impossible_kernel_durations" not in _result_codes(result)


def test_denoise_steps_come_from_profiler_steps_not_annotations(tmp_path, capsys):
    """Regression: the divisor must be denoise steps, not annotation windows.

    ``annotation_window_count`` counts user_annotation ranges. Under
    source-level instrumentation there are far more of those than denoise
    steps, and using them silently scaled every per-step figure.
    """
    events = list(_TRACE_EVENTS)
    # 3 real denoise steps ...
    for i in range(3):
        events.append({"cat": "gpu_user_annotation", "ph": "X", "name": f"ProfilerStep#{i}", "ts": 1000 + i, "dur": 1})
    # ... but many more annotation windows, as dense instrumentation produces.
    for i in range(25):
        events.append({"cat": "gpu_user_annotation", "ph": "X", "name": f"block_{i}", "ts": 1000 + i, "dur": 1})

    trace = _write_events(tmp_path / "ann.trace.json", events)
    _rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    steps = result.get("num_denoise_steps")
    assert steps == 3, f"expected the 3 ProfilerStep markers, got {steps}"
    assert steps != 28, "annotation_window_count must not be used as the step count"


def test_explicit_denoise_steps_wins_and_is_flagged(tmp_path, capsys):
    """An operator's explicit --num-denoise-steps is authoritative.

    Hyperloom cannot know what a user's ``prof.step()`` brackets, so a declared
    count must win over the trace-inferred one -- and the reported value must
    match the divisor actually used, which previously disagreed.
    """
    events = list(_TRACE_EVENTS) + [
        {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#0", "ts": 1000, "dur": 1}
    ]
    trace = _write_events(tmp_path / "ann2.trace.json", events)
    _rc, result, _ = _run(_base_argv(tmp_path, str(trace), extra=["--num-denoise-steps", "9"]), capsys)
    assert result.get("num_denoise_steps") == 9
    # The 1 inferred step disagrees with the requested 9, which must be flagged.
    assert "bypass_denoise_steps_mismatch" in _result_codes(result)
    msg = next(w["message"] for w in result["trace_health_warnings"] if w["code"] == "bypass_denoise_steps_mismatch")
    assert "requested count wins" in msg


def test_denoise_steps_read_the_file_the_reader_analyzed(tmp_path, capsys):
    """Directory input must not count markers in a different file.

    ``count_profiler_steps`` globs ``*.json`` and takes the first, so a stray
    ``config.json`` sorts ahead of the real trace and returns 0 while the
    reader resolved the actual trace.
    """
    d = tmp_path / "torch_trace"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"not": "a trace", "pad": "x" * 10}), encoding="utf-8")
    events = list(_TRACE_EVENTS) + [
        {"cat": "gpu_user_annotation", "ph": "X", "name": f"ProfilerStep#{i}", "ts": 1000 + i, "dur": 1}
        for i in range(4)
    ]
    (d / "trace.json").write_text(json.dumps({"traceEvents": events}), encoding="utf-8")

    _rc, result, _ = _run(_base_argv(tmp_path, str(d)), capsys)
    assert result.get("num_denoise_steps") == 4, "must count steps in the analyzed trace, not config.json"


def test_duration_warning_severity_is_graded(tmp_path, capsys):
    """A materially wrong ranking warns; a mild perturbation only informs."""
    base = [
        {"cat": "kernel", "ph": "X", "name": f"k{i}", "ts": 1000 + i * 10_000, "dur": 10_000, "pid": 2, "tid": 3}
        for i in range(40)
    ]
    severe = [dict(base[0], name="corrupt", dur=900_000)] + base[1:]
    trace = _write_events(tmp_path / "severe.trace.json", severe)
    _rc, result, _ = _run(_base_argv(tmp_path, str(trace)), capsys)
    warn = next((w for w in result["trace_health_warnings"] if w["code"] == "bypass_impossible_kernel_durations"), None)
    assert warn is not None and warn["severity"] == "warning"
    assert "unreliable" in warn["message"]
    assert "% of the" in warn["message"], "share of summed device time should be reported"
