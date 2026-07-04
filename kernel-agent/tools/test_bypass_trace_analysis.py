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

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bypass_trace_analysis as bta  # noqa: E402

_TRACE_EVENTS = [
    {"cat": "cpu_op", "name": "aten::paged_attn", "args": {"External id": 100}},
    {"cat": "cpu_op", "name": "aten::mm", "args": {"External id": 200}},
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 5, "External id": 100}},
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 7, "External id": 200}},
    {"cat": "kernel", "ph": "X", "name": "paged_attention_v1", "ts": 1000, "dur": 300, "args": {"correlation": 5}},
    {"cat": "kernel", "ph": "X", "name": "Cijk_Alik_Bljk_HHS", "ts": 1300, "dur": 200, "args": {"correlation": 7}},
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


def _analyze(*, kernels=None, attributed_pct=100.0, steady_status=None):
    """Build a minimal analyze dict for the quality-warning unit tests."""
    out = {
        "kernels": kernels if kernels is not None else [],
        "attribution": {"attributed_pct": attributed_pct},
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


def test_xdit_marks_estimated_and_auto_steady(tmp_path, capsys, monkeypatch):
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
    assert result["estimated"] is True
    # xDiT auto-enables steady-state; the repeating ProfilerStep window is found.
    assert result["aggregation_scope"] == "steady_state"
    codes = {w["code"] for w in result["trace_health_warnings"]}
    assert "bypass_xdit_estimated" in codes
    # estimated flag flows into summary + manifest.
    summ = json.loads(Path(result["artifact_paths"]["tracelens_summary"]).read_text())
    assert summ["estimated"] is True
    manifest = json.loads(Path(result["artifact_paths"]["trace_input_manifest"]).read_text())
    assert manifest["estimated"] is True and manifest["aggregation_scope"] == "steady_state"


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
