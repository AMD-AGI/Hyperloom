###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Tests for emit_kernel_candidates.py (offline analysis.md → kernel_candidates.json)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TOOL_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

import emit_kernel_candidates as ekc  # noqa: E402


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _exec_summary_md(*, idle_pct: float = 0.0, compute_pct: float = 100.0, comm_pct: float = 0.0) -> str:
    """Minimal analysis.md Executive Summary table for the idle/compute extractors."""
    return (
        "# Performance Analysis Report\n\n"
        "## Executive Summary\n\n"
        "| Metric | Value |\n"
        "| --- | --- |\n"
        f"| Idle % | {idle_pct}% |\n"
        f"| Compute % | {compute_pct}% |\n"
        f"| Exposed Communication % | {comm_pct}% |\n"
    )


def _analysis_output(tmp_path: Path, *, md: str | None = None) -> Path:
    """Create an analysis_output directory with analysis.md."""
    if md is None:
        md = _exec_summary_md()
    _write(tmp_path / "analysis.md", md)
    return tmp_path


def _stub_emit_deps(monkeypatch, *, parsed=None, recovered=None, finalized=None):
    """Replace TraceLens/Hyperloom emit helpers with in-process stubs."""
    parsed = list(parsed or [])
    recovered = list(recovered or [])
    if finalized is None:
        finalized = list(parsed) + list(recovered)

    monkeypatch.setattr(ekc, "parse_analysis_md", lambda *_a, **_k: list(parsed))
    monkeypatch.setattr(ekc, "recover_other_bucket_candidates", lambda *_a, **_k: list(recovered))
    monkeypatch.setattr(ekc, "_inject_collective_candidates", lambda _d, cands, **_k: list(cands))
    monkeypatch.setattr(ekc, "_extract_total_time_us_from_gpu_timeline", lambda *_a, **_k: 10890.0)
    monkeypatch.setattr(ekc, "_finalize_candidates", lambda cands, **_k: list(finalized))
    monkeypatch.setattr(ekc, "load_roofline_results", lambda *_a, **_k: {})
    monkeypatch.setattr(ekc, "merge_roofline_into_candidates", lambda *_a, **_k: None)

    written: dict[str, object] = {}

    def _write_reports(run_dir, **kwargs):
        written["run_dir"] = run_dir
        written["candidates"] = list(kwargs.get("candidates") or [])
        written["kwargs"] = kwargs
        path = Path(run_dir) / "kernel_candidates.json"
        path.write_text("{}", encoding="utf-8")
        return {"kernel_candidates": str(path)}

    monkeypatch.setattr(ekc, "write_reports", _write_reports)
    return written


def test_overlay_metrics_kernel_names_replaces_truncated_symbols(tmp_path):
    metrics = {
        "operations": [
            {
                "name": "aten::mm",
                "kernel_name": "Kernel 1: Cijk_B_PostGSU2<br>Kernel 2: Cijk_Ailk_Bljk_FULL",
                "args": "(256,131072) bf16<br>(131072,512) bf16",
            }
        ]
    }
    _write(tmp_path / "category_data" / "gemm_metrics.json", json.dumps(metrics))
    cands = [
        {
            "name": "aten::mm",
            "device_kernel_name": "Cijk_Ailk_Bljk_A...",
            "device_kernel_names": ["Cijk_B_PostGSU2", "Cijk_Ailk_Bljk_A..."],
            "shapes": [],
        }
    ]
    ekc.overlay_metrics_kernel_names(cands, tmp_path)
    assert cands[0]["device_kernel_name"] == "Cijk_B_PostGSU2"
    assert cands[0]["device_kernel_names"] == ["Cijk_B_PostGSU2", "Cijk_Ailk_Bljk_FULL"]
    assert cands[0]["shapes"] == ["(256,131072) bf16", "(131072,512) bf16"]


def test_overlay_metrics_kernel_names_skips_fusion_sidecar(tmp_path):
    fusion = {
        "operations": [
            {"name": "aten::mm", "kernel_name": "Kernel 1: should_not_apply", "args": "(1,1) bf16"}
        ]
    }
    gemm = {
        "operations": [
            {"name": "aten::mm", "kernel_name": "Kernel 1: real_symbol", "args": "(2,2) bf16"}
        ]
    }
    _write(tmp_path / "category_data" / "kernel_fusion_metrics.json", json.dumps(fusion))
    _write(tmp_path / "category_data" / "gemm_metrics.json", json.dumps(gemm))
    cands = [
        {
            "name": "aten::mm",
            "device_kernel_name": "trunc...",
            "device_kernel_names": ["trunc..."],
            "shapes": ["already"],
        }
    ]
    ekc.overlay_metrics_kernel_names(cands, tmp_path)
    assert cands[0]["device_kernel_name"] == "real_symbol"
    assert cands[0]["shapes"] == ["already"]


def test_infer_model_name_ignores_placeholder(tmp_path):
    _write(
        tmp_path / "metadata" / "model_info.json",
        json.dumps({"model": "Cannot be inferred from trace"}),
    )
    assert ekc._infer_model_name(tmp_path) == ""
    _write(tmp_path / "metadata" / "model_info.json", json.dumps({"model": "n/a"}))
    assert ekc._infer_model_name(tmp_path) == ""


def test_infer_model_name_returns_real_name(tmp_path):
    _write(tmp_path / "metadata" / "model_info.json", json.dumps({"model": "Llama-3-70B"}))
    assert ekc._infer_model_name(tmp_path) == "Llama-3-70B"


def test_infer_platform_and_analysis_mode_from_manifest():
    manifest = {"platform": "MI300X", "comparison_scope": "standalone"}
    assert ekc._infer_platform(manifest, "MI355X") == "MI300X"
    assert ekc._infer_analysis_mode(manifest, "default") == "standalone"
    assert ekc._infer_platform({}, "MI355X") == "MI355X"
    assert ekc._infer_analysis_mode({}, "default") == "default"


def test_resolve_trace_input_prefers_explicit_then_manifest(tmp_path):
    explicit = tmp_path / "explicit.json"
    explicit.write_text("{}", encoding="utf-8")
    manifest_trace = tmp_path / "from_manifest.json"
    manifest_trace.write_text("{}", encoding="utf-8")
    analysis_output = tmp_path / "analysis_output"
    analysis_output.mkdir()
    resolved = ekc._resolve_trace_input(
        explicit=str(explicit),
        manifest={"trace_path": str(manifest_trace)},
        analysis_output=analysis_output,
    )
    assert resolved == explicit.resolve()
    resolved = ekc._resolve_trace_input(
        explicit="",
        manifest={"trace_path": str(manifest_trace)},
        analysis_output=analysis_output,
    )
    assert resolved == manifest_trace
    resolved = ekc._resolve_trace_input(explicit="", manifest={}, analysis_output=analysis_output)
    assert resolved == analysis_output.resolve()


def test_emit_requires_analysis_md(tmp_path):
    with pytest.raises(FileNotFoundError, match="analysis.md is required"):
        ekc.emit_kernel_candidates_from_analysis_output(tmp_path, tmp_path / "out")


def test_emit_happy_path_writes_reports(tmp_path, monkeypatch):
    analysis = _analysis_output(tmp_path / "analysis_output")
    _write(
        analysis / "category_data" / "category_manifest.json",
        json.dumps({"platform": "MI300X", "comparison_scope": "standalone"}),
    )
    parsed = [{"name": "aten::mm", "duration_us": 10890.0, "device_kernel_name": "k..."}]
    written = _stub_emit_deps(monkeypatch, parsed=parsed, finalized=parsed)
    out_dir = tmp_path / "out"
    result = ekc.emit_kernel_candidates_from_analysis_output(analysis, out_dir)
    assert result["tool"] == "emit_kernel_candidates"
    assert result["hot_kernel_count"] == 1
    assert result["report_source"] == "analysis.md"
    assert result["idle_pct"] == 0.0
    assert result["compute_pct"] == 100.0
    assert written["candidates"] == parsed
    assert Path(result["kernel_candidates_path"]) == out_dir / "kernel_candidates.json"


def test_emit_recovers_other_bucket_when_analysis_md_empty(tmp_path, monkeypatch):
    analysis = _analysis_output(tmp_path / "analysis_output")
    recovered = [{"name": "fused_moe_kernel", "duration_us": 6700.0}]
    written = _stub_emit_deps(monkeypatch, parsed=[], recovered=recovered, finalized=recovered)
    result = ekc.emit_kernel_candidates_from_analysis_output(analysis, tmp_path / "out")
    assert result["report_source"] == "analysis.md+other_bucket_fallback"
    assert result["hot_kernel_count"] == 1
    assert written["candidates"][0]["name"] == "fused_moe_kernel"


def test_emit_empty_analysis_md_returns_empty_hot_kernels(tmp_path, monkeypatch):
    """No P-item rows and no other-bucket recovery is a valid empty emit."""
    analysis = _analysis_output(tmp_path / "analysis_output")
    written = _stub_emit_deps(monkeypatch, parsed=[], recovered=[], finalized=[])
    result = ekc.emit_kernel_candidates_from_analysis_output(analysis, tmp_path / "out")
    assert result["hot_kernel_count"] == 0
    assert result["report_source"] == "analysis.md"
    assert written["candidates"] == []


def test_emit_raises_when_finalize_drops_all_candidates(tmp_path, monkeypatch):
    analysis = _analysis_output(tmp_path / "analysis_output")
    _stub_emit_deps(
        monkeypatch,
        parsed=[{"name": "aten::mm", "duration_us": 1.0}],
        recovered=[],
        finalized=[],
    )
    with pytest.raises(RuntimeError, match="No hot-kernel candidates"):
        ekc.emit_kernel_candidates_from_analysis_output(analysis, tmp_path / "out")


def test_emit_high_idle_suppresses_hot_kernels(tmp_path, monkeypatch):
    analysis = _analysis_output(
        tmp_path / "analysis_output",
        md=_exec_summary_md(idle_pct=95.0, compute_pct=5.0),
    )
    parsed = [{"name": "aten::mm"}]
    written = _stub_emit_deps(monkeypatch, parsed=parsed, finalized=parsed)
    result = ekc.emit_kernel_candidates_from_analysis_output(analysis, tmp_path / "out")
    assert result["hot_kernel_count"] == 0
    assert "skipped:high_gpu_idle_pct" in result["report_source"]
    assert any(w.get("code") == "high_gpu_idle_pct" for w in result["trace_health_warnings"])
    assert written["candidates"] == []


def test_emit_low_compute_suppresses_hot_kernels(tmp_path, monkeypatch):
    analysis = _analysis_output(
        tmp_path / "analysis_output",
        md=_exec_summary_md(idle_pct=0.0, compute_pct=3.0, comm_pct=80.0),
    )
    parsed = [{"name": "aten::mm"}]
    written = _stub_emit_deps(monkeypatch, parsed=parsed, finalized=parsed)
    result = ekc.emit_kernel_candidates_from_analysis_output(analysis, tmp_path / "out")
    assert result["hot_kernel_count"] == 0
    assert "skipped:low_gpu_compute_pct" in result["report_source"]
    assert any(w.get("code") == "low_gpu_compute_pct" for w in result["trace_health_warnings"])
    assert written["candidates"] == []


def test_emit_graph_under_recorded_keeps_candidates(tmp_path, monkeypatch):
    analysis = _analysis_output(
        tmp_path / "analysis_output",
        md=_exec_summary_md(idle_pct=95.0, compute_pct=3.0),
    )
    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    parsed = [{"name": "aten::mm", "duration_us": 1.0}]
    written = _stub_emit_deps(monkeypatch, parsed=parsed, finalized=parsed)
    graph_warning = {
        "code": "graph_under_recorded",
        "severity": "warning",
        "message": "profiler captured ~1 of N graph replays",
    }
    monkeypatch.setattr(
        ekc,
        "_evaluate_idle_gate_with_graph_guard",
        lambda *_a, **_k: (80.0, {"code": "high_gpu_idle_pct"}, graph_warning),
    )
    result = ekc.emit_kernel_candidates_from_analysis_output(
        analysis,
        tmp_path / "out",
        probe_graph=True,
        trace_input=str(trace),
    )
    assert result["hot_kernel_count"] == 1
    assert result["report_source"] == "analysis.md"
    assert result["trace_health_warnings"] == [graph_warning]
    assert written["candidates"] == parsed


def test_build_parser_requires_analysis_output():
    parser = ekc.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    ns = parser.parse_args(["--analysis-output", "/tmp/out"])
    assert ns.analysis_output == "/tmp/out"
    assert ns.out_dir == ""


def test_main_prints_json_summary(tmp_path, monkeypatch, capsys):
    analysis = _analysis_output(tmp_path / "analysis_output")
    parsed = [{"name": "aten::mm", "duration_us": 1.0}]
    _stub_emit_deps(monkeypatch, parsed=parsed, finalized=parsed)
    monkeypatch.setattr(
        sys,
        "argv",
        ["emit_kernel_candidates.py", "--analysis-output", str(analysis), "--out-dir", str(tmp_path / "out")],
    )
    assert ekc.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "emit_kernel_candidates"
    assert payload["hot_kernel_count"] == 1
