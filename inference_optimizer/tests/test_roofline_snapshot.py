"""Structured roofline snapshot extraction for final.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors.report import (
    _build_summary_dict,
    _format_roofline_comparison_section,
)
from inference_optimizer.orchestrator.roofline_snapshot import (
    build_roofline_comparison,
    build_roofline_snapshot,
    extract_top_kernel,
    extract_workload_summary,
    format_roofline_metrics_table,
)
from inference_optimizer.orchestrator.shared_state import SharedState


FIXTURE_MD = Path(__file__).resolve().parents[2] / (
    "kernel-agent/tests/fixtures/tracelens_v03_llama70b_analysis.md"
)


def test_extract_workload_summary_from_fixture():
    wl = extract_workload_summary(FIXTURE_MD)
    assert wl["compute_pct"] == pytest.approx(99.30)
    assert wl["idle_pct"] == pytest.approx(0.25)
    assert wl["comm_pct"] == pytest.approx(0.42)
    assert wl["top_bottleneck"] == "GEMM"


def test_extract_top_kernel_from_session_metrics(tmp_path):
    tracelens = tmp_path / "tracelens"
    cat = tracelens / "category_data"
    cat.mkdir(parents=True)
    (tracelens / "analysis.md").write_text("## Executive Summary\n", encoding="utf-8")
    (cat / "moe_fused_metrics.json").write_text(
        json.dumps({
            "category": "moe_fused",
            "operations": [{
                "name": "aiter::fmoe_fp8_blockscale_g1u1",
                "percent_of_total": 21.29,
                "efficiency": {
                    "efficiency_percent": 29.68,
                    "bound_type": "compute",
                },
            }],
        }),
        encoding="utf-8",
    )
    top = extract_top_kernel(tracelens / "analysis.md")
    assert top["name"] == "aiter::fmoe_fp8_blockscale_g1u1"
    assert top["gpu_pct"] == pytest.approx(21.29)
    assert top["efficiency_pct"] == pytest.approx(29.68)
    assert top["bound_type"] == "compute"


def test_build_roofline_comparison_before_after(tmp_path):
    base_md = tmp_path / "base" / "tracelens" / "analysis.md"
    opt_md = tmp_path / "opt" / "tracelens" / "analysis.md"
    for md, compute in ((base_md, "60.0"), (opt_md, "75.0")):
        md.parent.mkdir(parents=True, exist_ok=True)
        (md.parent / "category_data").mkdir(exist_ok=True)
        md.write_text(
            f"## Executive Summary\n\n"
            f"| Metric | Value |\n"
            f"| Compute % | {compute}% |\n"
            f"| Idle % | 5.0% |\n"
            f"| Exposed Communication % | 8.0% |\n"
            f"| Top Bottleneck Category | MoE_fused (20.0%) |\n",
            encoding="utf-8",
        )
        (md.parent / "category_data" / "gemm_metrics.json").write_text(
            json.dumps({
                "category": "gemm",
                "operations": [{
                    "name": "gemm_kernel",
                    "percent_of_total": 10.0,
                    "efficiency": {"efficiency_percent": 30.0, "bound_type": "compute"},
                }],
            }),
            encoding="utf-8",
        )
    cmp = build_roofline_comparison(
        {
            "roofline_snapshot_id": 1,
            "analysis_md_path": str(base_md),
            "ts": "2026-05-21T10:00:00+00:00",
        },
        {
            "snapshot_id": 2,
            "analysis_md_path": str(opt_md),
            "ts": "2026-05-22T10:00:00+00:00",
        },
    )
    assert cmp["mode"] == "before_after"
    assert cmp["baseline"]["compute_pct"] == pytest.approx(60.0)
    assert cmp["latest"]["compute_pct"] == pytest.approx(75.0)
    assert cmp["delta"]["compute_pct"] == pytest.approx(15.0)
    assert cmp["delta"]["top_kernel_efficiency_pct"] == pytest.approx(0.0)


def test_build_summary_dict_includes_structured_roofline(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text(
        "## Executive Summary\n\n"
        "| Metric | Value |\n"
        "| Compute % | 86.2% |\n"
        "| Idle % | 5.3% |\n"
        "| Exposed Communication % | 8.5% |\n"
        "| Top Bottleneck Category | MoE_fused (28.78%) |\n",
        encoding="utf-8",
    )
    state = SharedState()
    state.session_id = "test"
    state.last_trace_analyze_baseline = {
        "roofline_snapshot_id": 1,
        "analysis_md_path": str(md),
        "ts": "2026-05-21T16:49:08+00:00",
    }
    state.last_trace_analyze = {
        "roofline_snapshot_id": 1,
        "analysis_md_path": str(md),
        "ts": "2026-05-21T16:49:08+00:00",
    }
    summary = _build_summary_dict(state, {}, [])
    rc = summary["roofline_comparison"]
    assert rc["mode"] == "single_snapshot"
    assert rc["baseline"]["compute_pct"] == pytest.approx(86.2)
    assert rc["baseline"]["top_bottleneck"] == "MoE_fused"


def test_format_comparison_section_renders_table():
    cmp = {
        "mode": "before_after",
        "baseline": {
            "snapshot_id": 1,
            "compute_pct": 86.2,
            "idle_pct": 5.3,
            "comm_pct": 8.5,
            "top_bottleneck": "MoE_fused",
            "top_kernel": {
                "name": "aiter::fmoe",
                "gpu_pct": 21.3,
                "efficiency_pct": 29.7,
                "bound_type": "compute",
            },
        },
        "latest": {
            "snapshot_id": 2,
            "compute_pct": 88.1,
            "idle_pct": 3.8,
            "comm_pct": 8.2,
            "top_bottleneck": "MoE_fused",
            "top_kernel": {
                "name": "aiter::fmoe",
                "gpu_pct": 19.8,
                "efficiency_pct": 34.9,
                "bound_type": "compute",
            },
        },
        "delta": {
            "compute_pct": 1.9,
            "idle_pct": -1.5,
            "comm_pct": -0.3,
            "top_kernel_efficiency_pct": 5.2,
        },
    }
    text = "\n".join(_format_roofline_comparison_section(cmp))
    assert "| Metric | Base | Opt | Δ |" in text
    assert "29.7%" in text
    assert "34.9%" in text
    assert "+5.2" in text
    table = "\n".join(format_roofline_metrics_table(cmp))
    assert "+1.9" in table
