# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for structured roofline snapshot extraction + rendering."""

from __future__ import annotations

import json

from inference_optimizer.orchestrator import roofline_snapshot as rs


_EXEC_MD = """\
# analysis
| Metric | Value |
|--------|-------|
| Compute % | 70.5% |
| Idle % | 12.0% |
| Exposed Communication % | 3.2% |
| Top Bottleneck Category | MoE_fused (28.78%) |
| Memory % | 85.0% |
"""


# ---- small parsers ----

def test_parse_pct():
    assert rs._parse_pct("28.78%") == 28.78
    assert rs._parse_pct("1,234.5") == 1234.5
    assert rs._parse_pct(None) is None
    assert rs._parse_pct("n/a") is None


def test_parse_executive_table_skips_header():
    rows = rs._parse_executive_table(_EXEC_MD)
    assert rows["Compute %"] == "70.5%"
    assert "Metric" not in rows


def test_parse_top_bottleneck():
    assert rs._parse_top_bottleneck("MoE_fused (28.78%)") == "MoE_fused"
    assert rs._parse_top_bottleneck(None) is None
    assert rs._parse_top_bottleneck("   ") is None


# ---- extract_workload_summary ----

def test_extract_workload_summary_missing(tmp_path):
    out = rs.extract_workload_summary(tmp_path / "no.md")
    assert out["compute_pct"] is None


def test_extract_workload_summary_full(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text(_EXEC_MD, encoding="utf-8")
    out = rs.extract_workload_summary(md)
    assert out["compute_pct"] == 70.5
    assert out["idle_pct"] == 12.0
    assert out["comm_pct"] == 3.2
    assert out["top_bottleneck"] == "MoE_fused"


# ---- derive_saturation_per_direction ----

def test_derive_saturation_empty():
    out = rs.derive_saturation_per_direction("")
    assert out == {"compute": 0.0, "memory": 0.0, "host_overhead": 0.0, "comm": 0.0}


def test_derive_saturation_full():
    out = rs.derive_saturation_per_direction(_EXEC_MD)
    assert out["compute"] == 70.5
    assert out["memory"] == 85.0
    assert out["host_overhead"] == 12.0  # via "Idle %"
    assert out["comm"] == 3.2


# ---- extract_top_kernel ----

def test_extract_top_kernel_no_dir(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text("x", encoding="utf-8")
    assert rs.extract_top_kernel(md) is None


def test_extract_top_kernel_picks_highest(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text("x", encoding="utf-8")
    cat = tmp_path / "category_data"
    cat.mkdir()
    (cat / "gemm_metrics.json").write_text(json.dumps({
        "category": "gemm",
        "operations": [
            {"name": "small", "percent_of_total": 5.0},
            {"name": "big", "percent_of_total": 40.0,
             "efficiency": {"efficiency_percent": "65%", "bound_type": "compute"}},
        ],
    }), encoding="utf-8")
    (cat / "bad_metrics.json").write_text("{not json", encoding="utf-8")
    top = rs.extract_top_kernel(md)
    assert top["name"] == "big"
    assert top["gpu_pct"] == 40.0
    assert top["efficiency_pct"] == 65.0
    assert top["bound_type"] == "compute"


def test_extract_top_kernel_unnamed_returns_none(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text("x", encoding="utf-8")
    cat = tmp_path / "category_data"
    cat.mkdir()
    (cat / "x_metrics.json").write_text(json.dumps({
        "operations": [{"name": "", "percent_of_total": 10.0}],
    }), encoding="utf-8")
    assert rs.extract_top_kernel(md) is None


# ---- _compute_within_and_gap ----

def test_compute_within_and_gap():
    assert rs._compute_within_and_gap(peak=0, achieved=10) == (None, None)
    within, gap = rs._compute_within_and_gap(peak=100, achieved=80)
    assert within == 80.0
    assert gap == 20.0


# ---- build_roofline_snapshot ----

def test_build_roofline_snapshot_no_analysis():
    snap = rs.build_roofline_snapshot(
        snapshot_id=1, ts="t0", analysis_md_path="",
        theoretical_peak_tok_per_sec=100.0, achieved_tok_per_sec=80.0,
    )
    assert snap["within_roofline_pct"] == 80.0
    assert snap["theoretical_peak_tok_per_sec"] == 100.0
    assert snap["compute_pct"] is None


def test_build_roofline_snapshot_with_analysis(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text(_EXEC_MD, encoding="utf-8")
    cat = tmp_path / "category_data"
    cat.mkdir()
    (cat / "g_metrics.json").write_text(json.dumps({
        "operations": [{"name": "k", "percent_of_total": 30.0,
                         "efficiency": {"efficiency_percent": "50%"}}],
    }), encoding="utf-8")
    snap = rs.build_roofline_snapshot(snapshot_id=2, ts="t1", analysis_md_path=str(md))
    assert snap["compute_pct"] == 70.5
    assert snap["top_kernel"]["name"] == "k"


# ---- _snapshot_id_from_meta / _num_delta ----

def test_snapshot_id_from_meta():
    assert rs._snapshot_id_from_meta({"snapshot_id": 3}) == 3
    assert rs._snapshot_id_from_meta({"roofline_snapshot_id": 4}) == 4
    assert rs._snapshot_id_from_meta({"snapshot_id": "x"}) is None


def test_num_delta():
    assert rs._num_delta(10.0, 7.0) == 3.0
    assert rs._num_delta(None, 1.0) is None


# ---- build_roofline_comparison_from_history ----

def test_comparison_from_history_empty():
    assert rs.build_roofline_comparison_from_history(None) is None
    assert rs.build_roofline_comparison_from_history([]) is None


def test_comparison_from_history_single():
    snaps = [{"snapshot_id": 1, "compute_pct": 50.0}]
    out = rs.build_roofline_comparison_from_history(snaps)
    assert out["mode"] == "single_snapshot"
    assert "delta" not in out


def test_comparison_from_history_before_after():
    snaps = [
        {"snapshot_id": 1, "compute_pct": 50.0, "top_kernel": {"efficiency_pct": 40.0}},
        {"snapshot_id": 2, "compute_pct": 60.0, "top_kernel": {"efficiency_pct": 55.0}},
    ]
    out = rs.build_roofline_comparison_from_history(snaps)
    assert out["mode"] == "before_after"
    assert out["delta"]["compute_pct"] == 10.0
    assert out["delta"]["top_kernel_efficiency_pct"] == 15.0


# ---- build_roofline_comparison ----

def test_build_comparison_no_paths():
    assert rs.build_roofline_comparison({}, {}) is None


def test_build_comparison_before_after(tmp_path):
    base_md = tmp_path / "base" / "analysis.md"
    latest_md = tmp_path / "latest" / "analysis.md"
    base_md.parent.mkdir()
    latest_md.parent.mkdir()
    base_md.write_text(_EXEC_MD, encoding="utf-8")
    latest_md.write_text(_EXEC_MD.replace("70.5", "80.0"), encoding="utf-8")
    out = rs.build_roofline_comparison(
        {"analysis_md_path": str(base_md), "ts": "t0", "snapshot_id": 1},
        {"analysis_md_path": str(latest_md), "ts": "t1", "snapshot_id": 2},
    )
    assert out["mode"] == "before_after"
    assert out["delta"]["compute_pct"] == 9.5


def test_build_comparison_single_snapshot(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text(_EXEC_MD, encoding="utf-8")
    out = rs.build_roofline_comparison(
        {"analysis_md_path": str(md), "ts": "t0", "snapshot_id": 1},
        {"analysis_md_path": str(md), "ts": "t0", "snapshot_id": 1},
    )
    assert out["mode"] == "single_snapshot"


# ---- formatters ----

def test_fmt_helpers():
    assert rs._fmt_delta(None) == "—"
    assert rs._fmt_delta(1.2) == "+1.2"
    assert rs._fmt_delta(-1.0) == "-1.0"
    assert rs._fmt_tput(None) == "—"
    assert rs._fmt_tput(0) == "—"
    assert rs._fmt_tput(12.34) == "12.3 tok/s"
    assert rs._fmt_pct_cell(None) == "—"
    assert rs._fmt_pct_cell(50.0) == "50.0%"


def test_format_table_single_snapshot():
    cmp = {
        "mode": "single_snapshot",
        "baseline": {
            "compute_pct": 70.0, "idle_pct": 10.0, "comm_pct": 2.0,
            "top_bottleneck": "moe", "top_kernel": {"efficiency_pct": 50.0, "name": "k"},
            "theoretical_peak_tok_per_sec": 100.0,
            "achieved_tok_per_sec": 80.0, "within_roofline_pct": 80.0,
            "gap_to_roofline_pct": 20.0,
        },
    }
    lines = rs.format_roofline_metrics_table(cmp)
    body = "\n".join(lines)
    assert "Theoretical peak" in body
    assert "| Compute % | 70.0% |" in body
    assert "`k`" in body


def test_format_table_before_after():
    cmp = {
        "mode": "before_after",
        "baseline": {"compute_pct": 70.0, "top_kernel": {"name": "a", "efficiency_pct": 40.0},
                      "theoretical_peak_tok_per_sec": 100.0},
        "latest": {"compute_pct": 80.0, "top_kernel": {"name": "b", "efficiency_pct": 55.0}},
        "delta": {"compute_pct": 10.0, "top_kernel_efficiency_pct": 15.0},
    }
    lines = rs.format_roofline_metrics_table(cmp)
    body = "\n".join(lines)
    assert "| Metric | Base | Opt | Δ |" in body
    assert "+10.0" in body
    assert "`a`" in body and "`b`" in body
