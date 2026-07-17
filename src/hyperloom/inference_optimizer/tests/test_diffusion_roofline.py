# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the diffusion workload-level roofline aggregator.

``diffusion_roofline`` aggregates a per-kernel TraceLens CSV dir into a single
workload roofline (kernel efficiency, gpu busy ratio, per denoise-step timings).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TOOL_DIR = Path(__file__).resolve().parents[4] / "src" / "hyperloom" / "agents" / "kernel" / "tools"
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

import diffusion_roofline as dr  # noqa: E402


def _write_csvs(csv_dir: Path, *, with_timeline: bool = True) -> None:
    unified = csv_dir / dr.UNIFIED_CSV
    header = [dr.COL_NAME, dr.COL_CATEGORY, dr.COL_BOUND, dr.COL_OP_COUNT, dr.COL_ROOFLINE_TIME, dr.COL_KERNEL_TIME_SUM]
    rows = [
        # ideal = roofline_time_first * op_count
        ["aten::mm", "GEMM", "COMPUTE_BOUND", "10", "50", "1000"],  # ideal 500 / actual 1000
        ["sdpa", "SDPA_fwd", "COMPUTE_BOUND", "5", "40", "1000"],  # ideal 200 / actual 1000
        ["triton_fused", "triton", "", "2", "0", "500"],  # no perf model
    ]
    with unified.open("w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    if with_timeline:
        (csv_dir / dr.GPU_TIMELINE_CSV).write_text(
            "type,time ms,percent\n"
            "computation_time,2400,96.0\n"
            "busy_time,2450,98.0\n"
        )


def test_diffusion_roofline_aggregation(tmp_path):
    _write_csvs(tmp_path)
    report = dr.build_report(tmp_path, num_denoise_steps=25, top_k=3)
    t = report["totals"]
    assert t["sigma_actual_kernel_us"] == pytest.approx(2500.0)
    assert t["sigma_ideal_roofline_us"] == pytest.approx(700.0)  # 500 + 200 + 0
    assert t["kernel_roofline_efficiency"] == pytest.approx(700.0 / 2500.0)
    assert t["compute_bound_us"] == pytest.approx(2000.0)
    assert t["no_perf_model_us"] == pytest.approx(500.0)
    assert report["gpu_busy_ratio"] == pytest.approx(0.98)
    assert report["end_to_end_efficiency_estimate"] == pytest.approx((700.0 / 2500.0) * 0.98)
    assert report["per_step"]["actual_kernel_us"] == pytest.approx(2500.0 / 25)
    assert report["top_kernels"][0]["kernel_time_us"] == pytest.approx(1000.0)


def test_diffusion_roofline_missing_timeline_is_fail_soft(tmp_path):
    _write_csvs(tmp_path, with_timeline=False)
    report = dr.build_report(tmp_path, num_denoise_steps=None, top_k=5)
    assert report["gpu_busy_ratio"] is None
    assert report["end_to_end_efficiency_estimate"] is None
    assert "per_step" not in report


def test_diffusion_roofline_missing_unified_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dr.build_report(tmp_path, num_denoise_steps=None, top_k=5)


def test_dit_analytic_flops_formula():
    # 1 layer, 1 token, 1 step, h=2, ffn=4 -> linear = 2*(4+8)*4 = 96;
    # attention = 2*(2*1*2) = 8; total = 104.
    flops = dr.dit_analytic_flops(hidden_size=2, num_layers=1, num_tokens=1, num_denoise_steps=1, ffn_ratio=4.0)
    assert flops["linear_flops"] == pytest.approx(96.0)
    assert flops["attention_flops"] == pytest.approx(8.0)
    assert flops["total_flops"] == pytest.approx(104.0)


def test_dit_analytic_ceiling_and_reconcile():
    # 1e12 FLOPs @ 1000 TFLOPS -> ideal 1e12/1e15 s = 1e-3 s = 1000 us.
    ceiling = dr.dit_analytic_ceiling({"total_flops": 1e12}, achievable_tflops=1000.0)
    assert ceiling["ideal_compute_us"] == pytest.approx(1000.0)
    totals = {"sigma_ideal_roofline_us": 500.0, "sigma_actual_kernel_us": 2000.0}
    recon = dr.reconcile(totals, ceiling)
    assert recon["analytic_vs_trace_ideal_ratio"] == pytest.approx(2.0)
    assert recon["analytic_achieved_efficiency"] == pytest.approx(0.5)


def test_dit_analytic_ceiling_zero_inputs_empty():
    assert dr.dit_analytic_ceiling({"total_flops": 0.0}, 1000.0) == {}
    assert dr.reconcile({"sigma_ideal_roofline_us": 100.0}, {}) == {}


def test_build_report_includes_analytic_ceiling(tmp_path):
    _write_csvs(tmp_path)
    report = dr.build_report(
        tmp_path,
        num_denoise_steps=25,
        top_k=3,
        dit_geometry={"hidden_size": 3072, "num_layers": 38, "num_tokens": 4096, "ffn_ratio": 4.0},
        achievable_tflops=1686.0,
    )
    assert "analytic_dit_ceiling" in report
    assert report["analytic_dit_ceiling"]["total_flops"] > 0
    assert "reconciliation" in report


def test_build_report_no_geometry_skips_analytic(tmp_path):
    _write_csvs(tmp_path)
    report = dr.build_report(tmp_path, num_denoise_steps=25, top_k=3)
    assert "analytic_dit_ceiling" not in report
    assert "reconciliation" not in report


def test_print_summary_full_report_smoke(tmp_path, capsys):
    """A report with every optional section (per-step + analytic ceiling +
    reconciliation) must render without raising."""
    _write_csvs(tmp_path)
    report = dr.build_report(
        tmp_path,
        num_denoise_steps=25,
        top_k=3,
        dit_geometry={"hidden_size": 3072, "num_layers": 38, "num_tokens": 4096, "ffn_ratio": 4.0},
        achievable_tflops=1686.0,
    )
    dr.print_summary(report)
    out = capsys.readouterr().out
    assert "diffusion workload roofline" in out
    assert "top kernels by time" in out
    assert "per denoise-step" in out
    assert "a-priori DiT ceiling" in out


def test_print_summary_minimal_report_smoke(tmp_path, capsys):
    """No timeline -> ``gpu_busy_ratio`` None exercises the ``_fmt_pct`` n/a
    branch; no geometry -> optional sections are omitted."""
    _write_csvs(tmp_path, with_timeline=False)
    report = dr.build_report(tmp_path, num_denoise_steps=None, top_k=3)
    dr.print_summary(report)
    out = capsys.readouterr().out
    assert "n/a" in out
    assert "per denoise-step" not in out
    assert "a-priori DiT ceiling" not in out


def test_fmt_pct_none_and_value():
    assert dr._fmt_pct(None) == "n/a"
    assert dr._fmt_pct(0.5) == "50.0%"


def test_safe_float_non_numeric_is_zero():
    assert dr.safe_float("not-a-number") == 0.0
    assert dr.safe_float(None) == 0.0
    assert dr.safe_float("12.5") == pytest.approx(12.5)


def test_aggregate_unified_memory_bound_split():
    rows = [
        {dr.COL_NAME: "gemm", dr.COL_BOUND: "COMPUTE_BOUND", dr.COL_OP_COUNT: "1",
         dr.COL_ROOFLINE_TIME: "10", dr.COL_KERNEL_TIME_SUM: "100"},
        {dr.COL_NAME: "copy", dr.COL_BOUND: "MEMORY_BOUND", dr.COL_OP_COUNT: "1",
         dr.COL_ROOFLINE_TIME: "5", dr.COL_KERNEL_TIME_SUM: "50"},
    ]
    totals = dr.aggregate_unified(rows)
    assert totals["compute_bound_us"] == pytest.approx(100.0)
    assert totals["memory_bound_us"] == pytest.approx(50.0)


def test_build_report_analytic_geometry_missing_key_is_fail_soft(tmp_path):
    """A dit_geometry dict missing a required key hits the guarded
    ``except (KeyError, TypeError, ValueError)`` path without raising."""
    _write_csvs(tmp_path)
    report = dr.build_report(
        tmp_path,
        num_denoise_steps=25,
        top_k=3,
        dit_geometry={"num_layers": 38, "num_tokens": 4096},  # no hidden_size
        achievable_tflops=1686.0,
    )
    assert "analytic_dit_ceiling" not in report


def test_print_summary_renders_analytic_ceiling_section(capsys):
    report = {
        "totals": {
            "sigma_actual_kernel_us": 2000.0,
            "sigma_ideal_roofline_us": 500.0,
            "kernel_roofline_efficiency": 0.25,
            "compute_bound_us": 1500.0,
            "memory_bound_us": 300.0,
            "no_perf_model_us": 200.0,
        },
        "gpu_busy_ratio": 0.9,
        "end_to_end_efficiency_estimate": 0.225,
        "top_kernels": [],
        "analytic_ceiling": {
            "total_flops": 5.0e12,
            "family": "flux",
            "hidden": 3072,
            "layers": 38,
            "num_steps": 28,
            "cfg_batch": 1,
            "ideal_ms": 12.3,
            "peak_tflops": 2516.6,
            "precision": "bf16",
        },
        "analytic_within_pct": 61.5,
        "reconciliation": {
            "analytic_vs_trace_ideal_ratio": 1.5,
            "analytic_achieved_efficiency": 0.4,
        },
    }
    dr.print_summary(report)
    out = capsys.readouterr().out
    assert "analytic absolute ceiling (approach a)" in out
    assert "within-roofline" in out
    assert "reconciliation" in out


def _write_model_dir(root: Path) -> Path:
    """Minimal SD3-like diffusers denoiser so diffusion_flops resolves geometry."""
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    (root / "transformer" / "config.json").write_text(
        '{"_class_name": "SD3Transformer2DModel", "num_layers": 4, '
        '"num_attention_heads": 8, "attention_head_dim": 8, "patch_size": 2}',
        encoding="utf-8",
    )
    return root


def test_main_end_to_end_with_model_dir_and_output(tmp_path, monkeypatch, capsys):
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    _write_csvs(csv_dir)
    model_dir = _write_model_dir(tmp_path / "model")
    out_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diffusion_roofline",
            "--perf-csv-dir", str(csv_dir),
            "--num-denoise-steps", "25",
            "--top-k", "3",
            "--model-dir", str(model_dir),
            "--precision", "bf16",
            "--output", str(out_path),
        ],
    )
    rc = dr.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "diffusion workload roofline" in out
    assert out_path.is_file()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert "analytic_ceiling" in saved
    assert saved["analytic_ceiling"]["total_flops"] > 0
    assert "analytic_within_pct" in saved


def test_main_with_dit_geometry_flags(tmp_path, monkeypatch, capsys):
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    _write_csvs(csv_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diffusion_roofline",
            "--perf-csv-dir", str(csv_dir),
            "--num-denoise-steps", "25",
            "--dit-hidden-size", "3072",
            "--dit-num-layers", "38",
            "--dit-num-tokens", "4096",
            "--achievable-tflops", "1686",
        ],
    )
    rc = dr.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "a-priori DiT ceiling" in out


def test_main_target_platform_resolves_achievable(tmp_path, monkeypatch, capsys):
    """``--target-platform`` (without an explicit --achievable-tflops) exercises
    the HW_SPECS_ACHIEVABLE resolution branch (fail-soft on any import error)."""
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    _write_csvs(csv_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diffusion_roofline",
            "--perf-csv-dir", str(csv_dir),
            "--num-denoise-steps", "25",
            "--dit-hidden-size", "3072",
            "--dit-num-layers", "38",
            "--dit-num-tokens", "4096",
            "--target-platform", "MI355X",
        ],
    )
    rc = dr.main()
    assert rc == 0
    assert "diffusion workload roofline" in capsys.readouterr().out
