# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the diffusion workload-level roofline aggregator."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.trace_analysis import diffusion_roofline as dr


def _write_csvs(csv_dir: Path, *, with_timeline: bool = True) -> None:
    unified = csv_dir / dr.UNIFIED_CSV
    header = [dr.COL_NAME, dr.COL_CATEGORY, dr.COL_BOUND, dr.COL_OP_COUNT, dr.COL_ROOFLINE_TIME, dr.COL_KERNEL_TIME_SUM]
    rows = [
        # ideal is roofline_time_first * op_count
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
            "type,time ms,percent\ncomputation_time,2400,96.0\nbusy_time,2450,98.0\n"
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


def test_aggregate_unified_reads_a_non_finite_or_junk_cell_as_zero():
    rows = [
        {
            dr.COL_NAME: "gemm",
            dr.COL_BOUND: "COMPUTE_BOUND",
            dr.COL_OP_COUNT: "not-a-number",
            dr.COL_ROOFLINE_TIME: "nan",
            dr.COL_KERNEL_TIME_SUM: "100",
        }
    ]
    totals = dr.aggregate_unified(rows)
    assert totals["sigma_ideal_roofline_us"] == 0.0
    assert totals["kernel_roofline_efficiency"] == 0.0


def test_aggregate_unified_memory_bound_split():
    rows = [
        {
            dr.COL_NAME: "gemm",
            dr.COL_BOUND: "COMPUTE_BOUND",
            dr.COL_OP_COUNT: "1",
            dr.COL_ROOFLINE_TIME: "10",
            dr.COL_KERNEL_TIME_SUM: "100",
        },
        {
            dr.COL_NAME: "copy",
            dr.COL_BOUND: "MEMORY_BOUND",
            dr.COL_OP_COUNT: "1",
            dr.COL_ROOFLINE_TIME: "5",
            dr.COL_KERNEL_TIME_SUM: "50",
        },
    ]
    totals = dr.aggregate_unified(rows)
    assert totals["compute_bound_us"] == pytest.approx(100.0)
    assert totals["memory_bound_us"] == pytest.approx(50.0)
