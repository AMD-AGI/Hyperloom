###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the bypass analytical per-kernel roofline (_bypass_roofline)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bypass_roofline import compute_roofline  # noqa: E402


def test_large_square_gemm_is_compute_bound():
    # (4096,4096)x(4096,4096) bf16: AI ~= 4096/3 ~= 1365 FLOPs/byte, well above the
    # MI300X bf16 machine balance (~247) -> compute bound; efficiency in (0,100].
    r = compute_roofline(
        category="GEMM", shape_str="(4096,4096) bf16<br>(4096,4096) bf16",
        gpu_time_us=500.0, call_count=1, gpu_type="mi300x",
    )
    assert r is not None
    assert r["bound_type"] == "compute_bound"
    assert r["arithmetic_intensity"] > 1000
    assert r["roofline_source"] == "analytical"
    assert 0.0 < r["efficiency_percent"] <= 100.0


def test_skinny_gemm_can_be_memory_bound():
    # A skinny GEMM (small M) has low AI -> memory bound.
    r = compute_roofline(
        category="GEMM", shape_str="(8,2560) bf16<br>(2560,2560) bf16",
        gpu_time_us=50.0, call_count=1, gpu_type="mi300x",
    )
    assert r is not None
    assert r["bound_type"] == "memory_bound"


def test_elementwise_is_memory_bound():
    r = compute_roofline(
        category="Elementwise", shape_str="(4096,4096) bf16<br>(4096,4096) bf16",
        gpu_time_us=100.0, call_count=1, gpu_type="mi300x",
    )
    assert r is not None
    assert r["bound_type"] == "memory_bound"
    assert r["arithmetic_intensity"] < 1.0
    assert "bandwidth_utilization_pct" in r


def test_convolution_estimates_bound():
    # VAE-style conv: input (2,320,64,64), weight (320,320,3,3).
    r = compute_roofline(
        category="Convolution", shape_str="(2,320,64,64) bf16<br>(320,320,3,3) bf16",
        gpu_time_us=800.0, call_count=1, gpu_type="mi300x",
    )
    assert r is not None
    assert r["bound_type"] in ("compute_bound", "memory_bound")
    assert r["arithmetic_intensity"] > 0


def test_vendor_gemm_gets_bound_even_without_source():
    # The whole point of the xDiT gap fix: a vendor GEMM (non-rewritable) still
    # gets an analytical bound purely from shapes + measured time.
    r = compute_roofline(
        category="GEMM", shape_str="(2048,2240) bf16<br>(2240,2240) bf16",
        gpu_time_us=300.0, call_count=1, gpu_type="mi300x",
    )
    assert r is not None and r["bound_type"] in ("compute_bound", "memory_bound")


def test_unestimable_returns_none():
    # Unknown category / no usable shapes -> None (leave placeholder).
    assert compute_roofline(category="Others", shape_str="(128,128) bf16", gpu_time_us=10.0) is None
    assert compute_roofline(category="GEMM", shape_str="", gpu_time_us=10.0) is None
    assert compute_roofline(category="GEMM", shape_str="(128,) bf16", gpu_time_us=10.0) is None  # only 1-D


def test_efficiency_capped_flag_when_estimate_overshoots():
    # Implausibly tiny time -> estimated achieved FLOPS >> peak -> clamped to 100%
    # AND flagged, so a capped 100% isn't mistaken for a real measurement.
    r = compute_roofline(
        category="GEMM", shape_str="(4096,4096) bf16<br>(4096,4096) bf16",
        gpu_time_us=0.001, call_count=1, gpu_type="mi300x",
    )
    assert r["efficiency_percent"] == 100.0
    assert r.get("roofline_estimate_capped") is True


def test_efficiency_not_capped_or_flagged_in_normal_case():
    r = compute_roofline(
        category="GEMM", shape_str="(4096,4096) bf16<br>(4096,4096) bf16",
        gpu_time_us=500.0, call_count=1, gpu_type="mi300x",
    )
    assert r["efficiency_percent"] < 100.0
    assert "roofline_estimate_capped" not in r


def test_sdpa_flops_match_attention_formula():
    # Q/K/V (B,H,S,D): attention flops = 4*B*H*S*S*D; bytes = 3*B*H*S*D*dtype_bytes.
    b, h, s, d = 2, 8, 1024, 64
    shp = f"({b},{h},{s},{d}) bf16<br>({b},{h},{s},{d}) bf16<br>({b},{h},{s},{d}) bf16"
    r = compute_roofline(category="SDPA", shape_str=shp, gpu_time_us=100.0, call_count=1, gpu_type="mi300x")
    expected_ai = round((4.0 * b * h * s * s * d) / (2.0 * 3 * b * h * s * d), 4)
    assert r["arithmetic_intensity"] == expected_ai


def test_gpu_and_dtype_change_peak():
    # fp8 has ~2x the bf16 peak on MI300X -> same GEMM/time yields lower efficiency%.
    common = dict(category="GEMM", shape_str="(4096,4096) X<br>(4096,4096) X", gpu_time_us=500.0, call_count=1, gpu_type="mi300x")
    bf16 = compute_roofline(**{**common, "dtype": "bf16"})
    fp8 = compute_roofline(**{**common, "dtype": "fp8"})
    assert bf16 and fp8
    assert fp8["efficiency_percent"] < bf16["efficiency_percent"]
