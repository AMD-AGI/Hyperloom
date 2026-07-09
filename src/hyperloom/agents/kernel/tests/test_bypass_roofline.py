###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the bypass analytical per-kernel roofline (_bypass_roofline)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

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


def test_roofline_attainment_is_binding_side():
    # compute-bound GEMM: attainment == compute utilization.
    g = compute_roofline(
        category="GEMM", shape_str="(4096,4096) bf16<br>(4096,4096) bf16",
        gpu_time_us=500.0, call_count=1, gpu_type="mi300x",
    )
    assert g["bound_type"] == "compute_bound"
    assert g["roofline_attainment_pct"] == g["compute_utilization_pct"]
    # memory-bound elementwise: attainment == bandwidth utilization, NOT the
    # compute-side efficiency_percent (which reads ~0 for a memory-bound kernel).
    e = compute_roofline(
        category="Elementwise", shape_str="(4096,4096) bf16<br>(4096,4096) bf16",
        gpu_time_us=200.0, call_count=1, gpu_type="mi300x",
    )
    assert e["bound_type"] == "memory_bound"
    assert e["roofline_attainment_pct"] == e["bandwidth_utilization_pct"]
    assert e["roofline_attainment_pct"] != e["efficiency_percent"]


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


# ── conv FLOPs use the weight channel dim (Cin/groups), not input channels ────


def test_depthwise_conv_uses_group_channels_not_input_channels():
    # Real Sana k004: input (2,11200,32,32), weight (Cout=11200, Cin/groups=1, 3, 3).
    # FLOPs must use wc=1 (depthwise), NOT the 11200 input channels -> the dense
    # formula overcounts by groups=Cin=11200x, faking a compute-bound eff=100%.
    B, C, HW, Cout, wc, R, S = 2, 11200, 32, 11200, 1, 3, 3
    r = compute_roofline(
        category="Convolution",
        shape_str=f"({B},{C},{HW},{HW}) bf16<br>({Cout},{wc},{R},{S}) bf16",
        gpu_time_us=2862.0, call_count=20, gpu_type="mi325x",
    )
    assert r is not None
    out_hw = HW * HW
    flops = 2.0 * B * Cout * out_hw * wc * R * S
    dbytes = 2.0
    nbytes = dbytes * (B * C * out_hw + Cout * wc * R * S + B * Cout * out_hw)
    assert r["arithmetic_intensity"] == round(flops / nbytes, 4)
    # low AI -> memory-bound (the dense-formula bug made it compute-bound eff=100%).
    assert r["bound_type"] == "memory_bound"


def test_dense_conv_flops_unchanged_by_wc_fix():
    # Guard: a dense conv has wc == Cin, so switching c -> wc must NOT change it.
    B, Cin, HW, Cout, R, S = 2, 320, 64, 320, 3, 3
    r = compute_roofline(
        category="Convolution",
        shape_str=f"({B},{Cin},{HW},{HW}) bf16<br>({Cout},{Cin},{R},{S}) bf16",
        gpu_time_us=800.0, call_count=1, gpu_type="mi300x",
    )
    assert r is not None
    out_hw = HW * HW
    flops = 2.0 * B * Cout * out_hw * Cin * R * S  # wc == Cin for a dense conv
    dbytes = 2.0
    nbytes = dbytes * (B * Cin * out_hw + Cout * Cin * R * S + B * Cout * out_hw)
    assert r["arithmetic_intensity"] == round(flops / nbytes, 4)


# ── SDPA layout inference (B,S,H,D vs B,H,S,D; Sq != Skv) ─────────────────────


def test_sdpa_cross_attention_infers_bshd_layout():
    # Real Sana k006 cross-attn: Q(B,Sq,H,D), K/V(B,Skv,H,D), score(B,H,Sq,Skv).
    # The head dim is shared between Q/K middle dims -> layout is resolved exactly
    # (not the hardcoded B,H,S,D), and FLOPs use Sq*Skv (Skv=300 != Sq=1024).
    B, Sq, H, D, Skv = 2, 1024, 20, 112, 300
    shp = (f"({B},{Sq},{H},{D}) bf16<br>({B},{Skv},{H},{D}) bf16<br>"
           f"({B},{Skv},{H},{D}) bf16<br>({B},{H},{Sq},{Skv}) bf16")
    r = compute_roofline(category="SDPA", shape_str=shp, gpu_time_us=200.0, call_count=1, gpu_type="mi300x")
    assert r is not None
    flops = 4.0 * B * H * Sq * Skv * D
    dbytes = 2.0
    nbytes = dbytes * (B * Sq * H * D + B * Skv * H * D + B * Skv * H * D)
    assert r["arithmetic_intensity"] == round(flops / nbytes, 4)
    assert "roofline_layout_inferred" not in r  # exact (shared head dim)


def test_sdpa_self_attention_ambiguous_layout_is_marked_inferred():
    # Self-attn Q=K=V=(B,H,S,D) with no score tensor: Q/K middle dims are
    # identical so H vs S can't be resolved from shapes -> heuristic (H=smaller)
    # AND the row is flagged roofline_layout_inferred so the estimate is honest.
    B, H, S, D = 2, 8, 1024, 64
    shp = f"({B},{H},{S},{D}) bf16<br>({B},{H},{S},{D}) bf16<br>({B},{H},{S},{D}) bf16"
    r = compute_roofline(category="SDPA", shape_str=shp, gpu_time_us=100.0, call_count=1, gpu_type="mi300x")
    assert r is not None
    # heuristic picks H=min(8,1024)=8, Sq=Skv=1024 -> classic 4*B*H*S*S*D.
    flops = 4.0 * B * H * S * S * D
    dbytes = 2.0
    nbytes = dbytes * (3 * B * H * S * D)
    assert r["arithmetic_intensity"] == round(flops / nbytes, 4)
    assert r.get("roofline_layout_inferred") is True


def test_sdpa_score_tensor_disambiguates_shared_seqlen():
    # Equal-seq cross-attn with different Q/K head counts (Hq=16, Hkv=4): the
    # only value Q/K middle dims share is the SEQ length, so shared-dim inference
    # would wrongly treat seq as the head. The authoritative score (B,Hq,Sq,Skv)
    # must resolve it -> H=Hq, and it must NOT be silently mis-labeled "exact".
    B, S, Hq, Hkv, D = 2, 256, 16, 4, 64
    shp = (f"({B},{S},{Hq},{D}) bf16<br>({B},{S},{Hkv},{D}) bf16<br>"
           f"({B},{S},{Hkv},{D}) bf16<br>({B},{Hq},{S},{S}) bf16")
    r = compute_roofline(category="SDPA", shape_str=shp, gpu_time_us=100.0, call_count=1, gpu_type="mi300x")
    assert r is not None
    flops = 4.0 * B * Hq * S * S * D  # NOT 4*B*S*Hq*Hkv*D (the shared-dim mistake)
    nbytes = 2.0 * (B * S * Hq * D + B * S * Hkv * D + B * S * Hkv * D)
    assert r["arithmetic_intensity"] == round(flops / nbytes, 4)


def test_sdpa_cross_attention_shared_dim_without_score():
    # Cross-attn Q/K/V only (no score operand): the shared head dim resolves the
    # layout exactly (Sq=1024 != Skv=300) -> not inferred.
    B, Sq, H, D, Skv = 2, 1024, 20, 112, 300
    shp = (f"({B},{Sq},{H},{D}) bf16<br>({B},{Skv},{H},{D}) bf16<br>({B},{Skv},{H},{D}) bf16")
    r = compute_roofline(category="SDPA", shape_str=shp, gpu_time_us=200.0, call_count=1, gpu_type="mi300x")
    assert r is not None
    flops = 4.0 * B * H * Sq * Skv * D
    nbytes = 2.0 * (B * Sq * H * D + B * Skv * H * D + B * Skv * H * D)
    assert r["arithmetic_intensity"] == round(flops / nbytes, 4)
    assert "roofline_layout_inferred" not in r


def test_efficiency_uses_achievable_peak_not_vendor():
    # Per-kernel efficiency% must use the max-achievable peak (708 TFLOPS bf16
    # mi300x) -- the SAME convention as the session roofline ceiling -- NOT the
    # vendor dense peak (1307.4), which would understate efficiency ~1.85x.
    r = compute_roofline(
        category="GEMM", shape_str="(4096,4096) bf16<br>(4096,4096) bf16",
        gpu_time_us=500.0, call_count=1, gpu_type="mi300x",
    )
    assert r is not None
    achieved_flops = (2.0 * 4096 ** 3) / 500e-6
    eff_achievable = achieved_flops / (708.0e12) * 100.0     # ~38.8%
    eff_vendor = achieved_flops / (1307.4e12) * 100.0        # ~21.0%
    assert abs(r["efficiency_percent"] - eff_achievable) < 0.5
    assert abs(r["efficiency_percent"] - eff_vendor) > 5.0
