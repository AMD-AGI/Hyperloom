###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Analytical per-kernel roofline for the bypass analysis backend.

Computes ``bound_type`` (compute- vs memory-bound), ``arithmetic_intensity``
(FLOPs/byte), and efficiency/utilization for EVERY hot kernel whose operand
shapes allow a FLOP/byte estimate — including vendor kernels (hipBLASLt GEMM,
MIOpen conv) that the opt-in rocprof enrichment skips. This is the analytical
half of the roofline (parity with TraceLens' per-op perf-model bound); the
rocprof enrichment remains the *measured* refinement for rewritable kernels.

Independent + GPU-free: it uses the kernel's real per-launch ``gpu_time_us``
(from the trace) plus FLOPs/bytes estimated from the captured operand shapes and
a compact AMD peak-spec table (reimplemented, not imported from the orchestrator
so the bypass tool stays standalone). ``arithmetic_intensity`` vs the machine
balance point (peak_flops / peak_bw) gives the bound; ``estimated_flops /
gpu_time`` vs peak gives efficiency.
"""

from __future__ import annotations

import re
from typing import Any

# Compact AMD peak specs (mirrors inference_optimizer roofline_ceiling.HW_SPECS;
# reimplemented so this tool imports nothing from the orchestrator package).
_PEAK_TFLOPS_MI300: dict[str, float] = {
    "bf16": 1307.4, "bfloat16": 1307.4, "f16": 1307.4, "fp16": 1307.4, "float16": 1307.4,
    "fp8": 2614.9, "f8": 2614.9, "float8_e4m3fn": 2614.9, "float8_e5m2": 2614.9,
    "fp32": 163.4, "f32": 163.4, "float32": 163.4,
}
_PEAK_TFLOPS_MI355: dict[str, float] = {
    "bf16": 2516.6, "bfloat16": 2516.6, "f16": 2516.6, "fp16": 2516.6, "float16": 2516.6,
    "fp8": 5033.2, "f8": 5033.2, "float8_e4m3fn": 5033.2, "float8_e5m2": 5033.2,
    "mxfp4": 10066.4, "fp4": 10066.4, "float4": 10066.4,
}
_HW_SPECS: dict[str, dict[str, Any]] = {
    "mi300x": {"hbm_bw_gbps": 5300.0, "peak_tflops": _PEAK_TFLOPS_MI300},
    "mi308x": {"hbm_bw_gbps": 5300.0, "peak_tflops": _PEAK_TFLOPS_MI300},
    "mi325x": {"hbm_bw_gbps": 6000.0, "peak_tflops": _PEAK_TFLOPS_MI300},
    "mi355x": {"hbm_bw_gbps": 8000.0, "peak_tflops": _PEAK_TFLOPS_MI355},
}
_DEFAULT_GPU = "mi300x"

_DTYPE_BYTES: dict[str, float] = {
    "f32": 4.0, "fp32": 4.0, "float32": 4.0,
    "bf16": 2.0, "bfloat16": 2.0, "f16": 2.0, "fp16": 2.0, "float16": 2.0,
    "f8": 1.0, "fp8": 1.0, "float8_e4m3fn": 1.0, "float8_e5m2": 1.0,
    "f4": 0.5, "fp4": 0.5, "mxfp4": 0.5, "float4": 0.5,
}

_OPERAND_RE = re.compile(r"\(([\d,\s]*)\)\s*(\w+)?")


def _dtype_bytes(tag: str) -> float:
    return _DTYPE_BYTES.get((tag or "").strip().lower(), 2.0)


def _parse_operands(shape_str: str) -> list[tuple[tuple[int, ...], str]]:
    """Parse ``"(M,K) bf16<br>(K,N) bf16"`` -> ``[((M,K),"bf16"), ((K,N),"bf16")]``.

    Scalar/empty operands (``()``) are dropped.
    """
    operands: list[tuple[tuple[int, ...], str]] = []
    for tok in (shape_str or "").split("<br>"):
        m = _OPERAND_RE.search(tok)
        if not m:
            continue
        body = m.group(1).strip()
        if not body:
            continue
        try:
            dims = tuple(int(d.strip()) for d in body.split(",") if d.strip())
        except ValueError:
            continue
        if dims:
            operands.append((dims, (m.group(2) or "").strip()))
    return operands


def _numel(dims: tuple[int, ...]) -> int:
    n = 1
    for d in dims:
        n *= d
    return n


def _estimate_flops_bytes(category: str, operands: list[tuple[tuple[int, ...], str]], dbytes: float) -> tuple[float, float] | None:
    """Estimate (flops, bytes) for one representative call, or ``None``.

    Category-specific closed forms from operand shapes. Returns ``None`` when the
    operands do not support a trustworthy estimate (caller leaves it unknown).
    """
    if not operands:
        return None
    cat = (category or "").lower()

    if cat == "gemm":
        two_d = [d for d, _ in operands if len(d) >= 2]
        if len(two_d) < 2:
            return None
        # A = (..., M, K), B = (..., K, N) sharing the inner K.
        a, b = two_d[0], two_d[1]
        m, k = a[-2], a[-1]
        # pick the B operand whose leading (of last two) dim matches K.
        bmat = next((d for d in two_d[1:] if d[-2] == k), b)
        n = bmat[-1]
        batch = _numel(a[:-2]) or 1
        flops = 2.0 * batch * m * n * k
        nbytes = dbytes * (batch * m * k + batch * k * n + batch * m * n)
        return flops, nbytes

    if cat == "convolution":
        # input (N,C,H,W), weight (K,C,R,S). Assume stride 1 / same spatial
        # (output HxW ~= input HxW) — a documented approximation (stride/padding
        # are not in Input Dims). Good enough for a bound classification.
        four_d = [d for d, _ in operands if len(d) == 4]
        if len(four_d) < 2:
            return None
        inp, wt = four_d[0], four_d[1]
        n, c, h, w = inp
        kk, wc, r, s = wt
        out_hw = h * w
        flops = 2.0 * n * kk * out_hw * c * r * s
        nbytes = dbytes * (_numel(inp) + _numel(wt) + n * kk * out_hw)
        return flops, nbytes

    if cat in ("sdpa",):
        # Attention: Q/K/V ~ (B,H,S,D) (or (B,S,H,D)); 2 matmuls (QK^T, AV).
        four_d = [d for d, _ in operands if len(d) == 4]
        if not four_d:
            return None
        q = four_d[0]
        b, h, s, d = q[0], q[1], q[2], q[3]
        flops = 2.0 * (2.0 * b * h * s * s * d)  # QK^T + A·V
        nbytes = dbytes * sum(_numel(x) for x in four_d[:3])
        return flops, nbytes

    if cat in ("elementwise", "normalization", "quantization", "kvcachestore", "memcpy"):
        # Memory-bound: ~1 flop/element, read all operands + write the largest.
        total = sum(_numel(d) for d, _ in operands)
        out = max(_numel(d) for d, _ in operands)
        flops = float(total)
        nbytes = dbytes * (total + out)
        return flops, nbytes

    return None


def compute_roofline(
    *,
    category: str,
    shape_str: str,
    gpu_time_us: float,
    call_count: int = 1,
    gpu_type: str = "",
    dtype: str = "",
) -> dict[str, Any] | None:
    """Analytical roofline for one kernel aggregate, or ``None`` when unestimable.

    Args:
        category: Kernel category (GEMM/Convolution/SDPA/Elementwise/...).
        shape_str: One representative call's ``"(dims) dtype<br>..."`` string.
        gpu_time_us: TOTAL device time for the aggregate (all launches).
        call_count: Number of launches (to get per-call time for efficiency).
        gpu_type: GPU key (mi300x/mi325x/mi355x); defaults to mi300x.
        dtype: Compute dtype tag (defaults from the first operand / bf16).

    Returns:
        ``{bound_type, arithmetic_intensity, flops_per_byte, efficiency_percent,
        compute_utilization_pct, bandwidth_utilization_pct, roofline_source}`` or
        ``None`` when the shapes do not support an estimate.
    """
    operands = _parse_operands(shape_str)
    if not operands:
        return None
    op_dtype = (dtype or operands[0][1] or "bf16").strip().lower()
    dbytes = _dtype_bytes(op_dtype)
    est = _estimate_flops_bytes(category, operands, dbytes)
    if est is None:
        return None
    flops, nbytes = est
    if flops <= 0 or nbytes <= 0:
        return None

    spec = _HW_SPECS.get((gpu_type or "").strip().lower()) or _HW_SPECS[_DEFAULT_GPU]
    peak_tflops = spec["peak_tflops"].get(op_dtype, spec["peak_tflops"].get("bf16", 0.0))
    peak_flops = peak_tflops * 1e12
    peak_bw = spec["hbm_bw_gbps"] * 1e9

    ai = flops / nbytes  # arithmetic intensity (FLOPs/byte)
    machine_balance = (peak_flops / peak_bw) if peak_bw > 0 else 0.0
    bound_type = "compute_bound" if (machine_balance > 0 and ai >= machine_balance) else "memory_bound"

    out: dict[str, Any] = {
        "bound_type": bound_type,
        "arithmetic_intensity": round(ai, 4),
        "flops_per_byte": round(ai, 4),
        "roofline_source": "analytical",
    }
    # Per-call achieved throughput from the real measured time -> efficiency.
    # ``roofline_estimate_capped`` flags when a util exceeds 100% (the FLOP/byte
    # estimate over-shot, e.g. the Convolution stride-1 approximation) and was
    # clamped -- so a capped 100% is not mistaken for a real "perfectly efficient"
    # measurement.
    calls = max(int(call_count or 1), 1)
    per_call_s = (float(gpu_time_us) / calls) / 1e6 if gpu_time_us else 0.0
    if per_call_s > 0 and peak_flops > 0:
        raw_eff = (flops / per_call_s) / peak_flops * 100.0
        out["efficiency_percent"] = round(min(raw_eff, 100.0), 3)
        out["compute_utilization_pct"] = out["efficiency_percent"]
        if raw_eff > 100.0:
            out["roofline_estimate_capped"] = True
    if per_call_s > 0 and peak_bw > 0:
        raw_bw = (nbytes / per_call_s) / peak_bw * 100.0
        out["bandwidth_utilization_pct"] = round(min(raw_bw, 100.0), 3)
        if raw_bw > 100.0:
            out["roofline_estimate_capped"] = True
    return out


__all__ = ["compute_roofline"]
