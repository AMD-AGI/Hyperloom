# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 dense linear for AMD RDNA4 riding the FP8 WMMA datapath (W4A8-style).

Design lineage
--------------
The core idea this kernel is built on -- folding the per-K-group scale into the
fp32 accumulator *after* ``tl.dot`` rather than dequantizing operands up front,
on WMMA v2 -- follows Rob Smith's (``tcclaviger``) ``_matmul_fp8_ogs`` from the
vLLM 0.24 line, where it was done for W8A8.  This project's RDNA4 work descends
from his gfx1201 enablement, and without it none of this exists.

What is added here on top of that idea: the MXFP4 nibble -> e4m3 bit-pattern
unpack (his was already 8-bit, so there was nothing to unpack), the E8M0 block
scale in place of a per-tensor fp8 scale, and the three-regime dispatch below
(fused decode / weight-only mid-M / exact bf16 dequant + hipBLASLt prefill).

RDNA4 (gfx1200/gfx1201) has no microscaling datapath, but it does have native
FP8 (e4m3fn) WMMA at roughly twice the bf16 rate.  ``RdnaMxfp4LinearKernel``
unpacks MXFP4 tiles to bf16 inside the GEMM and pays for it on every token; this
kernel keeps the weights packed in VRAM and instead:

* decode / small M (M <= FUSED_MAX_M=128): a fused Triton GEMM that unpacks each
  E2M1 nibble to its exact e4m3 bit pattern, does ``tl.dot`` on 32-wide K
  blocks and applies the E8M0 block scale to the fp32 partial *after* the dot
  (``acc += dot(x8, w8) * 2**(s-127)``), with per-token e4m3 activations
  (row scale applied in the epilogue);
* mid M (128 < M <= 512): the weight-only in-kernel bf16 path (RdnaMxfp4LinearKernel's op) --
  it ties the fused kernel there and skips the activation-quant pass;
* prefill / large M (> 512): an exact bf16 dequant of the packed weights into a reused
  scratch buffer, then hipBLASLt's bf16 GEMM -- activations stay bf16, so
  prefill numerics equal the weight-only kernel's.

Decode activations are quantized to e4m3 dynamically **per (token, 32-wide K
group)** -- finer than the stock FP8 (W8A8) path's per-token-per-128 groups.
(A first cut with per-token scales cost ~8 pts of gsm8k strict-match; the
group scales fixed it.)  Numerics: the fused path matches
``F.linear(x_q, dequant(w))`` to bf16 rounding.

Measured on R9700 (2026-08-15) vs RdnaMxfp4LinearKernel: 1.4-2.3x faster at
M<=128, 2.4-3x at M>=1024.  Disable with VLLM_RDNA_MXFP4_FP8=0.
"""
import os
from functools import cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.platforms import current_platform

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig
from .rdna import (
    MXFP4_GROUP_SIZE,
    _dequant_mxfp4_reference,
    _is_rdna4_platform,
)

logger = init_logger(__name__)

FUSED_MAX_M = 128          # fused e4m3 kernel wins clearly up to here (1.3-2x)
MID_MAX_M = 512            # (FUSED_MAX_M, MID_MAX_M]: weight-only in-kernel bf16 path (ties fused, no quant pass)
_ENV = "VLLM_RDNA_MXFP4_FP8"
_ENV_SKIP = "VLLM_RDNA_MXFP4_FP8_SKIP"   # comma-separated prefix substrings routed to bf16 activations (exact) for all M


def _skip_patterns():
    v = os.environ.get(_ENV_SKIP, "").strip()
    return [p for p in (x.strip() for x in v.split(",")) if p]


def _enabled() -> bool:
    return os.environ.get(_ENV, "1") not in ("0", "false", "False")


# ─────────────────────────── Triton kernels ────────────────────────────────
@triton.jit
def _nib_to_e4m3_bits(n):
    """E2M1 nibble -> e4m3fn bit pattern of the same value (exact)."""
    mag = n & 7
    sign = (n >> 3) & 1
    b = tl.where(mag == 0, 0, tl.where(mag == 1, 0x30, 0x38 + (mag - 2) * 4))
    return (b | (sign << 7)).to(tl.uint8)


@triton.jit
def _groupquant_fp8_kernel(X, X8, SX, K, stride_x, BK: tl.constexpr):
    """Per-(token, 32-wide K group) dynamic e4m3 quant. SX is [M, K//32] fp32."""
    row = tl.program_id(0)
    ok = tl.arange(0, BK)
    m = ok < K
    x = tl.load(X + row * stride_x + ok, mask=m, other=0.0).to(tl.float32)
    xg = tl.reshape(x, (BK // 32, 32))
    amax = tl.maximum(tl.max(tl.abs(xg), axis=1), 1e-12)          # [BK//32]
    sg = amax / 448.0
    q = (xg / sg[:, None]).to(tl.float8e4nv)
    tl.store(X8 + row * K + ok, tl.reshape(q, (BK,)), mask=m)
    og = tl.arange(0, BK // 32)
    tl.store(SX + row * (K // 32) + og, sg, mask=og < (K // 32))


@triton.jit
def _dot_block(x, w, om, mask_m, on, mask_n, SX, stride_sxm, WS, stride_wsn, kb, acc):
    part = tl.dot(x, tl.trans(w))
    s = tl.load(WS + on * stride_wsn + kb, mask=mask_n, other=127).to(tl.int32)
    sf = (s << 23).to(tl.float32, bitcast=True)      # 2^(s-127)
    sxb = tl.load(SX + om * stride_sxm + kb, mask=mask_m, other=0.0)
    return acc + part * (sxb[:, None] * sf[None, :])


@triton.jit
def _mxfp4_fp8_gemm_kernel(X8, SX, WP, WS, Y, M, N, K,
                           stride_xm, stride_sxm, stride_wn, stride_wsn, stride_ym,
                           BM: tl.constexpr, BN: tl.constexpr, NUM_KB: tl.constexpr,
                           UNROLL: tl.constexpr, EVEN_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    mask_m = om < M
    mask_n = on < N
    KW: tl.constexpr = 32 * UNROLL
    ok = tl.arange(0, KW)
    okp = tl.arange(0, KW // 2)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for it in range(0, NUM_KB // UNROLL):
        xp = X8 + om[:, None] * stride_xm + (it * KW + ok)[None, :]
        if EVEN_M:
            x = tl.load(xp)
        else:
            x = tl.load(xp, mask=mask_m[:, None], other=0.0)
        w = tl.load(WP + on[:, None] * stride_wn + (it * (KW // 2) + okp)[None, :],
                    mask=mask_n[:, None], other=0)
        lo = _nib_to_e4m3_bits(w & 0x0F)
        hi = _nib_to_e4m3_bits((w >> 4) & 0x0F)
        w8 = tl.reshape(tl.join(lo, hi), (BN, KW)).to(tl.float8e4nv, bitcast=True)
        if UNROLL == 1:
            acc = _dot_block(x, w8, om, mask_m, on, mask_n, SX, stride_sxm, WS, stride_wsn, it, acc)
        elif UNROLL == 2:
            xa, xb = tl.split(tl.permute(tl.reshape(x, (BM, 2, 32)), (0, 2, 1)))
            wa, wb = tl.split(tl.permute(tl.reshape(w8, (BN, 2, 32)), (0, 2, 1)))
            acc = _dot_block(xa, wa, om, mask_m, on, mask_n, SX, stride_sxm, WS, stride_wsn, it * 2 + 0, acc)
            acc = _dot_block(xb, wb, om, mask_m, on, mask_n, SX, stride_sxm, WS, stride_wsn, it * 2 + 1, acc)
        else:
            x01, x23 = tl.split(tl.permute(tl.reshape(x, (BM, 2, 64)), (0, 2, 1)))
            xa, xb = tl.split(tl.permute(tl.reshape(x01, (BM, 2, 32)), (0, 2, 1)))
            xc, xd = tl.split(tl.permute(tl.reshape(x23, (BM, 2, 32)), (0, 2, 1)))
            w01, w23 = tl.split(tl.permute(tl.reshape(w8, (BN, 2, 64)), (0, 2, 1)))
            wa, wb = tl.split(tl.permute(tl.reshape(w01, (BN, 2, 32)), (0, 2, 1)))
            wc, wd = tl.split(tl.permute(tl.reshape(w23, (BN, 2, 32)), (0, 2, 1)))
            acc = _dot_block(xa, wa, om, mask_m, on, mask_n, SX, stride_sxm, WS, stride_wsn, it * 4 + 0, acc)
            acc = _dot_block(xb, wb, om, mask_m, on, mask_n, SX, stride_sxm, WS, stride_wsn, it * 4 + 1, acc)
            acc = _dot_block(xc, wc, om, mask_m, on, mask_n, SX, stride_sxm, WS, stride_wsn, it * 4 + 2, acc)
            acc = _dot_block(xd, wd, om, mask_m, on, mask_n, SX, stride_sxm, WS, stride_wsn, it * 4 + 3, acc)
    tl.store(Y + om[:, None] * stride_ym + on[None, :], acc.to(tl.bfloat16),
             mask=mask_m[:, None] & mask_n[None, :])


# ─────────────────────────── host-side helpers ─────────────────────────────
def _fused_cfg(m: int):
    """(BM, BN, UNROLL, num_warps, num_stages) tuned on R9700 (2026-08-15)."""
    if m <= 16:
        return 16, 64, 4, 4, 2
    if m <= 32:
        return 32, 64, 4, 4, 2
    if m <= 64:
        return 64, 32, 2, 4, 2
    return 128, 32, 2, 8, 2


def _groupquant(x2d: torch.Tensor):
    m, k = x2d.shape
    x8 = torch.empty((m, k), device=x2d.device, dtype=torch.float8_e4m3fn)
    sx = torch.empty((m, k // 32), device=x2d.device, dtype=torch.float32)
    bk = triton.next_power_of_2(k)
    _groupquant_fp8_kernel[(m,)](x2d, x8, sx, k, x2d.stride(0), BK=bk, num_warps=8 if bk >= 8192 else 4)
    return x8, sx


@triton.jit
def _nib_to_bf16(nb):
    mag = nb & 7
    sgn = (nb >> 3) & 1
    e = mag >> 1
    m = mag & 1
    bits = tl.where(e == 0, m * 0x3F00, ((e + 126) << 7) | (m << 6))
    return ((sgn << 15) | bits).to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _dequant_bf16_kernel(WP, WS, OUT, N, K, stride_wn, stride_wsn, stride_on, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    on = pid_n * BN + tl.arange(0, BN)
    okp = pid_k * (BK // 2) + tl.arange(0, BK // 2)
    mask_n = on < N
    w = tl.load(WP + on[:, None] * stride_wn + okp[None, :], mask=mask_n[:, None], other=0).to(tl.int32)
    v = tl.reshape(tl.join(_nib_to_bf16(w & 0x0F), _nib_to_bf16((w >> 4) & 0x0F)), (BN, BK))
    okb = pid_k * BK + tl.arange(0, BK)
    sc = tl.load(WS + on[:, None] * stride_wsn + (okb // 32)[None, :], mask=mask_n[:, None], other=127).to(tl.int32)
    sf = (sc << 23).to(tl.float32, bitcast=True)
    tl.store(OUT + on[:, None] * stride_on + okb[None, :], (v.to(tl.float32) * sf).to(tl.bfloat16), mask=mask_n[:, None])


def _dequant_bf16(weight, weight_scale, out):
    n, packed_k = weight.shape
    k = packed_k * 2
    bk = 256 if k % 256 == 0 else 128
    grid = (triton.cdiv(n, 64), k // bk)
    _dequant_bf16_kernel[grid](weight, weight_scale, out, n, k, weight.stride(0), weight_scale.stride(0), out.stride(0),
                               BN=64, BK=bk, num_warps=4)
    return out


_SCRATCH: dict = {}


def _scratch(numel: int, device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    buf = _SCRATCH.get(key)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, device=device, dtype=torch.uint8)
        _SCRATCH[key] = buf
    return buf


if _is_rdna4_platform():
    from vllm.utils.torch_utils import direct_register_custom_op

    def rdna_mxfp4_fp8_gemm(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        exact: bool,
    ) -> torch.Tensor:
        """``x @ dequant(weight).T`` on RDNA4: fused e4m3 GEMM (per-32-group activation
        scales) for M <= FUSED_MAX_M, exact bf16 dequant + hipBLASLt above."""
        m, k = x.shape
        n = weight.shape[0]
        if (exact and m <= MID_MAX_M) or (FUSED_MAX_M < m <= MID_MAX_M):
            # bf16 activations, weights dequantized inside the GEMM (weight-only kernel)
            return torch.ops.vllm.rdna_mxfp4_gemm(x, weight, weight_scale)
        if m <= FUSED_MAX_M:
            xb = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
            x8, sx = _groupquant(xb)
            bm, bn, un, nw, ns = _fused_cfg(m)
            y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
            grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
            _mxfp4_fp8_gemm_kernel[grid](x8, sx, weight, weight_scale, y, m, n, k,
                                         x8.stride(0), sx.stride(0), weight.stride(0), weight_scale.stride(0), y.stride(0),
                                         BM=bm, BN=bn, NUM_KB=k // 32, UNROLL=un, EVEN_M=(m % bm == 0),
                                         num_warps=nw, num_stages=ns)
        else:
            buf = _scratch(n * k * 2, x.device)[: n * k * 2].view(torch.bfloat16).view(n, k)
            _dequant_bf16(weight, weight_scale, buf)
            y = F.linear(x, buf)
        if y.dtype != x.dtype:
            y = y.to(x.dtype)
        return y

    def rdna_mxfp4_fp8_gemm_fake(x, weight, weight_scale, exact) -> torch.Tensor:
        return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=x.dtype, device=x.device)

    direct_register_custom_op(
        op_name="rdna_mxfp4_fp8_gemm",
        op_func=rdna_mxfp4_fp8_gemm,
        mutates_args=[],
        fake_impl=rdna_mxfp4_fp8_gemm_fake,
        dispatch_key=current_platform.dispatch_key,
    )


class RdnaMxfp4Fp8LinearKernel(MxFp4LinearKernel):
    """MXFP4 weights x e4m3 activations on RDNA4 FP8 WMMA (see module docstring)."""

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "not running on ROCm"
        if not _is_rdna4_platform():
            return False, "requires an AMD RDNA4 GPU (gfx1200/gfx1201)"
        if not _enabled():
            return False, f"disabled via {_ENV}=0"
        if not hasattr(torch, "float8_e4m3fn"):
            return False, "torch has no float8_e4m3fn"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key not in (None, kMxfp4Dynamic):
            return False, "supports unquantized or MXFP4-dynamic activation configs only"
        if config.input_size is not None and config.input_size % 128 != 0:
            return False, f"needs K to be a multiple of 128 (got {config.input_size})"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        weight_scale = layer.weight_scale.data
        if weight.dtype != torch.uint8 or weight_scale.dtype != torch.uint8:
            raise ValueError("RdnaMxfp4Fp8LinearKernel expects packed uint8 weights and uint8 E8M0 scales")
        if weight.ndim != 2 or weight_scale.ndim != 2:
            raise ValueError("RdnaMxfp4Fp8LinearKernel expects 2-D weights and scales")
        n, packed_k = weight.shape
        k = packed_k * 2
        if k % 128 != 0:
            raise ValueError(f"RdnaMxfp4Fp8LinearKernel needs K ({k}) to be a multiple of 128")
        if tuple(weight_scale.shape) != (n, k // MXFP4_GROUP_SIZE):
            raise ValueError(f"unexpected MXFP4 scale shape {tuple(weight_scale.shape)}")
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.contiguous(), requires_grad=False)
        prefix = getattr(layer, "prefix", "") or ""
        layer.rdna_fp8_exact = any(p in prefix for p in _skip_patterns())
        if layer.rdna_fp8_exact:
            logger.info_once("RdnaMxfp4Fp8LinearKernel: %s -> bf16-activation (exact) path for all M", prefix)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        n, packed_k = layer.weight.shape
        k = packed_k * 2
        out_shape = (*x.shape[:-1], n)
        if x.shape[-1] != k:
            raise ValueError(f"activation K ({x.shape[-1]}) does not match the packed weight K ({k})")
        if x.dtype not in (torch.bfloat16, torch.float16):
            dq_w = _dequant_mxfp4_reference(layer.weight, layer.weight_scale, x.dtype)
            return F.linear(x, dq_w, bias)
        x_2d = x.reshape(-1, k)
        if x_2d.shape[0] == 0:
            return torch.empty(out_shape, dtype=x.dtype, device=x.device)
        if x_2d.stride(-1) != 1:
            x_2d = x_2d.contiguous()
        y = torch.ops.vllm.rdna_mxfp4_fp8_gemm(x_2d, layer.weight, layer.weight_scale,
                                               bool(getattr(layer, "rdna_fp8_exact", False)))
        if bias is not None:
            y = y + bias
        return y.reshape(out_shape)
