# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Skinny bf16/fp16 M=1 GEMV for the DENSE (unquantized) linears on RDNA3.

Profiling gemma-4-26B-A4B decode showed the per-layer DENSE MLP (bf16, unquantized shared expert,
gate/up/down at intermediate 2112) is ~43% of the per-token weight bytes -- the single biggest
memory consumer. On ROCm it goes through rocm_unquantized_gemm -> wvSplitK (the skinny M=1 GEMM,
ABSENT on our Windows build) -> torch.nn.functional.linear (rocBLAS), which is GEMM-tuned and
under-utilises the 84 CUs at M=1 for these small (2112/2816-row) shapes (rocBLAS ~330 GB/s cached).

This is a small-tile GEMV (few output rows per program -> many programs -> the CUs stay busy),
streaming the weight once. Patches UnquantizedLinearMethod.apply for the M==1 decode step only;
everything else (prefill, M>1) falls through to the original path. Enable VLLM_WIN_BF16_GEMV=1.
"""
import logging
import os

import torch

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)


@triton.jit
def _bf16_gemv_kernel(x_ptr, w_ptr, b_ptr, o_ptr, K, N,
                      HAS_BIAS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out[n] = sum_k w[n,k]*x[k] (+ bias[n]).  w is [N, K] row-major; x is [K].
    # One program handles BLOCK_N contiguous output rows; loops the K reduction. Small BLOCK_N ->
    # many programs -> full CU occupancy for the small dense-MLP shapes.
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = n < N
    acc = tl.zeros((BLOCK_N,), tl.float32)
    wrow = w_ptr + n[:, None] * K
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        kmask = kk < K
        x = tl.load(x_ptr + kk, mask=kmask, other=0.0).to(tl.float32)
        w = tl.load(wrow + kk[None, :], mask=nmask[:, None] & kmask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    if HAS_BIAS:
        acc += tl.load(b_ptr + n, mask=nmask, other=0.0).to(tl.float32)
    tl.store(o_ptr + n, acc.to(o_ptr.type.element_ty), mask=nmask)


def bf16_gemv(x_row, weight, bias):
    # x_row [K] contiguous; weight [N, K]; bias [N] or None -> out [N]
    N, K = weight.shape
    o = torch.empty(N, dtype=x_row.dtype, device=x_row.device)
    grid = lambda m: (triton.cdiv(N, m["BLOCK_N"]),)
    _bf16_gemv_kernel[grid](
        x_row, weight, bias if bias is not None else x_row, o, K, N,
        HAS_BIAS=bias is not None, BLOCK_N=16, BLOCK_K=256, num_warps=4)
    return o


_PATCHED = False


def patch_unquantized_linear() -> None:
    global _PATCHED
    if _PATCHED or os.environ.get("VLLM_WIN_BF16_GEMV", "0") != "1":
        return
    try:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    except Exception as e:  # noqa: BLE001
        logger.warning("bf16_gemv patch warning: %r", e)
        return
    validate = os.environ.get("VLLM_WIN_BF16_GEMV_VALIDATE", "0") == "1"
    _orig = UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None):
        w = layer.weight
        if (x.dim() >= 2 and x.shape[:-1].numel() == 1 and w.dim() == 2
                and w.dtype in (torch.bfloat16, torch.float16)
                and x.dtype == w.dtype and x.shape[-1] == w.shape[1]):
            xr = x.reshape(-1)  # [K]
            y = bf16_gemv(xr.contiguous(), w, bias)
            out = y.reshape(*x.shape[:-1], w.shape[0])
            if validate:
                ref = _orig(self, layer, x, bias)
                rel = (out.float() - ref.float()).abs().max().item() / (
                    ref.float().abs().max().item() + 1e-6)
                print(f"BF16_GEMV_VALIDATE N={w.shape[0]} K={w.shape[1]} rel={rel:.3e}")
                return ref
            return out
        return _orig(self, layer, x, bias)

    UnquantizedLinearMethod.apply = apply
    _PATCHED = True
    logger.info("patched UnquantizedLinearMethod.apply (M=1 bf16 skinny GEMV)")
