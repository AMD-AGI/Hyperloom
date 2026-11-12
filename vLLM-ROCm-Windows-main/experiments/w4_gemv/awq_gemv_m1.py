# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Dedicated M=1 AWQ W4A16 dequant-GEMV (Triton) for single-stream decode on RDNA3.

Why: the exllama gptq_gemm and conch W4 GEMM are throughput kernels (split-K + atomicAdd +
zero-init, or tl.dot padded to block_m>=16). At M=1 they waste ~4x-21x of bandwidth. A true
GEMV reads each 4-bit weight exactly once, unpacks + dequantizes in registers, multiply-
accumulates over K, writes once -- no split-K, no atomicAdd, no zero-init. Bandwidth-bound.

AWQ packing (matches vllm/model_executor/layers/quantization/awq_triton.py):
  qweight : [K, N//8]      int32, 8 output cols packed per int32, reverse-order [0,4,1,5,2,6,3,7]
  qzeros  : [K//G, N//8]   int32, same packing along N
  scales  : [K//G, N]      fp16
  dequant : w[k,n] = (nibble(qweight) - nibble(qzeros)) * scales[k//G, n]
This kernel assumes BLOCK_K == group_size (one quant group per K-iteration) so scales/zeros are
loaded once per iteration -- the common case (G=128).
"""
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _awq_gemv_m1_kernel(
    a_ptr, qw_ptr, scales_ptr, qz_ptr, c_ptr,
    K, N, group_size,
    BLOCK_N8: tl.constexpr,   # number of int32 columns per program -> BLOCK_N8*8 output cols
    BLOCK_K: tl.constexpr,    # == group_size
):
    pid = tl.program_id(0)
    N8 = N // 8
    col8 = pid * BLOCK_N8 + tl.arange(0, BLOCK_N8)        # int32 column indices [BLOCK_N8]
    col8_mask = col8 < N8

    # reverse-AWQ order -> shifts for the 8 nibbles, tiled across BLOCK_N8 int32 cols
    rev = ((tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]).reshape(8)  # [0,4,1,5,2,6,3,7]
    shifts = (rev * 4)                                     # [8]
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_N8, 8))
    shifts = tl.reshape(shifts, (BLOCK_N8 * 8,))          # [BLOCK_N8*8]

    out_n = pid * BLOCK_N8 * 8 + tl.arange(0, BLOCK_N8 * 8)   # full-N output indices
    out_mask = out_n < N

    acc = tl.zeros((BLOCK_N8 * 8,), dtype=tl.float32)

    num_groups = tl.cdiv(K, BLOCK_K)
    for g in range(0, num_groups):
        k0 = g * BLOCK_K
        offs_k = k0 + tl.arange(0, BLOCK_K)               # [BLOCK_K]
        k_mask = offs_k < K

        a = tl.load(a_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_K]

        # qweight tile [BLOCK_K, BLOCK_N8] int32
        qw = tl.load(qw_ptr + offs_k[:, None] * N8 + col8[None, :],
                     mask=k_mask[:, None] & col8_mask[None, :], other=0)
        # expand each int32 -> 8 (repeat) via interleave x3, then shift+mask -> [BLOCK_K, BLOCK_N8*8]
        qw = tl.interleave(qw, qw)
        qw = tl.interleave(qw, qw)
        qw = tl.interleave(qw, qw)                        # [BLOCK_K, BLOCK_N8*8]
        w = (qw >> shifts[None, :]) & 0xF                 # nibbles [BLOCK_K, BLOCK_N8*8]

        # zeros/scales for this group (one row, group index = g since BLOCK_K==group_size)
        qz = tl.load(qz_ptr + g * N8 + col8, mask=col8_mask, other=0)        # [BLOCK_N8] int32
        qz = tl.interleave(qz, qz)
        qz = tl.interleave(qz, qz)
        qz = tl.interleave(qz, qz)                        # [BLOCK_N8*8]
        z = (qz >> shifts) & 0xF                          # [BLOCK_N8*8]

        s = tl.load(scales_ptr + g * N + out_n, mask=out_mask, other=0.0).to(tl.float32)  # [BLOCK_N8*8]

        wdq = (w.to(tl.float32) - z.to(tl.float32)[None, :]) * s[None, :]    # [BLOCK_K, BLOCK_N8*8]
        acc += tl.sum(a[:, None] * wdq, axis=0)

    tl.store(c_ptr + out_n, acc.to(c_ptr.type.element_ty), mask=out_mask)


def awq_gemv_m1(a, qweight, scales, qzeros, group_size, BLOCK_N8=8, num_warps=4):
    # a: [1, K] fp16 ; qweight: [K, N//8] int32 ; scales: [K//G, N] fp16 ; qzeros: [K//G, N//8] int32
    K = a.shape[-1]
    N = scales.shape[1]
    c = torch.empty((1, N), dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(N // 8, BLOCK_N8),)
    _awq_gemv_m1_kernel[grid](
        a, qweight, scales, qzeros, c, K, N, group_size,
        BLOCK_N8=BLOCK_N8, BLOCK_K=group_size, num_warps=num_warps,
    )
    return c


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    dev = "cuda"
    G = 128
    for (K, N) in [(3584, 3584), (3584, 18944), (18944, 3584)]:
        a = torch.randn(1, K, device=dev, dtype=torch.float16) * 0.1
        qweight = torch.randint(0, 2**31 - 1, (K, N // 8), device=dev, dtype=torch.int32)
        qzeros = torch.randint(0, 2**31 - 1, (K // G, N // 8), device=dev, dtype=torch.int32)
        scales = (torch.randn(K // G, N, device=dev, dtype=torch.float16) * 0.01)

        # reference dequant via vLLM's awq_dequantize_triton, then matmul
        from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton
        w_ref = awq_dequantize_triton(qweight, scales, qzeros)   # [K, N] fp16
        c_ref = (a.float() @ w_ref.float())

        c = awq_gemv_m1(a, qweight, scales, qzeros, G)
        err = (c.float() - c_ref).abs().max().item()
        rel = err / (c_ref.abs().max().item() + 1e-6)
        print(f"K={K} N={N}  max_abs_err={err:.4e} rel={rel:.4e}  {'OK' if rel < 1e-2 else 'FAIL'}")

        # autotune sweep: BLOCK_N8 x num_warps
        wbytes = K * N * 0.5
        best = None
        for bn8 in (2, 4, 8, 16, 32):
            for nw in (1, 2, 4, 8):
                try:
                    for _ in range(15):
                        awq_gemv_m1(a, qweight, scales, qzeros, G, BLOCK_N8=bn8, num_warps=nw)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(200):
                        awq_gemv_m1(a, qweight, scales, qzeros, G, BLOCK_N8=bn8, num_warps=nw)
                    torch.cuda.synchronize()
                    dt = (time.perf_counter() - t0) / 200
                    gbps = wbytes / dt / 1e9
                    if best is None or gbps > best[0]:
                        best = (gbps, dt, bn8, nw)
                except Exception:
                    pass
        gbps, dt, bn8, nw = best
        print(f"   BEST gemv {dt*1e6:7.1f} us  {gbps:6.1f} GB/s  (BLOCK_N8={bn8} num_warps={nw}; ideal {wbytes/800e9*1e6:.0f}us)")
    print("GEMV_DONE")
