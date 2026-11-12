# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Reshape-free M=1 W4 GEMV: unpack the 8 nibbles in a static loop (tensors only [ROWS, BLOCK_N],
never [GROUP, BLOCK_N]) so register pressure stays low and BLOCK_N can be >= 32 (wave32 width)
for coalesced N-contiguous loads. Autotuned over BLOCK_N / num_warps / waves_per_eu."""
import torch
import triton
import triton.language as tl


def _configs():
    cfgs = []
    for bn in (32, 64, 128, 256):
        for nw in (1, 2, 4, 8):
            for we in (0, 2):
                cfgs.append(triton.Config({"BLOCK_N": bn, "waves_per_eu": we}, num_warps=nw))
    return cfgs


@triton.autotune(configs=_configs(), key=["K", "N"])
@triton.jit
def _gemv_rf(a_ptr, qw_ptr, s_ptr, z_ptr, c_ptr, K, N,
            GROUP: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = n < N
    ROWS: tl.constexpr = GROUP // 8
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for g in range(0, K // GROUP):
        k0 = g * GROUP
        row = (k0 // 8) + tl.arange(0, ROWS)
        qw = tl.load(qw_ptr + row[:, None] * N + n[None, :], mask=nmask[None, :], other=0)  # [ROWS,BN]
        a_group = tl.load(a_ptr + k0 + tl.arange(0, GROUP)).to(tl.float32)  # [GROUP]
        asum = tl.sum(a_group)
        gacc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for p in tl.static_range(8):
            qp = ((qw >> (p * 4)) & 0xF).to(tl.float32)                       # [ROWS,BN]
            ap = tl.load(a_ptr + k0 + p + tl.arange(0, ROWS) * 8).to(tl.float32)  # [ROWS], k=row*8+p
            gacc += tl.sum(ap[:, None] * qp, axis=0)
        s = tl.load(s_ptr + g * N + n, mask=nmask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + g * N + n, mask=nmask, other=0.0).to(tl.float32)
        acc += (gacc - z * asum) * s
    tl.store(c_ptr + n, acc.to(c_ptr.type.element_ty), mask=nmask)


def gemv(a, w_q, w_s, w_zp, group_size):
    K = a.shape[-1]
    N = w_s.shape[1]
    c = torch.empty((1, N), dtype=a.dtype, device=a.device)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _gemv_rf[grid](a, w_q, w_s, w_zp, c, K, N, GROUP=group_size)
    return c
