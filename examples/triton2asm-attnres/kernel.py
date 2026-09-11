# SPDX-FileCopyrightText: 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Triton score extracted from AITER PR #4863; see README for the pinned source."""

import triton
import triton.language as tl


@triton.jit
def _score_kernel_ref(
    prefix_ptr,
    bank_ptr,
    cw_ptr,
    scores_ptr,
    NVB,
    EPS: tl.constexpr,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_j = tl.program_id(1)

    offs_h = tl.arange(0, BLOCK_H)
    sumsq = tl.zeros([BLOCK_H], tl.float32)
    dotv = tl.zeros([BLOCK_H], tl.float32)

    for bh in range(0, tl.cdiv(H, BLOCK_H)):
        h0 = bh * BLOCK_H
        if pid_j < NVB:
            v = tl.load(bank_ptr + pid_t * stride_bm + pid_j * stride_bb + h0 + offs_h)
        else:
            v = tl.load(prefix_ptr + pid_t * stride_pm + h0 + offs_h)
        vf = v.to(tl.float32)
        cw = tl.load(cw_ptr + h0 + offs_h)
        sumsq += vf * vf
        dotv += vf * cw

    ss = tl.sum(sumsq, 0)
    dv = tl.sum(dotv, 0)
    rrms = tl.rsqrt(ss / H + EPS)
    score = dv * rrms
    tl.store(scores_ptr + pid_t * stride_sm + pid_j, score)


class Score:
    """Original Triton specialization with the same public API as the ASM port."""

    def __call__(self, prefix, bank, weight, output):
        _score_kernel_ref[(64, 9)](
            prefix,
            bank,
            weight,
            output,
            8,
            1e-6,
            prefix.stride(0),
            bank.stride(0),
            bank.stride(1),
            output.stride(0),
            H=7168,
            BLOCK_H=1024,
        )

    def close(self):
        pass
