# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One Triton vector-add specialization with an ordinary bracket launch."""

from __future__ import annotations

import triton
import triton.language as tl

N = 4103


@triton.jit
def _vector_add(A, B, C, N, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + index, index < N, other=0)
    b = tl.load(B + index, index < N, other=0)
    tl.store(C + index, a + b, index < N)


class VectorAdd:
    def __call__(self, a, b, out):
        return _vector_add[(triton.cdiv(N, 256),)](a, b, out, N, BLOCK=256, num_warps=4)

    def close(self):
        pass
