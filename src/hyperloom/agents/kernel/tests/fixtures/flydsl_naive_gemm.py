# Copyright Advanced Micro Devices, Inc. All rights reserved.

# Naive GEMM FlyDSL fixture (issue #211 §5): text sample for classifier patchability/metadata tests; not executed.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.utils.smem_allocator import SmemAllocator


@flyc.kernel
def naive_gemm(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    M: fx.Int,
    N: fx.Int,
    K: fx.Int,
    BLOCK_M: fx.Constexpr = 128,
    BLOCK_N: fx.Constexpr = 128,
    BLOCK_K: fx.Constexpr = 32,
):
    pid_m = fx.program_id(0)
    pid_n = fx.program_id(1)

    smem = SmemAllocator()
    a_smem = smem.alloc((BLOCK_M, BLOCK_K), dtype=A.dtype)
    b_smem = smem.alloc((BLOCK_K, BLOCK_N), dtype=B.dtype)

    acc = fx.zeros((BLOCK_M, BLOCK_N), dtype=fx.float32)
    a_buf = rocdl.make_buffer_tensor(A)
    b_buf = rocdl.make_buffer_tensor(B)

    for k in range(0, K, BLOCK_K):
        a_tile = a_buf.load((pid_m * BLOCK_M, k), (BLOCK_M, BLOCK_K))
        b_tile = b_buf.load((k, pid_n * BLOCK_N), (BLOCK_K, BLOCK_N))
        a_smem.store(a_tile)
        b_smem.store(b_tile)
        fx.barrier()
        acc += fx.dot(a_smem.load(), b_smem.load())
        fx.barrier()

    c_off = (pid_m * BLOCK_M, pid_n * BLOCK_N)
    C[c_off : c_off + (BLOCK_M, BLOCK_N)] = acc.to(C.dtype)
