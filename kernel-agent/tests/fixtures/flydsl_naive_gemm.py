# Naive GEMM kernel written with FlyDSL.
#
# Lifted from the AMD-AGI/kernel_playground `flydsl/naive_gemm_flydsl/`
# example referenced in Hyperloom issue #211 §5 as the canonical FlyDSL
# regression sample. Kept minimal so the file remains self-contained and
# parseable by ``tracelens_analysis._looks_like_flydsl_source`` /
# ``_flydsl_kernel_params`` without bringing the upstream FlyDSL runtime
# into Hyperloom's test environment. Trimmed to the markers we actually
# assert on:
#
#   * ``@flyc.kernel`` decorator    -> source-type sniff signal
#   * ``flydsl.compiler`` import    -> source-type sniff signal
#   * ``flydsl.expr`` import        -> source-type sniff signal
#   * ``SmemAllocator`` reference   -> FLYDSL_USES_SMEM=True
#   * ``rocdl.make_buffer_tensor``  -> FLYDSL_USES_BUFFER_LOAD=True
#
# This file is not executed; it is loaded as text by the FlyDSL classifier
# during patchability + metadata enrichment tests.

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
