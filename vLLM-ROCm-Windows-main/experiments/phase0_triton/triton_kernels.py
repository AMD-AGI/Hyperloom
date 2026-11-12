# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Phase-0+ Triton hardening on gfx1100: a tl.dot matmul (must lower to WMMA on RDNA3)
and a row-wise softmax — beyond the trivial vector-add. Validated vs torch.

Run with plain `python triton_kernels.py` (triton-windows JIT finds its own clang/HIP toolchain).
"""
import torch
import triton
import triton.language as tl

print("torch", torch.__version__, "| triton", triton.__version__,
      "| dev", torch.cuda.get_device_name(0))


# ---------------- Triton matmul (tl.dot -> WMMA on RDNA3) ----------------
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def triton_matmul(a, b):
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c


# ---------------- Triton row softmax ----------------
@triton.jit
def softmax_kernel(x_ptr, o_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(o_ptr + row * n_cols + cols, num / den, mask=mask)


def triton_softmax(x):
    rows, cols = x.shape
    o = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(cols)
    softmax_kernel[(rows,)](x, o, cols, BLOCK=BLOCK)
    return o


if __name__ == "__main__":
    torch.manual_seed(0)

    # matmul
    try:
        M = N = K = 512
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        c = triton_matmul(a, b)
        ref = a.float() @ b.float()
        rel = (c - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
        print("TRITON_MATMUL_OK", bool(rel < 0.02), "rel", round(rel, 5))
    except Exception as e:
        print("TRITON_MATMUL_OK", False, "err", repr(e))

    # softmax
    try:
        x = torch.randn(128, 2048, device="cuda", dtype=torch.float32)
        o = triton_softmax(x)
        ref = torch.softmax(x, dim=-1)
        err = (o - ref).abs().max().item()
        print("TRITON_SOFTMAX_OK", bool(err < 1e-5), "max_err", err)
    except Exception as e:
        print("TRITON_SOFTMAX_OK", False, "err", repr(e))
