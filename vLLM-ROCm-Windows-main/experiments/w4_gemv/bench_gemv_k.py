# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Microbench the packed-along-K M=1 W4 GEMV (the plugin kernel) at real model shapes, with a
correctness check vs a torch reference. Self-contained (kernel copied here for fast iteration).

Layout (conch-normalized): w_q [K//8, N] int32 packed-along-K straight order, w_s [K//G, N] fp16,
w_zp [K//G, N] uint8 unpacked. Dequant w[k,n] = (q(k,n) - z(g,n)) * s(g,n).
"""
import time
import torch
import triton
import triton.language as tl

G = 128


@triton.jit
def _gemv_k_kernel(a_ptr, qw_ptr, s_ptr, z_ptr, c_ptr, K, N,
                   BLOCK_N: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = n < N
    ROWS: tl.constexpr = GROUP // 8
    shifts = (tl.arange(0, 8) * 4).to(tl.int32)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    num_groups = K // GROUP
    for g in range(0, num_groups):
        k0 = g * GROUP
        row = (k0 // 8) + tl.arange(0, ROWS)
        qw = tl.load(qw_ptr + row[:, None] * N + n[None, :], mask=nmask[None, :], other=0)
        q = (qw[:, None, :] >> shifts[None, :, None]) & 0xF
        q = tl.reshape(q, (GROUP, BLOCK_N)).to(tl.float32)
        a = tl.load(a_ptr + k0 + tl.arange(0, GROUP)).to(tl.float32)
        contrib = tl.sum(a[:, None] * q, axis=0)
        asum = tl.sum(a)
        s = tl.load(s_ptr + g * N + n, mask=nmask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + g * N + n, mask=nmask, other=0.0).to(tl.float32)
        acc += (contrib - z * asum) * s
    tl.store(c_ptr + n, acc.to(c_ptr.type.element_ty), mask=nmask)


def gemv_k(a, w_q, w_s, w_zp, group_size, BLOCK_N=64, num_warps=2):
    K = a.shape[-1]; N = w_s.shape[1]
    c = torch.empty((1, N), dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(N, BLOCK_N),)
    _gemv_k_kernel[grid](a, w_q, w_s, w_zp, c, K, N,
                         BLOCK_N=BLOCK_N, GROUP=group_size, num_warps=num_warps)
    return c


def ref_dequant_matmul(a, w_q, w_s, w_zp, group_size):
    K8, N = w_q.shape
    K = K8 * 8
    shifts = (torch.arange(8, device=w_q.device, dtype=torch.int32) * 4)
    q = (w_q.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF      # [K8,8,N]
    q = q.reshape(K, N).float()
    gidx = torch.arange(K, device=w_q.device) // group_size
    w = (q - w_zp.float()[gidx]) * w_s.float()[gidx]          # [K,N]
    return a.float() @ w


SHAPES = {
    "14B qkv  K5120 N7168":  (5120, 7168),
    "14B o    K5120 N5120":  (5120, 5120),
    "14B gate K5120 N27648": (5120, 27648),
    "14B down K13824 N5120": (13824, 5120),
}


def make(K, N):
    a = torch.randn(1, K, device="cuda", dtype=torch.float16) * 0.1
    w_q = torch.randint(0, 2**31 - 1, (K // 8, N), device="cuda", dtype=torch.int32)
    w_s = torch.randn(K // G, N, device="cuda", dtype=torch.float16) * 0.01
    w_zp = torch.randint(0, 16, (K // G, N), device="cuda", dtype=torch.uint8)
    return a, w_q, w_s, w_zp


def time_cold(K, N, bn, nw, ref):
    # round-robin over enough weight copies to exceed the 96MB Infinity Cache -> cold DRAM reads
    wbytes = K * N * 0.5
    ncopies = max(3, int(160e6 // wbytes) + 1)
    a = torch.randn(1, K, device="cuda", dtype=torch.float16) * 0.1
    copies = []
    for _ in range(ncopies):
        copies.append((
            torch.randint(0, 2**31 - 1, (K // 8, N), device="cuda", dtype=torch.int32),
            torch.randn(K // G, N, device="cuda", dtype=torch.float16) * 0.01,
            torch.randint(0, 16, (K // G, N), device="cuda", dtype=torch.uint8),
        ))
    # correctness on copy 0
    wq, ws, wz = copies[0]
    c = gemv_k(a, wq, ws, wz, G, BLOCK_N=bn, num_warps=nw)
    rel = (c.float() - ref(a, wq, ws, wz)).abs().max().item() / (ref(a, wq, ws, wz).abs().max().item() + 1e-6)
    if rel > 1e-2:
        return None
    for i in range(ncopies):
        wq, ws, wz = copies[i % ncopies]
        gemv_k(a, wq, ws, wz, G, BLOCK_N=bn, num_warps=nw)
    torch.cuda.synchronize()
    iters = ncopies * 8
    t0 = time.perf_counter()
    for i in range(iters):
        wq, ws, wz = copies[i % ncopies]
        gemv_k(a, wq, ws, wz, G, BLOCK_N=bn, num_warps=nw)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return wbytes / dt / 1e9, dt, rel


if __name__ == "__main__":
    print("RX 7900 XT ~800 GB/s DRAM, 96MB Infinity Cache. M=1 W4 GEMV (packed-along-K), CACHE-COLD:")
    ref = lambda a, wq, ws, wz: ref_dequant_matmul(a, wq, ws, wz, G)
    for name, (K, N) in SHAPES.items():
        wbytes = K * N * 0.5
        best = None
        for bn in (8, 16, 32, 64):
            for nw in (1, 2, 4):
                try:
                    r = time_cold(K, N, bn, nw, ref)
                    if r is None:
                        continue
                    gbps, dt, rel = r
                    if best is None or gbps > best[0]:
                        best = (gbps, dt, bn, nw, rel)
                except Exception:
                    pass
        if best:
            gbps, dt, bn, nw, rel = best
            print(f"  {name:24} {dt*1e6:7.1f}us {gbps:6.1f} GB/s  bn={bn} nw={nw} rel={rel:.0e} "
                  f"({gbps/800*100:.0f}% of DRAM peak)")
        else:
            print(f"  {name:24} FAIL")
    print("BENCH_GEMV_K_DONE")
