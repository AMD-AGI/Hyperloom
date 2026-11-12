# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""v7: split-K + reshape-free + load hints. The research-designed bandwidth kernel.

- SPLIT-K (grid = N-tiles x K-splits) -> floods 84 CUs for small-N shapes (the occupancy fix).
- reshape-free unpack (8-nibble static loop, tensors only [ROWS,BLOCK_N]) -> low VGPR -> the extra
  split-K programs actually get occupancy.
- load hints: weight load cache_modifier=".cv" (glc|slc|dlc streaming, one-shot read) +
  tl.max_contiguous/tl.multiple_of on N to force wide vectorized loads.
- fp32 atomic-add partials into an [N] accumulator, then cast to fp16.
Autotuned over BLOCK_N / SPLIT_GROUPS / num_warps per (K,N). Run with AMDGCN_USE_BUFFER_OPS=1.
"""
import torch
import triton
import triton.language as tl

def _cfgs():
    return [triton.Config({"BLOCK_N": bn, "SPLIT_GROUPS": sg}, num_warps=nw)
            for bn in (32, 64) for sg in (8, 16, 32, 64) for nw in (1, 2)]


@triton.autotune(configs=_cfgs(), key=["K", "N"], reset_to_zero=["cacc_ptr"])
@triton.jit
def _splitk_kernel(a_ptr, qw_ptr, s_ptr, z_ptr, cacc_ptr, K, N,
                   GROUP: tl.constexpr, SPLIT_GROUPS: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n = tl.max_contiguous(tl.multiple_of(n, BLOCK_N), BLOCK_N)
    nmask = n < N
    ROWS: tl.constexpr = GROUP // 8
    num_groups = K // GROUP
    g0 = pid_k * SPLIT_GROUPS
    g1 = min(g0 + SPLIT_GROUPS, num_groups)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for g in range(g0, g1):
        k0 = g * GROUP
        row = (k0 // 8) + tl.arange(0, ROWS)
        qw = tl.load(qw_ptr + row[:, None] * N + n[None, :], mask=nmask[None, :], other=0,
                     cache_modifier=".cv")                                       # [ROWS,BLOCK_N]
        a_group = tl.load(a_ptr + k0 + tl.arange(0, GROUP)).to(tl.float32)
        asum = tl.sum(a_group)
        gacc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for j in tl.static_range(8):
            qv = ((qw >> (j * 4)) & 0xF).to(tl.float32)                          # [ROWS,BLOCK_N]
            ap = tl.load(a_ptr + k0 + j + tl.arange(0, ROWS) * 8).to(tl.float32)  # [ROWS]
            gacc += tl.sum(ap[:, None] * qv, axis=0)
        s = tl.load(s_ptr + g * N + n, mask=nmask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + g * N + n, mask=nmask, other=0.0).to(tl.float32)
        acc += (gacc - z * asum) * s
    tl.atomic_add(cacc_ptr + n, acc, mask=nmask)


def gemv(a, w_q, w_s, w_zp, group_size):
    K = a.shape[-1]
    N = w_s.shape[1]
    cacc = torch.zeros((N,), dtype=torch.float32, device=a.device)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),
                         triton.cdiv(K // group_size, meta["SPLIT_GROUPS"]))
    _splitk_kernel[grid](a, w_q, w_s, w_zp, cacc, K, N, GROUP=group_size)
    return cacc.to(a.dtype).view(1, N)


if __name__ == "__main__":
    torch.manual_seed(0)
    Gp = 128
    for (K, N) in [(5120, 5120), (5120, 7168), (13824, 5120), (5120, 27648)]:
        a = torch.randn(1, K, device="cuda", dtype=torch.float16) * 0.1
        wq = torch.randint(0, 2**31 - 1, (K // 8, N), device="cuda", dtype=torch.int32)
        ws = torch.randn(K // Gp, N, device="cuda", dtype=torch.float16) * 0.01
        wz = torch.randint(0, 16, (K // Gp, N), device="cuda", dtype=torch.uint8)
        shifts = (torch.arange(8, device="cuda", dtype=torch.int32) * 4)
        q = ((wq.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(K, N).float()
        gidx = torch.arange(K, device="cuda") // Gp
        ref = a.float() @ ((q - wz.float()[gidx]) * ws.float()[gidx])
        c = gemv(a, wq, ws, wz, Gp)
        rel = (c.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
        print(f"K={K} N={N} rel={rel:.2e} {'OK' if rel < 1e-2 else 'FAIL'}")
    print("VARIANT_OK")
