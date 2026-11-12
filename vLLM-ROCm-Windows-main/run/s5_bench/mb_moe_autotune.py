# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Verify the moe_decode autotune change: correctness (vs fp32 reference) + which BLOCK_N autotune picks +
GB/s, on synthetic full-expert tensors (no model). gate_up (wide N) and down (narrow N)."""
import os, time
os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")
import torch
import vllm_windows_rocm.moe_decode as md

dev = "cuda"
G = 32
BIAS = 8.0

def ref(x, w, s, ids):
    NE, N, Kp = w.shape
    K = x.shape[-1] if x.dim() == 1 else x.shape[1]
    out = torch.empty(len(ids), N, device=dev, dtype=torch.float32)
    for i, eid in enumerate(ids.tolist()):
        wp = w[eid].to(torch.int32)
        lo = (wp & 0xF).float() - BIAS
        hi = ((wp >> 4) & 0xF).float() - BIAS
        codes = torch.stack([lo, hi], dim=-1).reshape(N, K)          # [N,K]
        sfull = s[eid].float().repeat_interleave(G, dim=1)           # [N,K]
        wdeq = codes * sfull
        xv = (x if x.dim() == 1 else x[i]).float()
        out[i] = wdeq @ xv
    return out

def one(tag, N, K, x_per_expert):
    NE = 128
    w = torch.randint(0, 256, (NE, N, K // 2), device=dev, dtype=torch.uint8)
    s = (torch.randn(NE, N, K // G, device=dev, dtype=torch.float16) * 0.02)
    ids = torch.randperm(NE, device=dev)[:8].to(torch.int32)
    x = (torch.randn(8, K, device=dev, dtype=torch.float16) * 0.3) if x_per_expert \
        else (torch.randn(K, device=dev, dtype=torch.float16) * 0.3)
    out = md._moe_gemv_batched(x, w, s, ids, G, x_per_expert)
    orf = ref(x, w, s, ids)
    rel = (out.float() - orf).abs().max() / (orf.abs().max() + 1e-6)
    best = md._moe_gemv_batched_kernel.best_config
    # timing (cache-cold-ish: big tensors)
    def run(): md._moe_gemv_batched(x, w, s, ids, G, x_per_expert)
    for _ in range(20): run()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(200): run()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 200
    active_bytes = 8 * N * (K // 2) + 8 * N * (K // G) * 2  # w4 + scales for 8 experts
    gbs = active_bytes / dt / 1e9
    print(f"{tag:8s} N={N} K={K}: rel={rel.item():.3e} {'OK' if rel<2e-2 else 'MISMATCH'} | "
          f"picked {best} | {gbs:.0f} GB/s")

one("gate_up", 3072, 2560, x_per_expert=False)  # ERNIE-ish wide N
one("down",    2560, 1536, x_per_expert=True)   # narrow N
print("MB_MOE_AUTOTUNE_DONE")
