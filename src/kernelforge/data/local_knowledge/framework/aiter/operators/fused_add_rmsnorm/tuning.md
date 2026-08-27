---
title: fused_add_rmsnorm — tuning
kind: technique
operator: fused_add_rmsnorm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/triton/normalization/rmsnorm.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# fused_add_rmsnorm — tuning

Same bandwidth-bound playbook as rmsnorm/tuning.md, with **2× the traffic** (read
x + residual, write residual_out + y). The fusion itself is the primary win; after that, saturate
bandwidth.

## 1. Traffic accounting (why fuse)
| | reads | writes | passes |
|---|---|---|---|
| add then norm (2 kernels) | x, r (add); r' (norm) | r' (add); y (norm) | 5 |
| **fused** | x, r | r', y | 4 |
The fused kernel saves the `r'` round-trip (write+read of the residual). At N=8192 bf16, M=16k that's
~0.25 GB saved per call, ×2 per layer ×n_layers.

## 2. The kernel structure (aiter Triton, verified)
`_fused_add_rmsnorm_kernel` (`grid = min(rows, num_sms)`):
1. load `x` and `residual_in` (both bf16, vectorized 128-bit),
2. `r' = x + residual_in` (in IO dtype), **store `residual_out = r'`**,
3. `Σ(r')²` in fp32, `rsqrt`, `y = r'·rsqrt·γ`, store `y`.
Single-pass when N ≤ block (`65536/elt`); two-pass blocked otherwise (re-reads `r'`, but from the just-
written buffer — keep `r'` in registers/LDS for the single-pass case to avoid the re-read).

## 3. Knob table (same as rmsnorm)
| knob | setting | note |
|---|---|---|
| `num_warps` | 2–4 | memory-bound |
| `BLOCK_SIZE` | next_pow2(N) | full wave reduce, 128-bit loads |
| grid | `min(M, num_sms)` persistent | fill 304 CUs |
| `num_stages` | 2 | block pipelining |
| `waves_per_eu` | 3–4 | VGPR-light |
| `cache_modifier` | `.cg` on x, residual | read-once |

## 4. Keep r' on-chip for the single-pass case
Since `residual_out` is needed both as output AND as the input to the norm reduction, hold it in registers/
LDS after the add — don't write-then-reload from HBM (that defeats the fusion). The aiter kernel does the
add, the reduction, and both stores from the same on-chip `r'`.

## 5. Stack the quant fusion
The biggest further win is folding the downstream fp8 quant in (`rmsnorm2d_fwd_with_add_dynamicquant`) so
`y` is written as fp8 — see fused_norm_quant and [fusion.md](fusion.md).

## Sources
- `_fused_add_rmsnorm_kernel` structure (add, store residual_out, fp32 Σ, persistent grid): `aiter/ops/triton/normalization/rmsnorm.py`.
- bandwidth / 128-bit loads / ≥1024 grid: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
