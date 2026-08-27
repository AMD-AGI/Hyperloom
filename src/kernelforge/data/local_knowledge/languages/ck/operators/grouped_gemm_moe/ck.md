---
title: grouped_gemm_moe on ck — SOTA card
kind: sota_card
operator: grouped_gemm_moe
backend: ck
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, training]
status: competitive
updated: 2026-06-05
sources:
  - https://github.com/ROCm/composable_kernel
  - https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html
  - https://github.com/ROCm/aiter
---

# grouped_gemm_moe × ck

## TL;DR
> Composable Kernel provides a **fused grouped-GEMM device op** that processes all experts in one launch;
> it's aiter's fallback when FlyDSL is absent and the backbone of training-side fused MoE (Primus-Turbo).
> Strong for both fwd/bwd; for serving decode peak, aiter asm usually wins.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| CK fused grouped GEMM (single launch, all experts) | `ROCm/composable_kernel@HEAD` (grouped_gemm device op) | gfx942/950; bf16, fp8 | Primus-Turbo reports CK fused grouped GEMM outperforms multi-stream (scheduling overhead) — no first-party tok/s reproduced | MoE training fwd/bwd; aiter fallback |
| Primus-Turbo grouped GEMM (backend-selecting) | https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html | gfx942/950 | picks fastest grouped backend per kernel in fwd/bwd | training |

## Config space / knobs
- CK tile config: `MPerBlock/NPerBlock/KPerBlock`, `MPerXDL/NPerXDL` (mfma 16x16 vs 32x32), pipeline
  stages, block cluster; tune per group-M distribution.
- Single fused launch over expert offsets; select grouped backend per fwd/bwd kernel (Primus-Turbo).

## Numerics / parity
- fp32 accumulate; per-expert scales for fp8 → [numerics.md](numerics.md).

## Integration (rebind seam)
- Used via aiter (CK fallback) or directly through Primus-Turbo's grouped-GEMM op in training. Verify by
  CK kernel name in trace.

## Pitfalls & anti-patterns
- CK instances are shape-specialized; an uncovered (M-dist, N, K, dtype) compiles a new instance or falls
  back — pre-build the instances you need.

## How to verify
- A/B vs aiter/triton with per-expert dense oracle ([numerics.md](numerics.md)).

## Alternatives / cross-links
aiter · triton · hip · tilelang · [overview.md](overview.md)

## Sources
- Composable Kernel: https://github.com/ROCm/composable_kernel
- Primus-Turbo fused CK grouped GEMM: https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html
- AITER CK fallback: https://github.com/ROCm/aiter
