---
title: splitk_streamk_gemm on ck — SOTA card
kind: sota_card
operator: splitk_streamk_gemm
backend: ck
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-05
sources:
  - https://github.com/ROCm/composable_kernel
  - https://arxiv.org/abs/2301.03598
---

# splitk_streamk_gemm × ck

## TL;DR
> Composable Kernel ships **split-K GEMM device ops** (and stream-K-style tile mapping) as templated
> instances; pick the split-K instance for large-K/small-M shapes. Authorable at the template level, good
> when you need CK's fusion/layout control; for the live serving path it's reachable via aiter's CK fallback.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| CK split-K GEMM instance (atomic or workspace reduce) | `ROCm/composable_kernel@HEAD` (gemm split-k device ops) | gfx942/950; bf16, fp16, fp8 | no first-party number reproduced; shape-dependent | large-K small-M, CK-fused pipelines |

## Config space / knobs
- `KBatch` (= split-K factor), tile `MPerBlock/NPerBlock/KPerBlock`, `MPerXDL/NPerXDL` (mfma 16x16/32x32),
  reduction mode, pipeline stages.

## Numerics / parity
- fp32 accumulate; KBatch reduction order affects determinism (atomic vs workspace) → [numerics.md](numerics.md).

## Integration (rebind seam)
- Via aiter CK fallback or a direct CK op; verify CK kernel name in trace.

## Pitfalls & anti-patterns
- Shape-specialized instances: an uncovered (M,N,K,dtype,KBatch) compiles fresh or misses — pre-build.

## How to verify
- A/B vs dense, dense fp32 oracle ([numerics.md](numerics.md)).

## Alternatives / cross-links
triton · hipblaslt · asm · hip · [overview.md](overview.md)

## Sources
- Composable Kernel (split-K device ops): https://github.com/ROCm/composable_kernel
- Stream-K: https://arxiv.org/abs/2301.03598
