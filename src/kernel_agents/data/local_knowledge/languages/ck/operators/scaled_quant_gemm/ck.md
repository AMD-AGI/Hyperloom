---
title: scaled_quant_gemm on ck — SOTA card
kind: sota_card
operator: scaled_quant_gemm
backend: ck
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp4_e2m1, fp6_e2m3]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-05
sources:
  - https://github.com/ROCm/composable_kernel
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# scaled_quant_gemm × ck

## TL;DR
> Composable Kernel provides **fp8/mxfp block-scaled GEMM instances** that drive the CDNA4 scaled-MFMA with
> CK's tile/scale-pipeline templates. The route when you need CK's epilogue/layout control or as aiter's
> fallback; for the live serving path reach it via aiter.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| CK block-scaled GEMM instance (f8f6f4 scaled MFMA) | `ROCm/composable_kernel@HEAD` (scaled gemm device ops) | gfx950 mxfp; gfx942 fp8 | no first-party number reproduced | CK-fused low-bit pipelines, aiter fallback |

## Config space / knobs
- Tile `MPerBlock/NPerBlock/KPerBlock` (K aligned to 32-elem blocks), `MPerXDL/NPerXDL` (scaled MFMA
  shape), scale-LDS staging, pipeline stages, split-K.

## Numerics / parity
- E8M0 block scales, scale-after-dot, fp32 accumulate; accuracy gate → [numerics.md](numerics.md).

## Integration (rebind seam)
- Via aiter CK fallback or direct CK op; verify CK scaled-GEMM kernel name in trace.

## Pitfalls & anti-patterns
- Shape/dtype-specialized instances — pre-build the (M,N,K,dtype,scale) you need; uncovered → fallback.

## How to verify
- bf16 accuracy gate + TFLOPS vs peak ([numerics.md](numerics.md)).

## Alternatives / cross-links
triton · aiter · hip · asm · hipblaslt · [overview.md](overview.md)

## Sources
- Composable Kernel: https://github.com/ROCm/composable_kernel
- Matrix Core (scaled MFMA): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
