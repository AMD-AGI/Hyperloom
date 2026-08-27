---
title: scaled_quant_gemm on hip — SOTA card
kind: sota_card
operator: scaled_quant_gemm
backend: hip
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp4_e2m1, fp6_e2m3, fp6_e3m2]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-05
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
---

# scaled_quant_gemm × hip

## TL;DR
> Hand-written HIP/C++ using the scaled-MFMA builtins
> (`__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4`, 16x16x128 variant) is the lowest-level authorable
> path and the reference for understanding the scale pipeline. Use for bespoke fusions/research; production
> serving should use aiter/triton.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| HIP scaled MFMA GEMM (f8f6f4 builtins + LDS scale staging) | Matrix Core blog + CDNA4 ISA | gfx950 mxfp; gfx942 fp8 FNUZ | no published number — reference/authoring | custom low-bit fusion, asm reference |

## Config space / knobs
- Scaled-MFMA variant (32x32x64 / 16x16x128), fp4 256-bit-wide operand packing (upper 128 bits zero),
  scale LDS re-layout pipeline, K aligned to 32-elem blocks, register tiling.

## Numerics / parity
- E8M0, scale-after-dot, fp32 accumulate; FNUZ (gfx942) vs OCP (gfx950) encoding → [numerics.md](numerics.md).

## Integration (rebind seam)
- Custom op / hipModule at the GEMM call site; verify kernel name in trace.

## Pitfalls & anti-patterns
- Feeding the global-memory scale layout directly to scaled MFMA (skipping LDS re-layout) → wrong results.
- gfx942 lacks native block-scaled MFMA — the builtins target gfx950; don't assume cross-gen.

## How to verify
- bf16 accuracy gate ([numerics.md](numerics.md)) + TFLOPS vs peak; A/B vs triton scaled matmul.

## Alternatives / cross-links
triton · aiter · ck · asm · hipblaslt · [overview.md](overview.md)

## Sources
- Matrix Core programming: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- CDNA4 ISA: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
