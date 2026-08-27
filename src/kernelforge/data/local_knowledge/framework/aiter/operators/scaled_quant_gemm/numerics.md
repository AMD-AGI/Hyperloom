---
title: scaled_quant_gemm — numerics
kind: technique
operator: scaled_quant_gemm
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp4_e2m1, fp6_e2m3, fp6_e3m2]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
---

# scaled_quant_gemm — numerics

## TL;DR
> Accuracy is governed by the **block-scale scheme**: 32 elements share one E8M0 (exponent-only) scale,
> applied **after the dot product, before accumulation**. Get the application order and the E8M0 semantics
> right, then gate on a task metric — low-bit GEMM is the op most likely to silently regress quality.

## Considerations
- **Scale application order (CDNA4)**: scaled MFMA multiplies the fp32 partial dot by `sA*sB` before
  accumulating into the fp32 accumulator. Applying scales in the wrong order or pre-dot changes results.
- **E8M0 scale**: 8-bit exponent, no mantissa → scales are powers of two. The 32-element block must use the
  exact block the quantizer produced; misalignment corrupts a whole block.
- **dtype ranges**: fp8 e4m3 (more mantissa, less range) vs e5m2 (more range, less mantissa); fp6
  (e3m2/e2m3) gives extra mantissa at FP4 rate on CDNA4; fp4 e2m1 is most aggressive. Choose by where the
  tensor's dynamic range/precision sensitivity sits.
- **FNUZ (CDNA3) vs OCP (CDNA4)**: gfx942 fp8 is FNUZ (finite, no inf, single NaN encoding); gfx950 adds
  OCP fp8/MXFP. Reference quantizers must match the target encoding or parity fails.
- **Accumulate fp32** always; cast to out dtype at epilogue.

## Parity / accuracy gate
- Oracle = bf16 (or fp32) dense reference. Use a relative-error band (e.g. err_ratio threshold) per shape,
  AND a downstream task gate (PPL / MMLU delta within tolerance) — per-tensor MSE alone can hide systematic
  bias from low-bit quant. For grouped/MoE use the same gate per expert
  (grouped_gemm_moe/numerics.md).

## Sources
- Matrix Core blog (scale-after-dot, FNUZ vs OCP): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- CDNA4 ISA (scaled MFMA semantics, E8M0): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
