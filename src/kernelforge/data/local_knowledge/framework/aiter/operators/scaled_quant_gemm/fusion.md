---
title: scaled_quant_gemm — fusion
kind: technique
operator: scaled_quant_gemm
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp4_e2m1, fp6_e2m3]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://github.com/ROCm/aiter
---

# scaled_quant_gemm — fusion

## TL;DR
> The high-value fusions are at the **quant boundaries**: fuse input quantization (norm→quant) before the
> GEMM and output quantization after it, so activations never materialize in high precision between ops.
> The scale pipeline itself is "fused" by staging scales through LDS alongside the tile loads.

## Fusion neighbors
- **Upstream norm+quant**: rmsnorm/layernorm producing fp8/fp4 + per-block scales directly into the GEMM
  input → fused_norm_quant/overview.md,
  quant_fp4_mxfp/overview.md,
  quant_dequant_fp8/overview.md.
- **Epilogue bias/activation + output quant**: dequant accumulator, add bias, activation, re-quantize
  output to fp8/fp4 with new block scales → gemm_epilogue_fused/overview.md.
- **MoE**: scaled grouped GEMM with per-expert/per-block scales →
  grouped_gemm_moe/fusion.md.
- **KV-cache quant** shares the fp8 quant machinery → kv_cache_quant/overview.md.

## Scale-pipeline note
- Scales must be staged Global→LDS(re-layout)→LDS-read to feed scaled MFMA; there is no register-direct
  scale feed (Gluon tutorial). Treat this staging as part of the kernel, pipelined with A/B loads.

## Sources
- Gluon GEMM tutorial (scale pipeline): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- AITER fused quant GEMM paths: https://github.com/ROCm/aiter
