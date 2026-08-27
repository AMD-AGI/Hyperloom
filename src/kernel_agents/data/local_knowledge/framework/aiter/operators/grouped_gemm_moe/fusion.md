---
title: grouped_gemm_moe — fusion
kind: technique
operator: grouped_gemm_moe
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/fused_moe.py
  - https://github.com/ROCm/aiter
  - https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html
---

# grouped_gemm_moe — fusion

## TL;DR
> The big fused unit is the whole **fused_moe** op: sort/align → grouped gate/up GEMM → act-and-mul →
> grouped down GEMM → weighted combine, ideally as few launches as possible (AITER asm path fuses most of it).

## Fusion neighbors
- **act-and-mul (silu/gelu) between the two grouped GEMMs**: fuse as an epilogue of gate/up to avoid a
  round trip → act_and_mul_silu_gelu/overview.md.
- **Per-token/per-block quant epilogue**: quantize gate/up output to fp8/fp4 before the down GEMM, fused
  with the activation → fused_norm_quant/overview.md,
  scaled_quant_gemm/fusion.md.
- **Routing + align&sort upstream**: moe_routing_topk/overview.md,
  [../moe_dispatch_combine/overview.md](../moe_dispatch_combine/overview.md).
- **Weighted combine / scatter** as the down-GEMM epilogue (multiply by gate weight, scatter to token
  order) avoids a separate combine kernel.
- **Shared expert**: fuse the always-on shared expert with the routed experts →
  shared_expert_fusion/overview.md.
- **EP communication**: with expert parallelism, dispatch/combine all-to-all overlaps with the grouped
  GEMM → [../moe_dispatch_combine/overview.md](../moe_dispatch_combine/overview.md).

## What AITER fuses
- AITER asm fused_moe is reported best-on-AMD by fusing the grouped GEMMs + activation + combine; FlyDSL
  kernels handle mixed-precision (A4W4) MoE, falling back to CK when FlyDSL is absent.

## Sources
- AITER fused_moe / asm MoE: https://github.com/ROCm/aiter , https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html
