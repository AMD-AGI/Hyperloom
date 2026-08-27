---
title: fused_moe_grouped_gemm — fusion
kind: technique
operator: fused_moe_grouped_gemm
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, int8, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/fused_moe.py
  - ROCm/aiter@b467ce342:aiter/configs/tuned_fmoe.csv
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
---

# fused_moe_grouped_gemm — fusion

The whole point of this operator is that it is **already a mega-fusion**: sort → grouped GEMM (gate+up) →
activation → grouped GEMM (down) → weighted combine in one `fused_moe` pipeline. The remaining levers are
*what else* you fold into the two GEMM stages.

## Fusions that exist on AMD today
| fusion | what it merges | where | payoff |
|---|---|---|---|
| **gate+up (g1u1)** | two stage-1 GEMMs into one | `use_g1u1`, kernel `..._g1u1_...` | one GEMM, shared X load |
| **SwiGLU act_and_mul → stage-1 epilogue** | `act(G)⊙U` fused after gate+up | stage-1 asm kernel | no separate activation pass over `[M,inter]` |
| **routed-weight multiply → GEMM epilogue** | `topk_weights` multiply in stage-1 (`doweight_stage1`) or stage-2 (`MulRoutedWeight1`) | DB-selected | removes a weighted-combine kernel |
| **fp8 quant/dequant → epilogue** | dequant scales applied in the GEMM epilogue (block-scale) | CK `moe_ck2stages_*` | no separate dequant pass; scaled_quant_gemm |
| **token sort fused with histogram** | align&sort builds the grouped layout in one kernel | `moe_sorting` | 7× align&sort (moe_routing_topk) |
| **shared-expert into the grouped GEMM** | shared MLP run as synthetic routed experts in the same kernel | `fused_moe_dp_shared_expert` (shared_expert_fusion) | no separate shared Linear + residual add |

## The two-stage kernel anatomy (real on-box names)
- stage-1 fp8: `_ZN5aiter48fmoe_stage1_bf16_pertokenFp8_g1u1_64x128_2tg_pf3E` — hand-tuned **asm**, g1u1.
- stage-2 fp8: `moe_ck2stages_gemm2_256x64x128x256_1x4_MulABScaleExpertWeight_v3_Nswizzle0_Quant2_
  MulRoutedWeight1_F8_F8_B16` — **CK 2-stage**, `MulRoutedWeight1` folds the router weight, `Quant2`
  applies scales, output bf16.
So stage-1 = asm, stage-2 = CK, and the activation/weight/quant are **already** epilogue-fused.

## The dispatch ↔ GEMM seam (EP)
Under EP the grouped GEMM consumes MoRI-EP's **3D packed layout** (`packed_recv_x`, `packed_recv_count`,
…) directly — no re-materialization. This is the moe_dispatch_combine hand-off
(`ENABLE_STANDARD_MOE_ADAPT=ON`, `MoriAll2AllManager`).

## What is NOT yet fused on AMD
- **combine reduction → down-proj epilogue** (single-kernel MoE, FlashDMoE-style) — the north-star; MoRI-EP
  zero-copy buffers approach it but no single fused combine+GEMM kernel ships on AMD (moe_dispatch_combine).
- **stage-1 → stage-2 without a global round-trip of the intermediate** — the `[M,inter]` activation is
  materialized between stages (it's large); keeping it in LDS/registers across both GEMMs is not done.

## Cross-links
moe_routing_topk (feeds sort) · moe_dispatch_combine (EP feed) · shared_expert_fusion ·
scaled_quant_gemm · grouped_gemm_moe · `backends/aiter/fmoe.md`.

## Sources
- on-box kernel names + fusion flags: `ROCm/aiter@b467ce342:aiter/fused_moe.py`, `aiter/configs/tuned_fmoe.csv`.
- shared-expert fusion + EP co-design: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
