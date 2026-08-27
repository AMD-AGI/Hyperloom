---
title: layout_shuffle — fusion (shuffle+quantize at load; bpreshuffle GEMM/MoE)
kind: technique
operator: layout_shuffle
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1, int8]
regimes: [both]
updated: 2026-06-08
---

# layout_shuffle — fusion

## TL;DR
An offline "pay-once-where-it's-cold" weight transform: fuse it with load-time weight quantization (write the weight to its final fp8/fp4 MFMA layout once) and use it to unlock the runtime `bpreshuffle` GEMM/MoE kernels that load operands with no in-kernel reshuffle. Never shuffle in-kernel on the hot path.

The shuffle is an offline weight transform; its "fusion" is about (a) folding it with weight quantization at
load and (b) the runtime GEMM/MoE kernels it unlocks.

## Fusion targets
| pattern | how | link |
|---|---|---|
| **shuffle + weight quantize** (load-time) | quantize bf16→fp8/fp4 **and** shuffle into the MFMA layout in the same offline pass — the weight is written to its final on-device form once | [[operators/quant_dequant_fp8/overview.md]], [[operators/quant_fp4_mxfp/overview.md]] |
| **bpreshuffle → dense GEMM** | a shuffled `B` (`is_shuffled=True`) flips `bpreshuffle` in the 9-tuple key → the GEMM dispatches to the asm/CK/FlyDSL **bpreshuffle kernel** that loads operands with no in-kernel reshuffle | [[operators/dense_gemm/backends/aiter.md]], [[backends/aiter/tuned_gemm.md]] |
| **bpreshuffle → fused MoE** | FP4/FP8 expert weights pre-shuffled (`shuffle_weight_a16w4` + `shuffle_scale_a16w4`) feed the fast FP4 BpreShuffle MoE kernels (`f4gemm_*_BpreShuffle_*`) | [[backends/aiter/fmoe.md]], [[operators/fused_moe_grouped_gemm/overview.md]] |

## The trade (same as KV shuffle)
This is the **"pay once where it's cold"** pattern: the shuffle/quant runs **once at model load** (cold) so
**every GEMM** (hot, every forward) reads its operand fragment conflict-free and vectorized. Compare:
- [[operators/paged_kv_copy/fusion.md]] — pay layout cost on the KV *write* to make the attention *read* free.
- [[operators/transpose/fusion.md]] — a pre-shuffle is an **amortized one-time transpose** into MFMA order.

## Why not fuse it at runtime
You could shuffle in-kernel, but that re-pays the permute every call and risks LDS bank conflicts — the whole
point is to move it out of the hot path. Keep it offline/at-load.

## Cross-links
[[operators/layout_shuffle/tuning.md]] · [[operators/dense_gemm/backends/aiter.md]] ·
[[backends/aiter/tuned_gemm.md]] · [[backends/aiter/configs_db.md]] · [[backends/aiter/fmoe.md]] ·
[[operators/transpose/fusion.md]] · [[operators/paged_kv_copy/fusion.md]].
