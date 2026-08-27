---
title: grouped_gemm_moe — tuning
kind: technique
operator: grouped_gemm_moe
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-06-09
---

# grouped_gemm_moe — tuning

## TL;DR
> The dominant knob is **tile-M vs the per-expert M distribution**: pick a tile that minimizes padding waste
> on small groups while keeping MFMA throughput on large groups; then balance tiles across CUs so no expert
> serializes.

## The levers
- **BLOCK_M / BLOCK_N / BLOCK_K**: small BLOCK_M (16/32/64) cuts padding waste for decode's tiny groups;
  larger BLOCK_M (128/256) for prefill compute-bound groups. BLOCK_K typically 64/128 to feed MFMA.
- **MFMA shape**: `mfma_16x16` favors small-M / skinny groups (less wasted lanes); `mfma_32x32` higher
  throughput for large groups. See [../skinny_gemv_decode/overview.md](../skinny_gemv_decode/overview.md)
  for the same tradeoff in the M=1..8 limit.
- **MoE align block size**: tokens are padded so each expert's M is a multiple of BLOCK_M (align&sort). Too
  large → padding waste; too small → poor MFMA utilization. Tune jointly with BLOCK_M.
- **Tile balancing across CUs**: assign expert-tiles round-robin across 304 CUs (MI300X) / 256 CUs (MI350X)
  so a hot expert doesn't bottleneck; this is the grouped-GEMM analog of stream-K
  (→ [../splitk_streamk_gemm/overview.md](../splitk_streamk_gemm/overview.md)).
- **waves_per_eu / num_stages / num_warps** (triton): standard occupancy/pipeline knobs.
- **Single fused launch**: process all experts in one kernel (Primus-Turbo fused CK grouped GEMM) to kill
  per-expert launch + scheduling overhead.

## Quant-path levers (the dominant win, version-tagged)
- **fp8 block-scale** is the MI300X workhorse: aiter fused MoE is **up to 3× on DeepSeek-V3** vs an unfused
  stack (AMD). The align&sort redesign already collapsed the **sort step 10×** — don't re-tune the sort.
- **A4W4 / MXFP4** is the MI355X (CDNA4) lever: the FlyDSL path cuts latency **1.6× at concurrency 512**;
  MXFP4 GEMMs are ≈**62%** of Llama2-70B e2e, so low-bit weights dominate the budget. Kimi-K2.5 on MI300X
  with FlyDSL + SGLang/AITER: **−65% TTFT, −69% TPOT, +162% throughput** (vendor).
- **MoRI** fuses the cross-GPU EP dispatch/combine into the FusedMoE kernel (see
  [../../../../framework/aiter/operators/moe_dispatch_combine/backends/mori.md](../../../../framework/aiter/operators/moe_dispatch_combine/backends/mori.md)).

## aiter selection
- aiter dispatches the grouped path through the same per-shape tuned mechanism as dense GEMM; the 9-tuple
  key `(cu_num, padded_M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)` selects an impl, and small
  `padded_M` per group can route to the skinny path. See
  [../dense_gemm/backends/aiter.md](../dense_gemm/backends/aiter.md) for the capture/tune/deploy recipe.

## Pitfalls
- Tuning with synthetic uniform M while real routing is skewed → tiles tuned for the wrong M distribution.
  Capture real `topk_ids`/`expert_ids` and tune against the live distribution.
- Empty experts still cost a launch slot in multi-launch designs — prefer fused single-launch.

## Verify
- Profile per-grouped-GEMM time vs padding ratio; confirm tile count ≈ CU count multiples and no single
  expert dominates the timeline.
