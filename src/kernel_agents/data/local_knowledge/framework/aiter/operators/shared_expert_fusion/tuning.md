---
title: shared_expert_fusion — tuning
kind: technique
operator: shared_expert_fusion
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/fused_moe_dp_shared_expert.py
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
---

# shared_expert_fusion — tuning

## What you actually tune
Whether to fuse at all (flag), **where** the shared compute overlaps the routed compute, the shared
expert's own GEMM tiling, and (under EP) how the shared expert is injected into routing/dispatch.

## The two fusion strategies
1. **Fuse into the grouped GEMM + atomic-add** (aiter `fused_moe_dp_share_expert`): the shared expert runs
   as stage-1/stage-2 over the full token range and **atomic-adds** into the routed MoE result buffer
   (`a16 ... result here, it will atomic add to it`). Uses `get_dp_shared_expert_stage1/stage2_moe_sorting_
   result` to build a sorting layout for the shared (dense, all-token) part. **Knobs**: `block_size_M`
   (shared M is all tokens → big tile, fill 304 CUs), `get_padded_M` bucketing, DP token-range split
   (`get_dp_shared_expert_token_range(token_num, dp_size, rank)`).
2. **Synthetic routed expert (EP)**: inject the shared expert as an expert every token selects (weight 1)
   via `grouped_topk` so it rides the same **dispatch** and grouped GEMM. **Knob**: which top-k slot the
   shared expert occupies; preserves a single fused dispatch for shared+routed.

## Overlap scheduling (the perf lever)
The shared expert is **always on** (dense), the routed experts are **sparse** — they can overlap:
- run the shared dense GEMM on a **separate stream / HW queue** (`GPU_MAX_HW_QUEUES=2`) concurrently with
  the routed grouped GEMM, then atomic-add — hides the shared cost under the routed compute.
- or interleave in one kernel (aiter fused path) so the MFMA units stay busy across both.
- HIP-graph-capture the whole thing on decode.

## DP / EP token-range split
Under data parallelism the shared expert is computed per-DP-rank over its **token slice**
(`get_dp_shared_expert_token_range`), avoiding redundant shared compute across ranks. Tune `dp_size` and
ensure the slice boundaries align with `block_size_M`.

## fp8 / quant
The shared expert can be fp8 block-scale like the routed experts (same `quant_type`); keep its quant
consistent with the routed path so they share the dequant epilogue. fnuz on gfx942.

## Pitfalls in tuning
- **`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` is incompatible with MoRI** — under MoRI Wide-EP the fusion
  is done MoRI-side (synthetic routed expert), not via the vLLM flag. Don't set both.
- Atomic-add contention: many blocks atomic-adding into the same result buffer can serialize — ensure the
  routed and shared writes target disjoint token ranges where possible, or accept the atomic.
- Shared M is large (all tokens) → don't tile it like a skinny per-expert GEMM; treat it as a dense GEMM.

## How to verify a tuning win
- Isolated: `fused_moe_dp_share_expert` vs (separate shared Linear + routed `fused_moe`) timing.
- rocprof: confirm shared and routed kernels **overlap** (separate queues) and the atomic-add isn't a
  serialization bottleneck.
- e2e DeepSeek tok/s with the fusion on/off; greedy parity.

## Sources
- aiter shared-expert fusion internals (token-range, stage1/2 sorting, atomic-add, padded M): `ROCm/aiter@b467ce342:aiter/fused_moe_dp_shared_expert.py`.
- shared-as-routed + MoRI incompatibility: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
- multi-queue overlap (`GPU_MAX_HW_QUEUES`): allreduce / RCCL tuning, MI300X workload guide.
