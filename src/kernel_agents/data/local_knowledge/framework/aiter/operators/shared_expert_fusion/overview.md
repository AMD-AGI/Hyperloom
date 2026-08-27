---
title: shared_expert_fusion — overview
kind: operator_overview
operator: shared_expert_fusion
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/fused_moe_dp_shared_expert.py
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
---

# shared_expert_fusion  (shared-expert + routed-expert overlap/fusion)

## TL;DR
DeepSeek-style MoE (V2/V3/R1) runs a **shared expert** (a dense MLP) for **every** token alongside the
routed experts. Naively this is a separate Linear + activation + residual add per layer. Shared-expert
fusion **folds the shared MLP into the fused-MoE grouped GEMM** so shared + routed experts compute in one
pipeline (and, under EP, one dispatch). On AMD this is `aiter.fused_moe_dp_share_expert` (single-GPU/DP)
and the Wide-EP "shared-as-synthetic-routed-expert" trick (EP). The single most important fact: it
**removes a separate dense MLP pass per layer** and lets the always-on shared compute **overlap** the
sparse routed compute — preserving numerics (atomic-add into the routed result).

## Math contract
For each token `t`: `y_t = Σ_{e∈topk(t)} g_{t,e}·Expert_e(x_t)  +  SharedExpert(x_t)`.
- **SharedExpert** is a dense SwiGLU MLP run for all tokens (no routing weight, weight=1).
- Fusion: treat the shared expert as an **extra expert that every token selects** (synthetic top-k slot,
  weight 1) so it rides the same grouped GEMM + sort; or compute it in the same kernel and **atomic-add**
  its output into the routed accumulator (aiter `fused_moe_dp_share_expert` — the shared result is added to
  the no-shared MoE result buffer). dtype/acc identical to fused_moe_grouped_gemm.

## Shape regimes
- Shared expert M = **all tokens** (dense), routed expert M = sorted/padded per-expert tiles (sparse). So
  the shared MLP is a **large dense GEMM** (dense_gemm) while the routed part is grouped — fusion must
  schedule both efficiently on 304 CUs.
- DeepSeek-V3: H=7168, shared inter ~2048, E=256 routed + 1 shared, top-8.

## Where it matters (Amdahl)
On DeepSeek-family models the shared expert runs **every layer for every token** — it's a non-trivial fixed
cost. Folding it into the fused MoE removes a per-layer Linear+add and overlaps it with the sparse routed
compute; combined with EP it's part of AMD's Wide-EP path that hit **32.3k in / 12.4k out tok/s per node**
(32× MI300X). The win is largest when the shared expert is a meaningful fraction of the MoE-layer FLOPs.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (`fused_moe_dp_share_expert`, flag-gated) | [aiter.md](aiter.md) |
| hip | 🟢 (the fused stage-1/stage-2 + atomic-add kernels) | hip |
| triton | 🟡 (separate shared MLP + add; overlap via streams; editable) | triton |

## Fusion neighbors
- **shared → grouped GEMM** (this operator) + **shared → dispatch** (EP, synthetic routed expert,
  moe_dispatch_combine) + **shared routing into the gate** (moe_routing_topk fusion). Builds on
  fused_moe_grouped_gemm. See [fusion.md](fusion.md).

## Numerics
Designed to **preserve** the unfused math (shared output atomic-added to the routed sum). The atomic-add
reduction order differs from a sequential add → small bf16 deltas; fp8 shared expert is a quant gate. See
[numerics.md](numerics.md).

## How to bench
aiter `fused_moe_dp_share_expert` vs unfused (separate shared Linear + fused routed MoE) at the model's
(H, inter, E, topk, tokens); e2e DeepSeek tok/s with the fusion flag on/off. Oracle = greedy parity vs the
unfused stack.

## Sources
- aiter shared-expert fusion (`fused_moe_dp_share_expert`, atomic-add into result): `ROCm/aiter@b467ce342:aiter/fused_moe_dp_shared_expert.py`.
- Wide-EP shared-as-routed fusion + 32-GPU numbers: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
- aiter fused MoE / shared-expert: https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
