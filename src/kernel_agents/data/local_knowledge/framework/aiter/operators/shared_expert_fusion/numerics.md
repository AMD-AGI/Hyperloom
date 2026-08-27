---
title: shared_expert_fusion — numerics
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

# shared_expert_fusion — numerics & parity

## The design intent
Shared-expert fusion is **math-preserving by construction**: `y = routed + shared` regardless of whether
the shared MLP runs separately or fused. The two numeric subtleties are the **atomic-add reduction order**
and any **fp8 quant** of the shared expert.

## Atomic-add ordering
aiter's `fused_moe_dp_share_expert` **atomic-adds** the shared output into the routed result buffer (the
docstring: the no-shared MoE result is passed in and "it will atomic add to it"). Atomic-add over many
blocks is **non-deterministic in order** → tiny bf16 rounding differences vs a sequential `routed + shared`
add. This is benign (numerical-equivalence-class) but means **byte parity will not hold** — gate with
greedy/temp=0 over ≥10 prompts.

## Weight = 1 for the shared expert
The shared expert has **no routing weight** (or weight 1). A bug when injecting it as a synthetic routed
expert is to apply a routed weight to it — verify the shared slot carries weight 1 and is **not** renormed
together with the routed weights (DeepSeek renorms only the routed top-k).

## fp8 / quant interaction
If the shared expert is fp8 block-scale (to match the routed path's dequant epilogue), it adds quant error
on the always-on dense MLP — a quant gate. fnuz on gfx942 (wrong dialect = 2× off). Keeping the shared
expert bf16 while routed is fp8 is valid but loses the shared dequant fusion; decide per accuracy budget.

## Residual / order
The shared expert output is added to the routed sum **before** the layer residual add. Preserve that order;
adding shared after the residual (or double-counting it) is a real regression. Under EP, the shared
contribution must land in the **combine** at the right token positions (it's a synthetic expert) — verify
the inverse map places it correctly.

## Verification recipe
1. Isolated: `fused_moe_dp_share_expert(x, routed_result=0)` shared-only output vs a torch dense MLP
   reference within tolerance.
2. Full MoE layer: fused vs (separate shared Linear + routed `fused_moe`) — small bf16 deltas only
   (atomic-add order), no systematic divergence.
3. e2e DeepSeek greedy parity + a small eval if the shared expert is fp8.

## Sources
- atomic-add into routed result, token-range, weight handling: `ROCm/aiter@b467ce342:aiter/fused_moe_dp_shared_expert.py`.
- math-preserving shared-as-routed fusion: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
- fnuz fp8 / quant gate: fused_moe_grouped_gemm numerics, scaled_quant_gemm.
