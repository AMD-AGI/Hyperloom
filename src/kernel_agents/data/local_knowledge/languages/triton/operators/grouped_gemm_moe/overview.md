---
title: grouped_gemm_moe — overview
kind: operator_overview
operator: grouped_gemm_moe
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - https://github.com/ROCm/aiter
  - https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html
---

# grouped_gemm_moe

## TL;DR
> A batch of independent GEMMs with variable per-group M (one GEMM per MoE expert), executed in a single
> fused kernel launch keyed by a routing/offset table — the single biggest win is **doing all experts in
> one launch over a sorted/aligned token layout** instead of one launch per expert.

## Math contract
- For each expert `e` in `[0..E)`: `Y[rows_e] = X[rows_e] @ W_e (+ bias_e)`, where `rows_e` are the
  tokens routed to `e`. Equivalent to a block-diagonal GEMM with per-group `M_e` and shared `N,K`.
- MoE FFN uses two grouped GEMMs: gate/up `(K=hidden, N=2*inter)` then down `(K=inter, N=hidden)`,
  with an act-and-mul (silu/gelu) between → see [fusion.md](fusion.md) and
  act_and_mul_silu_gelu/overview.md.
- Inputs are pre-sorted by expert and padded so each group's M aligns to the tile (MoE align & sort);
  an `expert_ids`/`num_tokens_post_pad` offset table drives tile→group mapping.
- dtype: bf16/fp16 weights, or fp8/fp4 block-scaled (→ scaled_quant_gemm/overview.md).
  Accumulate fp32.

## Shape regimes
- **Prefill**: large total token count → most groups have `M_e ≫ tile`; behaves like many dense GEMMs,
  compute-bound, tile-efficiency dominated.
- **Decode**: few tokens, top-k routing → many groups with tiny `M_e` (1..few tiles), memory-bound and
  highly imbalanced; tile balancing and avoiding empty-tile waste dominate.

## Where it matters (Amdahl)
- MoE FFN grouped GEMMs are the dominant GPU cost in MoE models (e.g. DeepSeek/Qwen-MoE). AITER's asm
  fused_moe path is reported as the best on AMD, with ~3x fused-MoE acceleration attributed to the
  grouped-GEMM-of-different-shapes structure (Bruce Xu, AITER/SGLang integration blog) — so moving this
  op moves e2e materially.

## Backend landscape (link table → SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟡 competitive | [triton.md](triton.md) |
| ck | 🟡 competitive | ck |
| hip | 🟡 competitive | hip |
| aiter | 🟢 sota | aiter |
| tilelang | 🧪 experimental | tilelang |
| flydsl | 🟢 sota (2-stage gate+up / down GEMM, fp8/bf16, JIT) | flydsl |

## Fusion neighbors
- Upstream: moe_routing_topk/overview.md (topk + sort/align),
  [../../../../framework/aiter/operators/moe_dispatch_combine/overview.md](../../../../framework/aiter/operators/moe_dispatch_combine/overview.md).
- Inline: act-and-mul, per-token/per-block quant epilogue → [fusion.md](fusion.md).
- Downstream: weighted combine/scatter back to token order. See also
  shared_expert_fusion/overview.md.

## Numerics
- fp8/fp4 expert weights with per-group scales; routing weight multiply in fp32 → [numerics.md](numerics.md).

## How to bench
- Isolate the two grouped GEMMs with a fixed routing table (real captured `topk_ids`), median of ≥3 warm
  reps; oracle = per-expert dense reference `X_e @ W_e` gathered back. Compare against
  dense_gemm/overview.md summed over groups for a sanity ceiling.

## Sources
- AITER repo + DeepSeek-R1/SGLang integration blog (grouped GEMM = MoE win): https://github.com/ROCm/aiter ,
  https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html
- MoE align & sort design: https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
