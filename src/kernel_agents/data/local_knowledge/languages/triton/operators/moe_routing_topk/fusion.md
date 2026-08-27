---
title: moe_routing_topk — fusion
kind: technique
operator: moe_routing_topk
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/topk.py
  - https://github.com/vllm-project/vllm/pull/17955
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
---

# moe_routing_topk — fusion

The router is a chain of tiny kernels; almost every win is **collapsing launches** and **pushing the
routed-weight multiply downstream**.

## Fusions that exist on AMD today
| fusion | what it merges | where | payoff |
|---|---|---|---|
| **gate + bias + grouped top-k** | softmax/sigmoid + correction_bias + group-select + within-group top-k + renorm/scale | `moe_fused_gate` / `biased_grouped_topk` (one HIP/DPP kernel) | 1 launch instead of 4–5; the DPP path is 1.66× over CK |
| **routed-weight multiply → grouped-GEMM epilogue** | the `topk_weights` multiply folded into stage-1 (`doweight_stage1`) or stage-2 (`MulRoutedWeight1`) of fused MoE | aiter fused_moe / CK `moe_ck2stages_*` | removes a separate weighted-combine pass |
| **shared-expert as synthetic routed experts** | shared MLP injected into the top-k slots so it dispatches with routed experts | Wide-EP / shared_expert_fusion | one fused dispatch for shared+routed; no separate Linear+add |
| **softmax routing + shared-expert sigmoid in one launch** | when `topk_softmax` supports sigmoid fusion, the routing softmax and the shared-expert sigmoid gate share a kernel | vLLM AITER fused-MoE V1 (PR #17955) | 1 launch; else a fallback injects shared weights into the AITER buffer |
| **align&sort fused with histogram** | per-expert count + padded sorted permutation in one multi-block kernel | `moe_align_block_size` | the 7× MI300X win |

## The hand-off seam (router → next stage)
The router's real product for fusion is the **align&sort output**:
`(sorted_token_ids, expert_ids, num_tokens_post_pad)`. Two consumers:
- **single-GPU**: feeds fused_moe_grouped_gemm directly (contiguous per-expert tiles).
- **expert-parallel**: feeds moe_dispatch_combine — MoRI-EP `dispatch(input, indices)` takes the
  int32 `topk_ids` directly; the per-token routing weight is then **multiplied during combine**, not in
  the router. So under EP the router should emit weights and **stop** — don't pre-multiply.

## What is NOT yet fused on AMD
- **Router fully fused into the dispatch kernel** (gate→select→permute→send in one GPU-initiated kernel)
  is the north-star (cf. single-kernel a2a studies) but not shipped; the router and dispatch are separate
  launches today.
- **Gate-GEMM (router linear `[H,E]`) + softmax+topk** is usually two kernels — the router GEMM is a tiny
  dense GEMM (dense_gemm) and the gate is a separate elementwise+reduce; fusing them is rarely worth it
  because the GEMM is bandwidth-bound on the `[H,E]` weight.

## Decode-path discipline
On decode, fuse aggressively and **HIP-graph-capture the whole router** (gate+topk+align&sort) so the
3–5 launches become one graph replay; hoist all `torch.empty`/memset out (see [tuning.md](tuning.md)).

## Cross-links
shared_expert_fusion · moe_dispatch_combine · fused_moe_grouped_gemm ·
aiter · `backends/aiter/fmoe.md`.

## Sources
- aiter fused gate / biased_grouped_topk: `ROCm/aiter@a6bb49937:aiter/ops/topk.py`, `csrc/kernels/moe_fused_gate.cu`.
- softmax routing + shared-expert sigmoid single launch: https://github.com/vllm-project/vllm/pull/17955
- shared-expert-as-routed + prob-mult-in-combine (Wide-EP): https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
