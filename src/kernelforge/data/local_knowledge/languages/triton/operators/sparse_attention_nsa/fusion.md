---
title: sparse_attention_nsa — fusion
kind: technique
operator: sparse_attention_nsa
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://github.com/fla-org/native-sparse-attention
  - https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
---

# sparse_attention_nsa — fusion

## What fuses (and what deliberately doesn't)
| fusion | where | note |
|---|---|---|
| **selected + sliding-window branches** | one Triton kernel | FLA ships a fused kernel combining selected attention with the sliding branch (2025-02 release) |
| **compressed-KV pooling** | pre-step | block-mean/learned pooling into block summaries; feeds both the compressed branch and the selection score |
| **indexer / fp8 MQA logits** | the selection score | DeepSeek runs it in fp8; its output ranks blocks (top-k). Kept as its own kernel (it's the hot path) |
| **online top-k** | inside the indexer epilogue | avoids materializing the full score matrix (FLA) |
| **fnuz quantise-and-insert KV** | KV-cache write | fused quantise + paged insert so cache bytes match the fnuz dialect (V4 bring-up) |
| **RoPE / qk-norm** | before scoring | standard (rope, rmsnorm) |
| **gather of selected blocks** | the selected-branch load | indexed/masked `buffer_load`; the non-contiguous gather is the perf-sensitive seam (gather_scatter) |

## What FSA deliberately *splits* (anti-fusion)
Flash Sparse Attention found that **decoupling** is faster than one mega-kernel for the selected branch:
- online-softmax statistics → a **separate pre-compute kernel**;
- attention accumulation → **another kernel**;
- plus an **early-return** on empty index tensors.
This avoids loop-order stalls from non-contiguous query access — so for the selected branch, *less*
fusion can win on AMD. Benchmark both.

## Cross-links
- Sliding branch: sliding_window_attention · selection: argmax_topk · gather: gather_scatter /
  paged_kv_copy · KV quant: kv_cache_quant · MLA base: mla_attention.
- Languages: triton_amd (portable path), hip_cpp (gather/indexer helpers), tilelang.
- aiter dispatcher: aiter.

## Sources
- FLA fused selected+sliding, online top-k: https://github.com/fla-org/native-sparse-attention
- FSA decoupled softmax/accumulate + early-return: https://arxiv.org/html/2508.18224v1
- fnuz quantise-and-insert KV: https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
