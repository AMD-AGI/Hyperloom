---
title: sparse_attention_nsa — overview
kind: operator_overview
operator: sparse_attention_nsa
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://github.com/fla-org/native-sparse-attention
  - https://arxiv.org/abs/2502.11089
  - https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
  - https://arxiv.org/html/2508.18224v1
---

# sparse_attention_nsa  (Native Sparse Attention / block-sparse attention)

## TL;DR
NSA (Native Sparse Attention, DeepSeek, arXiv 2502.11089) replaces full attention with **three parallel
branches per query** — **compressed** (coarse pooled KV), **selected** (top-k *blocks* chosen by a
learned/derived importance score), and **sliding-window** (local) — gated and summed. The kernel win is
that **selected** attention only reads the top-k KV blocks, making prefill O(seq·k·block) instead of
O(seq²). On AMD this is **primarily Triton-portable** (the reference `fla-org/native-sparse-attention`
is CUDA-targeted but Triton runs on MI300X via triton_amd); production DeepSeek-style sparse MLA on
MI300X uses **aiter Triton kernels** (`unified_attention_sparse_mla`, `fp8_mqa_logits`) with some paths
**falling back to generic Triton on gfx942** because the tuned/asm path is gfx950-only or broken on CDNA3.
The single most important fact: the cost is dominated by the **indexer/top-k selection + gather**, not the
attention matmul.

## Math contract
Per query `q_i`, output is a gated combination of three branches:
```
o_i = g_cmp·Attn(q_i, K_cmp, V_cmp)        # compressed: pool KV into block summaries
    + g_slc·Attn(q_i, K[S_i], V[S_i])      # selected: S_i = top-k KV blocks by importance score
    + g_swa·Attn(q_i, K_win, V_win)        # sliding window: last W tokens (see sliding_window_attention)
```
- **Block size** (reference): `block_size=64`, `window_size=64`, top-k selected blocks per query; the
  number of selected blocks can **vary per query/position/batch** (`block_indices`, `block_counts`).
- **Selection score** = compressed-branch attention logits reused (NSA "natively trainable"), or a
  separate **lightning indexer / fp8 MQA logits** (DeepSeek-V3.2/V4 sparse MLA).
- Online softmax inside each branch, fp32 accumulate; gates `g_*` from a small projection.

## Shape regimes
- **Prefill**: long seq (32K–128K+) — where sparsity pays. Selected branch reads `k·block_size` keys per
  query instead of all. The hard kernel problems (Flash Sparse Attention, arXiv 2508.18224) are
  **non-contiguous gather** of selected query/KV tokens and online-softmax over a ragged block set.
- **Decode**: `sq=1`, paged KV; the indexer scores all (or compressed) KV, picks top-k blocks, then a
  paged gather + attention over those blocks. The **paged MQA logits** kernel is the indexer hot path.

## Where it matters (Amdahl)
On DeepSeek-V3.2/V4-Flash-class models the sparse MLA + MoE layers are the **most expensive** layers
(MI300X bring-up profiling); making the indexer/gather efficient is what moves e2e. The reference
bring-up moved +8.6% (2485→2699 tok/s/GPU) by fixing the *bookkeeping* around the sparse matmuls (static
capture-safe metadata), not the matmul itself — confirming the selection/gather is the lever.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢 (portable SOTA on AMD; aiter sparse-MLA + FLA/FSA reference) | [triton.md](triton.md) |
| hip | 🟡 (paged MQA-logits / gather helpers; gfx942 fallback glue) | hip |
| tilelang | 🧪 (CK `50_sparse_attn` jenga/VSA example; CDNA3-validated, not a tuned NSA path) | tilelang |
| ck | 🧪 (`example/ck_tile/50_sparse_attn`) — no dedicated card; via tilelang/hip notes | — |
| aiter | 🟢 (dispatcher; `unified_attention_sparse_mla`, `pa_mqa_logits`) | see triton card + aiter |

## Fusion neighbors
Compressed-KV pooling, the **indexer** (fp8 MQA logits), top-k block selection (argmax_topk),
paged-KV gather (gather_scatter / paged_kv_copy), RoPE, fp8 KV quant. The sliding-window branch
is sliding_window_attention. See [fusion.md](fusion.md).

## Numerics
Three-branch gated sum + top-k selection introduces **discrete selection** (argmax-style tie risk) on
top of softmax — see [numerics.md](numerics.md). fp8 indexer on gfx942 = fnuz dialect.

## How to bench
Reference (FLA): `B,T,H,HQ,D = 4,2048,4,64,64`, `block_size=64`, `window_size=64`, top-k. Bench the
selected-branch kernel + indexer in isolation; oracle = dense attention masked to the same selected
blocks. See [tuning.md](tuning.md).

## Sources
- NSA paper (3 branches, hardware-aligned, trainable): https://arxiv.org/abs/2502.11089
- FLA NSA reference (`parallel_nsa`, online top-k, block_size=64, CUDA-targeted): https://github.com/fla-org/native-sparse-attention
- Flash Sparse Attention (kernel challenges: non-contiguous gather, decoupled softmax): https://arxiv.org/html/2508.18224v1
- DeepSeek-V4-Flash MI300X bring-up (sparse MLA, indexer, gfx942 Triton fallback, +8.6%): https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
