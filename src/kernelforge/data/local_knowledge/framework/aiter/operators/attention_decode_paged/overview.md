---
title: attention_decode_paged — overview
kind: operator_overview
operator: attention_decode_paged
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/paged_attn.py
  - https://github.com/vllm-project/vllm/tree/main/csrc/rocm
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - https://arxiv.org/abs/2205.14135
---

# attention_decode_paged  (paged-KV decode attention / flash-decoding)

## TL;DR
The single-token-per-step attention of the autoregressive decode phase: `sq=1` query attends over a
**paged KV-cache** of all prior tokens. It is **bandwidth- and latency-bound** (no big GEMM — it streams
K/V from HBM), so the dominant trick is **splitKV / flash-decoding**: split the long KV history across
CUs and reduce the partials, to fill 304 CUs when batch is small. On AMD serving the default is the
**aiter / vLLM-custom-HIP paged-attn** kernel; the lever is the **partition (split) size** and the
**KV-cache layout/dtype**.

## Math contract
`O = softmax(q·Kᵀ·scale + mask)·V`, with `q[b,h,1,d]` and K/V gathered through a **block/page table**
(`block_tables[b]` → physical block ids; `seq_lens[b]` → valid length). KV-cache stored in pages of
`block_size` tokens. dtype: q/K/V bf16/fp16 (or fp8 KV-cache), **fp32 online-softmax accumulate**, bf16
out. No causal mask needed within a step (the query is the newest token) — only the valid-length mask.

## KV-cache layout (the load-bearing detail)
- **NHD** = `[num_blocks, block_size, num_kv_heads, head_dim]` (token-major within a page).
- **HND** = `[num_blocks, num_kv_heads, head_dim, block_size]` (head-major) — vLLM's ROCm custom path
  uses a reshaped `[2, num_blocks, block_size*num_kv_heads*head_size]` with an inner `x = 16/elt_size`
  split on K for coalesced 128-bit reads. Layout choice drives load coalescing; the wrong layout for the
  kernel = strided loads = bandwidth loss. `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` reshuffles for the
  AITER FA path at concurrency ≥32.
- **page_size / block_size**: 16 is common; larger pages → fewer block-table indirections but coarser
  memory. `_PARTITION_SIZE_ROCM = 256` in aiter's custom paged-attn (the split-KV chunk).

## Shape regimes
Decode: `sq=1`, `batch = running concurrency` (1..256+), KV length = context (1k..128k). GQA collapses
`num_kv_heads ≪ num_q_heads` (see ../gqa_mqa_attention/overview.md). Distinct from
../attention_prefill_fmha/overview.md (long sq, GEMM-bound).

## Where it matters (Amdahl)
Decode latency (TPOT) is what users feel in chat; paged-attn is a top-3 decode kernel alongside the
skinny GEMMs. AITER FA / AITER MLA backends report **1.2–4.4× higher TPS** vs generic FA on AMD — most
of that is the decode path. Small-batch decode is launch/latency-bound, so splitKV + kernel fusion
(persistent kernels) matter more than raw FLOPs.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (default; asm/CK) | [aiter.md](aiter.md) |
| hip (vLLM custom paged-attn) | 🟢 sota (editable, strong decode) | hip |
| vllm_kernels | 🟢 (ROCM_ATTN / ROCM_AITER_FA dispatch) | vllm_kernels |
| triton | 🟡 (universal fallback; flash-decoding) | triton |
| ck | 🟡 (CK-Tile paged-KV FMHA decode) | ck |
| fa_rocm | 🟡 (Triton-backend paged; AITER FA preferred) | fa_rocm |

## Fusion neighbors
KV-cache write fuses with RoPE+norm+quant (pre-decode); flash-decoding reduce kernel fuses the per-split
partials; persistent-kernel decode amortizes launch. See [fusion.md](fusion.md).

## Numerics
fp32 online-softmax accumulate; fp8 KV-cache is the common quant (FNUZ on gfx942) — accuracy-gate. See
[numerics.md](numerics.md).

## How to bench
Isolated decode bench: fix `(batch, num_q_heads, num_kv_heads, head_dim, context_len, block_size, dtype)`,
median ≥3 warm reps; e2e via `--attention-backend` swap at the served concurrency, TPOT + parity gate.

## Sources
- aiter paged-attn (partition 256, supported head sizes 64/80/96/112/120/128/192/256, gqa_ratio 1-32, max_seq_len ≤65536, KV layout split): on-box `ROCm/aiter@b467ce342:aiter/paged_attn.py`.
- vLLM custom HIP paged-attn (`paged_attention_ll4mi_*`), KV layout, shuffle: https://github.com/vllm-project/vllm/tree/main/csrc/rocm ; `backends/vllm_kernels/rocm_kernels.md`.
- 1.2–4.4× TPS (vendor): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- flash-decoding / online softmax: https://arxiv.org/abs/2205.14135
