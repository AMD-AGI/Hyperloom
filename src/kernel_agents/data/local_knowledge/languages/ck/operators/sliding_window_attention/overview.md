---
title: sliding_window_attention — overview
kind: operator_overview
operator: sliding_window_attention
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://github.com/Dao-AILab/flash-attention
  - https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# sliding_window_attention  (SWA / local attention)

## TL;DR
Flash-attention restricted to a **local band**: each query `i` attends only to keys in
`[i - window_left, i + window_right]` instead of the full causal prefix. It is the same online-softmax
FMHA kernel as attention_prefill_fmha / attention_decode_paged plus a **band mask** that lets the
kernel **skip whole KV blocks** outside the window — so for long sequences SWA is O(seq·window) not
O(seq²). On MI300X the **safe SWA path today is the CK backend** (`VLLM_USE_TRITON_FLASH_ATTN=0`); the
Triton FA backend's SWA was historically WIP, though aiter/sglang Triton attention now carry a
`sliding_window` argument. The single most important fact: SWA's win is **KV-block skipping**, so the
mask must prune blocks (not just zero scores) or you pay full-attention cost.

## Math contract
`O = softmax( (Q·Kᵀ)·scale + mask_swa )·V`, with
`mask_swa[i,j] = 0 if (i-j) ≤ window_left AND (j-i) ≤ window_right else -inf`.
- FA-style API: `window_size = (left, right)`; `(-1,-1)` = full attention; causal SWA = `(W-1, 0)`.
- CK-Tile mask vocabulary (`example/ck_tile/01_fmha/mask.hpp`): `mask_top_left`, `mask_bottom_right`,
  `window_generic` with explicit `left, right` (FA-style), plus an **attention-sink** size (`sink_size`)
  for StreamingLLM-style "keep first N + last W".
- Optional **logit soft-cap** and **attention sink** tokens fuse into the same mask pass (Gemma-2/3,
  Mistral, Qwen2/3, Phi-3, GPT-OSS use SWA layers).
- dtype bf16/fp16 in, fp32 online-softmax accumulate, bf16/fp16 out; fp8 variants via scaled inputs.

## Shape regimes
- **Prefill**: long `sq=sk` (e.g. 8K–128K). The whole point — block-skipping makes the kernel grid scale
  with `window`, not `seq`. Window 1024–4096 is typical.
- **Decode**: `sq=1`, paged KV, but the KV scan is **truncated to the last `window` tokens** → the
  paged-attention loop only reads the in-window pages (KV-cache eviction can drop older pages entirely).
- Hybrid models interleave SWA layers with full-attention layers (Gemma-2 1:1; Qwen "every N").

## Where it matters (Amdahl)
On a pure-SWA or SWA-heavy model at long context, the local kernel can be the dominant attention cost,
and block-skipping is the only thing that keeps TTFT/throughput linear in seq. On hybrid models the SWA
layers are cheap per-layer but numerous, so a correct (not silently-full) SWA kernel matters for the
long-context regime where MI300X's 192 GB HBM is the selling point. Note the known ROCm long-seq FA gap
(~20–25% vs CUDA at 32K+, vLLM blog) — SWA mitigates it by shrinking the KV scan.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| ck (ck_tile) | 🟢 sota (the reliable SWA path on ROCm) | [ck.md](ck.md) |
| fa_rocm | 🟢 (CK default; Triton SWA maturing) | fa_rocm |
| aiter | 🟡 (Triton FA + paged decode carry `sliding_window`) | aiter |
| triton | 🟡 (editable; band-mask + block-skip; upstream SWA WIP) | triton |
| hip / asm / tilelang | ⚪/🧪 (no dedicated SWA card; portable via the above) | — |

## Fusion neighbors
RoPE pre-step, qk-norm, logit soft-cap, attention-sink tokens, fp8 KV-cache quant
(kv_cache_quant). See [fusion.md](fusion.md).

## Numerics
Band mask + online softmax; same fp32-accumulate parity rules as full FMHA. The risk is **correctness,
not precision**: a wrong window edge or an off-by-one sink silently changes which tokens are attended.
See [numerics.md](numerics.md).

## How to bench
Reference: b=8, h=32, sq=sk=8192, d=128, window=1024, causal. Bench CK vs Triton at the same shape;
oracle = full-attention with an explicit Python band mask, greedy temp=0 parity ≥10 prompts. See
[tuning.md](tuning.md).

## Sources
- FA window_size API + ROCm two-backend (CK default, head_dim≤256; Triton SWA WIP): https://github.com/Dao-AILab/flash-attention
- vLLM ROCm: CK for SWA (`VLLM_USE_TRITON_FLASH_ATTN=0`), long-seq gap, 7 backends: https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
- CK-Tile mask vocabulary (window_generic, sink): `ROCm/composable_kernel:example/ck_tile/01_fmha/mask.hpp` (on-box via aiter 3rdparty).
- MI300X attention tuning: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
