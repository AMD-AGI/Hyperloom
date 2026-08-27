---
title: gqa_mqa_attention — fusion
kind: operator_overview
operator: gqa_mqa_attention
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/paged_attn.py
  - https://github.com/ROCm/aiter
  - https://github.com/Dao-AILab/flash-attention
---

# gqa_mqa_attention — fusion

GQA/MQA has the same fusion neighbors as MHA — it is a KV-broadcast trait, not a different op. The one
GQA-specific point is that the broadcast itself is an **intra-kernel fusion** (the shared KV head feeds R
query heads in one pass), and it must compose with the pre-attention KV-write fusion sized for
`num_kv_heads`.

## The broadcast is intra-kernel (not a separate op)
The right kernels read each KV head once and reuse it across its R query heads inside the `q·Kᵀ`/`P·V`
loop — there is no "expand KV" op to fuse or eliminate, because a correct GQA kernel never materializes
the expanded KV. The anti-pattern is a *separate* `repeat_kv` op before attention; the fusion is simply
**not doing that** (broadcast in-register).

## Pre-attention KV-write must be sized for num_kv_heads
The fused **RoPE + KV-cache write + (fp8) quant** kernel writes only `num_kv_heads` worth of K/V (not
`num_q_heads`) — so the GQA memory saving starts at the write, not just the read. aiter's KV-write path
takes `num_kv_heads`; make sure the cache is allocated at `num_kv_heads` (the `get_kv_cache_shape` uses
`num_kv_heads`).

## Shared neighbors with MHA
| fusion | applies to GQA? |
|---|---|
| RoPE + KV-write + fp8 quant (pre, sized to num_kv_heads) | yes |
| causal/SWA/ALiBi mask trait | yes |
| splitKV + reduce (decode) | yes |
| O in MFMA layout → next GEMM (`OPTIMIZE_EPILOGUE`) | yes |
| persistent / unified prefill+decode | yes |

## Where fusion moves e2e
The GQA bandwidth win (KV read shared by R heads) multiplies with fp8 KV and with the launch-fusion
decode wins — for GQA models (Llama-3, Qwen, Mistral) this *is* the attention e2e contribution. Nothing
GQA-specific to fuse beyond getting the broadcast right and sizing the KV-write to `num_kv_heads`.

## Sources
- aiter KV-cache shape uses `num_kv_heads`, gqa_ratio broadcast: on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/paged_attn.py`.
- RoPE+KV-write+quant fusion: https://github.com/ROCm/aiter ; `operators/attention_decode_paged/fusion.md`.
- MQA/GQA FA feature: https://github.com/Dao-AILab/flash-attention
