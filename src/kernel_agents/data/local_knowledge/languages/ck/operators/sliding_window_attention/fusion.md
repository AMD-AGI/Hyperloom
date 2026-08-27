---
title: sliding_window_attention — fusion
kind: technique
operator: sliding_window_attention
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://github.com/Dao-AILab/flash-attention
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# sliding_window_attention — fusion

## What fuses into the SWA kernel
SWA is FMHA + band mask, so it inherits the full-attention fusion neighbors and adds a few SWA-specific
ones:

| fusion | where | benefit |
|---|---|---|
| **RoPE / qk-norm pre-step** | before QKᵀ | one less HBM round-trip; standard FA fusion (rope, rmsnorm) |
| **Band mask + block-skip** | the KV loop bound itself | the SWA win — see [tuning.md](tuning.md) |
| **Logit soft-cap** | fp32, before mask add | Gemma-2/3 (`tanh` cap); fused into the score |
| **Attention sink** | mask coordinate builder | StreamingLLM / GPT-OSS keep first-N + last-W |
| **fp8 KV-cache quant/dequant** | KV load | bandwidth; fnuz on gfx942 (kv_cache_quant) |
| **Output projection** | epilogue (rarely) | usually a separate GEMM; kept distinct on AMD |

## Hybrid-model layer routing (the macro-fusion)
The bigger "fusion" is architectural: SWA-heavy hybrids (Gemma-2 1:1 SWA/full, Qwen "every N", Mistral)
share the same KV cache manager but route per-layer to SWA vs full attention. On ROCm, getting both
paths on the **same backend** (so the scheduler/graph capture is uniform) matters more than micro-fusion.
vLLM's `ROCM_AITER_FA` routes Decode/Prefill/Extend per request; SWA layers ride the same dispatcher.

## Cross-links
- Core attention fusion: attention_prefill_fmha · attention_decode_paged
- Chunked/prefix-cache interplay: chunked_prefill
- KV quant: kv_cache_quant · RoPE: rope

## Sources
- FA SWA + soft-cap + sink: https://github.com/Dao-AILab/flash-attention
- MI300X 2-GEMM attention fusion / OPTIMIZE_EPILOGUE: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
