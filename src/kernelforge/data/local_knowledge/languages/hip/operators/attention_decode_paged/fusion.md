---
title: attention_decode_paged — fusion
kind: operator_overview
operator: attention_decode_paged
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/paged_attn.py
  - https://github.com/ROCm/aiter
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# attention_decode_paged — fusion

Decode is launch- and bandwidth-bound, so the fusion wins are about **removing kernel launches and HBM
round-trips**, not about combining big GEMMs.

## The decode-step pipeline
```
new token ─► [QKV proj GEMM] ─► [RoPE + KV-write + quant] ─► [paged-attn (splitKV)] ─► [reduce] ─► [O proj]
                                  └─ fuse pre ─┘                └──── fuse the splits ────┘
```

### Pre-attention: fused RoPE + KV-cache write + quant
The new token's K/V must be rotated, optionally fp8-quantized, and written into the paged cache before
attention. aiter fuses **RoPE + KV-cache-write + (fp8) quant** into one kernel (`write_to_paged_cache`
path) — one pass instead of three. For fp8 KV this is also where the per-head scale is applied. This is
the highest-value pre-decode fusion.

### Intra-attention: splitKV + reduce
flash-decoding *is* a fusion of the per-split partials into the reduce kernel
(`paged_attention_ll4mi_reduce_kernel`). The single-pass `paged_attention_v1` avoids the separate reduce
entirely when the context is short enough to fit one partition (no split needed) — fewer launches at
small context.

### Persistent / unified kernels (launch fusion)
- **Persistent MLA decode** (`SGLANG_AITER_MLA_PERSIST=1`) keeps the kernel resident across the decode
  loop, amortizing launch.
- **Unified attention** (`SGLANG_USE_AITER_UNIFIED_ATTN=1`) runs chunked-prefill + decode in **one
  kernel** — eliminating the separate prefill and decode launches for mixed batches. See
  ../chunked_prefill/overview.md.
- **HIP-graph capture** folds the whole decode step's kernels into one replayable graph.

## What does NOT fuse
- The **QKV projection GEMM** stays separate from paged-attn (different shape, consumed by RoPE first).
- The **O-projection GEMM** stays separate (attention output feeds it).
- Fusing the two projection GEMMs into paged-attn would spill registers and lose their tuned tiling — the
  cheap wins are the small ops (rope/quant/kv-write) and launch amortization.

## Backend support
| fusion | aiter | vLLM custom HIP | Triton |
|---|---|---|---|
| RoPE + KV-write + fp8 quant (pre) | **yes** (asm) | partial | yes (Triton kernel) |
| splitKV + reduce | yes (`v2`) | yes (`ll4mi_reduce`) | yes (stage-2) |
| persistent decode | yes (`SGLANG_AITER_MLA_PERSIST`) | no | no |
| unified prefill+decode | yes (unified_attention) | no (separate) | yes (unified_attention) |

## Where fusion moves e2e
At small/medium concurrency decode is launch-bound, so the launch-fusion wins (persistent, unified,
HIP-graph) and the fp8-KV bandwidth cut are where TPOT improves — this is most of the reported AITER
1.2–4.4× TPS over generic FA.

## Sources
- aiter RoPE+KV-write+quant fusion, paged-attn v1/v2 split+reduce: on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/paged_attn.py`, `aiter/rotary_embedding.py`.
- persistent MLA / unified attention envs: `backends/sglang_kernels/attention_backends.md`.
- 1.2–4.4× TPS from AITER decode path: https://vllm.ai/blog/2026-02-27-rocm-attention-backend
