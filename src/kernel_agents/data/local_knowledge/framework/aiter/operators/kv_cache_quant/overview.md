---
title: kv_cache_quant — overview
kind: operator_overview
operator: kv_cache_quant
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, int8, bf16, fp16]
regimes: [decode, prefill, both]
updated: 2026-07-14
sources:
  - vllm-project/vllm@HEAD:csrc/cache_kernels.cu
  - ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_rope_cache_quant.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# kv_cache_quant  (store K/V to the paged cache in FP8/INT8)

## TL;DR
Quantize the K and V tensors to **FP8 (or INT8)** as they are **written into the paged KV cache**, and
dequantize on read in the attention kernel. This **halves KV-cache memory** vs bf16 → roughly **2× the
context length or batch** for the same HBM, which is the dominant memory pressure in long-context decode.
The store-quant is fused into the `reshape_and_cache` write (and, in aiter, into the QK-norm+RoPE+write
chain) so the bf16 KV never round-trips HBM. The accuracy lever is the **k_scale / v_scale** granularity
(per-tensor static is standard; block/per-token is more accurate). Paged layout (`slot_mapping`) is
preserved; only the dtype of the cache changes.

## Math contract
- **Store**: for each token's K/V (after RoPE), `kv_fp8 = scaled_convert(kv_bf16, scale)` written to the
  cache slot `slot_mapping[token]`; cache dtype `kv_cache_dtype ∈ {auto, fp8, fp8_e4m3, fp8_e5m2, int8}`.
- **Load**: the paged-attention kernel reads `kv_fp8` and applies `* k_scale / * v_scale` into fp32 before
  the QK·V math (online softmax in fp32).
- vLLM: `reshape_and_cache[_flash]` does `fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt, k_scale)`
  inline (`csrc/cache_kernels.cu`); `k_scale`/`v_scale` are scalar (per-tensor) by default.

## Scale granularity (the accuracy/perf axis)
| granularity | scale | when |
|---|---|---|
| **per-tensor static** | scalar k_scale, v_scale (calibrated) | the standard, cheapest, paged-friendly |
| per-head / per-channel | `[num_heads]` / `[head_dim]` | better; some models need per-head |
| **block / per-token** | per KV block (paged block) or per token | best accuracy; aiter `*_block_quant_shuffle` |

## Paged layout
The cache is paged: `[num_blocks, block_size, num_heads, head_dim]` (+ an `x`-split for the K layout in
vLLM). The quant writes into `slot_mapping[token]` so quantization is **per-slot** and never reorders the
paged structure. KV layout shuffle (`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`) interacts with the FA backend at
concurrency ≥32. → operators/paged_kv_copy, operators/attention_decode_paged.

## Shape regimes
- **decode (dominant)**: one (or few) new token's K/V per step → tiny write, fused into the cache write;
  the *read* side (paged attention dequant) is the hot path.
- **prefill**: write the whole prompt's K/V → larger store; still fused into reshape_and_cache.

## Where it matters (Amdahl)
KV quant is not about compute time — it is about **memory capacity**. FP8 KV ≈ 2× context/batch, which
raises throughput by enabling larger batches (more concurrency) at the same HBM. The quant/dequant cost is
small and fused; the win is system-level. Accuracy is the constraint, not speed.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (fused QK-norm+RoPE+KV-write+quant, pts/block) | [aiter.md](aiter.md) |
| vllm_kernels | 🟢 sota (`reshape_and_cache[_flash]` scaled_convert) | vllm_kernels |
| hip | 🟢 sota (editable cache-write HIP source) | hip |
| triton | 🟡 competitive (Triton paged-attn FP8 KV read/write) | triton |

## Fusion neighbors
QK-norm + RoPE + KV-write + quant (`fused_qk_norm_rope_cache_quant`), dequant fused into paged-attention
read. → fusion.md, operators/rope, operators/attention_decode_paged, operators/paged_kv_copy.

## Numerics
fp8 fnuz vs ocp (off-by-2× trap on gfx942), k_scale/v_scale, per-tensor vs block, online-softmax fp32
accumulate, accuracy gate (gsm8k loss observed with some MLA+KV combos) → numerics.md.

## How to bench
Isolated: time `reshape_and_cache` / `fused_qk_norm_rope_cache_quant` for a decode step. e2e: enable
`--kv-cache-dtype fp8`, measure max batch/context AND task accuracy (the gate is accuracy, not tok/s).

## Sources
- vLLM reshape_and_cache + scaled_convert: `vllm-project/vllm@HEAD:csrc/cache_kernels.cu`.
- aiter fused QK-norm+RoPE+KV-quant (pts/block, slot_mapping): `ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_rope_cache_quant.py`.
- KV shuffle env, kv-cache-dtype: https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
- FP8 fnuz/ocp: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
