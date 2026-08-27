---
title: kv_cache_quant — fusion
kind: operator_overview
operator: kv_cache_quant
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [decode, prefill, both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_rope_cache_quant.py
  - ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_mrope_cache_quant.py
  - vllm-project/vllm@HEAD:csrc/cache_kernels.cu
---

# kv_cache_quant — fusion

> KV quant is almost never run standalone — it is fused into the **producer chain** (QK-norm → RoPE → cache
> write) on the store side, and into the **paged-attention read** on the load side. The bf16 K/V never
> touches HBM; only the FP8 cache does.

## 1. QK-norm + RoPE + KV-write + quant (the big fused kernel) → operators/rope
aiter's `fused_qk_norm_rope_cache_quant` family does the entire pre-attention KV chain in one kernel:
- `fused_qk_norm_rope_cache_quant_shuffle` — QK-norm + RoPE + KV-write + FP8 quant + KV-layout shuffle.
- `fused_qk_norm_rope_cache_pts_quant_shuffle` — **per-tensor-scale** KV quant variant (scalar
  `per_tensor_k_scale`/`per_tensor_v_scale`).
- `fused_qk_norm_rope_cache_block_quant_shuffle` — **block-scale** KV quant variant.
- `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` — the **mRoPE** (3D multimodal) variant
  (`aiter/ops/fused_qk_norm_mrope_cache_quant.py`) → operators/mrope.
They take `slot_mapping` (paged), `cos_sin_cache`, `qw`/`kw` (norm weights), `kv_cache_dtype`, and the
scales, and write the quantized K/V straight into the paged cache. This removes: a norm pass, a RoPE pass,
a KV write, and a separate quant — collapsed to one launch.

## 2. quant fused into reshape_and_cache (vLLM) → operators/paged_kv_copy
vLLM's `reshape_and_cache` / `reshape_and_cache_flash` do `fp8::scaled_convert<cache_t, scalar_t,
kv_dt>(tgt_key, k_scale)` inline as they scatter K/V into the paged slots — the quant is fused into the
cache-copy, never a separate op (`csrc/cache_kernels.cu`).

## 3. dequant fused into the paged-attention read → operators/attention_decode_paged
The store-side quant has a matching **read-side dequant** fused into the paged-attention kernel: the FP8 KV
is loaded and multiplied by `k_scale`/`v_scale` into fp32 before the QK·V dot. There is no standalone KV
dequant op — it lives in the attention kernel (`paged_attention_ll4mi_*` reads FP8 KV with scaled
conversion). → backends/vllm_kernels rocm_kernels.

## Fusion decision
| stage | fuse into |
|---|---|
| K/V after projection, before cache | `fused_qk_norm_rope_cache_quant*` (#1) or `reshape_and_cache` (#2) |
| multimodal (mRoPE) | `fused_qk_norm_mrope_cache_quant` (#1) |
| KV read in decode | the paged-attention kernel's dequant (#3) |
| standalone | almost never — use #1/#2 |

## Pitfalls
- The KV-layout **shuffle** chosen on store must match the FA backend's read layout
  (`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`).
- per-tensor vs block scale must be consistent between the fused write and the attention read.
- FNUZ↔OCP dialect must match between the write kernel and the attention read (numerics.md).
- mRoPE needs the mrope variant (position handling differs).

## Sources
- aiter fused QK-norm+RoPE+KV-quant (pts/block/mrope, slot_mapping, scales): `ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_rope_cache_quant.py`, `aiter/ops/fused_qk_norm_mrope_cache_quant.py`.
- vLLM reshape_and_cache inline scaled_convert: `vllm-project/vllm@HEAD:csrc/cache_kernels.cu`.
