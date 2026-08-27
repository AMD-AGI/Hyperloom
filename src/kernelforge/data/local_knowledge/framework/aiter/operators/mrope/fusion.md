---
title: mrope — fusion neighbors
kind: technique
operator: mrope
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_mrope_cache_quant.py
  - https://github.com/sgl-project/sglang/issues/18466
---

# mrope — fusion

Same fusion family as rope — the attention-entry mega-fusion, specialized for 3D positions.

## 1. QK-norm + mRoPE + KV-cache write + quant (+ shuffle)
aiter `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle`: Q/K RMSNorm → 3D mRoPE (per `mrope_section`) →
KV-cache write (paged, with positions) → optional fp8 quant → KV-layout shuffle, all in one kernel. The
VLM analog of the Qwen3 `fused_qk_norm_rope_cache_quant_shuffle` win (#18466). Cross-link rmsnorm,
fused_norm_quant, kv_cache_quant, paged_kv_copy, layout_shuffle.

## 2. mRoPE + KV write
The cached mRoPE variants apply mRoPE to K and write to the cache in one pass (the 3D analog of
`rope_cached_*`). Cross-link paged_kv_copy.

## 3. inside VLM attention prefill
mRoPE applied at the QKᵀ load for image/video tokens. Cross-link attention_prefill_fmha.

## Fusion table
| form | impl | folds in |
|---|---|---|
| QK-norm+mRoPE+KV+quant+shuffle | `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` | norm, 3D rope, kv-write, fp8, shuffle |
| mRoPE + KV write | cached mrope variant | kv-write |

## Sources
- aiter fused QK-norm+mrope+KV+quant+shuffle: `aiter/ops/fused_qk_norm_mrope_cache_quant.py`.
- analogous Qwen3 RoPE fusion win: https://github.com/sgl-project/sglang/issues/18466.
