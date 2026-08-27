---
title: rope — fusion neighbors
kind: technique
operator: rope
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/fused_qk_norm_rope_cache_quant.py
  - /sgl-workspace/aiter/aiter/ops/triton/fusions/fused_kv_cache.py
  - https://github.com/sgl-project/sglang/issues/18466
---

# rope — fusion

RoPE sits at the **attention entry** and fuses heavily with everything around it. Standalone it's an extra
Q/K round-trip; fused it's near-free.

## 1. QK-norm + RoPE + KV-cache write + quant  (the Qwen3 mega-fusion)
The dominant production fusion: apply Q/K RMSNorm, then RoPE, then write K/V to the paged KV cache, with
optional fp8 quant — all in one kernel.
- `fused_qk_norm_rope_cache_quant` (aiter, standard RoPE).
- `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` (mrope/3D variant → mrope).
SGLang cites this fused QKNorm+RoPE+KV-set kernel as a significant Qwen3/Qwen3-VL per-layer win (#18466).
Cross-link rmsnorm, fused_norm_quant, kv_cache_quant, paged_kv_copy.

## 2. RoPE + KV-cache write
`rope_cached_*` and `fused_kv_cache` (Triton `fusions/fused_kv_cache.py`, `fused_bmm_rope_kv_cache.py`)
apply RoPE to K and write it to the cache in one pass. `rope_cached_positions_*` take positions inline.
Cross-link paged_kv_copy.

## 3. QKV-split + RoPE
`fused_qkv_split_qk_rope` (Triton): split the fused QKV projection output and apply RoPE to Q,K in one
kernel. Cross-link dense_gemm (QKV proj).

## 4. inside attention prefill
Some FMHA kernels apply RoPE on-the-fly at the QKᵀ load. Cross-link attention_prefill_fmha.

## Fusion table
| form | impl | folds in |
|---|---|---|
| QK-norm+RoPE+KV+quant | `fused_qk_norm_rope_cache_quant` | norm, rope, kv-write, fp8 |
| mrope variant | `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` | + 3D rope → mrope |
| RoPE + KV write | `rope_cached_*`, `fused_kv_cache` | kv-write |
| QKV-split + RoPE | `fused_qkv_split_qk_rope` | qkv split |

## torch.compile
vLLM `VLLM_ROCM_USE_AITER_TRITON_ROPE` and the RoPE custom-op registration keep the fused kernel through
Inductor. See backends/vllm_kernels/aiter_integration.

## Sources
- QK-norm+RoPE+KV+quant fusion: `/sgl-workspace/aiter/aiter/ops/fused_qk_norm_rope_cache_quant.py`.
- RoPE+KV / QKV-split+RoPE (Triton): `/sgl-workspace/aiter/aiter/ops/triton/fusions/fused_kv_cache.py`, `rope/fused_qkv_split_qk_rope.py`.
- Qwen3 fused QKNorm+RoPE+KV win: https://github.com/sgl-project/sglang/issues/18466.
