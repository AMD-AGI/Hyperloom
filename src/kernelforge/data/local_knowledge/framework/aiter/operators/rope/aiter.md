---
title: rope on aiter — SOTA card
kind: sota_card
operator: rope
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/rope.py
  - ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_rope_cache_quant.py
  - https://github.com/sgl-project/sglang/issues/18466
  - https://docs.vllm.ai/en/latest/design/fusions/
---

# rope × aiter

## TL;DR
aiter is the live RoPE path with the most complete variant set on AMD: cached/uncached, 1-channel/2-channel,
thd (varlen), 2d, positions/offsets — plus the **fused QK-norm+RoPE+KV+quant** attention-entry kernel that
is the Qwen3 win. On the serving path you use the fused variant, not standalone `rope_fwd`.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `rope_fwd` / `rope_fwd_inplace` (HIP/C++) | `aiter/ops/rope.py` (`module_rope_*`; kernels `csrc/kernels/rope/rope_common.h`) | gfx942/950, bf16/fp16 | bandwidth-bound; in-place halves write | standalone RoPE |
| `rope_cached_*` / `rope_cached_positions_*` | same | gfx942/950 | RoPE + cos/sin cache (+ positions inline) | cached path |
| `rope_thd_*` (varlen) / `rope_2d_*` | same | gfx942/950 | ragged/2D layouts | varlen prefill / 2D |
| `fused_qk_norm_rope_cache_quant_shuffle` / `..._block_quant_shuffle` / `fused_qk_rope_concat_and_cache_mla` | `aiter/ops/fused_qk_norm_rope_cache_quant.py` | gfx942/950, fp8 | Qwen3 per-layer win (#18466) | **the serving attention entry** |

## Config space / knobs
- Variant select = entrypoint (cached vs uncached, 1c/2c, thd/2d, positions/offsets, fused).
- `is_neox` / rotation style; `rotary_dim` for partial; cos/sin cache (scaling baked in).
- JIT `module_rope_*` on first call (many small modules — one per variant).

## Numerics / parity
cos/sin fp32; `is_neox` matches config; `rotary_dim` bound for partial; scaling cache correct. RoPE is
deterministic → greedy parity should be token-identical (divergence = style/dim bug). See
[numerics.md](numerics.md).

## Integration (rebind seam)
- vLLM: `VLLM_ROCM_USE_AITER=1` (+ `VLLM_ROCM_USE_AITER_TRITON_ROPE` for the Triton rope variant);
  RoPE layer routes to aiter. vLLM also has a **ROCm-only RoPE + KV-cache fusion** pass (O1+, auto) that
  folds the cache write into RoPE. ⚠ Inductor torch-op quant can auto-fuse some quant patterns, so adjacent
  norm/quant passes may be obsolete except custom-op cases — RoPE itself stays a custom op (the fused
  QK-norm+RoPE+KV+quant kernel is the win, not an Inductor pattern).
- SGLang: on by default; Qwen3 attention uses `fused_qk_norm_rope_cache_quant`.
- Verify: `AITER_LOG_MORE=1` dispatch; rocprofv3 fused QK-norm+RoPE kernel.

## Pitfalls & anti-patterns
- ⚠ Wrong `is_neox` / partial-rotary bound → garbage attention (silent).
- ⚠ Standalone `rope_fwd` when the fused entry exists → extra Q/K round-trip.
- ⚠ Scaling cache mismatch (YaRN/NTK) — verify against config.

## How to verify
`op_tests/test_rope.py`; isolated vs fp64 rotation; greedy parity (token-identical); rocprofv3.

## Alternatives / cross-links
triton · hip · vllm_kernels · mrope ·
fused_norm_quant · kv_cache_quant.

## Sources
- aiter rope variants (`ROCm/aiter@b467ce342`): `aiter/ops/rope.py`.
- fused QK-norm+RoPE+KV+quant (exported as `fused_qk_norm_rope_cache_quant_shuffle`,
  `fused_qk_norm_rope_cache_block_quant_shuffle`, `fused_qk_rope_concat_and_cache_mla`):
  `aiter/ops/fused_qk_norm_rope_cache_quant.py`.
- Qwen3 win: https://github.com/sgl-project/sglang/issues/18466.
- vLLM ROCm RoPE+KV-cache fusion pass (O1+, auto); torch-op quant auto-fuse / custom-op caveat: https://docs.vllm.ai/en/latest/design/fusions/.
