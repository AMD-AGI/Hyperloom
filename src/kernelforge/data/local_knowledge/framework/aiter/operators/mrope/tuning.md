---
title: mrope — tuning
kind: technique
operator: mrope
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_mrope_cache_quant.py
  - https://github.com/vllm-project/vllm/pull/22593
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# mrope — tuning

Same memory-bound playbook as rope/tuning.md. The only delta is the **section split**:
each dim's cos/sin comes from its axis's position.

## 1. Section split: precompute the per-dim axis map
`mrope_section = [s_t, s_h, s_w]` partitions `rotary_dim`. Rather than branch per dim, **precompute** which
axis each dim uses (or interleave the cos/sin cache so a single gather gives the right angle). The vLLM
MRoPE Triton kernel builds the cos/sin for the 3 axes and selects per section — keep that selection in
constexpr/section-tile form, not a per-lane branch.

## 2. Positions tensor is `[3, ...]`
Load all 3 axis positions for the token; for text tokens they're equal (degenerates to RoPE). The cos/sin
lookup is per (position, dim) — reuse across heads (same as RoPE).

## 3. Knobs (inherit RoPE)
| knob | setting | note |
|---|---|---|
| `num_warps` | 4 | memory-bound |
| `BLOCK_S` | 32 (seq-tile) | per RoPE |
| grid | `(b, h, cdiv(s,32))` | fills chip prefill |
| in-place | yes | halve write |
| section map | precomputed constexpr | avoid per-lane branch |
| cos/sin reuse | per seq-tile, across heads | per RoPE |

## 4. The real lever: fuse into the attention entry
aiter `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` folds QK-norm + mRoPE + KV-write + quant — the only
way mRoPE is near-free on a VLM decode. See [fusion.md](fusion.md).

## Sources
- aiter fused mrope kernel (mrope_section, 3D, cache+quant): `aiter/ops/fused_qk_norm_mrope_cache_quant.py`.
- MRoPE Triton section handling + partial fix: https://github.com/vllm-project/vllm/pull/22593.
- 128-bit loads / grid: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
