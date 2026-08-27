---
title: mrope — numerics & parity
kind: technique
operator: mrope
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-07-14
sources:
  - https://github.com/vllm-project/vllm/pull/22593
  - https://github.com/vllm-project/vllm/issues/39625
  - ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_mrope_cache_quant.py
---

# mrope — numerics & parity

Inherits rope/numerics.md (cos/sin fp32, is_neox, partial-rotary, scaling, exact
rotation) plus the section-split specifics.

## 1. `mrope_section` must match the model
The `[s_t, s_h, s_w]` partition is a model constant (Qwen2.5-VL has specific section sizes). A wrong
partition rotates dims by the wrong axis's position → image/video tokens get scrambled positional info →
degraded VLM accuracy (subtle, not a crash). Carry it from the model config.

## 2. Per-axis positions
The `[3, ...]` positions tensor: text tokens use the same position on all 3 axes (degenerates to RoPE);
image/video tokens have distinct h/w/temporal positions. Feeding text positions to image tokens (or
mis-broadcasting) is a correctness bug.

## 3. Partial rotation (the recurring MRoPE crash)
vLLM #22593: the Qwen2-VL MRoPE Triton kernel assumed `head_size == rotary_dim`; GLM-4.1V uses partial
rotation → **illegal memory access**. #39625: MiniMaxM2+NVFP4 partial-rotary shape mismatch with TP=2.
Bound the rotation by `rotary_dim` and handle the partition relative to `rotary_dim`, not `head_size`.

## 4. Parity
Exact rotation → token-identical greedy parity on a fixed VLM input. A divergence = wrong section /
positions / partial bound, not rounding.

## Parity gate
1. isolated vs fp64 per-section rotation; confirm `mrope_section` and per-axis positions.
2. partial-rotary shape test (rotary_dim < head_size, TP>1).
3. e2e VLM greedy parity with image inputs.

## Sources
- partial-rotary / section bugs: https://github.com/vllm-project/vllm/pull/22593, https://github.com/vllm-project/vllm/issues/39625.
- mrope_section in the fused kernel: `aiter/ops/fused_qk_norm_mrope_cache_quant.py`.
