---
title: sliding_window_attention — numerics
kind: technique
operator: sliding_window_attention
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://github.com/Dao-AILab/flash-attention
  - https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
---

# sliding_window_attention — numerics

## The dominant risk is correctness, not precision
SWA uses the same fp32 online-softmax accumulate as full FMHA, so per-element precision is identical to
attention_prefill_fmha. The failures that bite are **masking/semantics bugs**, which silently change
*which* tokens are attended:

1. **Window-edge off-by-one.** `window_size=(left,right)` is inclusive of `left` past tokens. Causal SWA
   with window `W` is `(W-1, 0)`, not `(W, 0)`. CK's `window_generic` splits an even window as
   `left = W/2`, `right = W-1-left` (see `mask.hpp`) — a *symmetric* window, different from a causal
   left-only window. Match the model's definition (HF config `sliding_window`) exactly.
2. **Attention sink.** StreamingLLM / GPT-OSS keep the first `sink` tokens always-visible plus the last
   `W`. Dropping the sink changes long-context behavior; CK carries `sink_size` in the mask coordinate
   builder. A pure band mask (no sink) is a different model.
3. **Silent full-attention.** If block-skipping is broken the kernel is numerically *correct* but slow —
   and if the mask is broken the other way (window applied where the model wants full attention on a
   full-attention layer of a hybrid model), outputs are wrong. Per-layer SWA-vs-full routing must match
   the architecture.
4. **Logit soft-cap** (Gemma): `score = cap·tanh(score/cap)` must be applied **before** the mask add,
   inside fp32. Skipping it changes the distribution.

## FP8 KV cache
fp8 SWA decode (fnuz on gfx942) quantizes the KV cache; the off-by-one exponent bias vs OCP means a
wrong-dialect read is off by ~2×. Always accuracy-gate fp8 SWA. See kv_cache_quant.

## Parity gate
- Oracle: dense attention with an explicit Python band mask (`+ sink` if applicable), fp32, same RoPE.
- Greedy temp=0, fixed seed, ≥10 prompts, ≥2 of them longer than the window (so the band actually
  truncates) — a window bug is invisible on prompts shorter than `W`.
- Cross-backend bf16 argmax flips on long greedy decode are benign (numerical-equivalence-class), the
  same caveat as full FMHA — distinguish from a real mask regression by checking logits, not just tokens.

## Sources
- FA window_size semantics: https://github.com/Dao-AILab/flash-attention
- CK window/sink mask builder: `ROCm/composable_kernel:example/ck_tile/01_fmha/mask.hpp` (on-box).
- ROCm SWA path / CK fallback: https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
