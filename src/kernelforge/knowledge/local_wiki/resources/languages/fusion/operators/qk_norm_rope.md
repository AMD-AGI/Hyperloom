---
title: qk_norm_rope — Fuse per-head Q/K RMSNorm (+ any grouped blend / temperature) with RoPE into one kernel.
kind: operator
scope: languages/fusion
updated: 2026-08-14
---

# qk_norm_rope

Fuse per-head Q/K RMSNorm (+ any grouped blend / temperature) with RoPE into one kernel.

| | |
|---|---|
| Env flag | `FUSED_QK` |
| Trigger categories | `rmsnorm`, `rope` |
| Minimum trigger share | 0.12 |
| Frameworks | sglang, vllm, vllm-aiter |

## What to fuse

Collapse the per-(token,k-head) QK post-processing chain -- grouped-mean blend (if present) -> RMSNorm(rsqrt) -> optional temperature -> (optionally RoPE) -- into one Triton kernel. A natural grid is one program per (token, k-head) looping the GQA q-heads inside; outputs match the eager fp32 dtype.

## How to localize it in the source

Grep the model file for these anchors and fuse the chain they mark:

- `q_norm`
- `k_norm`
- `_normalize_qk`
- `_add_grouped_qk_means`
- `rotary_emb(`
- `apply_qk_norm`
- `clamp_temp`

## Correctness reference

Reference = the model's real eager QK methods (e.g. `_add_grouped_qk_means` + `_normalize_qk`, or `q_norm`/`k_norm` + `rotary_emb`). Import and call them directly on representative q/k tensors; do NOT re-derive the math.

## Already-fused markers

If the source already matches one of these, the chain is fused and there is
nothing to claim here:

- `fused_qk_norm`
- `fused_qk_norm_rope`
- `fused_qk_norm_mrope`

## ROCm constraint

Author a ROCm-native Triton (or aiter) kernel. Do not reuse a framework
CUDA-only fused op such as `fused_qk_norm_rope`. Verify the kernel builds and
runs on the target GPU, not only that it matches numerically.
