---
title: residual_add_rmsnorm — Fold the residual-add into the following RMSNorm (fused add+rmsnorm, llama-style residual threading).
kind: operator
scope: languages/fusion
updated: 2026-08-14
---

# residual_add_rmsnorm

Fold the residual-add into the following RMSNorm (fused add+rmsnorm, llama-style residual threading).

| | |
|---|---|
| Env flag | `FUSED_RESIDUAL` |
| Trigger categories | `add`, `rmsnorm` |
| Minimum trigger share | 0.10 |
| Frameworks | sglang, vllm, vllm-aiter |

## What to fuse

For each decoder layer, replace the standalone `x = x + residual; y = norm(x)` with a fused add+rmsnorm `y, residual = norm(x, residual)`. Thread `residual` across layers; close the final add into the last norm. Prefer the framework's fused add+rmsnorm ONLY if it has a ROCm (aiter/HIP) implementation; otherwise author a Triton kernel computing rmsnorm(x + residual) in one pass.

## How to localize it in the source

Grep the model file for these anchors and fuse the chain they mark:

- `hidden_states = hidden_states + residual`
- `+ residual`
- `RMSNorm(`
- `input_layernorm`
- `post_attention_layernorm`
- `ffn_norm`

## Correctness reference

Reference = the framework's own RMSNorm eager forward applied to (x + residual). Import the real RMSNorm class from the framework and call its forward; do NOT re-implement rmsnorm.

## Already-fused markers

If the source already matches one of these, the chain is fused and there is
nothing to claim here:

- `fused_add_rmsnorm`
- `add_rmsnorm`
- `norm\([^)\n]*,\s*residual`

## ROCm constraint

Author a ROCm-native Triton (or aiter) kernel. Do not reuse a framework
CUDA-only fused op such as `fused_qk_norm_rope`. Verify the kernel builds and
runs on the target GPU, not only that it matches numerically.
