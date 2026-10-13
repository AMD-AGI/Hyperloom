---
title: scaled_residual_add_rmsnorm — Fuse per-branch `residual + branch*scalar` then RMSNorm (Granite muP residual_multiplier).
kind: operator
scope: languages/fusion
updated: 2026-08-14
---

# scaled_residual_add_rmsnorm

Fuse per-branch `residual + branch*scalar` then RMSNorm (Granite muP residual_multiplier).

| | |
|---|---|
| Env flag | `GRANITE_FUSED_RESIDUAL` |
| Trigger categories | `add`, `mul`, `rmsnorm` |
| Minimum trigger share | 0.08 |
| Frameworks | sglang, vllm, vllm-aiter |

## What to fuse

Fuse `new_residual = branch*scale + residual; out = rmsnorm(new_residual, w)` into one Triton kernel (`scaled_add_rmsnorm`), plus a `scaled_add` for the final branch that has no immediately-following norm. For residual-threaded models (Granite dense) fold the scalar into the NEXT layer's `input_layernorm` and the final `model.norm` by returning the RAW branch output.

## How to localize it in the source

Grep the model file for these anchors and fuse the chain they mark:

- `residual_multiplier`
- `attention_multiplier`
- `* self.residual_multiplier`
- `residual + `
- `input_layernorm`
- `post_attention_layernorm`

## Correctness reference

Reference = import the framework RMSNorm and compare `rmsnorm(x*scale + r)` on representative tensors. Author template: kernel/docs/fusion_templates/granite_fused.py.

## Already-fused markers

If the source already matches one of these, the chain is fused and there is
nothing to claim here:

- `scaled_add_rmsnorm`
- `GRANITE_FUSED`

## ROCm constraint

Author a ROCm-native Triton (or aiter) kernel. Do not reuse a framework
CUDA-only fused op such as `fused_qk_norm_rope`. Verify the kernel builds and
runs on the target GPU, not only that it matches numerically.
