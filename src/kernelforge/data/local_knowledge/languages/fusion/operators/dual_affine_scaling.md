---
title: dual_affine_scaling — Fuse a dual (x + bias) * scale affine on the hidden (and residual) streams into one kernel.
kind: operator
scope: languages/fusion
updated: 2026-08-14
---

# dual_affine_scaling

Fuse a dual (x + bias) * scale affine on the hidden (and residual) streams into one kernel.

| | |
|---|---|
| Env flag | `FUSED_RESIDUAL_SCALE` |
| Trigger categories | `add`, `elementwise`, `mul` |
| Minimum trigger share | 0.06 |
| Frameworks | sglang, vllm, vllm-aiter |

## What to fuse

Fuse `(x + bias) * scale` applied per-row over the hidden dim on both the hidden and (when present) residual streams into a single Triton kernel; fp32 output.

## How to localize it in the source

Grep the model file for these anchors and fuse the chain they mark:

- `ResidualScaling`
- `residual_scale`
- `residual_bias`
- `* scale`
- `(x + bias)`
- `has_residual`

## Correctness reference

Reference = the model's eager affine (e.g. `ResidualScaling.forward`). Import and call it on representative tensors; do NOT re-implement the affine.

## Already-fused markers

If the source already matches one of these, the chain is fused and there is
nothing to claim here:

- `fused_residual_scaling`

## ROCm constraint

Author a ROCm-native Triton (or aiter) kernel. Do not reuse a framework
CUDA-only fused op such as `fused_qk_norm_rope`. Verify the kernel builds and
runs on the target GPU, not only that it matches numerically.
