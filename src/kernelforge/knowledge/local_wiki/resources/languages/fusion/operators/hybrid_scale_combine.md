---
title: hybrid_scale_combine — Fuse hybrid attn+mamba input-prescale and output-combine scalar muls (Falcon-H1).
kind: operator
scope: languages/fusion
updated: 2026-08-14
---

# hybrid_scale_combine

Fuse hybrid attn+mamba input-prescale and output-combine scalar muls (Falcon-H1).

| | |
|---|---|
| Env flag | `FALCON_H1_FUSED_SCALES` |
| Trigger categories | `add`, `mul` |
| Minimum trigger share | 0.06 |
| Frameworks | sglang, vllm, vllm-aiter |

## What to fuse

(a) prescale: read `hidden` once, emit `hidden*attn_in_mult` and `hidden*ssm_in_mult` (2 muls + 2 reads -> 1 kernel); (b) combine: `attn_out*attn_out_mult + mamba_out*ssm_out_mult` in one kernel.

## How to localize it in the source

Grep the model file for these anchors and fuse the chain they mark:

- `attn_in_mult`
- `ssm_in_mult`
- `attn_out_mult`
- `ssm_out_mult`
- `key_multiplier`
- `* self.attention_in_multiplier`

## Correctness reference

Reference = the eager scalar muls / combine on representative tensors. Author template: kernel/docs/fusion_templates/falcon_h1_fused.py.

## Already-fused markers

If the source already matches one of these, the chain is fused and there is
nothing to claim here:

- `FALCON_H1_FUSED`
- `fused_scales`

## ROCm constraint

Author a ROCm-native Triton (or aiter) kernel. Do not reuse a framework
CUDA-only fused op such as `fused_qk_norm_rope`. Verify the kernel builds and
runs on the target GPU, not only that it matches numerically.
