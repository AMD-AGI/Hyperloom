---
title: act_and_mul_silu_gelu — numerics & parity
kind: technique
operator: act_and_mul_silu_gelu
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1, mxfp4]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/activation.py
  - /sgl-workspace/aiter/aiter/ops/triton/activation.py
  - https://github.com/vllm-project/vllm/blob/main/csrc/activation.cu
---

# act_and_mul_silu_gelu — numerics & parity

## 1. fp32 activation compute
`sigmoid`/`erf`/`tanh` are computed in **fp32** even with bf16 IO; the multiply by the up half and the
convert to out dtype happen last. bf16 sigmoid loses precision near 0 → small drift. The standard:
`silu(fp32(gate)) * fp32(up)` → convert.

## 2. Exact-erf vs tanh-approx GeLU
Two GeLU definitions ship and are **not bit-equal**:
- **exact** `gelu(z) = z·Φ(z)` (erf): `gelu_and_mul`.
- **tanh approx** `0.5z(1+tanh(√(2/π)(z+0.044715z³)))`: `gelu_tanh_and_mul`.
Models are trained with one or the other (GPT-2/BERT = tanh approx; some use exact). **Match the
checkpoint** — a mismatch is a silent accuracy bug (vLLM #43326: GELU_TANH unsupported in a MoE path).
`gelu_fast`/`gelu_quick` are further approximations for non-gated activations.

## 3. mul_and_silu vs silu_and_mul (which half is gated)
`silu_and_mul`: `silu(x[:d])·x[d:]`. `mul_and_silu`: `x[:d]·silu(x[d:])`. The concat order of the fused
gate/up Linear decides which is correct — get it wrong and the gate is applied to the up projection.
Confirm against the model's `gate_up_proj` packing.

## 4. fp8/fp4 quant-fused output (→ fused_norm_quant)
- **fp8** (`act_mul_and_fp8_group_quant`): per-group scale = `max(|y|)/fp8_max` in fp32; **fnuz on
  gfx942** (off-by-2× if wrong dialect). Group size must match the down-proj GEMM's dequant.
- **mxfp4** (`act_mul_and_mxfp4_quant`): block-scaled with e8m0 scale, block=32; **CDNA4 (gfx950) HW**;
  on gfx942 FP4 has no HW path (vLLM FP4BMM crashes gfx942 → `VLLM_ROCM_USE_AITER_FP4BMM=0`).
- Gate at the **task** level (gsm8k/MMLU), not allclose.

## Parity gate
1. isolated vs fp64: rel-err band; confirm correct GeLU variant + correct gated half.
2. fp8/fp4 fused: task eval delta within noise; fnuz dialect on gfx942; FP4 only on gfx950.
3. greedy e2e parity after a backend/fusion swap (reduction-free op, but quant scale order can differ).

## Sources
- aiter activation variants (silu/gelu/gelu_tanh/scaled): `/sgl-workspace/aiter/aiter/ops/activation.py`.
- act+quant (fp8 group, mxfp4 e8m0 block=32): `/sgl-workspace/aiter/aiter/ops/triton/activation.py`.
- GELU_TANH variant mismatch bug: https://github.com/vllm-project/vllm/issues/43326.
- FP4 gfx942 crash: perf_knowledge backends/vllm_kernels/overview.
