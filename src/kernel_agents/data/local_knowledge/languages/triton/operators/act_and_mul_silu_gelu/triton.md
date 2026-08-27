---
title: act_and_mul_silu_gelu on triton — SOTA card
kind: sota_card
operator: act_and_mul_silu_gelu
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, mxfp4]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/activation.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# act_and_mul_silu_gelu × triton

## TL;DR
Triton is the authorable SOTA, especially for the **act+quant fused** variants aiter ships in
`ops/triton/activation.py` (`act_mul_and_fp8_group_quant`, `act_mul_and_mxfp4_quant`). Memory-bound →
Triton matches hand-written; the value is expressing the activation+quant fusion the C++ path may lack.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `act_mul_and_fp8_group_quant` | `aiter/ops/triton/activation.py` | gfx942/950, fp8 out | per-group fp8 quant fused | act → fp8 down-proj |
| `act_mul_and_mxfp4_quant` | same | gfx950, mxfp4 out | e8m0 block-32 scale | act → mxfp4 (CDNA4) |
| Triton silu/gelu + mul (in fused-MoE) | `aiter/ops/triton/moe/` | gfx942/950 | inside Triton fused-MoE | Triton MoE path |

## Config space / knobs
- `BLOCK_SIZE_N = min(256, next_pow2(N_half))`, ≥32, ×32 if mxfp4.
- `BLOCK_SIZE_M = min(8, next_pow2(M))`; `NUM_WARPS = 1 if BLOCK_M<4 else 4`.
- grid `(cdiv(M, BLOCK_M), cdiv(N_half, BLOCK_N·NUM_ITER))`.
- `activation ∈ {"silu","gelu","gelu_tanh"}` (Literal arg); fp32 act compute.
- `waves_per_eu=3–4`; 128-bit loads (vectorized halves).

## Numerics / parity
fp32 act; correct GeLU variant + gated half; fp8 fnuz gfx942; mxfp4 gfx950. Task gate for quant. See
[numerics.md](numerics.md).

## Integration (rebind seam)
- Direct import from `aiter.ops.triton.activation`.
- torch.compile: Inductor can codegen the gated activation; the act+quant fusion is best from the aiter
  Triton kernel (Inductor won't synthesize the fp8 group quant).

## Pitfalls & anti-patterns
- `num_warps=8` over-warps a memory-bound op → spill; use 1/4.
- mxfp4 on gfx942 → no FP4 HW; only gfx950.
- Wrong `activation` literal vs checkpoint.

## How to verify
`TRITON_PRINT_AUTOTUNING=1`; isolated bench at `(M, 2d)`; fp64 oracle; task eval for quant; greedy parity.

## Alternatives / cross-links
aiter · hip · vllm_kernels ·
fused_norm_quant · quant_fp4_mxfp · triton_levers/patterns §5.

## Sources
- aiter Triton act+quant kernels: `/sgl-workspace/aiter/aiter/ops/triton/activation.py`.
- memory-bound knobs / 128-bit loads: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
