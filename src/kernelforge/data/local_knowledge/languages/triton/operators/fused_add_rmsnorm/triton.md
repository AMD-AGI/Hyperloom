---
title: fused_add_rmsnorm on triton — SOTA card
kind: sota_card
operator: fused_add_rmsnorm
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py
  - /sgl-workspace/aiter/aiter/ops/triton/normalization/fused_add_rmsnorm_pad.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# fused_add_rmsnorm × triton

## TL;DR
Triton is the authorable SOTA; aiter's own `_fused_add_rmsnorm_kernel` (and a padded variant
`fused_add_rmsnorm_pad.py`) is the reference. Memory-bound → matches CK. Use for fused variants the library
lacks or torch.compile codegen.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `_fused_add_rmsnorm_kernel` | `aiter/ops/triton/normalization/rmsnorm.py` | gfx942/950, bf16/fp16 | persistent `min(rows,num_sms)`, add→store residual_out→fp32 Σ | the Triton tier |
| `fused_add_rmsnorm_pad` | `aiter/ops/triton/normalization/fused_add_rmsnorm_pad.py` | gfx942/950 | padded-N variant | non-pow2 / padded hidden |
| `_quant_fused_add_rmsnorm_kernel` | same dir | gfx942/950, fp8 y | + dynamic quant | fused_norm_quant |

## Config space / knobs
- `block_size = min(65536//elt, next_pow2(N))`, `use_blocked = N>block`.
- Grid `min(rows, num_sms)` persistent; `num_warps=2–4`; `num_stages=2`; `.cg` loads; fp32 Σ(r')².
- Keep `r'` on-chip (registers/LDS) for single-pass to avoid HBM re-read.

## Numerics / parity
add in IO dtype, store residual_out; fp32 Σ; γ fp32-promote. Reduction order vs CK differs → greedy
re-gate. See [numerics.md](numerics.md).

## Integration (rebind seam)
- Direct: `from aiter.ops.triton.normalization.rmsnorm import rmsnorm2d_fwd_with_add`.
- torch.compile: Inductor emits Triton for the add+rmsnorm pattern; aiter ops preferred for fused quant.

## Pitfalls & anti-patterns
- `num_warps=8` → spill.
- Writing `r'` to HBM then reloading for the reduction → defeats the fusion; hold on-chip.
- fp32 accumulate omitted → drift (compounds via residual).

## How to verify
`TRITON_PRINT_AUTOTUNING=1`; `residual_out` parity vs `x+r`; ISA `dwordx4`; greedy e2e.

## Alternatives / cross-links
aiter · hip · vllm_kernels · rmsnorm/backends/triton ·
triton_levers/patterns §5.

## Sources
- aiter Triton fused-add (+pad, +quant) kernels: `/sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py`, `fused_add_rmsnorm_pad.py`.
- memory-bound knobs: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
