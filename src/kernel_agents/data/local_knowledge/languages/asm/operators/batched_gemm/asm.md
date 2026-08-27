---
title: batched_gemm on asm — SOTA card
kind: sota_card
operator: batched_gemm
backend: asm
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
---

# batched_gemm × asm

## TL;DR
> Hand-written MFMA assembly (or Gluon) gives **peak** per-shape batched GEMM — reserve it for the few
> batched shapes that dominate Amdahl and where tuned hipBLASLt leaves a real gap, or to exploit CDNA4
> block-scaled fp4/fp6 MFMA before libraries do. Highest authoring cost; engage via aiter's race or a
> direct call. For standalone attention matmuls, prefer the FMHA kernels instead.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter asm batched kernels (raced in tuned_gemm) | `ROCm/aiter@HEAD` | gfx942; bf16/fp8 | wins specific skinny/decode batched shapes in the per-shape race @ MI300X, 2026-06-08 | shapes where asm beats hipBLASLt |
| Gluon near-peak GEMM (carried to batched) | ROCm Gluon GEMM tutorial | gfx950; bf8/mxfp4 | BF8 3257 TFLOPS (99.72%), MXFP4 5255 TFLOPS (92.41%) single-GEMM peak; batched of small shapes lower (occupancy) | fp4/fp8 near-peak on CDNA4 |

## Config space / knobs
- MFMA `16×16×16`/`32×32×8`; CDNA4 `mfma_scale_f32_32x32x64_f8f6f4` (256-bit operands; pad fp4 to
  `fp4x64_t`), E8M0 per-32 scales; scale pipeline GR→LW→LR.
- Batch on a grid axis; ≥1024 total WGs; same-XCD per batch; deep K-pipeline; `OPTIMIZE_EPILOGUE`.

## Numerics / parity
fp32 accumulate per batch; fp4/fp6 block-scaled → task-gated; bf16 parity-safe. See [numerics.md](numerics.md).

## Integration (rebind seam)
Engage via aiter (raced candidate) or direct call. No env-overlay for a standalone asm blob.

## Pitfalls & anti-patterns
- Authoring asm for non-Amdahl batched shapes — cost without e2e move.
- fp4 scale-layout (GR→LW→LR) errors → wrong results/stalls; missing 256-bit operand padding.
- Re-implementing attention matmuls in asm instead of using FMHA.

## How to verify
Microbench the kernel (TFLOP/s = 2·B·M·N·K / t) vs `hipblaslt-bench --batch_count`; fp4 vs dequant ref + eval.

## Alternatives / cross-links
aiter (dispatches asm) · hipblaslt · ck ·
[overview.md](overview.md) · attention operators/attention_prefill_fmha/overview.md ·
language refs languages/asm_mfma/..., languages/hipkittens/....

## Sources
- CDNA3/4 matrix-core programming + scaled-MFMA: ROCm matrix-cores-cdna blog.
- BF8 99.72% / MXFP4 92.41% TFLOPS: ROCm Gluon GEMM tutorial.
- aiter asm dispatch: `ROCm/aiter@HEAD` (see aiter.md).
