---
title: quant_dequant_fp8 on asm — SOTA card
kind: sota_card
operator: quant_dequant_fp8
backend: asm
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/quant_kernels.cu
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# quant_dequant_fp8 × asm

## TL;DR
Standalone FP8 quant is **bandwidth-bound**, so hand-written assembly buys almost nothing over a
well-vectorized HIP kernel — there is no matrix-core math to schedule. The place raw asm/intrinsics
matter for FP8 quant is **inside fused MFMA kernels**: the FP8 *conversion* and *block-scale apply* are
done with the conversion intrinsics (`v_cvt_pk_fp8_f32`, and on CDNA4 the `v_mfma_scale_*` that applies
the E8M0 scale after the dot product) so the FP8 operands feed the matrix core with no separate pass.
This card is `competitive` only as a sub-component; standalone, prefer hip.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `v_cvt_pk_fp8_f32` packed conversion (in fused GEMM/quant µkernels) | aiter asm GEMM `gemm_a8w8_asm`; cast in `csrc/kernels/quant_kernels.cu` (HIP intrinsics) | gfx942/950, e4m3 | same-pass cast (no extra HBM) | fused into the consuming MFMA kernel |
| CDNA4 `v_mfma_scale_f32_*_f8f6f4` (applies E8M0 after dot) | hardware/cdna4_mi350 | gfx950, mxfp8 | block-scaled MFMA | block-scaled FP8/MXFP GEMM |

## Config space / knobs
- conversion intrinsic selection (`v_cvt_pk_fp8_f32` packs 2 f32→fp8); pack width.
- for the scaled MFMA: E8M0 scale operand layout (Ax/Bx) — verify with the matrix-instruction calculator.
- instruction scheduling (`sched_group_barrier`) so the cast overlaps the load.

## Numerics / parity
The conversion intrinsic honors the arch FP8 dialect (FNUZ gfx942 / OCP gfx950). The scaled MFMA applies
the scale **after the dot, before accumulate** (hardware/cdna4_mi350). Gate on task accuracy →
numerics.md.

## Integration (rebind seam)
Not a standalone Python op — it lives inside aiter's asm GEMM / fused kernels. The rebind seam is the
parent fused kernel, not a quant entrypoint.

## Pitfalls & anti-patterns
- Writing a standalone asm quant kernel — no benefit over vectorized HIP (memory-bound).
- Wrong E8M0 scale operand layout on the scaled MFMA → silent corruption.
- FNUZ↔OCP conversion-intrinsic mismatch.

## How to verify
`amd_matrix_instruction_calculator --architecture cdna4 --instruction v_mfma_scale_f32_32x32x64_f8f6f4
--get-register` for scale layout; disassemble the fused kernel and confirm `v_cvt_pk_fp8_f32` + no extra
HBM round-trip.

## Alternatives / cross-links
hip (prefer standalone) · aiter · languages/asm_mfma ·
hardware/cdna4_mi350 · [overview.md](overview.md).

## Sources
- HIP/intrinsic cast (`scaled_quant_impl`): `ROCm/aiter@a6bb49937:csrc/kernels/quant_kernels.cu`.
- `v_cvt_pk_fp8_f32` / `v_mfma_scale_*`, E8M0 apply: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
