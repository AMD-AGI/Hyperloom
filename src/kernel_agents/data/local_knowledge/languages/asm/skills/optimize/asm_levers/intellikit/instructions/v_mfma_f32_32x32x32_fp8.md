---
instruction: v_mfma_f32_32x32x32_fp8
category: MFMA
architecture: gfx950
tags: [MFMA, FP8, matrix-multiply, gfx950, accumulators]
---

# v_mfma_f32_32x32x32_fp8_fp8

## Syntax

```asm
v_mfma_f32_32x32x32_fp8_fp8 a[0:15], v[0:3], v[4:7], a[0:15]
```

## Description

32x32x32 matrix fused multiply-add with FP8 inputs and FP32 accumulators. Processes 32 K-elements per instruction, doubling K-throughput versus the 32x32x16 BF16/F16 variants.

## Operands

| Operand | Width | Location | Notes |
|---------|-------|----------|-------|
| Src A | 4 VGPRs (128 bits) | VGPRs only | 32 FP8 values |
| Src B | 4 VGPRs (128 bits) | VGPRs only | 32 FP8 values |
| Src/Dst C | 16 AGPRs (512 bits) | AGPRs or VGPRs | FP32 accumulators |

**Src must be VGPRs, NOT AGPRs.** Reading AGPRs as MFMA source requires `v_accvgpr_read_b32` with NOP penalties.

## Cycle Counts (Measured)

| Metric | Value |
|--------|-------|
| CPI | ~32 cycles |
| Effective CPI (production) | 33.8 cycles |
| Peak TFLOPS (MI355X) | 2517 TFLOPS |

## NOP Requirements

| Transition | NOPs Required |
|------------|---------------|
| VALU → this MFMA | s_nop 1 (2 NOP cycles) |
| s_waitcnt → this MFMA | 0 NOPs |
| This MFMA → v_accvgpr_read_b32 | 2x s_nop 15 (32 NOP cycles minimum) |
| This MFMA → ds_read consuming result | s_waitcnt provides sufficient gap |

## Co-Execution

During the 32-cycle execution window, the following can execute in parallel:
- Scalar operations (s_load, s_mov, s_waitcnt)
- ds_read/ds_write (LDS operations)
- buffer_load/global_load (memory loads)
- VALU on VGPRs NOT used by the MFMA

## Output Layout

FP32 results in 16 AGPRs (or VGPRs). Column-major: `a[k] = C[col_group*4+k, lane_row]`. Each accumulator VGPR/AGPR holds 4 rows in the same column, not 4 columns in the same row.

## FP8 Format Notes

- Native OCP FP8 (E4M3/E5M2). If data is in FNUZ format, apply 0.25x correction per operand (0.0625x total for A*B).
- Use `v_cvt_scalef32_pk_f16_fp8` for FP8→FP16 dequantization when needed outside MFMA.
- E8M0 scale factors encode as `(byte << 23)` for direct FP32 multiplication.
