---
instruction: v_mfma_scale_f32_16x16x128_f8f6f4
category: MFMA
architecture: gfx950
tags: [MFMA, scaled, FP8, FP6, FP4, microscaling, block-scaled, gfx950]
---

# v_mfma_scale_f32_16x16x128_f8f6f4 (Block-Scaled MFMA)

## Syntax

```asm
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3], v16, v17 op_sel_hi:[0,0]
```

## Description

Scaled 16x16x128 matrix multiply-add with per-block scale factors. Supports FP8, FP6, and FP4 data types with independent scaling per 32-element block. The `op_sel_hi` field selects the data format for each operand.

## Operands

| Operand | Width | Notes |
|---------|-------|-------|
| Dst/Src C | 4 AGPRs | FP32 accumulators |
| Src A | 8 VGPRs (256 bits) | 128 elements in FP8/FP6/FP4 |
| Src B | 8 VGPRs (256 bits) | 128 elements in FP8/FP6/FP4 |
| Scale A | 1 VGPR | Per-block scale factor |
| Scale B | 1 VGPR | Per-block scale factor |

## Format Selection (op_sel_hi)

| op_sel_hi | Format | Bits/Element | Elements in 256 bits |
|-----------|--------|-------------|---------------------|
| 0 | FP8 (E4M3) | 8 | 32 per VGPR |
| 1 | BF8 (E5M2) | 8 | 32 per VGPR |
| 2 | FP6 | 6 | ~42 per VGPR (packed) |
| 3 | FP4 | 4 | 64 per VGPR |

## FNUZ Correction

`v_mfma_scale` uses OCP FP8 natively. If input data is in FNUZ format, apply 0.25x correction per operand:
- FNUZ A × FNUZ B → multiply result by 0.0625 (0.25 × 0.25)
- This applies per-operand, so mixed OCP/FNUZ requires only one 0.25x

## Known Issues

1. **Scale factor VGPR lifetime.** Scale VGPRs must remain valid through the entire MFMA execution window. If ds_read overwrites the scale VGPR during co-execution, restore it at loop entry.

2. **Triton dot_scaled generates these.** When disassembling Triton-compiled kernels, these appear as the inner-loop MFMA. The disassembly round-trip may lose the `lds` modifier on associated buffer_loads.
