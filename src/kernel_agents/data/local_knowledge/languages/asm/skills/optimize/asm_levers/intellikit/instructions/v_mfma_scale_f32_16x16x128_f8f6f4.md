---
instruction: v_mfma_scale_f32_16x16x128_f8f6f4
category: MFMA
architecture: gfx950
hazard_severity: critical
tags: [MFMA, FP8, scaled, co-execution, AGPR, block-scale]
---

# v_mfma_scale_f32_16x16x128_f8f6f4

Scaled FP8 matrix fused multiply-add, 16x16 output tile, K=128. The scaled variant of v_mfma_f32_16x16x128_f8f6f4 that integrates per-block scaling into the MFMA operation. Co-execution window includes LD_SCALE at cycles 12-15.

---

## Quick Reference

| Field | Value |
|-------|-------|
| Opcode | `v_mfma_scale_f32_16x16x128_f8f6f4` |
| Tile Shape | 16 rows x 16 cols, K=128 |
| Input Type | FP8/FP6/FP4 (selected by CBSZ/BLGP) with per-block scale |
| Output Type | FP32 (accumulated in AGPRs) |
| Latency | 32 cycles (same pipeline as unscaled variant) |
| Issue Rate | 32 cycles |
| FLOPs | 16 * 16 * 128 * 2 = 65,536 (same compute, scaling is free) |
| Measured CPI | Not measured separately (expected same as unscaled: ~31.6) |

---

## Output Layout (Column-Major)

Same 16x16 column-major layout as all 16x16 MFMA instructions:

```
Lane L (0..63) has 4 AGPRs: a[k] = C[(L/16)*4 + k, L%16]   for k=0..3
```

---

## Operand Requirements

| Operand | Register | Count | Constraint |
|---------|----------|-------|------------|
| Src0 (A) | VGPRs | 8 | Must be VGPRs |
| Src1 (B) | VGPRs | 8 | Must be VGPRs |
| Src2/Dst (C/D) | AGPRs | 4 | Accumulator input/output |
| Scale A | Not measured | Not measured | Per-block scale factor for A matrix |
| Scale B | Not measured | Not measured | Per-block scale factor for B matrix |

The exact encoding of scale operands (register location, format) is not fully documented. E8M0 scale factors are encoded as `(byte << 23)`.

---

## Co-Execution Window

| Cycle Range | Activity |
|-------------|----------|
| 0-7 | Reads A, B, D operands. No co-execution. |
| 8-11 | Co-exec: TEX, LDS. |
| 12-15 | Co-exec: TEX, LDS, ALU0, **LD_SCALE**. |
| 16-31 | Full co-exec: TEX, LDS, ALU0. |
| 32+ | Can issue next MFMA. |

The LD_SCALE co-execution is available at cycles 12-15, which is specific to scaled MFMA variants. The unscaled variant also shows LD_SCALE at this window, suggesting the hardware pipeline slot exists for both but is only functional for the scaled opcode.

---

## NOP Requirements

Same as all MFMA instructions on gfx950:

| Transition | Required Wait |
|------------|---------------|
| `s_waitcnt` -> this MFMA | **0 NOPs** |
| VALU write -> this MFMA (same src VGPR) | **2 NOPs** (s_nop 1 x 2) |
| This MFMA -> another MFMA (diff AGPR dest) | **0 NOPs** |
| Last MFMA -> `v_accvgpr_read_b32` | **64 NOPs** (s_nop 15 x 4) |
| `v_accvgpr_write_b32` -> this MFMA (same AGPR) | **1 NOP** (s_nop 0) |

Additional hazard (gfx950-only): **CvtScaleForwardingHazard** -- not yet empirically characterized. Exercise caution with scale forwarding between MFMA instructions.

---

## CBSZ / BLGP Encoding

Same format selection as v_mfma_f32_16x16x128_f8f6f4:

| CBSZ[2:0] | A Format | BLGP[2:0] | B Format |
|-----------|----------|-----------|----------|
| 0 | FP8 (E4M3) | 0 | FP8 (E4M3) |
| 1 | BF8 (E5M2) | 1 | BF8 (E5M2) |
| 2 | FP6 (E3M2) | 2 | FP6 (E3M2) |
| 3 | FP6 (E2M3) | 3 | FP6 (E2M3) |
| 4 | FP4 (E2M1) | 4 | FP4 (E2M1) |

---

## Known Bugs / Gotchas

1. **Scale operand encoding not fully documented**: The exact register format and encoding of scale operands is not fully characterized. Disassemble a working kernel to verify.

2. **CvtScaleForwardingHazard**: A gfx950-specific hazard related to scale forwarding. Documented in internal arch docs but not yet empirically validated.

3. **E8M0 scale encoding**: Scale factors use E8M0 format, encoded as `(byte << 23)` to produce a floating-point power-of-two multiplier.

4. **Same accum_offset/AGPR aliasing bugs as all MFMA instructions.**

---

## Use Cases

- **Microscaling FP8 GEMM**: Block-scaled FP8 inference where each tile has its own scale factor.
- **Grouped GEMM with per-expert scaling**: Different experts may use different scale factors.
- **Future training workloads**: Block floating point formats for training with reduced memory footprint.
