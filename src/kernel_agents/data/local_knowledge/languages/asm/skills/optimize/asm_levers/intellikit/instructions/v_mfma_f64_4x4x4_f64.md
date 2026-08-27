---
instruction: v_mfma_f64_4x4x4_f64
category: MFMA
architecture: gfx950
tags: [v_mfma_f64_4x4x4_f64, MFMA, FP64, small-tile, AGPR]
---

# v_mfma_f64_4x4x4_f64

FP64 (double-precision) matrix fused multiply-add, 4x4 output tile, K=4. Small-tile FP64 MFMA variant.

---

## Quick Reference

| Field | Value |
|-------|-------|
| Opcode | `v_mfma_f64_4x4x4_f64` |
| Tile Shape | 4 rows x 4 cols, K=4 |
| Input Type | FP64 (double precision) |
| Output Type | FP64 (accumulated in AGPRs) |
| Latency | Not measured |
| Issue Rate | Not measured |
| FLOPs | 4 * 4 * 4 * 2 = 128 FP64 FLOPs |
| Measured CPI | ~3.9 shader cycles (isa-bench, raw FCLK CPI = 0.672) |

---

## Output Layout

Not measured. The 4x4 tile is small enough that each lane likely holds a subset of the 16 output elements.

---

## Operand Requirements

| Operand | Register | Count | Constraint |
|---------|----------|-------|------------|
| Src0 (A) | VGPRs | Not measured | FP64 requires 2 VGPRs per value |
| Src1 (B) | VGPRs | Not measured | FP64 requires 2 VGPRs per value |
| Src2/Dst (C/D) | AGPRs | Not measured | FP64 accumulators |

---

## Co-Execution Window

Not measured.

---

## NOP Requirements

Not measured. Expected to follow MFMA hazard rules.

---

## Known Bugs / Gotchas

1. **Very small tile**: Only 128 FP64 FLOPs per instruction. Many MFMAs needed for meaningful matrix sizes.
2. **No empirical validation beyond throughput CPI.**

---

## CBSZ / BLGP Encoding

Not applicable. FP64 has no format variants.

---

## Use Cases

- **Small FP64 matrix operations**: Batched small-matrix solvers, 4x4 transforms.
- **FP64 reduction operations**: Where small tiles are sufficient.
