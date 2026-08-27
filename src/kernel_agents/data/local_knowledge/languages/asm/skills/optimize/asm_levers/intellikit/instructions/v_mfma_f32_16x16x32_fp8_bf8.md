---
instruction: v_mfma_f32_16x16x32_fp8_bf8
category: MFMA
architecture: gfx950
hazard_severity: critical
tags: [MFMA, FP8, legacy, AGPR, backward-compat]
---

# v_mfma_f32_16x16x32_fp8_bf8

Legacy FP8 matrix fused multiply-add, 16x16 output tile, K=32. MI300X-era (gfx942) FP8 instruction also present on MI355X (gfx950) for backward compatibility. On MI355X, prefer v_mfma_f32_16x16x128_f8f6f4 which delivers 8x the FLOPs per instruction.

---

## Quick Reference

| Field | Value |
|-------|-------|
| Opcode | `v_mfma_f32_16x16x32_fp8_bf8` |
| Tile Shape | 16 rows x 16 cols, K=32 |
| Input Type | FP8 (E4M3) / BF8 (E5M2), selected by blgp |
| Output Type | FP32 (accumulated in AGPRs) |
| Latency | 16 cycles (empirical; some docs report 4 cycles) |
| Issue Rate | Not measured separately |
| FLOPs | 16 * 16 * 32 * 2 = 16,384 |
| Measured CPI | Not measured (isa-bench tests the f8f6f4 variant instead) |

### Latency Disagreement

- **gfx950-reference.md**: 16 cycles (empirical).
- **Some documentation** reports 4-cycle latency, 1-cycle throughput. This appears to be the MI300X pipeline spec, not MI355X.
- **Practical**: On MI355X, treat as 16 cycles. But this instruction is deprecated in favor of f8f6f4 regardless.

---

## Output Layout (Column-Major)

Same 16x16 column-major layout as v_mfma_f32_16x16x32_bf16:

```
Lane L (0..63) has 4 AGPRs: a[k] = C[(L/16)*4 + k, L%16]   for k=0..3
```

---

## Operand Requirements

| Operand | Register | Count | Constraint |
|---------|----------|-------|------------|
| Src0 (A) | VGPRs | 4 | Must be VGPRs |
| Src1 (B) | VGPRs | 4 | Must be VGPRs |
| Src2/Dst (C/D) | AGPRs | 4 | Accumulator input/output |

---

## Co-Execution Window

Not measured for this specific opcode. Expected to be similar to v_mfma_f32_16x16x32_bf16 (cycles 8-15 free for co-execution).

---

## NOP Requirements

| Transition | Required Wait |
|------------|---------------|
| `s_waitcnt` -> this MFMA | **0 NOPs** |
| VALU write -> this MFMA (same src VGPR) | **2 NOPs** (s_nop 1 x 2) |
| This MFMA -> another MFMA (diff AGPR dest) | **0 NOPs** |
| Last MFMA -> `v_accvgpr_read_b32` | **64 NOPs** (s_nop 15 x 4) |
| `v_accvgpr_write_b32` -> this MFMA (same AGPR) | **1 NOP** (s_nop 0) |

---

## CBSZ / BLGP Encoding

Unlike the f8f6f4 variants which use both CBSZ and BLGP, this legacy instruction uses only `blgp` for format selection:

| blgp | Format |
|------|--------|
| 0 | FP8 x FP8 (both A and B are E4M3) |
| 1 | FP8 x BF8 (A=E4M3, B=E5M2) |

Limited to FP8 and BF8 formats only. No FP6 or FP4 support.

---

## Known Bugs / Gotchas

1. **Deprecated on MI355X**: This instruction delivers only 16,384 FLOPs vs 65,536 for v_mfma_f32_16x16x128_f8f6f4. Switching to the native variant gives 1.67-1.76x speedup with no algorithmic changes.

2. **Same tile shape, 4x fewer FLOPs**: The output tile is identical (16x16, 4 AGPRs). The only difference is K=32 vs K=128. Four of these instructions = one native f8f6f4 instruction in terms of work done.

3. **Latency documentation conflicts**: 4 cycles (docs) vs 16 cycles (empirical). The 4-cycle number may be MI300X-specific or from an older pipeline model.

4. **Same accum_offset/AGPR aliasing bugs as all MFMA instructions.**

---

## Use Cases

- **MI300X backward compatibility**: Code that must run on both MI300X (gfx942) and MI355X (gfx950).
- **Variable-K wgrad (legacy path)**: Used in grouped GEMM wgrad before upgrading to f8f6f4.
- **Recommendation**: On MI355X, migrate to v_mfma_f32_16x16x128_f8f6f4 for 4x FLOPs per instruction and 1.67-1.76x end-to-end speedup.
