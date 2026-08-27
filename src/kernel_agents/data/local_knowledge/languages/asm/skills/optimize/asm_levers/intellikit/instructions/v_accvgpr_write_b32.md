---
instruction: v_accvgpr_write_b32
category: AGPR
architecture: gfx950
tags: [v_accvgpr_write_b32, AGPR, NOP, MFMA, accum_offset]
---

# v_accvgpr_write_b32

Write a value from a VGPR (or immediate) into an AGPR (accumulator VGPR).

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP3 |
| Measured round-trip (write+read) CPI | 1.07 ref-clk (median 274 ticks / 256 iters) |
| Normalized round-trip latency | ~6.2 shader cycles (write + read combined) |
| Category | AGPR access |

## Hazards

### VALU write -> MFMA SrcA/SrcB: 2 NOPs required

When `v_accvgpr_write_b32` writes to an AGPR that is subsequently used as an MFMA SrcC (accumulator input), 2 NOPs are needed between the write and the MFMA:

```asm
v_accvgpr_write_b32 a0, v10      ; initialize accumulator
s_nop 1                           ; hazard guard
s_nop 0                           ; (2 NOPs total)
v_mfma_f32_16x16x32_bf16 a[0:3], v[src_a], v[src_b], a[0:3]
```

However, for MFMA -> MFMA chaining with the same SrcC accumulator (no intervening accvgpr_write), zero NOPs are needed -- the hardware forwards internally.

## Known Bugs / Gotchas

### accum_offset aliasing

Same as `v_accvgpr_read_b32`: VGPRs at index >= `accum_offset` alias AGPRs. Writing to a "VGPR" above accum_offset actually writes an AGPR. This is the source of silent corruption when `accum_offset` is set too low.

### MFMA accumulator initialization

To zero accumulators before a GEMM, either:
1. Use `v_accvgpr_write_b32 a0, 0` for each AGPR (requires NOP before first MFMA)
2. Use MFMA with literal `0` as SrcC: `v_mfma ... a[0:3], v[src_a], v[src_b], 0` (zeroes accumulators implicitly, no extra instruction)

Option 2 is preferred because it avoids the write + NOP overhead.

### BWD attention K-data reload

In backward attention, after `v_perm_b32` clobbers K-data VGPRs for BF16 packing, the K data must be reloaded from AGPRs (where it was saved) or from LDS. If using AGPRs, `v_accvgpr_read` is needed with full NOP drain.

## Common Usage Patterns

### Accumulator initialization
```asm
; Zero all accumulators before GEMM:
v_accvgpr_write_b32 a0, 0
v_accvgpr_write_b32 a1, 0
; ... for all AGPRs
s_nop 1   ; before first MFMA using these accumulators
```

### VGPR-to-AGPR data save
```asm
; Save VGPR data to AGPR for later recovery:
v_accvgpr_write_b32 a16, v_data
; ... K data in v_data can now be clobbered
; Later recovery:
s_nop 15
s_nop 15
v_accvgpr_read_b32 v_data, a16
```

## Sources

- Accumulator initialization patterns
- MFMA NOP table (VALU->MFMA: 2 NOPs)
- K-data AGPR save/restore in BWD attention
- isa-bench: agpr_write_read round-trip latency kernel
