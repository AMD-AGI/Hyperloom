---
instruction: s_nop
category: scalar
architecture: gfx950
tags: [s_nop, NOP, hazard, transcendental, MFMA, v_readlane]
---

# s_nop

Scalar no-operation. Consumes 1 cycle per NOP unit (s_nop N = N+1 cycles of delay).

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | SOPP |
| Overhead CPI | 0.168 ref-clk (median 172 ticks / 1024 iters) |
| Normalized overhead | ~1 shader cycle per s_nop 0 |
| Category | Scalar flow control |

## Hazards

None -- this IS the hazard mitigation instruction.

## Known Bugs / Gotchas

### s_nop N provides N+1 cycles of delay

`s_nop 0` = 1 cycle, `s_nop 1` = 2 cycles, `s_nop 3` = 4 cycles, `s_nop 15` = 16 cycles. The operand is the number of additional idle cycles beyond the 1-cycle instruction itself.

### NOPs between MFMAs are unnecessary on gfx950

gfx950 handles MFMA pipeline hazards via hardware interlock. Compiler-inserted NOPs between MFMA instructions (common in Triton output) have zero performance impact and zero correctness impact when removed. This was independently verified across multiple kernel families.

However, NOPs execute "for free" during MFMA co-execution windows, so removing them does not improve performance either. The NOPs are neither helpful nor harmful on gfx950.

### Required NOP contexts on gfx950

Despite MFMA NOPs being free, several instruction sequences DO require explicit NOPs:

| Hazard | NOP Required | Notes |
|--------|-------------|-------|
| Transcendental -> consumer | s_nop 3 | v_exp_f32, v_rcp_f32, v_rsq_f32 -- no HW interlock |
| VALU write -> v_readlane_b32 | s_nop 1 | Returns stale value without NOP |
| MFMA -> v_accvgpr_read_b32 | s_nop 15 x 2 | 32 cycles total, no HW interlock |
| VALU -> MFMA SrcA/SrcB | s_nop 1 (2 NOPs) | May be handled by HW on gfx950 |
| M0 write -> buffer_load ... lds | s_nop 0 | 1-cycle M0 settling delay |
| v_cmp -> v_cndmask (some paths) | s_nop 1 | VCC write-to-read hazard |

### Calibration reference for isa-bench

`s_nop 0` is used as the cycle counter calibration reference in the isa-bench measurement methodology. The ratio between `s_memrealtime` ticks and `s_nop 0` cycles establishes the FCLK-to-shader-clock normalization factor (~5.814x on MI355X at 1600 MHz FCLK).

### Trailing NOP sled after s_endpgm

Triton-compiled kernels often have 50-300 `s_nop 0` instructions after `s_endpgm` as alignment padding. These never execute. When patching assembly, these trailing NOPs can be removed to make room for new instructions (maintaining constant .text section size).

## Common Usage Patterns

### Transcendental hazard guard
```asm
v_exp_f32 v10, v10
s_nop 3              ; 4 cycles -- wait for transcendental result
v_mul_f32 v11, v10, v11
```

### MFMA accumulator read drain
```asm
v_mfma_f32_32x32x16_bf16 a[0:15], ...
s_nop 15             ; 16 cycles
s_nop 15             ; 16 more cycles (32 total)
v_accvgpr_read_b32 v0, a0
```

### v_readlane hazard guard
```asm
v_mul_f32 v7, v7, v12
s_nop 1              ; 2 cycles
v_readlane_b32 s19, v7, 0
```

## Sources

- NOP requirement table for gfx950
- MFMA NOP removal confirmed to have zero performance effect independently
- isa-bench: s_nop_overhead latency kernel
