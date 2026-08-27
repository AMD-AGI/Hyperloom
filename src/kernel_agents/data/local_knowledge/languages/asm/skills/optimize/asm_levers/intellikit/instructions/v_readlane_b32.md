---
instruction: v_readlane_b32
category: vector
architecture: gfx950
tags: [v_readlane_b32, cross-lane, hazard, s_nop, reduction]
---

# v_readlane_b32

Cross-lane read: extracts a single lane's VGPR value into an SGPR.

## Quick Facts

| Property | Value |
|----------|-------|
| Category | Cross-lane |
| Measured latency | 0.586 ref-clk CPI (median 150 ticks / 256 iters) |
| Normalized latency | ~3.4 shader cycles |
| Result destination | SGPR |

## Hazards

### VALU write -> v_readlane_b32: needs s_nop 1

`v_readlane_b32` does NOT interlock with preceding VALU writes to the source VGPR. Without a NOP, it returns the **pre-VALU-write** value.

**Required:** `s_nop 1` between any VALU write and v_readlane_b32 reading the same VGPR.

```asm
v_mul_f32 v7, v7, v12       ; VALU writes v7
s_nop 1                      ; MANDATORY
v_readlane_b32 s19, v7, 0   ; now reads post-multiply value
```

**How discovered:** Attention kernel produced all-NaN. Diagnostic kernel stored 13 intermediate values. `v_readlane_b32 s19, v7, 0` returned 128.0 (raw dot product) instead of 11.31 (scaled score), proving the preceding `v_mul_f32` hadn't committed when readlane executed.

**Measured in isa-bench:** hazard_readlane_nop0 kernel confirms the hazard exists (returns stale float value as raw bits in the cycle counter output).

## Known Bugs / Gotchas

### DPP row_shr is broken -- use v_readlane_b32 loop instead

DPP `row_shr` modifiers on `v_add_f32` are a complete no-op at runtime on gfx950 (assemble correctly, encode correctly, do nothing). The standard workaround for cross-lane reductions is a `v_readlane_b32` loop:

```asm
; Reduction: sum all 64 lanes of v4 into s12
s_mov_b32 s12, 0
.set i, 0
.rept 64
    v_readlane_b32 s13, v4, i
    s_add_f32 s12, s12, s13    ; NOTE: s_add_f32 not on gfx950, use VALU
    .set i, i+1
.endr
```

Since `s_add_f32` is not available on gfx950, the actual pattern requires more creativity (accumulate in SGPR as integer, or use a VGPR accumulator).

## Common Usage Patterns

### Lane 0 broadcast for wave-uniform values
```asm
; After a per-lane computation, broadcast lane 0's result:
v_mul_f32 v10, v10, v11
s_nop 1                        ; hazard guard
v_readlane_b32 s10, v10, 0    ; extract lane 0 to SGPR
; s10 now usable as uniform scalar
```

### Cross-lane reduction (replacement for broken DPP)
```asm
; Sum reduction across 64 lanes into s_acc
; Must insert s_nop 1 before first readlane if VALU just wrote v_src
s_nop 1
v_readlane_b32 s_tmp, v_src, 0
s_mov_b32 s_acc, s_tmp
v_readlane_b32 s_tmp, v_src, 1
; ... accumulate across all lanes
```

## Sources

- Attention kernel NaN debugging, DPP row_shr failure
- VALU->readlane hazard documentation
- isa-bench: crosslane_readlane latency kernel, hazard_readlane_nop0 correctness kernel
- gfx950-reference.md: NOP cheat sheet
