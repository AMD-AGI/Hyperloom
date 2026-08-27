---
instruction: v_mul_f32
category: vector
architecture: gfx950
tags: [v_mul_f32, VALU, multiply, softmax]
---

# v_mul_f32

Floating-point multiplication of two FP32 values.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP2 |
| Throughput CPI | 0.199 ref-clk (median 204 ticks / 1024 iters) |
| Normalized throughput | ~1.2 shader cycles |
| Category | VALU arithmetic |

## Hazards

Standard VALU forwarding latency (5 cycles). No special hazards.

## Known Bugs / Gotchas

### Attention output normalization

The forward attention epilogue uses 27 `v_mul_f32` instructions to multiply each O accumulator element by the inverse of `l_acc` (the softmax denominator):

```asm
v_mul_f32 v35, v34, v18    ; O_elem * (1/l_acc)
v_mul_f32 v36, v34, v19    ; next element
; ... 27 total
```

The reciprocal `1/l_acc` is computed via the full IEEE-754 `v_div_scale/v_rcp_f32/v_div_fmas/v_div_fixup` sequence (13 instructions) to avoid precision loss from `v_rcp_f32` alone.

### VALU write -> v_readlane_b32 hazard applies

If `v_mul_f32` writes to a VGPR that is immediately read by `v_readlane_b32`, the `s_nop 1` hazard applies:

```asm
v_mul_f32 v7, v7, v12       ; VALU writes v7
s_nop 1                      ; MANDATORY
v_readlane_b32 s19, v7, 0   ; now reads post-multiply value
```

## Common Usage Patterns

### O normalization (attention epilogue)
```asm
; v34 = 1/l_acc (precomputed via full division)
v_mul_f32 v_out, v34, v_accum     ; normalize each accumulator element
```

### Scale application (simpler than v_pk_mul_f32)
```asm
; When only a single value needs scaling:
v_mul_f32 v_scaled, v_scale, v_value
```

### Softmax score scaling
```asm
; scale = 1/sqrt(d_head), precomputed on host
v_mul_f32 v_score, s_scale, v_raw_score
```

## Sources

- O normalization (27 v_mul_f32 in epilogue)
- Attention kernel softmax, readlane hazard interaction
- isa-bench: valu_mul_f32 throughput kernel
