---
instruction: v_add_f32
category: vector
architecture: gfx950
tags: [v_add_f32, VALU, DPP, broken]
---

# v_add_f32

Floating-point addition of two FP32 values.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP2 |
| Throughput CPI | 0.203 ref-clk (median 208 ticks / 1024 iters) |
| Normalized throughput | ~1.2 shader cycles |
| Latency CPI | 0.178 ref-clk (median 182 ticks / 1024 iters) |
| Category | VALU arithmetic |

## Hazards

Standard VALU forwarding latency (5 cycles). No special hazards.

### DPP modifiers are broken

`v_add_f32` with DPP `row_shr` modifier is a complete no-op at runtime on gfx950. The instruction assembles and encodes correctly but does nothing. This affects all DPP modifiers on this instruction. See `v_readlane_b32` for the workaround (readlane loop for cross-lane reductions).

## Known Bugs / Gotchas

### DPP row_shr produces no effect

DPP `row_shr:N` on `v_add_f32` assembles correctly but does nothing at runtime on gfx950. This was discovered when implementing cross-lane reductions for softmax:

```asm
; BROKEN -- does nothing on gfx950:
v_add_f32 v4, v4, v4 row_shr:1 bound_ctrl:0

; WORKAROUND -- use v_readlane_b32 loop:
v_readlane_b32 s_tmp, v4, 0
; ... accumulate across all lanes
```

### Used in softmax row-sum chain

The forward attention kernel uses a 31-instruction serial `v_add_f32` chain to sum 32 P values within a single lane. No cross-lane communication needed because the MFMA output layout places all K-column values for a given Q-row in the same lane's VGPRs.

```asm
v_add_f32 v34, v65, v85    ; sum = P[0] + P[1]
v_add_f32 v34, v124, v34   ; sum += P[2]
; ... 31 total v_add_f32 ops
v_add_f32 v45, v55, v34    ; final sum of all 32 P values
```

## Common Usage Patterns

### Softmax row-sum accumulation
```asm
; Serial lane-local sum (no cross-lane needed):
v_add_f32 v_sum, v_P0, v_P1
v_add_f32 v_sum, v_P2, v_sum
; ... repeat for all P values
```

### Address computation (integer-like usage)
```asm
; Rarely used for addresses -- v_add_u32 preferred for integer work
```

## Sources

- DPP row_shr failure discovery
- Softmax row-sum chain analysis (31 ops for 32 values)
- isa-bench: valu_add_f32 throughput and latency kernels
