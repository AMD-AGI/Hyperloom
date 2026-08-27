---
instruction: v_max3_f32
category: vector
architecture: gfx950
tags: [v_max3_f32, softmax, reduction, three-input]
---

# v_max3_f32

Three-input maximum: returns max(src0, src1, src2). Reduces 3 values per instruction.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP3 |
| Throughput | ~1 shader cycle (standard VALU issue rate) |
| Category | VALU arithmetic |

## Hazards

Standard VALU forwarding latency (5 cycles). No special hazards.

## Known Bugs / Gotchas

### The key to efficient softmax on gfx950

Forward attention uses a 16-instruction `v_max3_f32` chain to reduce 32 S values to a per-lane max, completely replacing the 80-bpermute butterfly reduction used by naive implementations:

```asm
v_max3_f32 v85, v50, s10, v51     ; max(S[0], -inf, S[1])  -- s10 = -inf
v_max3_f32 v85, v85, v52, v53     ; max(prev, S[2], S[3])
v_max3_f32 v85, v85, v54, v55     ; max(prev, S[4], S[5])
; ... 16 total v_max3_f32 ops
v_max3_f32 v85, v85, v48, v49     ; final: max of all 32 S values
```

After the lane-local max, a single `ds_bpermute_b32` exchanges the max between lanes 0-31 and 32-63 (half-wave exchange), followed by one more `v_max3_f32` to combine:

```asm
ds_bpermute_b32 v92, v99, v85        ; exchange with partner lane (lane XOR 32)
v_max3_f32 v89, v121, v85, v92       ; max(prev_max, this_half, other_half)
```

**Total: 18 instructions** vs the naive approach's **192 instructions** (96 bpermutes + 96 v_max_f32).

### Why this works: MFMA output layout

The `v_mfma_f32_32x32x16_bf16` output layout places all 32 K-column values for a given Q-row in a single lane's VGPRs. This means the row-max can be computed entirely within a single lane using scalar (non-cross-lane) instructions. `v_max3_f32` is ideal because it reduces 3 values per instruction.

### First operand typically holds running accumulator

The standard pattern feeds the running max back as src0:

```asm
v_max3_f32 v_acc, v_acc, v_new0, v_new1   ; acc = max(acc, new0, new1)
```

The first instruction uses `-inf` (typically in an SGPR) as the initial accumulator value.

## Common Usage Patterns

### Softmax row-max (attention kernels)
```asm
; Reduce 32 S values to lane-local max:
v_max3_f32 v_max, v_S0, s_neg_inf, v_S1
v_max3_f32 v_max, v_max, v_S2, v_S3
; ... 16 total, then 1 bpermute for half-wave exchange
```

### General 3-way max
```asm
v_max3_f32 v_result, v_a, v_b, v_c   ; result = max(a, b, c)
```

## Sources

- Softmax row-max via v_max3_f32 chain (17 instructions total)
- Agents adopted v_max3_f32 pattern for softmax
- gfx950-reference.md: optimization playbook
