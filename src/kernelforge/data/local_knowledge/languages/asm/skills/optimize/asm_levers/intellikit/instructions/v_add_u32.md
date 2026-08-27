---
instruction: v_add_u32
category: vector
architecture: gfx950
tags: [v_add_u32, literal, MEMORY_APERTURE_VIOLATION, hazard]
---

# v_add_u32

Unsigned 32-bit integer addition. Does NOT set carry (use `v_add_co_u32` for carry).

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP2 |
| Throughput CPI | 0.199 ref-clk (median 204 ticks / 1024 iters) |
| Normalized throughput | ~1.2 shader cycles |
| Category | VALU integer |

## Hazards

Standard VALU forwarding latency (5 cycles). No special hazards beyond the literal bug below.

## Known Bugs / Gotchas

### Literal constant 128 causes MEMORY_APERTURE_VIOLATION

Using the literal value 128 as an inline operand to `v_add_u32` causes a GPU fault on gfx950:

```asm
; CRASHES with MEMORY_APERTURE_VIOLATION:
v_add_u32_e32 v10, 128, v10

; WORKAROUND -- use an SGPR:
s_mov_b32 s_tmp, 128
v_add_u32_e32 v10, s_tmp, v10
```

The value 128 is at the boundary between inline constants (0-127) and literal constants in the VOP2 encoding. The assembler may mis-encode it, causing the hardware to interpret subsequent bytes as an address or operand, leading to an aperture violation.

### Redundant MOV pattern: v_add_u32 vX, 0, vY

Triton's register allocator generates `v_add_u32_e32 vX, 0, vY` as a register-to-register copy (identity operation). In the GGEMM kernels, 4-6 such instructions at the inner loop top copy immutable LDS base addresses. These are provably redundant when the source registers (e.g., v176-v181) are never modified in the loop body. Eliminating them and using source registers directly saves ~0.25% on GGEMM wgrad.

```asm
; REDUNDANT (Triton-generated defensive copy):
v_add_u32_e32 v216, 0, v180    ; copy v180 -> v216
ds_read_b64_tr_b8 v[152:153], v216 offset:16384

; OPTIMIZED (use source directly):
ds_read_b64_tr_b8 v[152:153], v180 offset:16384
```

### Used for MFMA address computation

In both attention and GEMM kernels, `v_add_u32` computes per-thread tile offsets by adding the loop counter (SGPR) to a base address (VGPR):

```asm
v_add_u32_e32 v84, s14, v97   ; k_col = tile_offset + base_col
```

## Common Usage Patterns

### Loop induction variable update
```asm
v_add_u32_e32 v198, s35, v198   ; k_offset += k_stride
```

### Causal mask column index computation
```asm
v_add_u32 v85, s14, v97         ; k_col = tile_offset + base_col
v_cmp_gt_i32_e64 s[0:1], v85, v89  ; k_col > q_row_start?
```

## Sources

- Literal 128 MEMORY_APERTURE_VIOLATION discovery
- Redundant MOV elimination (+0.25%)
- Causal mask column index computation
