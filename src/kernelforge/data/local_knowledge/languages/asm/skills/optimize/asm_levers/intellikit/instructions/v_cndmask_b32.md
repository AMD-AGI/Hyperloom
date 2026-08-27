---
instruction: v_cndmask_b32
category: vector
architecture: gfx950
tags: [v_cndmask_b32, VCC, constant-bus, select]
---

# v_cndmask_b32

Conditional select: chooses between two source values per-lane based on a mask (VCC or explicit SGPR pair).

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP2 (VCC implicit) / VOP3 (explicit mask) |
| Throughput CPI | 0.254 ref-clk (median 260 ticks / 1024 iters) |
| Normalized throughput | ~1.5 shader cycles |
| Category | VALU bitwise/select |

## Hazards

Standard VALU forwarding latency (5 cycles).

### Constant bus restriction (VOP2 encoding)

In VOP2 encoding, `v_cndmask_b32` implicitly reads VCC as the mask. It cannot simultaneously use another SGPR as src0 or src1 due to the constant bus restriction. Using an SGPR source AND VCC causes a constant bus conflict:

```asm
; BROKEN -- constant bus conflict (2 SGPR reads: s10 + VCC):
v_cndmask_b32 v5, s10, v5, vcc

; CORRECT -- use VOP3 encoding with explicit mask:
v_cndmask_b32_e64 v5, s10, v5, vcc
; or load SGPR to VGPR first:
v_mov_b32 v_tmp, s10
v_cndmask_b32 v5, v_tmp, v5, vcc
```

The BWD attention kernel uses `v_cndmask_b32_e64` (VOP3) throughout to avoid this restriction.

## Known Bugs / Gotchas

### Causal masking uses 32 x v_cndmask_b32

The forward attention kernel applies causal masking per-element using `v_cndmask_b32` to replace masked positions with `-inf` (0xff800000). Each of the 32 S-matrix values in v[34:65] gets its own compare + cndmask:

```asm
v_cmp_gt_i32_e64 s[0:1], v85, v89     ; k_col > q_row?
v_cndmask_b32_e64 v50, v87, v50, s[0:1]  ; replace with -inf if masked
```

32 elements x 6 instructions = 192 instructions for the causal mask section. This is the most instruction-heavy part of the attention loop.

### s_nop sometimes needed between v_cmp and v_cndmask

It has been observed that `s_nop 1` is needed between `v_cmp_neq_f32` and `v_cndmask_b32` in the forward attention disassembly. This appears to be a VCC write-to-read hazard in specific instruction sequences. The kernel uses only 4 NOPs total, and 2 of them guard `v_cmp -> v_cndmask` sequences.

## Common Usage Patterns

### Causal mask application
```asm
v_cmp_gt_i32_e64 s[mask:mask+1], v_kcol, v_qrow
v_cndmask_b32_e64 v_S, v_neg_inf, v_S, s[mask:mask+1]
```

### Out-of-bounds sentinel selection
```asm
; Replace address with OOB sentinel if bounds check fails:
v_cmp_lt_u32 vcc, v_idx, s_limit
v_cndmask_b32 v_addr, v_sentinel, v_addr, vcc  ; keep addr if in-bounds
```

### NaN handling in softmax
```asm
v_cmp_neq_f32 vcc, v_max, v_max    ; is max NaN?
s_nop 1                              ; VCC write hazard
v_cndmask_b32 v_max, v_max, v_zero, vcc  ; replace NaN with 0
```

## Sources

- Constant bus restriction on v_cndmask_b32
- Causal masking analysis (192 instructions per iteration)
- Discovered during production kernel development: common causal mask bugs
- isa-bench: valu_cndmask_b32 throughput kernel
