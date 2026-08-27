---
instruction: v_add_co_u32
category: vector
architecture: gfx950
tags: [v_add_co_u32, carry-out, VCC, clobber]
---

# v_add_co_u32

Unsigned 32-bit integer addition with carry-out to VCC.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP2 (VOP3 for explicit carry dest) |
| Throughput | ~1 shader cycle (standard VALU issue rate) |
| Category | VALU integer |

## Hazards

Standard VALU forwarding latency (5 cycles).

### Clobbers VCC

`v_add_co_u32` writes carry-out to VCC (or an explicit SGPR pair in VOP3 encoding). This silently clobbers VCC, which may be live for subsequent `v_cndmask_b32` or branch instructions.

## Known Bugs / Gotchas

### VCC clobber breaks subsequent v_cndmask_b32

In attention kernels, bounds checking uses `v_cmp_*` to set VCC, followed by `v_cndmask_b32` to select values based on VCC. If `v_add_co_u32` is inserted between the compare and the cndmask for address computation, it overwrites VCC:

```asm
v_cmp_gt_i32 vcc, v_col, v_row    ; sets VCC for causal mask
; ... address computation ...
v_add_co_u32 v_addr, vcc, v_base, v_offset  ; CLOBBERS VCC!
v_cndmask_b32 v_val, v_neg_inf, v_val, vcc  ; reads WRONG VCC
```

**Fix:** Use `v_add_u32` (no carry) for address computation when VCC is live, or save/restore VCC around the carry-out add:

```asm
; Option 1: use v_add_u32 instead (preferred)
v_add_u32 v_addr, v_base, v_offset   ; no VCC clobber

; Option 2: use VOP3 with explicit carry dest
v_add_co_u32_e64 v_addr, s[tmp:tmp+1], v_base, v_offset  ; carry to s_tmp, not VCC
```

### 64-bit address computation

`v_add_co_u32` is required for the low half of 64-bit address arithmetic (where carry propagation matters). Always pair with `v_addc_co_u32` for the high half:

```asm
v_add_co_u32 v_addr_lo, vcc, v_base_lo, v_offset_lo
v_addc_co_u32 v_addr_hi, vcc, v_base_hi, 0, vcc  ; propagate carry
```

## Common Usage Patterns

### 64-bit pointer arithmetic
```asm
v_add_co_u32 v0, vcc, v0, s_offset_lo
v_addc_co_u32 v1, vcc, v1, s_offset_hi, vcc
```

### When to use v_add_u32 instead
```asm
; For 32-bit offsets where carry is irrelevant:
v_add_u32 v_offset, s_stride, v_offset   ; no VCC clobber
```

## Sources

- Bounds checking VCC clobber hazard
- Attention kernel address computation patterns
- gfx950-reference.md: NOP/hazard cheat sheet
