---
instruction: s_movk_i32
category: scalar
architecture: gfx950
tags: [s_movk_i32, SALU, sign-extension, hazard]
---

# s_movk_i32

Move a 16-bit inline constant into an SGPR. The constant is sign-extended to 32 bits.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | SOPK |
| Throughput | ~1 shader cycle (standard SALU issue rate) |
| Category | Scalar ALU |

## Hazards

None specific to this instruction.

## Known Bugs / Gotchas

### SIGN-EXTENDS values >= 0x8000

`s_movk_i32` sign-extends the 16-bit immediate to 32 bits. Values >= 0x8000 (32768 decimal) become negative:

```asm
; WRONG -- 32768 becomes -32768:
s_movk_i32 s10, 0x8000    ; s10 = 0xFFFF8000 = -32768 (NOT +32768!)

; CORRECT -- use s_mov_b32 for values >= 0x8000:
s_mov_b32 s10, 0x8000     ; s10 = 0x00008000 = +32768
```

This bug was discovered when computing LDS offsets. An LDS offset of 32768 (for the second half of a 64KB double-buffer) was loaded via `s_movk_i32`, producing a negative value that caused the LDS access to wrap or fault.

### Safe range: 0x0000 to 0x7FFF

Only use `s_movk_i32` for values in the range [0, 32767] (0x0000 to 0x7FFF). For larger values, use `s_mov_b32` which can encode a full 32-bit literal.

### Why s_movk_i32 exists

`s_movk_i32` is a compact encoding (4 bytes) compared to `s_mov_b32` with a 32-bit literal (8 bytes). The assembler may automatically choose `s_movk_i32` when the immediate fits in 16 bits. This is correct for signed values but surprising for unsigned values >= 0x8000.

## Common Usage Patterns

### Small loop strides and constants (safe)
```asm
s_movk_i32 s10, 64        ; s10 = 64 (safe, < 0x8000)
s_movk_i32 s11, 0x440     ; s11 = 1088 (LDS stride, safe)
```

### Large constants (use s_mov_b32)
```asm
s_mov_b32 s10, 0x8000     ; s10 = 32768 (must use s_mov_b32)
s_mov_b32 s11, 0x10000    ; s11 = 65536
s_mov_b32 s12, 0xff800000 ; s12 = -inf (FP32)
```

## Sources

- Sign-extension trap discovery
- Confirmed s_movk sign-extension hazard
- gfx950-reference.md: NOP/hazard cheat sheet
