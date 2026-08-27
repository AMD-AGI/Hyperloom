---
instruction: v_cvt_pk_bf16_f32
category: vector
architecture: gfx950
tags: [v_cvt_pk_bf16_f32, BF16, conversion, packing]
---

# v_cvt_pk_bf16_f32

Pack two FP32 values into a single BF16 pair. Native BF16 conversion with round-to-nearest-even.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP3 (op 0x268) |
| Measured throughput | 0.270 ref-clk CPI (median 276 ticks / 1024 iters) |
| Normalized throughput | ~1.6 shader cycles |
| Category | VALU conversion |

## Hazards

None specific to this instruction beyond standard VALU forwarding latency (5 cycles to dependent consumer).

## Known Bugs / Gotchas

### MANDATORY on gfx950 -- manual BF16 packing produces garbage

Manual BF16 packing via `v_lshrrev_b32 v, 16, v` (extracting upper 16 bits of F32 as BF16 truncation) produces **inf/NaN garbage** on gfx950. You MUST use the native conversion instruction.

```asm
; WRONG -- produces inf/NaN on gfx950:
v_lshrrev_b32 v10, 16, v10    ; "truncate" F32 to BF16
v_and_or_b32 v_packed, v11_shifted, v10  ; pack pair

; CORRECT:
v_cvt_pk_bf16_f32 v_packed, v10, v11
```

### Packing order: S0 = LOW bits, S1 = HIGH bits

`v_cvt_pk_bf16_f32 dst, S0, S1` places:
- S0 in the **LOW** 16 bits of dst
- S1 in the **HIGH** 16 bits of dst

This is the **opposite** of what many people assume. Multiple agents independently wasted hours on row-pair swaps caused by getting the operand order wrong.

### Replaces 10-instruction software rounding sequence

The compiler emits a 10-instruction RTNE software rounding pattern per F32 pair:
```
v_cmp_u_f32    (NaN check)
v_bfe_u32      (extract rounding bit)
v_add3_u32     (add rounding bias 0x7fff)
v_cndmask_b32  (NaN -> 0x7fff0000)
v_lshrrev_b32  (shift)
; repeat for second value
v_and_or_b32   (pack)
```

Single `v_cvt_pk_bf16_f32` replaces all 10 instructions. In BWD attention, this replaced 480 instructions (48 blocks x 10) with 48 instructions -- a 10% instruction count reduction. Also frees 3 constant VGPRs (v229=0xffff0000, v230=0x7fff0000, v231=0x7fff).

### "rtz" label is actually RTNE

Kernels labeled "rtz" (round-to-zero) actually implement RTNE in software. The `v_bfe_u32 + v_add3_u32` pattern is round-to-nearest-even. True RTZ would be a single `v_lshrrev_b32 v, 16, v`. The native `v_cvt_pk_bf16_f32` also does RTNE, so replacing the software sequence is semantically correct.

## Common Usage Patterns

### MFMA accumulator store (BF16 output)
```asm
; After MFMA drain, read accumulators and pack to BF16:
v_accvgpr_read_b32 v10, a0
v_accvgpr_read_b32 v11, a1
s_nop 1
s_nop 1
v_cvt_pk_bf16_f32 v10, v10, v11    ; pack a[0:1] into BF16 pair
global_store_dword v[addr], v10, off
```

### FP8 GEMM store path (64 packs)
```asm
; Scale FP32 accumulators, then pack to BF16:
v_pk_mul_f32 v[0:1], v[0:1], s[scale:scale+1]  ; apply scale
v_cvt_pk_bf16_f32 v0, v0, v1                     ; pack pair
buffer_store_dwordx2 v[0:1], ...                  ; store BF16
```

## Sources

- Mandatory native conversion discovery (manual packing -> inf/NaN)
- BWD attention BF16 optimization (480 -> 48 instructions)
- Discovered during production kernel development: packing order confusion (agents 1, 4, 8)
- isa-bench: valu_pk_bf16_f32 throughput kernel
