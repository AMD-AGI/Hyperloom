---
instruction: v_perm_b32
category: vector
architecture: gfx950
tags: [v_perm_b32, byte-permute, BF16, packing]
---

# v_perm_b32

Byte permutation: rearranges bytes from two source registers according to a selector constant.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP3 |
| Throughput | ~1 shader cycle (standard VALU issue rate) |
| Category | VALU bitwise |

## Hazards

Standard VALU forwarding latency (5 cycles). No special hazards.

## Known Bugs / Gotchas

### Used for FP32-to-BF16 packing in optimized kernels

The backward attention kernel uses `v_perm_b32` with a selector constant in an SGPR (s64) to pack FP32 softmax output (P matrix) to BF16 for MFMA consumption:

```asm
v_perm_b32 v164, v53, v52, s64   ; pack two FP32 -> BF16 pair
v_perm_b32 v165, v55, v54, s64
; ... 12 total v_perm_b32 per block
```

### Register pressure spike from batch computation

The BWD attention kernel precomputes all 12 perm results (v164-v175) before any MFMA consumption, creating a 12-VGPR pressure spike. Computing perm results just-in-time (2 at a time, interleaved with MFMA consumption) reduces peak pressure by ~8 VGPRs.

### Clobbers live data in destination range

In BWD attention, v_perm_b32 writes to v[164:175], which overlaps with K data that was moved from AGPRs for 16x16x32 MFMA conversion. The perm output silently clobbers the K tile data. Solution: reload K data from LDS before the next GEMM0 block.

## Common Usage Patterns

### BF16 pair packing (attention kernels)
```asm
; s64 holds the packing selector constant
; Pack pairs of FP32 values from softmax P output:
v_perm_b32 v_packed0, v_hi0, v_lo0, s_selector
v_perm_b32 v_packed1, v_hi1, v_lo1, s_selector
; Results used as MFMA BF16 source operands
```

### General byte rearrangement
```asm
; Selector byte values:
;   0-3: select byte 0-3 from S1 (first source)
;   4-7: select byte 0-3 from S0 (second source)
;   0x0C: force byte to 0x00
;   0x0D: force byte to 0xFF
v_perm_b32 v_dst, v_src0, v_src1, s_selector
```

## Sources

- BWD attention v_perm_b32 scheduling optimization, register pressure analysis
- BF16 packing constant identification (v229, v230, v231)
