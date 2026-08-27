---
instruction: v_accvgpr_read_b32
category: AGPR
architecture: gfx950
tags: [v_accvgpr_read_b32, AGPR, NOP, MFMA, accum_offset]
---

# v_accvgpr_read_b32

Read a single AGPR (accumulator VGPR) value into a regular VGPR.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | VOP3 |
| Measured round-trip (write+read) CPI | 1.07 ref-clk (median 274 ticks / 256 iters) |
| Normalized round-trip latency | ~6.2 shader cycles (write + read combined) |
| Category | AGPR access |

## Hazards

### MFMA -> v_accvgpr_read_b32: needs s_nop 15 x 2

After an MFMA instruction writes to accumulators, reading the result via `v_accvgpr_read_b32` requires two `s_nop 15` instructions (32 NOPs total). There is NO hardware interlock for this hazard.

```asm
v_mfma_f32_32x32x16_bf16 a[0:15], v[src_a], v[src_b], a[0:15]
s_nop 15          ; MANDATORY -- 16 cycles
s_nop 15          ; MANDATORY -- 16 more cycles (32 total)
v_accvgpr_read_b32 v10, a0    ; NOW safe to read
```

Without the NOPs, `v_accvgpr_read_b32` returns stale (pre-MFMA) accumulator values.

### Avoiding AGPRs entirely

The forward attention kernel uses zero AGPRs. All MFMA results are written to regular VGPRs. This avoids the expensive `s_nop 15; s_nop 15` drain sequence entirely. The tradeoff is that VGPR-destination MFMAs have different register pressure characteristics.

## Known Bugs / Gotchas

### accum_offset determines AGPR boundary

VGPRs at index >= `accum_offset` alias AGPRs. If `accum_offset` is set incorrectly in the kernel descriptor, regular VGPR reads above that boundary return AGPR data (garbage). The `.amdhsa_accum_offset` must be set to at least the highest VGPR index used + 1, or match the total VGPR count when no AGPRs are needed.

```
accum_offset = 224:
  v[0:223] = regular VGPRs
  v[224+]  = alias AGPRs (a[0], a[1], ...)
```

### next_free_vgpr must cover accum_offset + n_agprs

The kernel descriptor field `.amdhsa_next_free_vgpr` must be set to at least `accum_offset + n_agprs`, not just the count of regular VGPRs. If this is wrong, the hardware does not allocate enough register file entries and reads/writes to AGPRs corrupt other state.

### BWD attention store pattern

In backward attention kernels using AGPRs as MFMA accumulators, the store path reads accumulators to VGPRs for BF16 conversion:

```asm
v_accvgpr_read_b32 v10, a0
v_accvgpr_read_b32 v11, a1
s_nop 1                         ; gap for read settling
s_nop 1                         ; additional gap
v_cvt_pk_bf16_f32 v10, v10, v11  ; pack to BF16
global_store_dword v[addr], v10, off
```

## Common Usage Patterns

### MFMA accumulator drain (AGPR path)
```asm
; After MFMA chain completes:
s_nop 15
s_nop 15
v_accvgpr_read_b32 v0, a0
v_accvgpr_read_b32 v1, a1
; ... read all needed accumulators
; Then convert/store
```

### Avoid when possible
```asm
; PREFERRED: use VGPR destinations in MFMA:
v_mfma_f32_32x32x16_bf16 v[0:15], v[src_a], v[src_b], v[0:15]
; No accvgpr_read needed -- results already in VGPRs
```

## Sources

- AGPR count bug, accum_offset aliasing
- MFMA NOP requirements (s_nop 15 x 2)
- BWD attention MFMA drain timing
- Optimized kernels use zero AGPRs to avoid this hazard
- isa-bench: agpr_write_read round-trip latency kernel
