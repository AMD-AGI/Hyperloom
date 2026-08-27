---
instruction: ds_write2
category: memory
architecture: gfx950
tags: [ds_write2, lgkmcnt, LDS, non-contiguous, bank-conflict]
---

# ds_write2 Family

## Variants

| Opcode | Width per element | Total | Offset unit | Counter |
|--------|-------------------|-------|-------------|---------|
| `ds_write2_b32` | 32-bit | 64-bit (2 dwords) | 4 bytes | lgkmcnt |
| `ds_write2_b64` | 64-bit | 128-bit (4 dwords) | 8 bytes (qwords) | lgkmcnt |
| `ds_write2st64_b32` | 32-bit | 64-bit | 256 bytes (64*4) | lgkmcnt |
| `ds_write2st64_b64` | 64-bit | 128-bit | 512 bytes (64*8) | lgkmcnt |

## Syntax

```asm
ds_write2_b32  vAddr, vS0, vS1 offset0:N offset1:M
ds_write2_b64  vAddr, v[S0:S0+1], v[S1:S1+1] offset0:N offset1:M
ds_write2st64_b32 vAddr, vS0, vS1 offset0:N offset1:M
ds_write2st64_b64 vAddr, v[S0:S0+1], v[S1:S1+1] offset0:N offset1:M
```

Writes two non-contiguous elements to LDS in a single instruction. The two offsets are applied independently to vAddr.

For `ds_write2_b32`: effective addresses are `vAddr + offset0*4` and `vAddr + offset1*4`.
For `ds_write2_b64`: effective addresses are `vAddr + offset0*8` and `vAddr + offset1*8`.
For `st64` variants: multiply the unit by 64.

Offset fields are 8-bit unsigned (0-255).

## Cycle Counts (Measured on MI355X)

| Variant | Mode | CPI | Notes |
|---------|------|-----|-------|
| `ds_write2_b32` | throughput | ~0.93 | 1024 iters, very fast throughput |

## Counter

**lgkmcnt**. Each ds_write2 instruction counts as ONE lgkmcnt increment, despite writing to two separate addresses.

## FIFO Ordering

lgkmcnt FIFO. Single entry per instruction.

## Coherence Flags

None (LDS is CU-local). Cross-wavefront visibility requires lgkmcnt(0) + s_barrier.

## Known Hazards

### 1. lgkmcnt(0) Before s_barrier
Same as all ds_write variants.

### 2. Bank Conflicts Between Two Writes
The two offset addresses within a single ds_write2 can conflict with each other if they map to the same LDS bank.

## Known Bugs

None specific.

## Alignment Requirements

- `ds_write2_b32`: source registers are two separate single VGPRs
- `ds_write2_b64`: source registers are two separate even-numbered VGPR pairs
- Offsets are in units of the element size (4 bytes for b32, 8 bytes for b64)
- Maximum offset: 255

## LDS Variant

This IS an LDS instruction.

## Performance Notes

- **Efficient for non-contiguous LDS writes**: One lgkmcnt slot for two writes. More efficient than two separate ds_write_b32/b64 instructions.
- **GGEMM LDS staging**: The wgrad kernel uses `ds_write2st64_b64` to write buffer_load results into the double-buffered LDS tile structure with strided offsets.
- **K/V pattern**: After `v_perm_b32` byte shuffle, `ds_write2_b32` writes permuted K/V data to two non-contiguous LDS locations.

## Common Patterns

### GGEMM Wgrad LDS Stage
```asm
; Write LHS buffer_load data to LDS with stride-64 layout
s_waitcnt vmcnt(2)
ds_write2st64_b64 v196, v[128:129], v[130:131] offset0:0 offset1:1
; Write RHS buffer_load data
s_waitcnt vmcnt(0)
ds_write2st64_b64 v196, v[216:217], v[220:221] offset0:4 offset1:5
```

### Attention K/V Store
```asm
; After v_perm_b32 byte repack:
ds_write2_b32 v_lds_kv, v0, v1 offset0:0 offset1:64
```
