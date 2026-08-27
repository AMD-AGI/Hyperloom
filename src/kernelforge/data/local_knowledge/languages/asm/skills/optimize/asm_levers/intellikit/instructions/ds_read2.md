---
instruction: ds_read2
category: memory
architecture: gfx950
tags: [ds_read2, lgkmcnt, LDS, non-contiguous, bank-conflict]
---

# ds_read2 Family

## Variants

| Opcode | Width per element | Total | Offset unit | Counter |
|--------|-------------------|-------|-------------|---------|
| `ds_read2_b32` | 32-bit | 64-bit (2 dwords) | 4 bytes | lgkmcnt |
| `ds_read2_b64` | 64-bit | 128-bit (4 dwords) | 8 bytes (qwords) | lgkmcnt |
| `ds_read2st64_b32` | 32-bit | 64-bit | 256 bytes (64*4) | lgkmcnt |
| `ds_read2st64_b64` | 64-bit | 128-bit | 512 bytes (64*8) | lgkmcnt |

## Syntax

```asm
ds_read2_b32  v[D:D+1], vAddr offset0:N offset1:M
ds_read2_b64  v[D:D+3], vAddr offset0:N offset1:M
ds_read2st64_b32 v[D:D+1], vAddr offset0:N offset1:M
ds_read2st64_b64 v[D:D+3], vAddr offset0:N offset1:M
```

Reads two non-contiguous elements from LDS in a single instruction. The two offsets are applied independently to vAddr.

For `ds_read2_b32`: effective addresses are `vAddr + offset0*4` and `vAddr + offset1*4`.
For `ds_read2_b64`: effective addresses are `vAddr + offset0*8` and `vAddr + offset1*8`.
For `st64` variants: multiply the unit by 64 (e.g., `ds_read2st64_b64` uses `offset*512`).

Offset fields are 8-bit unsigned (0-255).

## Cycle Counts (Measured on MI355X)

| Variant | CPI | Notes |
|---------|-----|-------|
| `ds_read2_b32` | ~3.0 | 1024 iters, latency mode |

No separate measurement for ds_read2_b64; expect similar to ds_read_b128 (~3.0 CPI) since both read 128 bits.

## Counter

**lgkmcnt**. Each ds_read2 instruction counts as ONE lgkmcnt increment, despite reading two separate addresses. This is the key advantage -- one lgkmcnt slot for two data fetches.

## FIFO Ordering

lgkmcnt FIFO. Each ds_read2 is a single entry in the FIFO.

## Coherence Flags

None (LDS is CU-local).

## Known Hazards

### 1. Destination VGPR Clobber
Same as other ds_read variants. The destination range covers:
- ds_read2_b32: 2 VGPRs
- ds_read2_b64: 4 VGPRs

### 2. Bank Conflicts Between the Two Accesses
The two offsets within a single ds_read2 can conflict with each other if they map to the same LDS bank. This creates a within-instruction bank conflict, adding latency.

## Known Bugs

None specific.

## Alignment Requirements

- `ds_read2_b32`: destination is an even-numbered VGPR pair
- `ds_read2_b64`: destination is an even-numbered VGPR (base of 4)
- Offsets are in units of the element size (4 bytes for b32, 8 bytes for b64)
- Maximum offset: 255 (in element-size units)

## LDS Variant

This IS an LDS instruction.

## Performance Notes

- **Non-contiguous LDS access**: Use ds_read2 when the two data elements are at different, non-adjacent LDS addresses. This is more efficient than two separate ds_read_b32/b64 instructions because it uses only one lgkmcnt slot.
- **Attention pattern**: BWD kernels use ds_read2_b64 to load Q/K data from non-contiguous LDS locations in a single instruction.

## Common Patterns

### Load Two Non-Adjacent LDS Rows
```asm
; Read row 0 and row 64 of LDS tile (stride-64 pattern)
ds_read2st64_b64 v[0:3], v_lds_base offset0:0 offset1:1
s_waitcnt lgkmcnt(0)
; v[0:1] = data from vAddr + 0*512
; v[2:3] = data from vAddr + 1*512
```

### K/V Staged Load After Permute
```asm
; After v_perm_b32 byte shuffle, write two elements to non-contiguous LDS
ds_write2_b32 v_lds, v0, v1 offset0:0 offset1:64
s_barrier
; Read them back with ds_read2_b32
ds_read2_b32 v[4:5], v_lds offset0:0 offset1:64
```
