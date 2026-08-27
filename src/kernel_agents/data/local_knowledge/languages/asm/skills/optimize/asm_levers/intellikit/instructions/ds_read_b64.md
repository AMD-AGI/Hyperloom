---
instruction: ds_read_b64
category: memory
architecture: gfx950
tags: [ds_read_b64, lgkmcnt, LDS, VGPR-clobber]
---

# ds_read_b64

## Opcode

`ds_read_b64 v[D:D+1], vAddr offset:N`

Reads 64 bits (8 bytes, 2 dwords) from LDS at address `vAddr + offset` into 2 consecutive VGPRs.

## Cycle Counts (Measured on MI355X)

| Measurement | CPI | Notes |
|-------------|-----|-------|
| db.json (stride1) | ~3.5 | 512 iters, latency mode |
| Earlier measurement | ~6.1 | Older methodology |

## Counter

**lgkmcnt**. Same semantics as ds_read_b128.

## FIFO Ordering

lgkmcnt FIFO -- oldest first. Same rules as ds_read_b128. All ds_read and ds_write operations share the same lgkmcnt FIFO.

## Coherence Flags

None applicable (LDS is CU-local).

## Known Hazards

### 1. Destination VGPR Clobber
Same as ds_read_b128 but with 2 VGPRs instead of 4. If v[D:D+1] holds live data, it will be overwritten when the read completes.

### 2. Address Self-Clobber
Same as ds_read_b128 -- if vAddr is within the destination range, it will be overwritten.

## Known Bugs

None specific.

## Alignment Requirements

- Destination: even-numbered VGPR pair (e.g., v[4:5], v[146:147])
- Offset field: 0-65535 bytes
- Effective LDS address should be 8-byte aligned for optimal performance

## LDS Variant

This IS an LDS instruction.

## Performance Notes

- Used by BWD attention kernels for Q/K/V operand loading alongside ds_read_b128.
- Also used with ds_read2_b64 for non-contiguous access patterns.
- Same MFMA co-execution hiding applies -- latency is free in MFMA-heavy loops.

## Common Patterns

### BWD Attention Operand Load
```asm
ds_read_b64 v[0:1], v_lds_base offset:0     ; load 2 dwords
ds_read_b64 v[2:3], v_lds_base offset:512   ; next chunk
s_waitcnt lgkmcnt(0)
; Use v[0:3] as MFMA operand
```
