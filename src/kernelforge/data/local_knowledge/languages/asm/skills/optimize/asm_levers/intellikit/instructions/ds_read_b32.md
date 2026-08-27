---
instruction: ds_read_b32
category: memory
architecture: gfx950
tags: [ds_read_b32, lgkmcnt, LDS, bank-conflict, broadcast]
---

# ds_read_b32

## Opcode

`ds_read_b32 vD, vAddr offset:N`

Reads 32 bits (4 bytes, 1 dword) from LDS at address `vAddr + offset` into a single VGPR.

## Cycle Counts (Measured on MI355X)

| Measurement | CPI | Notes |
|-------------|-----|-------|
| db.json broadcast | ~2.9 | All lanes same address, FREE |
| db.json stride1 | ~3.5 | Stride-1 access, no conflict |
| db.json stride4 | ~3.2 | 4-byte stride |
| db.json stride16 | ~4.4 | 16-byte stride |
| db.json stride64 | ~5.2 | 64-byte stride |
| db.json 2-way conflict | ~3.5 | 2-way bank conflict |
| db.json 4-way conflict | ~4.4 | 4-way bank conflict |
| db.json 16-way conflict | ~5.8 | 16-way bank conflict |
| db.json segment conflict | ~7.0 | Segment-level conflict |
| Earlier measurement | ~12.2 | Older methodology |

### Bank Conflict Analysis

gfx950 has **64 LDS banks** (NOT 32 as on MI300X). Bank conflict cost scales approximately linearly:
- No conflict: ~3 CPI
- 2-way: ~3.5 CPI (+0.5)
- 4-way: ~4.4 CPI (+1.4)
- 16-way: ~5.8 CPI (+2.8)

Broadcast (all lanes same bank): FREE -- same cost as no-conflict.

## Counter

**lgkmcnt**. Same semantics as other ds_read variants.

## FIFO Ordering

lgkmcnt FIFO -- same as ds_read_b128.

## Coherence Flags

None applicable (LDS is CU-local).

## Known Hazards

### 1. Destination VGPR Clobber
Single VGPR destination. If vD holds a live value, it will be overwritten.

## Known Bugs

None specific.

## Alignment Requirements

- No VGPR alignment requirement for destination (single register)
- Offset field: 0-65535 bytes
- LDS address should be 4-byte aligned

## LDS Variant

This IS an LDS instruction.

## Performance Notes

- **Bank conflicts from scatter patterns**: In a rmsnorm kernel, `ds_write_b16` scatter caused 91% of all wait cycles due to bank conflicts. The 64-bank structure on gfx950 helps compared to 32-bank MI300X, but scatter patterns can still create severe conflicts.
- **Swizzle variant**: `ds_read_b32` + `ds_swizzle` has ~2.9 CPI (similar to broadcast), suggesting the swizzle logic adds minimal overhead.

## Common Patterns

### Scalar Broadcast from LDS
```asm
; All lanes read the same LDS address (broadcast = free)
ds_read_b32 v0, v_lds_base offset:0
s_waitcnt lgkmcnt(0)
; v0 now holds the broadcast value in all lanes
```

### LDS Reduction Tree
```asm
; Each lane reads its neighbor's value for tree reduction
ds_read_b32 v1, v_neighbor_addr offset:0
s_waitcnt lgkmcnt(0)
v_add_f32 v0, v0, v1     ; accumulate
```
