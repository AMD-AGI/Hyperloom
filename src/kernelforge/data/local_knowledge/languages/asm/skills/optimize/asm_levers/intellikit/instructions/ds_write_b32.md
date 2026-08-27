---
instruction: ds_write_b32
category: memory
architecture: gfx950
tags: [ds_write_b32, lgkmcnt, LDS, bank-conflict, s_barrier]
---

# ds_write_b32

## Opcode

`ds_write_b32 vAddr, vS offset:N`

Writes 32 bits (4 bytes, 1 dword) from a single VGPR to LDS at address `vAddr + offset`.

## Cycle Counts (Measured on MI355X)

| Measurement | CPI | Notes |
|-------------|-----|-------|
| db.json (stride1) | ~3.9 | 1024 iters, latency mode |
| Earlier measurement | ~13.6 | Older methodology |

## Counter

**lgkmcnt**. Same as all ds_write variants.

## FIFO Ordering

lgkmcnt FIFO -- shared with all LDS/SMEM operations.

## Coherence Flags

None (LDS is CU-local).

## Known Hazards

### 1. Bank Conflict Penalty (Severe for Scatter Patterns)
ds_write_b32 with scatter addressing patterns causes severe LDS bank conflicts. In a rmsnorm kernel, `ds_write_b16` scatter caused 91% of all wait cycles. The cost scales linearly with conflict degree.

### 2. lgkmcnt(0) Before s_barrier
Same as all ds_write variants.

## Known Bugs

None specific.

## Alignment Requirements

- No VGPR alignment requirement for source (single register)
- Offset field: 0-65535 bytes
- Effective LDS address should be 4-byte aligned

## LDS Variant

This IS an LDS instruction.

## Performance Notes

- Narrowest write -- use ds_write_b128 or ds_write_b64 where possible for better throughput.
- Bank conflict cost is the same per-dword regardless of write width, but wider writes (b128) amortize the LDS access overhead.

## Common Patterns

### LDS Reduction Write
```asm
; Write partial sum to LDS for tree reduction
ds_write_b32 v_lds_base, v0 offset:0
s_waitcnt lgkmcnt(0)
s_barrier
; Other waves can now read this value
```
