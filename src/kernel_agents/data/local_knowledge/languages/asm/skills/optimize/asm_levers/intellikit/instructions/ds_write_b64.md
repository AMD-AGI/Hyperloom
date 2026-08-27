---
instruction: ds_write_b64
category: memory
architecture: gfx950
tags: [ds_write_b64, lgkmcnt, LDS, s_barrier]
---

# ds_write_b64

## Opcode

`ds_write_b64 vAddr, v[S:S+1] offset:N`

Writes 64 bits (8 bytes, 2 dwords) from 2 consecutive VGPRs to LDS at address `vAddr + offset`.

## Cycle Counts (Measured on MI355X)

| Measurement | CPI | Notes |
|-------------|-----|-------|
| db.json (stride1) | ~3.6 | 512 iters, latency mode |

## Counter

**lgkmcnt**. Same as ds_write_b128.

## FIFO Ordering

lgkmcnt FIFO -- shared with all LDS/SMEM operations.

## Coherence Flags

None (LDS is CU-local). Requires `s_barrier` for cross-wavefront visibility.

## Known Hazards

Same as ds_write_b128:
1. lgkmcnt(0) required before s_barrier
2. Write-read ordering requires lgkmcnt drain + barrier

## Known Bugs

None specific.

## Alignment Requirements

- Source VGPRs: even-numbered VGPR pair
- Offset field: 0-65535 bytes
- Effective LDS address should be 8-byte aligned

## LDS Variant

This IS an LDS instruction.

## Performance Notes

- Less commonly used than ds_write_b128 in GEMM kernels (prefer wider writes for throughput).
- Used in some attention kernel patterns for partial writes.

## Common Patterns

### Paired with ds_write2st64_b64
See ds_write2 family for the strided variant that handles non-contiguous writes.
