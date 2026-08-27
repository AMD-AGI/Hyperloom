---
instruction: flat_store
category: memory
architecture: gfx950
tags: [flat_store, vmcnt, coherence, flat-vs-global, s_endpgm]
---

# flat_store Family

## Variants

| Opcode | Width | VGPRs | Counter |
|--------|-------|-------|---------|
| `flat_store_dword` | 32-bit | 1 VGPR data | vmcnt |
| `flat_store_dwordx2` | 64-bit | 2 VGPRs data | vmcnt |
| `flat_store_dwordx4` | 128-bit | 4 VGPRs data | vmcnt |

## Cycle Counts (Measured on MI355X)

| Variant | Mode | CPI | Notes |
|---------|------|-----|-------|
| `flat_store_dword` | throughput | ~0.34 | Fire-and-forget, 1024 iters |

## Counter

**vmcnt**. Same FIFO as all vmcnt operations.

## FIFO Ordering

vmcnt FIFO -- same as global_store.

## Coherence Flags

| Flag | Effect |
|------|--------|
| `sc0` | System coherent level 0 -- ensures L2 visibility |
| `sc1` | System coherent level 1 |
| `sc0 sc1` | Fully system-coherent for cross-CU/cross-kernel visibility |

For inter-workgroup barriers: use `flat_store sc0 sc1` to ensure data is visible to all CUs.

## Known Hazards

### 1. s_waitcnt vmcnt(0) Before s_endpgm
Same as all store types. MANDATORY on gfx950.

### 2. flat vs global Coherence Mismatch
Same as flat_load. Never mix flat_store with global_load for the same memory across kernels.

## Known Bugs

None specific.

## Alignment Requirements

- Data VGPRs: Even-numbered for dwordx2/dwordx4
- Address: 64-bit VGPR pair (even-aligned)

## LDS Variant

flat_store can write to LDS if the address falls in the LDS range, but prefer ds_write for explicit LDS access.

## Performance Notes

- Fire-and-forget, very fast issue (~0.34 CPI throughput).
- Used in inter-WG barrier patterns for publishing shared state.

## Common Patterns

### Inter-Workgroup Barrier Data Publish
```asm
; Publish reduction result visible to all CUs:
flat_store_dword v[8:9], v0 sc0 sc1     ; write with full coherence
s_waitcnt vmcnt(0)
; Now other CUs can see this value via flat_load sc0 sc1
```
