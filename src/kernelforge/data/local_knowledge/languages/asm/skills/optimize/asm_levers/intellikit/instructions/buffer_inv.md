---
instruction: buffer_inv
category: memory
architecture: gfx950
tags: [buffer_inv, cache-invalidation, TCP, vL1, coherence]
---

# buffer_inv

## Opcode

`buffer_inv`

Invalidates the TCP (vL1) cache. All subsequent loads from this CU will fetch fresh data from L2/HBM rather than returning potentially stale L1 cache entries.

No operands. No VGPR/SGPR inputs or outputs.

## Cycle Counts

Not directly measured in db.json. Expected to be a few cycles (SALU-class instruction). The performance impact comes from the cache misses that follow, not the instruction itself.

## Counter

Does not increment any counter. buffer_inv is a cache management instruction, not a memory access.

## FIFO Ordering

Not applicable. buffer_inv does not participate in the vmcnt or lgkmcnt FIFO.

## Coherence Flags

None. The instruction unconditionally invalidates the entire TCP (vL1) cache.

## Known Hazards

### 1. HARMFUL at Kernel Start
Issuing `buffer_inv` at the beginning of a kernel is actively harmful. The Triton compiler sometimes inserts this as a "safety" measure, but on gfx950 it causes NaN output in some kernels.

**Root cause**: The TCP cache may contain valid data from the current kernel's argument loading (kernarg preload). Invalidating it forces re-fetches that may race with other memory operations.

Empirically validated on MI355X.

### 2. Not Needed for Single-Stream Pipelines
For kernels in a single-stream execution pipeline (no concurrent kernels accessing the same memory), buffer_inv provides no benefit. The cache is naturally consistent within a single stream.

## Known Bugs

### Causes NaN When Used at Kernel Start
Removing the Triton-inserted `buffer_inv` at kernel entry fixes NaN output. The instruction destroys valid kernarg data in the L1 cache.

## Alignment Requirements

None (no operands).

## LDS Variant

None. buffer_inv affects only the TCP (vL1) cache for global/buffer memory. LDS is a separate memory and is not affected.

## Performance Notes

- **Remove it unless you have a specific multi-stream coherence need.** In all observed gfx950 assembly kernels (GEMM, attention, rmsnorm), buffer_inv was either harmful or unnecessary.
- If you need fresh data from L2, use `sc0` on individual loads instead of invalidating the entire cache.
- The cost is not the instruction itself but the subsequent cache misses (every load after buffer_inv will miss L1).

## Common Patterns

### When to Use (Rare)
```asm
; Only when switching between independent memory regions in a persistent kernel
; where the previous iteration's L1 contents are guaranteed stale:
buffer_inv
s_nop 0                    ; settling time (may be needed)
buffer_load_dwordx4 ...    ; now fetches from L2/HBM
```

### When NOT to Use (Common)
```asm
; DON'T do this at kernel entry (Triton sometimes generates it):
; buffer_inv    ; <-- REMOVE THIS, causes NaN
s_load_dwordx4 s[0:3], s[0:1], 0x0   ; kernarg load
```
