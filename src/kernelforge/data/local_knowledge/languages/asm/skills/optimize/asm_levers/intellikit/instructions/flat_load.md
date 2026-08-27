---
instruction: flat_load
category: memory
architecture: gfx950
tags: [flat_load, vmcnt, coherence, flat-vs-global, sc0]
---

# flat_load Family

## Variants

| Opcode | Width | VGPRs | Counter |
|--------|-------|-------|---------|
| `flat_load_dword` | 32-bit | 1 VGPR dest | vmcnt |
| `flat_load_dwordx2` | 64-bit | 2 VGPRs dest | vmcnt |
| `flat_load_dwordx4` | 128-bit | 4 VGPRs dest | vmcnt |

## Cycle Counts (Measured on MI355X)

| Variant | Mode | CPI | Notes |
|---------|------|-----|-------|
| `flat_load_dword` | latency | ~5.3 | 256 iters |
| `flat_load_dwordx4` | latency | ~5.8 | 256 iters |

Comparable to global_load performance (~6-7 CPI for L1 hit). Practical HBM latency ~300-400 cycles.

## Counter

**vmcnt**. Flat loads count against the same vmcnt counter as global_load and buffer_load.

## FIFO Ordering

vmcnt FIFO -- same as global_load. Oldest issued load drains first.

## Coherence Flags

| Flag | Effect |
|------|--------|
| (none) | Standard L1-cached load |
| `sc0` | System coherent level 0 -- bypasses L1 TCP |
| `sc1` | System coherent level 1 -- bypasses L2 |
| `sc0 sc1` | Fully system-coherent |

For inter-workgroup barriers:
- **Loads**: Use `flat_load sc0 sc1` to read from memory (bypass both caches)
- **Stores**: Use `flat_store sc0` or `flat_store sc0 sc1`

The inter-WG barrier pattern specifically requires: `flat_atomic sc0` for the atomic counter + `flat_load/store sc0 sc1` for data coherence. This achieves ~50ns per WG per barrier.

## Known Hazards

### 1. flat vs global Coherence Mismatch (CRITICAL)
Using `flat_store` in one kernel and `global_load` in the next kernel (or vice versa) to access the same memory causes stale reads. The flat and global instruction families use DIFFERENT cache coherence paths on gfx950.

**Symptom**: Reads return old data. Non-deterministic corruption.

**Rule**: ALWAYS use the same instruction family (flat OR global, never mixed) for the same memory region across kernels.

Empirically validated on MI355X.

### 2. WAW Hazard
Same as global_load -- destination VGPRs are written asynchronously when data arrives.

## Known Bugs

### flat_atomic_add_f32 Does NOT Support sc0/sc1
The `flat_atomic_add_f32` instruction does not accept sc0/sc1 coherence modifiers. This is a hardware limitation. See the flat_atomic card for details.

## Alignment Requirements

- `flat_load_dwordx4`: Even-numbered VGPR dest base
- `flat_load_dwordx2`: Even-numbered VGPR pair
- Address: 64-bit VGPR pair (even-aligned)
- Memory address should be naturally aligned

## LDS Variant

flat instructions can access both global memory and LDS (the hardware disambiguates based on the address range). However, for explicit LDS access, use ds_read/ds_write instructions instead.

## Performance Notes

- Performance is comparable to global_load. Choose based on coherence needs.
- flat is required when accessing memory that might be in either global or LDS address space (generic pointers).
- For known-global access in GEMM kernels, prefer `global_load` (simpler, well-characterized performance).

## Common Patterns

### Inter-Workgroup Barrier Data Load
```asm
; After atomic counter indicates all WGs have arrived:
flat_load_dword v0, v[8:9] sc0 sc1      ; load shared data, bypass caches
s_waitcnt vmcnt(0)
; v0 now has the latest cross-WG visible value
```

### Generic Pointer Load
```asm
; When the pointer might be global or LDS:
flat_load_dwordx4 v[0:3], v[8:9]
s_waitcnt vmcnt(0)
```
