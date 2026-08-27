---
instruction: global_store
category: memory
architecture: gfx950
tags: [global_store, vmcnt, s_endpgm, store-leak, fire-and-forget]
---

# global_store Family

## Variants

| Opcode | Width | VGPRs | Counter |
|--------|-------|-------|---------|
| `global_store_dword` | 32-bit (1 dword) | 1 VGPR data | vmcnt |
| `global_store_dwordx2` | 64-bit (2 dwords) | 2 VGPRs data | vmcnt |
| `global_store_dwordx4` | 128-bit (4 dwords) | 4 VGPRs data | vmcnt |

## Cycle Counts (Measured on MI355X)

| Variant | Mode | CPI | Notes |
|---------|------|-----|-------|
| `global_store_dword` | throughput | ~0.63 | Fire-and-forget, very fast issue |
| `global_store_dwordx4` | latency | ~13.4 | Higher latency for wide stores |

Stores are fire-and-forget: the instruction issues quickly and the data is written asynchronously. The CPI for throughput mode is very low because the store does not block the wavefront.

## Counter

**vmcnt**. Stores increment the vmcnt counter at issue time and decrement when the store completes (reaches the memory subsystem). Use `s_waitcnt vmcnt(N)` to ensure stores have committed.

## FIFO Ordering

Same FIFO semantics as global_load. Stores and loads share the same vmcnt FIFO. When mixing loads and stores, both contribute to the outstanding count.

## Coherence Flags

| Flag | Effect |
|------|--------|
| (none) | Standard write-through to L1 |
| `sc0` | System coherent level 0 |
| `sc1` | System coherent level 1 |
| `sc0 sc1` | Fully system-coherent |

For inter-workgroup barrier stores: use `sc0` on stores to ensure visibility at L2 (not stuck in L1 TCP write buffer). The inter-WG barrier pattern uses `flat_atomic sc0` + `flat_load/store sc0 sc1`.

## Known Hazards

### 1. MUST drain vmcnt before s_endpgm
**CRITICAL**: On gfx950, `s_endpgm` does NOT guarantee that outstanding stores complete. Without `s_waitcnt vmcnt(0)` before `s_endpgm`, stores can "leak" -- data may never reach memory. This causes intermittent corruption that is extremely difficult to debug.

```asm
; MANDATORY at every kernel exit point:
s_waitcnt vmcnt(0) lgkmcnt(0)
s_endpgm
```

Empirically validated on MI355X across multiple kernel projects. Triton sometimes omits this, creating latent bugs.

### 2. Exec Masking for Bounds Checking
When writing partial tiles (edge tiles where some threads are out of bounds), use exec masking to prevent out-of-bounds stores:

```asm
v_cmp_lt_u32_e64 s[2:3], v0, s4    ; thread_id < valid_count
s_and_saveexec_b64 s[6:7], s[2:3]
global_store_dwordx4 v[8:9], v[0:3], off
s_or_b64 exec, exec, s[6:7]
```

### 3. v_add_co_u32 Clobbers VCC
Address computation with `v_add_co_u32` clobbers VCC. If VCC holds a live value (e.g., from a comparison), save it first.

## Known Bugs

None specific to global_store beyond the s_endpgm store leak (which affects all store types).

## Alignment Requirements

- `global_store_dword`: No special VGPR alignment for data
- `global_store_dwordx2`: Data must be from an even-numbered VGPR pair
- `global_store_dwordx4`: Data must be from an even-numbered VGPR
- Address is always a 64-bit VGPR pair (even-aligned base register)

Memory address should be naturally aligned for best performance.

## LDS Variant

No LDS variant for global_store. Data must come from VGPRs.

## Performance Notes

- Stores are non-blocking (fire-and-forget). The wavefront can continue executing immediately after issuing a store.
- The vmcnt counter tracks store completion, but you only need to wait (vmcnt) before s_endpgm or before reading the stored data back.
- For GEMM output epilogues, `buffer_store_dwordx2` is commonly used instead of global_store for BF16 output (4 BF16 values = 8 bytes = dwordx2).

## Common Patterns

### GEMM Output Store (BF16)
```asm
; After scale + convert:
v_pk_mul_f32 v[0:1], v[138:139], v[0:1]   ; scale
v_cvt_pk_bf16_f32 v0, v0, v1               ; FP32 -> BF16 pair
global_store_dword v[200:201], v0, off      ; store 2 BF16 values
```

### Kernel Epilogue
```asm
; Store all output tiles
global_store_dwordx4 v[addr:addr+1], v[0:3], off
global_store_dwordx4 v[addr:addr+1], v[4:7], off offset:16
; ... more stores ...
s_waitcnt vmcnt(0)       ; MANDATORY before endpgm
s_endpgm
```
