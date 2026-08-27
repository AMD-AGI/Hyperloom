---
instruction: buffer_store
category: memory
architecture: gfx950
tags: [buffer_store, vmcnt, s_endpgm, store-leak]
---

# buffer_store Family

## Variants

| Opcode | Width | VGPRs | Counter |
|--------|-------|-------|---------|
| `buffer_store_dword` | 32-bit (1 dword) | 1 VGPR data | vmcnt |
| `buffer_store_dwordx2` | 64-bit (2 dwords) | 2 VGPRs data | vmcnt |
| `buffer_store_dwordx4` | 128-bit (4 dwords) | 4 VGPRs data | vmcnt |

## Cycle Counts (Measured on MI355X)

| Variant | Mode | CPI | Notes |
|---------|------|-----|-------|
| `buffer_store_dword` | throughput | ~0.28 | Fire-and-forget, very fast |
| `buffer_store_dwordx4` | throughput | ~0.73 | Fire-and-forget |

Stores issue quickly (non-blocking). The wavefront proceeds immediately.

## Counter

**vmcnt**. Stores enter the vmcnt FIFO at issue time. Shares the same counter as buffer_load and global_load/store.

## FIFO Ordering

Same FIFO as all vmcnt operations. Stores and loads are interleaved in issue order.

## Coherence Flags

Same as buffer_load: `sc0`, `sc1` for system coherence levels.

## Known Hazards

### 1. MUST drain vmcnt before s_endpgm
Same as global_store. `s_waitcnt vmcnt(0)` is MANDATORY before `s_endpgm` on gfx950 or stores may not commit.

### 2. Data VGPRs Must Be Ready
The data VGPRs are read at issue time. If they contain results from an in-flight MFMA or ds_read, ensure the appropriate waitcnt (lgkmcnt or MFMA drain) has completed before issuing the store.

## Known Bugs

None specific beyond the s_endpgm store leak.

## Alignment Requirements

- Buffer descriptor: 4-SGPR aligned (s[N:N+3])
- `buffer_store_dwordx2` data: even-numbered VGPR pair
- `buffer_store_dwordx4` data: even-numbered VGPR base

## LDS Variant

None. There is no buffer_store ... lds variant.

## Performance Notes

- **Primary output path for GEMM epilogues**: `buffer_store_dwordx2` is the standard choice for BF16 output (4 BF16 values = 8 bytes = 2 dwords). Used by optimized kernels and GGEMM output paths.
- Fire-and-forget semantics make stores nearly free from a latency perspective. Only pay for the vmcnt drain before s_endpgm.

## Common Patterns

### GEMM BF16 Output Store
```asm
; After MFMA accumulator -> BF16 conversion:
v_pk_mul_f32 v[0:1], v[138:139], v[0:1]    ; multiply by scale
v_cvt_pk_bf16_f32 v0, v0, v1                ; FP32 pair -> BF16 pair
buffer_store_dwordx2 v[0:1], v200, s[4:7], 0 offen  ; store 4 BF16 values

; At kernel end:
s_waitcnt vmcnt(0) lgkmcnt(0)
s_endpgm
```

### Attention Output Store (BWD)
```asm
; Store dQ gradients as BF16
buffer_store_dwordx2 v[0:1], v_offset, s[rsrc], 0 offen
buffer_store_dwordx2 v[2:3], v_offset, s[rsrc], 0 offen offset:256
; ... 16 stores per output tile ...
s_waitcnt vmcnt(0)
s_endpgm
```
