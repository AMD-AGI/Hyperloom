---
instruction: s_load
category: scalar
architecture: gfx950
tags: [s_load, lgkmcnt, SMEM, kernarg, SGPR-alignment]
---

# s_load Family

## Variants

| Opcode | Width | SGPRs | Counter |
|--------|-------|-------|---------|
| `s_load_dword` | 32-bit (1 dword) | 1 SGPR dest | lgkmcnt |
| `s_load_dwordx2` | 64-bit (2 dwords) | 2 SGPRs dest | lgkmcnt |
| `s_load_dwordx4` | 128-bit (4 dwords) | 4 SGPRs dest | lgkmcnt |
| `s_load_dwordx8` | 256-bit (8 dwords) | 8 SGPRs dest | lgkmcnt |
| `s_load_dwordx16` | 512-bit (16 dwords) | 16 SGPRs dest | lgkmcnt |

## Syntax

```asm
s_load_dwordx4 s[D:D+3], s[base:base+1], offset
```

Loads data from memory (typically kernarg segment or constant memory) into SGPRs. The base address is a 64-bit SGPR pair, and the offset is an immediate or SGPR.

## Cycle Counts (Measured on MI355X)

| Variant | Mode | CPI | Notes |
|---------|------|-----|-------|
| `s_load_dword` | latency | ~1.7 | 256 iters, very fast |
| `s_load_dwordx4` | latency | ~1.7 | 256 iters, same as dword! |

From earlier measurement (older methodology): ~9.8 CPI.

The db.json measurement shows remarkably low latency (~1.7 CPI), likely because the kernarg data is cached in the scalar L0/constant cache. Real-world latency for uncached SMEM access is higher (~10-20 cycles).

## Counter

**lgkmcnt**. s_load shares the lgkmcnt counter with ALL LDS operations (ds_read, ds_write). This is important -- `s_waitcnt lgkmcnt(0)` drains BOTH pending s_loads AND pending ds_read/ds_write operations.

When mixing s_load with ds_read in the same code region, track both types of operations in the lgkmcnt FIFO.

## FIFO Ordering

lgkmcnt FIFO -- s_load operations are interleaved with LDS operations in the same FIFO. Oldest drains first.

```
s_load_dwordx4 s[0:3]:     lgkmcnt = 1 (oldest)
ds_read_b128 v[0:3]:       lgkmcnt = 2
s_load_dwordx2 s[4:5]:     lgkmcnt = 3 (newest)

s_waitcnt lgkmcnt(1):      drains s_load s[0:3] and ds_read v[0:3]
                           s_load s[4:5] still pending
```

## Coherence Flags

| Flag | Effect |
|------|--------|
| `glc` | Non-temporal / bypass L1 (global coherent) |
| `dlc` | Device-level coherent |

Typically used without flags for kernarg loading (the constant cache provides the correct data).

## Known Hazards

### 1. SGPR Destination Clobber
s_load writes to SGPRs asynchronously. If the destination SGPRs hold live values (e.g., loop counters, buffer descriptors), they will be overwritten when the load completes.

**Common trap**: `s_load_dwordx2 s[2:3], s[0:1], 0x0` loads kernargs into s[2:3]. If the kernel descriptor uses kernarg preloading (USER_SGPR_COUNT=16), s[2] already holds `workgroup_id_x`. The s_load clobbers it.

### 2. lgkmcnt Shared with LDS
Forgetting that s_load and ds_read/ds_write share lgkmcnt is a source of miscounting bugs. If you have 3 ds_reads and 1 s_load pending, lgkmcnt is 4 (not 3).

## Known Bugs

### Kernarg Preloading vs s_load Conflict
Triton-compiled kernels may preload 14 dwords of kernarg into s[2:15] via the kernel descriptor (USER_SGPR_COUNT=16). The kernel code then uses s[2:15] directly without any s_load. If you hand-assemble with USER_SGPR_COUNT=2 (no preloading), the kernel expects s_load to populate these SGPRs, but the workgroup_id_x (now at s[2] instead of s[16]) gets clobbered by the first s_load.

**Rule**: Always use `--ref-co` patching to preserve the Triton kernel descriptor's preloading configuration. Never assemble standalone .co files for kernels that depend on kernarg preloading.

## Alignment Requirements

- `s_load_dword`: No alignment requirement for destination
- `s_load_dwordx2`: Destination must be 2-aligned SGPR pair (s[N:N+1] where N is even)
- `s_load_dwordx4`: Destination must be 4-aligned (s[N:N+3] where N is multiple of 4)
- `s_load_dwordx8`: Destination must be 8-aligned
- `s_load_dwordx16`: Destination must be 16-aligned
- Base address: 2-aligned SGPR pair

**SGPR alignment is mandatory.** Using a misaligned SGPR destination causes assembly errors or hardware faults.

Empirically validated on MI355X.

## LDS Variant

None. s_load always reads from global/constant memory through the scalar memory path. For LDS, use ds_read.

## Performance Notes

- **Very fast for cached data**: Kernarg and constant data is typically cached in the scalar L0 constant cache (~1.7 CPI measured).
- **Used sparingly**: In GEMM kernels, s_load is only used during the prologue (kernarg loading) and occasionally in the outer persistent loop (reading group offsets). Not a hot-path instruction.
- **Scalar path advantage**: s_load uses the scalar memory unit, which is independent of the vector memory unit. It does not compete with buffer_load/global_load for memory bandwidth.

## Common Patterns

### Kernarg Loading
```asm
; Load first 4 dwords of kernarg into SGPRs
; s[0:1] = kernarg_segment_ptr (set by hardware)
s_load_dwordx4 s[4:7],   s[0:1], 0x0    ; first 4 args
s_load_dwordx4 s[8:11],  s[0:1], 0x10   ; next 4 args
s_load_dwordx4 s[12:15], s[0:1], 0x20   ; next 4 args
s_waitcnt lgkmcnt(0)                      ; drain all
; Now s[4:15] hold 12 kernel arguments
```

### Buffer Descriptor Setup
```asm
; Load buffer descriptor components from kernarg
s_load_dwordx4 s[28:31], s[0:1], 0x30    ; buffer resource descriptor
s_waitcnt lgkmcnt(0)
; Now s[28:31] is ready for buffer_load/buffer_store
```

### Group Offset Lookup (Persistent GEMM)
```asm
; Load per-group offset from global offset table
s_lshl_b32 s_tmp, s_group_idx, 2         ; byte offset
s_load_dword s_offset, s[base_ptr], s_tmp
s_waitcnt lgkmcnt(0)
; s_offset now holds this group's starting offset
```
