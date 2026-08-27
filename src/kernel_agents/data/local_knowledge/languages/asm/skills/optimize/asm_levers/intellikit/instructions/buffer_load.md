---
instruction: buffer_load
category: memory
architecture: gfx950
tags: [buffer_load, vmcnt, WAW-hazard, direct-to-LDS, FIFO]
---

# buffer_load Family

## Variants

| Opcode | Width | VGPRs | Counter | Notes |
|--------|-------|-------|---------|-------|
| `buffer_load_dword` | 32-bit | 1 VGPR dest | vmcnt | |
| `buffer_load_dwordx2` | 64-bit | 2 VGPRs dest | vmcnt | |
| `buffer_load_dwordx4` | 128-bit | 4 VGPRs dest | vmcnt | Primary variant for GEMM |
| `buffer_load_dwordx4 ... lds` | 128-bit direct-to-LDS | 0 VGPRs (bypasses) | vmcnt | MI355X feature |
| `buffer_load_format_x` | format-converted | 1 VGPR dest | vmcnt | |

## Cycle Counts (Measured on MI355X)

| Variant | Mode | CPI | Notes |
|---------|------|-----|-------|
| `buffer_load_dword` | latency | ~4.0 | L1 hit, 256 iters |
| `buffer_load_dwordx4` | latency | ~5.7 | L1 hit, 256 iters |
| `buffer_load_format_x` | latency | ~5.1 | Format-converted load |

Practical HBM latency: ~300-400 cycles from issue to data arrival.

Earlier measurement (older methodology): buffer_load measured as ~16% slower than global_load for the same access pattern.

## Counter

**vmcnt**. Both the standard and `lds` variants count against vmcnt.

**Important**: The `lds` variant counts against vmcnt, NOT lgkmcnt, even though data is written to LDS. This is a common source of confusion.

## FIFO Ordering

vmcnt FIFO -- oldest issued load drains first. Same semantics as global_load. buffer_loads and global_loads share the same vmcnt FIFO if both are in flight.

When hoisting buffer_loads earlier in a loop, ALL downstream vmcnt values must be recounted:
```
buffer_load_dwordx4 v[128:131], ...  ; oldest, vmcnt position 4
buffer_load_dwordx4 v[132:135], ...  ; vmcnt position 3
buffer_load_dwordx4 v[216:219], ...  ; vmcnt position 2
buffer_load_dwordx4 v[220:223], ...  ; newest, vmcnt position 1
; ... compute ...
s_waitcnt vmcnt(2)                   ; drains oldest 2 (v[128:135])
s_waitcnt vmcnt(0)                   ; drains all remaining
```

## Coherence Flags

Same as global_load: `sc0`, `sc1`, `sc0 sc1` for system coherence levels.

## Known Hazards

### 1. WAW Hazard: Asynchronous Destination Write (CRITICAL)
**The most frequently rediscovered hazard in practice.** buffer_load writes its destination VGPRs when data ARRIVES from HBM (~300 cycles after issue), NOT when `s_waitcnt vmcnt()` executes.

If ANY instruction writes to the buffer_load destination VGPRs between the load issue and data arrival, the arriving load will silently overwrite that value:

- **buffer_load into MFMA source VGPRs**: Non-deterministic wrong results. The load data overwrites operands mid-MFMA-execution.
- **buffer_load into MFMA accumulator VGPRs**: Partial zeroing pattern (e.g., elements [0:4]=0, [4:12]=correct, [12:16]=0).
- **buffer_load into ds_read destination VGPRs**: LDS data overwritten by later-arriving HBM data.

**Rule**: buffer_load destinations must be completely dead (not read or written) between the load issue and the vmcnt drain.

### 2. Safe Self-Clobber (Address = Destination[0])
`buffer_load_dwordx4 v[X:X+3], vX, s[desc], 0 offen` where vX is both the address and the first destination register is SAFE. The address VGPR (vaddr) is read at issue time, before the destination write occurs on data arrival. This is the standard Triton-generated pattern.

### 3. NOP Between Consecutive buffer_loads
On gfx950, removing `s_nop 0` between consecutive `buffer_load_dwordx4` instructions causes crashes in some configurations (memory access faults). This is empirically inconsistent -- results are empirically inconsistent. **Conservative rule**: keep an `s_nop 0` between back-to-back buffer_loads using the same buffer descriptor.

### 4. s_endpgm Store Leak (applies to buffer_store too)
Must have `s_waitcnt vmcnt(0)` before `s_endpgm`. Same as global_store.

## Known Bugs

### Out-of-Bounds Returns Zero (Non-Swizzled Descriptors)
When the vaddr offset exceeds the buffer descriptor's size field, the load returns zero without faulting (for non-swizzled, non-format buffers). This is the standard sentinel pattern used for bounds checking:

```asm
v195 = v_bfrev_b32 1                    ; 0x80000000 = OOB sentinel
v_cndmask_b32 v128, v128, v195, vcc     ; if out-of-bounds, use sentinel
buffer_load_dwordx4 v[128:131], v128, s[28:31], 0 offen  ; returns 0 for OOB
```

If the buffer descriptor has different flags (swizzled), OOB behavior may fault instead of returning zero. Ensure consistent descriptor configuration between LHS and RHS.

## Alignment Requirements

- Buffer descriptor: 4-SGPR aligned (s[N:N+3] where N is multiple of 4)
- `buffer_load_dwordx4` dest: even-numbered VGPR base
- `buffer_load_dwordx2` dest: even-numbered VGPR pair
- vaddr: single VGPR (no alignment constraint)

## LDS Variant (buffer_load ... lds)

`buffer_load_dwordx4 vaddr, s[desc], offset offen lds` loads data directly from HBM into LDS, bypassing VGPRs entirely. The m0 register specifies the LDS write destination offset.

**Key properties**:
- Counts against **vmcnt** (not lgkmcnt)
- No VGPR is consumed for data (only vaddr for the address)
- Eliminates the traditional buffer_load -> ds_write pipeline
- Requires `s_nop 0` (or equivalent 1-cycle spacer) after writing m0 before issuing the load (M0 write hazard on gfx950)

```asm
s_mov_b32 m0, lds_offset                         ; set LDS write offset
s_nop 0                                           ; M0 settle hazard
buffer_load_dwordx4 v98, s[24:27], 0 offen lds   ; HBM -> LDS directly
```

**When to use**: The `use_block_pingpong=True` Triton variant uses this for 15-20% speedup over the VGPR-staged pipeline. However, it makes buffer_load hoisting inapplicable (no VGPR destinations to decouple).

## Performance Notes

- **16% slower than global_load** for GEMM data loads on gfx950. Measured empirically. Optimized kernels use buffer_load exclusively, while custom ASM kernels prefer global_load for hot-path data.
- **Buffer descriptor overhead**: Requires 4-SGPR descriptor setup. The descriptor must be restored if the outer loop clobbers it (common in persistent kernels where s6/s7 are reused for tile arithmetic).
- **Memory queue saturation**: Issuing 8+ buffer_loads simultaneously saturates the memory controller, causing 9-11% regression. Partial hoisting (6 loads spread across the loop) outperforms front-loading.
- **Hoisting sweet spot**: 2 of 4 B-side loads hoisted to after MFMA #8 gives +2-4% speedup. Full hoisting of all 4 causes regression.

## Common Patterns

### GEMM A/B Matrix Load
```asm
; 4-SGPR buffer descriptor in s[28:31]
; vaddr in v128 (byte offset for this thread)
buffer_load_dwordx4 v[128:131], v128, s[28:31], 0 offen  ; self-clobber OK
buffer_load_dwordx4 v[132:135], v132, s[28:31], 0 offen
; ... 128 MFMAs of compute ...
s_waitcnt vmcnt(0)
ds_write_b128 v196, v[128:131] offset:0     ; stage to LDS
ds_write_b128 v196, v[132:135] offset:8192  ; stage to LDS
```

### Direct-to-LDS Prefetch (Pingpong Variant)
```asm
s_mov_b32 m0, s8                              ; LDS bank offset
s_nop 0                                        ; M0 settle
buffer_load_dwordx4 v98, s[24:27], 0 offen lds  ; HBM -> LDS
; No ds_write needed
```

### Buffer Descriptor Construction
```asm
; s[28:31] = buffer resource descriptor
s_mov_b32 s28, base_addr_lo      ; bits [31:0] of base address
s_mov_b32 s29, base_addr_hi      ; bits [47:32] of base, swizzle config
s_mov_b32 s30, 0xFFFFFFFF        ; num_records (max = no bounds check)
s_mov_b32 s31, 0x00027000        ; DST_SEL, NUM_FORMAT, DATA_FORMAT
```
