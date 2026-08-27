---
instruction: global_load
category: memory
architecture: gfx950
tags: [global_load, vmcnt, WAW-hazard, FIFO, coherence, direct-to-LDS]
---

# global_load Family

## Variants

| Opcode | Width | VGPRs | Counter |
|--------|-------|-------|---------|
| `global_load_dword` | 32-bit (1 dword) | 1 VGPR dest | vmcnt |
| `global_load_dwordx2` | 64-bit (2 dwords) | 2 VGPRs dest (even-aligned) | vmcnt |
| `global_load_dwordx4` | 128-bit (4 dwords) | 4 VGPRs dest (even-aligned) | vmcnt |
| `global_load_lds_dword` | 32-bit direct-to-LDS | 0 VGPRs (data bypasses VGPRs) | vmcnt |

## Cycle Counts (Measured on MI355X)

| Variant | Mode | CPI (cycles/instruction) | Notes |
|---------|------|--------------------------|-------|
| `global_load_dword` (L1 hit) | latency | ~6.1 | L1 cache hit |
| `global_load_dword` (L2 hit) | latency | ~12.7 | L2 cache hit |
| `global_load_dword` (HBM) | latency | ~11.2 | HBM access (128 iters) |
| `global_load_dword` (NT) | latency | ~9.6 | Non-temporal (sc0) |
| `global_load_dword` (SC) | latency | ~9.2 | System-coherent (sc0 sc1) |
| `global_load_dwordx2` | latency | ~6.7 | L1 cache hit |
| `global_load_dwordx4` | latency | ~6.9 | L1 cache hit |
| `global_load_lds_dword` | latency | ~6.2 | Direct to LDS |

From earlier measurement (older methodology, 256-iteration):
- `global_load_dword`: ~25.3 CPI (likely includes more pipeline overhead)
- `global_load_dwordx4`: ~28.2 CPI

Practical HBM latency in kernels: ~300-400 cycles from issue to data arrival.

## Counter

**vmcnt**. All global_load variants decrement vmcnt when data arrives. Use `s_waitcnt vmcnt(N)` to drain.

Exception: `global_load_lds_dword` also counts against vmcnt (NOT lgkmcnt), despite writing to LDS.

## FIFO Ordering

vmcnt is FIFO -- oldest issued load drains first. `vmcnt(N)` means "wait until at most N loads remain outstanding." The oldest `(total - N)` loads are guaranteed complete.

**Critical rule**: When reordering global_loads (e.g., hoisting prefetches), all downstream vmcnt values must be recounted from scratch based on the new issue order, not the code position.

**Trap**: Issuing a prefetch load BEFORE a data load reverses their FIFO positions. If you then use `vmcnt(1)` expecting the data load to be drained, you actually drained the prefetch and the data load is still in flight.

## Coherence Flags

| Flag | Effect |
|------|--------|
| (none) | Standard L1-cached load |
| `sc0` | System coherent level 0 -- bypasses L1 TCP cache |
| `sc1` | System coherent level 1 -- bypasses L2 cache |
| `sc0 sc1` | Fully system-coherent, bypasses both caches |
| `nt` | Non-temporal hint -- deprioritized in cache |

For inter-CU visibility (e.g., inter-workgroup barriers): use `sc0` on loads (ensures read from L2, not stale L1).

## Known Hazards

### 1. Address Captured at Issue Time (Safe Self-Clobber)
The hardware reads the address VGPR(s) at issue time, before the destination is written. This means `global_load_dwordx4 v[X:X+3], v[X:X+1], ...` where the address register overlaps with the destination range is SAFE. The address is consumed before the destination is overwritten on data arrival (~300 cycles later).

### 2. WAW Hazard: Destination Clobbers Live Registers
`global_load` writes its destination VGPRs asynchronously when data arrives from memory (~300-400 cycles after issue). If ANY instruction writes to the same destination VGPRs between the load issue and the vmcnt drain, the load arrival will silently overwrite that value. This is the single most dangerous hazard.

**Example**: global_load_dwordx4 into v[0:3] at line 10, then ds_read into v[2:3] at line 50. If the global_load data arrives after the ds_read completes, v[2:3] contains HBM data instead of LDS data.

### 3. Dest Clobbers LDS Read Address
If the destination range of a global_load overlaps with a VGPR used as an LDS read address (ds_read base), the arriving load data will corrupt the address, causing subsequent ds_reads to access wrong LDS locations. Produces non-deterministic NaN.

Discovered during production kernel development: global_load dest regs clobber P LDS read addr, causes non-deterministic NaN.

## Known Bugs

### flat vs global Coherence Mismatch
Using `flat_store` in one kernel and `global_load` in the next kernel (or vice versa) to access the same memory can produce stale reads. The two instruction families use different cache coherence paths. Always use the same instruction family (flat or global) consistently for the same memory across kernels.

Empirically validated on MI355X.

## Alignment Requirements

- `global_load_dword`: No special VGPR alignment
- `global_load_dwordx2`: Dest must be an even-numbered VGPR pair (e.g., v[2:3], not v[3:4])
- `global_load_dwordx4`: Dest must be an even-numbered VGPR (e.g., v[4:7])
- Address is always a 64-bit VGPR pair (even-aligned base register)

The memory address itself should be naturally aligned for best performance (4-byte for dword, 8-byte for dwordx2, 16-byte for dwordx4), though unaligned accesses are supported with a performance penalty.

## LDS Variant

`global_load_lds_dword` loads data directly from global memory into LDS, bypassing VGPRs. The LDS destination offset is set via the `m0` register. Counts against **vmcnt** (not lgkmcnt). Requires `s_nop 0` or equivalent after writing m0 before issuing the load (m0 write hazard).

## Performance Notes

- **16% faster than buffer_load on gfx950** for GEMM data loads. Measured empirically. The global_load path has lower overhead than the buffer descriptor indirection used by buffer_load.
- **Optimal for software pipelining**: Issue global_load_dwordx4 at the top of the inner loop to maximize latency hiding (~3584 cycles of MFMA cover vs ~224 cycles if issued late).
- **Memory queue saturation**: Issuing more than 6-8 global/buffer loads simultaneously can saturate the memory controller, causing regression. Partial hoisting (some loads early, some late) outperforms front-loading all loads.
- **Throughput (stores)**: global_store_dword throughput is ~0.63 CPI (very fast fire-and-forget).

## Common Patterns

### Cooperative Global Load (rmsnorm / all-reduce kernels)
```asm
; Each thread loads its portion, exec mask for bounds
v_cmp_lt_u32_e64 s[2:3], v0, s4       ; thread_id < N
s_and_saveexec_b64 s[6:7], s[2:3]
global_load_dwordx4 v[4:7], v[8:9], off
s_or_b64 exec, exec, s[6:7]           ; restore exec
s_waitcnt vmcnt(0)
```

### Software-Pipelined GEMM Load
```asm
; Top of inner loop: issue all loads
global_load_dwordx4 v[128:131], v[200:201], off  ; A-tile
global_load_dwordx4 v[132:135], v[202:203], off  ; A-tile
; ... 128 MFMAs of compute ...
s_waitcnt vmcnt(0)                                ; drain all
ds_write_b128 v196, v[128:131] offset:0           ; stage to LDS
```

### Address Computation
```asm
v_add_co_u32_e32 v8, vcc, s0, v0     ; base_lo + thread_offset
v_addc_co_u32_e32 v9, vcc, 0, v1, vcc ; carry into base_hi
global_load_dwordx4 v[4:7], v[8:9], off
```
Note: `v_add_co_u32` clobbers VCC. Plan accordingly.
