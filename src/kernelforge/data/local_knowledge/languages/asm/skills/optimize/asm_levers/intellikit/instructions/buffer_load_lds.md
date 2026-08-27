---
instruction: buffer_load_lds
category: memory
architecture: gfx950
tags: [buffer_load, LDS, direct-to-LDS, GEMM, prefetch, vmcnt, m0]
---

# buffer_load_dwordx4 ... lds (Direct-to-LDS)

## Syntax

```asm
s_mov_b32 m0, <lds_offset>
buffer_load_dwordx4 v<offset>, s[<srd>:<srd+3>], 0 offen lds
```

## Description

Loads 128 bits from HBM directly into LDS, bypassing VGPRs entirely. The VGPR operand is the **buffer offset** (not a destination). The LDS write address is set via the `m0` register before each load.

## Counter

**vmcnt** (NOT lgkmcnt). Even though data arrives in LDS, the load is tracked by the global memory counter. This is the most common source of confusion with this instruction.

## Key Advantages

| Property | buffer_load + ds_write | buffer_load_lds |
|----------|----------------------|-----------------|
| VGPR pressure | +4 per in-flight load | 0 (bypasses VGPRs) |
| Instructions | buffer_load + s_waitcnt + ds_write | 1 instruction |
| Barriers needed | 2 per double-buffer flip | 1 per double-buffer flip |

Measured 17% speedup over disassembled buffer_load+ds_write path in GEMM inner loops.

## Usage Pattern (Double-Buffered GEMM Prefetch)

```asm
; Prefetch A tile to LDS buffer 0
s_mov_b32 m0, 0x0000                         ; LDS offset for buffer 0
buffer_load_dwordx4 v0, s[8:11], 0 offen lds ; A[0:15]
s_mov_b32 m0, 0x0010
buffer_load_dwordx4 v0, s[8:11], s4 offen lds ; A[16:31]

; Prefetch B tile to LDS buffer 0
s_mov_b32 m0, 0x1000
buffer_load_dwordx4 v1, s[12:15], 0 offen lds ; B[0:15]

s_waitcnt vmcnt(0)                            ; wait for ALL LDS loads
s_barrier                                     ; sync workgroup

; ds_read from LDS for MFMA operands
ds_read_b128 v[4:7], v2                       ; read A tile
ds_read_b128 v[8:11], v3                      ; read B tile
s_waitcnt lgkmcnt(0)

v_mfma_f32_32x32x16_bf16 ...
```

## Known Issues

1. **m0 must be set before each load.** The LDS write address comes from m0, not the instruction encoding. Forgetting to update m0 writes all loads to the same LDS offset.

2. **vmcnt, not lgkmcnt.** Use `s_waitcnt vmcnt(0)` to drain these loads, not lgkmcnt. They share the vmcnt FIFO with regular buffer_loads and global_loads.

3. **Disassembly round-trip breaks it.** `llvm-objdump` disassembles `buffer_load_dwordx4 ... lds` into separate `buffer_load_dwordx4` (into VGPRs) + `ds_write_b128` (to LDS). Reassembling from disassembly loses the direct-to-LDS optimization.

## Assembler Support

Confirmed to assemble on gfx950 with ROCm 7.2 llvm-mc:
```
/opt/rocm/llvm/bin/llvm-mc --triple=amdgcn-amd-amdhsa --mcpu=gfx950 -filetype=obj
```
