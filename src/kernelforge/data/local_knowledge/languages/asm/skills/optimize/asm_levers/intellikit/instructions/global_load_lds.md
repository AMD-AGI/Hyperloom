---
instruction: global_load_lds
category: memory
architecture: gfx950
tags: [global_load, LDS, direct-to-LDS, vmcnt, m0]
---

# global_load_dwordx4 ... lds (Direct-to-LDS via Global)

## Syntax

```asm
s_mov_b32 m0, <lds_offset>
global_load_dwordx4 v[<addr>:<addr+1>], off lds
```

## Description

Same concept as `buffer_load_dwordx4 ... lds` but uses the global address space instead of buffer resources. Loads data from HBM directly into LDS, bypassing VGPRs. The `m0` register provides the LDS destination offset.

## Counter

**vmcnt** (NOT lgkmcnt). Same as `buffer_load ... lds`.

## Differences from buffer_load_lds

| Property | buffer_load_lds | global_load_lds |
|----------|----------------|-----------------|
| Address source | Buffer resource descriptor (SRD) + VGPR offset | 64-bit flat address in VGPRs |
| Bounds checking | Hardware bounds check via SRD | No bounds check |
| Use case | Structured buffer access | Pointer-based access |

## Usage Notes

- Set `m0` before each load to specify the LDS write offset
- Prefer `buffer_load ... lds` when a buffer resource descriptor is available (better scheduling, bounds checking)
- Use `global_load ... lds` when working with raw pointers without an SRD setup
