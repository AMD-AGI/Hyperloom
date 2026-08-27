---
instruction: kernel_descriptor
category: metadata
architecture: gfx950
tags: [kernel-descriptor, accum_offset, next_free_vgpr, AGPR, metadata, .args, hipModuleLaunchKernel]
---

# Kernel Descriptor and AMDGPU Metadata

## Overview

Every hand-written gfx950 assembly kernel requires two things to launch correctly:
1. A **kernel descriptor** (`.amdhsa_kernel` directives) specifying register allocation and hardware resources
2. **AMDGPU metadata** (`.amdgpu_metadata` YAML section) describing the kernel interface

Both must be consistent or the kernel will fail at launch or produce garbage results.

## Critical Fields

### accum_offset and next_free_vgpr

On CDNA (gfx90a/gfx940/gfx950), VGPRs and AGPRs share one physical register file. `amdhsa_accum_offset` divides the allocated block:

```
Physical register file:
  [0 .. accum_offset-1]         → VGPRs (v0, v1, ...)
  [accum_offset .. next_free_vgpr-1]  → AGPRs (a0, a1, ...)
```

**The most common MFMA bug:** Setting `next_free_vgpr == accum_offset` allocates **zero AGPRs**. MFMA instructions that write to `a[N]` will silently write to unallocated registers and reads return garbage.

**Formula:** `amdhsa_next_free_vgpr = accum_offset + n_agprs`

Example for 128x128 BF16 GEMM (80 VGPRs, 64 AGPRs):
```
.amdhsa_next_free_vgpr 144     ; 80 + 64
.amdhsa_accum_offset 80
```

### AGPR Aliasing Hazard

VGPRs at or above `accum_offset` alias AGPRs: `v[accum_offset + N] == a[N]`. Writing to these VGPRs (e.g., via `ds_read_b128`) silently clobbers accumulator state.

```asm
; With accum_offset=64:
; v64 == a0, v65 == a1, v66 == a2, v67 == a3
ds_read_b128 v[64:67], v8     ; DESTROYS a[0:3] accumulator values
```

**Rule:** ALL compute VGPRs must be below accum_offset.

## .args Metadata (Required)

On ROCm 7.2.0/gfx950, `hipModuleLaunchKernel` returns error 701 ("too many resources requested for launch") if the `.args` array is missing from `.amdgpu_metadata`, even when VGPRs/SGPRs/LDS are within limits.

```yaml
.amdgpu_metadata:
  amdhsa.version: [1, 2]
  amdhsa.kernels:
    - .name: my_kernel
      .symbol: my_kernel.kd
      .kernarg_segment_size: 64
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .sgpr_count: 32
      .vgpr_count: 144
      .max_flat_workgroup_size: 256
      .args:
        - { .offset: 0,  .size: 8, .value_kind: global_buffer, .address_space: global }
        - { .offset: 8,  .size: 8, .value_kind: global_buffer, .address_space: global }
        - { .offset: 16, .size: 4, .value_kind: by_value }
```

Every kernarg must have `.offset`, `.size`, and `.value_kind`. Use `global_buffer` for pointers, `by_value` for scalars.

## s_waitcnt Before s_endpgm

Every gfx950 kernel must have `s_waitcnt vmcnt(0)` immediately before `s_endpgm`. Without it, global stores may not be visible to the next kernel on the same HIP stream, causing non-deterministic NaN.

```asm
; End of kernel — MANDATORY
s_waitcnt vmcnt(0)
s_endpgm
```

Check that ALL exit paths (including early-out branches to `.Lexit:`) flow through the waitcnt.

## Flat vs Global Consistency

Use consistent memory instruction types across kernels sharing buffers. Mixing `flat_store` in one kernel with `global_load` in a consumer kernel can cause stale reads on gfx950, even on the same HIP stream. Prefer `global_load/store` for global memory access.

## Minimal Template

```asm
.amdhsa_kernel my_kernel
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 64
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 144
  .amdhsa_next_free_sgpr 32
  .amdhsa_accum_offset 80
  .amdhsa_ieee_mode 0
  .amdhsa_dx10_clamp 0
  .amdhsa_wavefront_size32 0
.end_amdhsa_kernel
```
