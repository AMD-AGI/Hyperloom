# HIP Kernel Pitfalls

## Register Pinning Traps

### AGPR inline asm `"+a"` drops reg_idx=0
`asm volatile("..." : "+a"(fp32x16))` tells clang the AGPR is live, but the compiler
drops register index 0 from the live set. First row of MFMA output holds half the
correct value.
- SYMPTOM: SNR = 21 dB (should be >100 dB)
- FIX: Use builtins + empty-asm "+v" barrier:
  ```cpp
  c = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c, 0, 0, 0);
  asm volatile("" : "+v"(c));
  ```
- ACHIEVED: 142 dB SNR, 256 VGPR + 170 AGPR + 0 spill at 256 fp32/lane pressure

### Register allocator drift across loop iterations
Without explicit pinning, the compiler may reassign VGPR/AGPR slots between loop
iterations. This breaks hand-written `ds_read_pinned` and `read_agpr` templates
that assume fixed register positions.
- FIX: Use `reserve_vgpr_range<START, END>()` and `reserve_agpr_range<START, END>()`
  at kernel entry, before any computation
- FIX: Document the register layout in a comment block at kernel top

## Memory Access Traps

### BufferSRD bounds checking silent clamp
Buffer loads with SRD clamp out-of-bounds accesses to the last valid element instead
of faulting. This means boundary bugs produce slightly-wrong results, not crashes.
- SYMPTOM: Last row/column of output tile has duplicated values
- FIX: Validate SRD num_records field matches actual buffer size
- FIX: Test with non-power-of-2 matrix dimensions to catch boundary bugs

### Direct-to-LDS load size varies by architecture
- gfx950: supports 4, 12, and 16-byte `buffer_load_lds` per thread
- gfx942: only 4-byte `buffer_load_lds`
- FIX: Guard with `#if defined(__gfx950__)` and provide 4-byte fallback
- SYMPTOM: Compile error or illegal instruction on wrong architecture

### Shared memory bank conflicts with MFMA output
MFMA output lane layout creates systematic bank conflicts when writing to shared
memory with naive row-major addressing.
- FIX: Column swizzling: `col ^ (row >> 1)` before LDS store
- EVIDENCE: 15-20% throughput improvement on shared memory-heavy kernels

## Pipeline Traps

### Missing epilogue phases
Shifted-LDG pipeline requires 2-3 epilogue phases to drain the pipeline after the
main loop exits. Missing epilogue = last 2-3 K-tiles not computed.
- SYMPTOM: Output correct for small K, wrong for large K (last tiles missing)
- FIX: Implement explicit epilogue phases:
  1. Epilogue 1: last GMEM load, mixed LDS prefetch
  2. Epilogue 2: no GMEM load, LDS from both buffers
  3. Epilogue 3: pure MFMA, no loads

### wait_vmcnt too aggressive
Setting `wait_vmcnt<0>()` (wait for ALL outstanding GMEM ops) kills latency hiding.
The whole point of software pipelining is overlapping compute with memory.
- FIX: Use `wait_vmcnt<N>()` where N = number of in-flight loads you want to keep
- RULE: For double-buffered pipeline, `wait_vmcnt<12>` is a good starting point

### Barrier placement with double-buffered LDS
`__builtin_amdgcn_s_barrier()` must be placed AFTER all LDS reads from the current
buffer AND BEFORE any LDS writes to the next buffer. Wrong placement = data race.
- FIX: Pattern is: LDS_READ → BARRIER → GMEM_TO_LDS → MFMA (using read data)

## Occupancy Traps

### VGPR 256 cliff
occupancy=2 requires VGPR ≤ 256. At 257 VGPR, occupancy drops to 1 — this is a
step function, not gradual.
- EVIDENCE: 296→238 VGPR delivered −32% latency in CK production kernel
- FIX: Trade persistent register tiles for per-iteration LDS reloads
- FIX: Move accumulator from VGPR to AGPR (separate file, doesn't count toward limit)

### LDS 80KB cliff
occupancy=2 requires LDS ≤ 80KB per workgroup. Double-buffered tiles can easily
exceed this.
- FIX: Alias LDS slots when buffers have non-overlapping lifetimes
- CAUTION: aliasing can prevent async prefetch — prove LDS budget first

## HipKittens Traps

### Tile size must match MFMA instruction
HipKittens tile dimensions (e.g., `rt_16x32_s`) are coupled to MFMA instruction
sizes. Mismatched tiles compile but produce wrong results.
- FIX: Use standard tile sizes: 16x16, 16x32, 32x16 for bf16/fp16
- FIX: For FP8, use 32x32x64 tile shape

### Missing KITTENS_CDNA4 define
HipKittens defaults to CDNA3 code paths. Missing `-DKITTENS_CDNA4` on gfx950 uses
suboptimal instruction sequences.
- SYMPTOM: Works but 20-30% slower than expected
- FIX: Always pass `-DKITTENS_CDNA4` for gfx950 targets

### shared_allocator must be first in kernel
HipKittens `shared_allocator` manages dynamic shared memory. Creating it after other
shared memory declarations causes offset corruption.
- FIX: `shared_allocator al((int*)&__shm[0]);` as the FIRST line after shared memory extern

## Scheduling Traps

### sched_group_barrier mask confusion
The mask values are not intuitive:
- `0x08` = MFMA, `0x02` = VALU, `0x01` = VMEM, `0x400` = EXP
- Using wrong mask doesn't cause errors — it just doesn't schedule correctly
- SYMPTOM: No speedup from "optimized" scheduling
- FIX: Define named constants and verify with rocprof ISA analysis

## torch.Tensor Integration Traps

### clone() preserves non-contiguous strides
`tensor.transpose(0,1).clone()` produces a non-contiguous clone (preserve_format
default). Passing this to a HIP kernel that assumes contiguous layout causes hidden
`.contiguous()` copies in HBM.
- FIX: Use `.contiguous()` explicitly before kernel launch, OR
- FIX: Handle stride parameters in the kernel

## Build System Traps

### hipcc JIT cache stale artifacts
hipcc caches compiled kernels. Changing `#include`d headers may not trigger rebuild.
- FIX: `rm -rf /tmp/comgr_*` to clear JIT cache
- FIX: Use explicit `-c` + link steps for deterministic builds

### Ninja dependency tracking for headers
Similar to CK: ninja doesn't track transitive header dependencies.
- FIX: `rm -f *.o` when shared headers change
- FIX: Use `hipcc --write-dependencies` to generate .d files for make
