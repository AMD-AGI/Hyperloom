---
title: HIP kernel-language API — qualifiers, launch, built-ins (wave64)
kind: api_reference
gens: [gfx942, gfx950]
dtypes: [both]
regimes: [both]
status: sota
updated: 2026-07-09
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/cpp_language_extensions.html
---

# HIP kernel-language API reference

The device-side language surface: function qualifiers, launch syntax, indexing, synchronization, cross-lane
(wave64) built-ins, and atomics. This is the **API standard** ("what the calls are"); for *how to use them
fast* (MFMA/LDS/scheduling) see [../skills/optimize/hip_levers/](../skills/optimize/hip_levers/). Hardware
constants (wave size, VGPR/LDS/CU) are single-sourced in `local_knowledge/hardware/` — not repeated here.

## Function qualifiers
| Qualifier | Runs on | Callable from |
|---|---|---|
| `__global__` | device (kernel entry) | host (via `<<<>>>` or `hipLaunchKernelGGL`) |
| `__device__` | device | device |
| `__host__` | host | host (combine `__host__ __device__` for both) |
| `__forceinline__` / `__noinline__` | — | inlining control (early-inline matters for register control) |
| `__launch_bounds__(maxTPB, minWavesPerEU)` | on `__global__` | caps registers for occupancy (see hip_levers) |
| `__restrict__` | pointer params | enables wider vectorized loads + reordering |
| `__shared__` | in-kernel | static LDS allocation; dynamic LDS = `extern __shared__` + 3rd launch arg |

## Launch syntax
```cpp
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);          // triple-chevron
hipLaunchKernelGGL(kernel, gridDim, blockDim, sharedMemBytes, stream, args...);  // macro form
// Dynamic shared memory above default must be opted in BEFORE launch:
hipFuncSetAttribute((void*)kernel, hipFuncAttributeMaxDynamicSharedMemorySize, bytes);
// Cooperative launch (enables grid-wide sync):
hipLaunchCooperativeKernel((void*)kernel, gridDim, blockDim, args, sharedMemBytes, stream);
```
`dim3 gridDim/blockDim` are 3-D. `blockDim.x*.y*.z ≤ 1024` and should be a **multiple of 64** (wave64).

## Indexing built-ins
```cpp
threadIdx.{x,y,z}  blockIdx.{x,y,z}  blockDim.{x,y,z}  gridDim.{x,y,z}
int warpSize;                       // == 64 on CDNA (NOT 32)
int lane = threadIdx.x % warpSize;  // 0..63
int wave = threadIdx.x / warpSize;
```

## Synchronization
```cpp
__syncthreads();                 // block barrier (all threads must reach)
__syncthreads_count(pred);       // barrier + count of nonzero predicates
__syncthreads_and(pred);         // barrier + AND
__syncthreads_or(pred);          // barrier + OR
__threadfence_block();           // memory fence, block scope
__threadfence();                 // memory fence, device scope
__threadfence_system();          // memory fence, system (host-visible) scope
__builtin_amdgcn_s_barrier();    // low-level workgroup barrier
```
Divergent `__syncthreads()` (not all lanes reach it) deadlocks — see debug-hip-kernel.

## Cross-lane / warp built-ins (wave64 — masks are 64-bit)
```cpp
unsigned long long __ballot(int pred);   // 64-bit mask (bit i = lane i)
int __all(int pred);  int __any(int pred);
unsigned long long __activemask();
int __popcll(unsigned long long);        // popcount over 64 bits — NOT __popc
T __shfl(T v, int srcLane, int width=warpSize);
T __shfl_up(T v, unsigned d, int width=warpSize);
T __shfl_down(T v, unsigned d, int width=warpSize);
T __shfl_xor(T v, int laneMask, int width=warpSize);
T __reduce_add_sync(unsigned long long mask, T v);   // + _min_/_max_ variants
```
- Mask type **must be `unsigned long long`** (a 32-bit mask static-asserts on CDNA).
- Half-float `__shfl` is unsupported — shuffle as int/float and repack.
- These carry **no memory barrier** — add fences for side-effect ordering.
- Low-level equivalents: `__builtin_amdgcn_ds_bpermute/ds_permute/ds_swizzle/mov_dpp/permlane16/readlane`
  (see [../skills/optimize/hip_levers/hip_builtins.md](../skills/optimize/hip_levers/hip_builtins.md)).

## Atomics
```cpp
atomicAdd atomicSub atomicMin atomicMax atomicExch atomicCAS      // int + float
atomicAnd atomicOr atomicXor atomicInc atomicDec                  // int
safeAtomicAdd(addr, val);     // always numerically correct
unsafeAtomicAdd(addr, val);   // HW fp atomic when available (-munsafe-fp-atomics) — big for reductions/split-K
```
Atomics resolve at the **L2** coherence point; keep them out of inner loops.

## Cooperative groups
```cpp
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;
cg::thread_block b = cg::this_thread_block();
cg::grid_group   g = cg::this_grid();                       // needs hipLaunchCooperativeKernel
cg::thread_block_tile<64> w = cg::tiled_partition<64>(b);   // N = power of 2, ≤ 64 on CDNA
cg::coalesced_group a = cg::coalesced_threads();            // active lanes only
b.sync();  b.thread_rank();  b.size();
```

## Sources
- HIP kernel language (qualifiers, warpSize, __launch_bounds__, __shfl/__ballot, atomics): https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- HIP C++ language extensions (built-in vars, cooperative groups, fences): https://rocm.docs.amd.com/projects/HIP/en/latest/reference/cpp_language_extensions.html
- Hardware constants (wave64, VGPR/LDS/CU): `local_knowledge/hardware/` (single source of truth).
