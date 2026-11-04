# HIP API Reference for Kernel Development

## Low-Precision Data Types (gfx950 / CDNA4)

### FP4 (E2M1) — CDNA4 only
```cpp
#include <hip/hip_fp4.h>
// Classes: __hip_fp4_e2m1, __hip_fp4x2_e2m1, __hip_fp4x4_e2m1
// Conversion: __hip_cvt_float_to_fp4(), __hip_cvt_fp4_to_halfraw()
// Saturation: __hip_saturation_t with __HIP_SATFINITE
```

### FP6 (E3M2 / E2M3) — CDNA4 only
```cpp
#include <hip/hip_fp6.h>
// Classes: __hip_fp6_e2m3, __hip_fp6_e3m2 and vector variants
// E3M2 = wider range, E2M3 = higher precision
```

### FP8 (E4M3 / E5M2)
```cpp
#include <hip/hip_fp8.h>
// OCP format (gfx950): __hip_fp8_e4m3, __hip_fp8_e5m2
// FNUZ format (gfx942): __hip_fp8_e4m3_fnuz, __hip_fp8_e5m2_fnuz
// E4M3 = higher precision (inference weights)
// E5M2 = wider range (gradients)
```

### BF16 / FP16
```cpp
#include <hip/hip_bf16.h>   // __hip_bfloat16, __hip_bfloat162
#include <hip/hip_fp16.h>   // __half, __half2
// Conversions: __float2bfloat16(), __bfloat162float()
// Conversions: __float2half(), __half2float()
```

### HIP EXT Microscaling APIs (gfx950)
```cpp
// Scale type: __amd_scale_t (E8M0 format)
// Storage: __amd_fp8x2_storage_t, __amd_fp8x8_storage_t, __amd_fp4x2_storage_t
// Scale-aware: __amd_cvt_fp8x2_to_floatx2_scale()
// Stochastic rounding: _sr suffix APIs (require seed parameter)
// C++ structs: __hipext_ocp_fp8_e4m3, __hipext_ocp_fp8x2_e4m3, __hipext_ocp_fp6x32_e2m3
```

## Custom Vector Types for MFMA

```cpp
// Defined via __attribute__((vector_size(N)))
using float32x4 = __attribute__((vector_size(16))) float;   // 4 × fp32 (MFMA acc)
using int32x4   = __attribute__((vector_size(16))) int;     // 4 × int32 (SRD)
using int32x8   = __attribute__((vector_size(32))) int;     // 8 × int32 (MFMA operand)
```

## Device-Side Synchronization

```cpp
__syncthreads()                // Block barrier (all threads)
__syncthreads_count(pred)      // Barrier + count non-zero predicates
__syncthreads_and(pred)        // Barrier + all() predicate
__syncthreads_or(pred)         // Barrier + any() predicate
__threadfence_block()          // Memory fence, block scope
__threadfence()                // Memory fence, device scope
__threadfence_system()         // Memory fence, system scope (host-visible)
```

## Warp Cross-Lane Functions

```cpp
// Vote (return 64-bit masks on AMD)
int __all(int predicate)
int __any(int predicate)
unsigned long long __ballot(int predicate)
unsigned long long __activemask()

// Shuffle (warpSize=64 on CDNA)
T __shfl(T var, int srcLane, int width=warpSize)
T __shfl_down(T var, unsigned delta, int width=warpSize)
T __shfl_up(T var, unsigned delta, int width=warpSize)
T __shfl_xor(T var, int laneMask, int width=warpSize)

// Warp reductions
T __reduce_add_sync(unsigned long long mask, T var)
T __reduce_min_sync(unsigned long long mask, T var)
T __reduce_max_sync(unsigned long long mask, T var)
```

## Atomic Operations

```cpp
// Integer + Float: atomicAdd, atomicSub, atomicMin, atomicMax, atomicExch, atomicCAS
// Bitwise: atomicAnd, atomicOr, atomicXor
// Counter: atomicInc, atomicDec
// Safe FP: safeAtomicAdd(addr, val)    — always correct
// Fast FP: unsafeAtomicAdd(addr, val)  — hardware-accelerated if available
// NOTE: Atomics execute at L2 (device coherence point); avoid in inner loops
```

## Memory Management (Host APIs)

```cpp
hipMalloc(ptr, size)                          // Device memory
hipMallocManaged(ptr, size)                   // Unified memory
hipHostMalloc(ptr, size, flags)               // Pinned host memory
hipMemcpy(dst, src, size, kind)               // Sync copy
hipMemcpyAsync(dst, src, size, kind, stream)  // Async copy
hipMemcpyPeerAsync(dst, dDev, src, sDev, sz, stream)  // P2P
hipFree(ptr)
```

## Streams and Events

```cpp
hipStreamCreate(&stream)
hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)
hipStreamSynchronize(stream)
hipEventCreate(&event)
hipEventRecord(event, stream)
hipEventSynchronize(event)
hipStreamWaitEvent(stream, event, 0)  // stream waits for event
```

## Kernel Launch

```cpp
// Triple-chevron syntax
kernel<<<gridDim, blockDim, sharedMem, stream>>>(args...);

// Dynamic shared memory attribute (MUST call before launch if > default)
hipFuncSetAttribute(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, bytes);

// Launch bounds (compiler hint for register optimization)
__global__ void __launch_bounds__(MAX_THREADS, MIN_BLOCKS) kernel(...);
```

## Cooperative Groups

```cpp
#include <hip/hip_cooperative_groups.h>
using namespace cooperative_groups;

thread_block g = this_thread_block();          // Block scope
grid_group gg = this_grid();                   // Grid scope (cooperative launch)
thread_block_tile<32> tile = tiled_partition<32>(g);  // Sub-warp (power of 2)
coalesced_group active = coalesced_threads();  // Active lanes only

g.sync();           // Synchronize group
g.thread_rank();    // Lane position in group
g.size();           // Group size

// Cooperative launch (enables grid_group sync)
hipLaunchCooperativeKernel(kernel, gridDim, blockDim, sharedMem, stream, args);
```

## Hardware Constants (gfx950 / CDNA4)

| Property | Value |
|----------|-------|
| Warp size | 64 threads |
| Max threads/block | 1024 |
| Max warps/CU | 40 (theoretical) |
| VGPR per CU | 512 KiB (256 per thread × occupancy) |
| AGPR per CU | 512 KiB (separate accumulator file) |
| LDS per CU | 256 KB (CDNA4), 128 KB (CDNA3) |
| LDS banks | 64 (CDNA4), 32 (CDNA3) |
| LDS peak BW | 256 B/cycle (CDNA4), 128 B/cycle (CDNA3) |
| L1 cache | 16 KB per CU (write-through) |
| L2 cache | Shared across CUs, 32 channels, 256 B interleave |
| XCDs | 32 (gfx950), 8 CUs per XCD |
| Peak inst issue | 5/cycle: 1 VALU + 1 VMEM + 1 SALU/SMEM + 1 LDS + 1 branch |

## Occupancy Rules

| Resource | Occupancy=1 | Occupancy=2 | Occupancy=4 |
|----------|-------------|-------------|-------------|
| VGPR | ≤512 | ≤256 | ≤128 |
| LDS | ≤128KB | ≤80KB | ≤32KB |

- Occupancy is a step function — 257 VGPR = occupancy=1 regardless of other resources
- Higher occupancy hides memory latency; compute-bound kernels can run at occupancy=1
- Query occupancy: `hipOccupancyMaxActiveBlocksPerMultiprocessor()`
