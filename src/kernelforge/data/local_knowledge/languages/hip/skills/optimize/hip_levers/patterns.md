---
title: HIP / C++ — kernel patterns (reductions, GEMM, streams/graphs)
kind: language
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [both]
status: sota
updated: 2026-07-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html
  - https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
---

# HIP — kernel patterns

## 1. Wave / cross-lane primitives (64-bit masks)
This is where wave64 changes the API surface most vs CUDA.
```cpp
unsigned long long active = __ballot(pred);   // 64-bit mask (bit i = lane i in 0..63)
int count = __popcll(active);                  // popcount over 64 bits — NOT __popc
float down = __shfl_down(val, 1);              // width defaults to warpSize (64)
float xed  = __shfl_xor(val, 16);
unsigned long long m = 0xFFFFFFFFFFFFFFFFull;  // all 64 lanes
float r = __shfl_down_sync(m, val, 1);
```
- Mask type **must be 64-bit** (`unsigned long long`); a 32-bit mask triggers a static-assert in
  `amd_warp_sync_functions.h`.
- **Contiguous, hole-free masks are faster** (`0xFF` beats `0xFB`) — the backend uses faster cross-lane
  ops for prefix masks. Reduce over `0..N-1`, not scattered lanes.
- Intrinsics carry **no memory barrier** — add `__syncthreads()`/fences for ordering of side effects.
- Half-float `__shfl` is **not** supported — shuffle as int/float and repack.

### Wave64-correct block reduction
```cpp
__device__ float warp_reduce_sum(float v) {            // 64-lane reduce
    for (int off = warpSize/2; off > 0; off >>= 1)     // 32,16,8,4,2,1
        v += __shfl_down(v, off);
    return v;                                           // lane 0 holds the sum
}
__global__ void block_reduce(const float* in, float* out, int n) {
    __shared__ float partial[64];
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float v = (tid < n) ? in[tid] : 0.0f;
    v = warp_reduce_sum(v);                             // intra-wave (64 lanes)
    int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
    if (lane == 0) partial[wave] = v;
    __syncthreads();
    if (wave == 0) {
        int nw = blockDim.x / warpSize;
        v = (lane < nw) ? partial[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) atomicAdd(out, v);              // HW fp atomic (-munsafe-fp-atomics)
    }
}
```

## 2. Grid-stride loops (full-occupancy, coalesced)
```cpp
__global__ void saxpy(int n, float a, const float* x, float* y) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        y[i] = a*x[i] + y[i];                          // 128-bit coalesced if aligned
}
int blocks = 304 * 8;                                  // saturate the device
saxpy<<<blocks, 256, 0, stream>>>(n, 2.0f, x, y);
```
Consecutive lanes touch consecutive addresses → wave issues `global_load_dwordx4`. Use `float4`/`int4`.

## 3. Cooperative groups
```cpp
namespace cg = cooperative_groups;
cg::thread_block_tile<64> wave = cg::tiled_partition<64>(cg::this_thread_block());
for (int off = wave.size()/2; off > 0; off >>= 1) v += wave.shfl_down(v, off);
```
`thread_block_tile<N>`: N a power of 2, **≤ 64** on CDNA (`<64>` = full wave, `<32>` = half). Grid-wide
`cg::grid_group::sync()` needs `hipLaunchCooperativeKernel` + a resident grid — use sparingly.

## 4. Streams, async copy, graphs
```cpp
hipStream_t s; hipStreamCreate(&s);
float* h; hipHostMalloc(&h, bytes);                    // pinned → true async DMA
hipMemcpyAsync(d, h, bytes, hipMemcpyHostToDevice, s);
kernel<<<grid, block, 0, s>>>(d, n);
hipStreamSynchronize(s);
```
- Overlap copy/compute on **separate streams**; sequence with `hipEventRecord`/`hipStreamWaitEvent`.
- Multi-GPU: prefer **one process per GPU**; `GPU_MAX_HW_QUEUES=2`; disable NUMA balancing for training.
- **HIP graphs** kill per-launch overhead in decode loops: `hipStreamBeginCapture` →
  `hipGraphInstantiate` → `hipGraphLaunch`.

## 5. Tiled LDS GEMM skeleton (CDNA3-tuned, FMA path)
```cpp
#define TM 64
#define TN 64
#define TK 16
__global__ void __launch_bounds__(256, 2)              // 4 waves, 2 waves/EU (VGPR ≤ 256)
gemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
           float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TK][TM + 1];                  // +1 pad kills 32-bank conflicts
    __shared__ float Bs[TK][TN + 1];
    int tx = threadIdx.x, ty = threadIdx.y;            // 16×16
    int row0 = blockIdx.y*TM, col0 = blockIdx.x*TN;
    float acc[4][4] = {{0}};                            // 4×4 register micro-tile
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int i=ty;i<TM;i+=16) for (int kk=tx;kk<TK;kk+=16) As[kk][i]=A[(row0+i)*K+(k0+kk)];
        for (int kk=ty;kk<TK;kk+=16) for (int j=tx;j<TN;j+=16) Bs[kk][j]=B[(k0+kk)*N+(col0+j)];
        __syncthreads();
        #pragma unroll
        for (int kk=0;kk<TK;++kk) {
            float a[4],b[4];
            #pragma unroll
            for (int i=0;i<4;++i) a[i]=As[kk][ty*4+i];
            #pragma unroll
            for (int j=0;j<4;++j) b[j]=Bs[kk][tx*4+j];
            #pragma unroll
            for (int i=0;i<4;++i)
                #pragma unroll
                for (int j=0;j<4;++j) acc[i][j]+=a[i]*b[j];   // FMA-mapped
        }
        __syncthreads();
    }
    /* store acc with bounds check */
}
dim3 block(16,16);                                     // 256 threads = 4 waves
dim3 grid((N+TN-1)/TN, (M+TM-1)/TM);                   // aim ≥1024 blocks
```
AMD specifics: `__launch_bounds__(256,2)` keeps VGPR ≤ 256; LDS padded `+1`; 4×4 register blocking
makes the inner loop FMA-bound not LDS-bound. The peak path replaces the FMA loop with
`__builtin_amdgcn_mfma_*` + LDS double-buffering — see [intrinsics.md](intrinsics.md) and
[lds_async.md](lds_async.md).

## 6. MFMA microkernel (peak path)
Carry the `fp32x4` accumulator in a stable variable across the K-loop (stays in AGPRs), double-buffer
the LDS tiles, use `global_load_lds` to skip VGPR staging, and place `sched_group_barrier` to keep the
matrix core fed. Full annotated skeleton in [intrinsics.md](intrinsics.md) §1/§4 and the AMD lab notes.

## Sources
- 64-bit masks, __shfl, hole-free mask perf, half-float shuffle: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- Cooperative groups (tile ≤ warpSize): https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html
- Streams/graphs/multi-GPU (GPU_MAX_HW_QUEUES, ≥1024 grid): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Tiled LDS GEMM / MFMA upgrade path: https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
