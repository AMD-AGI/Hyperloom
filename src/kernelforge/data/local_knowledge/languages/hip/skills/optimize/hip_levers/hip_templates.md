---
title: HIP — starting bodies that are already wave64-correct
kind: language
lever: hip_templates
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html
  - https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
---

# HIP starting templates

## Route here when
You already know the kernel's shape — reduction, elementwise, GEMM, multi-stream — and you want a body
that is correct on CDNA before you start optimizing it.

**The reason this card exists:** porting a CUDA kernel and finding the AMD deltas one failure at a time
is slow, and some of the deltas do not fail loudly. A 32-lane assumption inside a reduction produces
wrong numbers, not a crash. Start from a body that is already right.

## 1. Cross-lane operations, where 64 lanes changes the API
```cpp
unsigned long long active = __ballot(pred);   // 64-bit mask; bit i is lane i, i in 0..63
int count = __popcll(active);                 // 64-bit popcount — __popc is wrong here
float down = __shfl_down(val, 1);             // width defaults to warpSize, which is 64
float xed  = __shfl_xor(val, 16);
unsigned long long m = 0xFFFFFFFFFFFFFFFFull; // all 64 lanes participating
float r = __shfl_down_sync(m, val, 1);
```

Four things that differ from CUDA habit:

- **Masks are 64 bits** (`unsigned long long`). Hand one a 32-bit value and
  `amd_warp_sync_functions.h` fires a static assert — this one at least fails at compile time.
- **Prefer contiguous masks.** `0xFF` outperforms `0xFB` because the backend can select faster
  cross-lane instructions for prefix-shaped masks. Structure reductions over lanes `0..N-1` rather than
  a scattered subset.
- **None of these imply a memory barrier.** If you are ordering side effects, you still need
  `__syncthreads()` or an explicit fence.
- **`__shfl` on half is unsupported.** Shuffle as int or float, then repack.

### A block reduction that is correct on 64 lanes
```cpp
__device__ float wave_reduce_sum(float v) {            // reduce across 64 lanes
    for (int off = warpSize/2; off > 0; off >>= 1)     // 32, 16, 8, 4, 2, 1
        v += __shfl_down(v, off);
    return v;                                          // the sum ends up in lane 0
}

__global__ void block_reduce(const float* in, float* out, int n) {
    __shared__ float partial[64];
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float v = (tid < n) ? in[tid] : 0.0f;
    v = wave_reduce_sum(v);                            // within the wave
    int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
    if (lane == 0) partial[wave] = v;
    __syncthreads();
    if (wave == 0) {
        int nw = blockDim.x / warpSize;
        v = (lane < nw) ? partial[lane] : 0.0f;
        v = wave_reduce_sum(v);
        if (lane == 0) atomicAdd(out, v);              // hardware fp atomic with -munsafe-fp-atomics
    }
}
```

The loop begins at `warpSize/2`, which is **32** here and 16 on NVIDIA. That single initializer is the
line most often carried over unchanged from a CUDA reduction, and when it is wrong the kernel reduces
only half of each wave — quietly, and with plausible-looking output.

## 2. Grid-stride loops
```cpp
__global__ void saxpy(int n, float a, const float* __restrict__ x, float* __restrict__ y) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        y[i] = a*x[i] + y[i];                          // 128-bit coalesced when aligned
}

int cu; hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, 0);   // 256 on gfx950
saxpy<<<cu * 8, 256, 0, stream>>>(n, 2.0f, x, y);
```

Because adjacent lanes read adjacent addresses, the wave can issue `global_load_dwordx4`. Declaring the
data as `float4` or `int4` makes that easier for the compiler to prove.

**Query the CU count rather than writing a literal.** 304 is MI300X, 256 is gfx950, and a kernel that
hardcodes either becomes wrong on the next part.

## 3. Cooperative groups
```cpp
namespace cg = cooperative_groups;
cg::thread_block_tile<64> wave = cg::tiled_partition<64>(cg::this_thread_block());
for (int off = wave.size()/2; off > 0; off >>= 1) v += wave.shfl_down(v, off);
```

`thread_block_tile<N>` needs N to be a power of two and **no larger than 64** on CDNA — `<64>` is a full
wave, `<32>` is half of one. Grid-wide `cg::grid_group::sync()` exists but requires
`hipLaunchCooperativeKernel` and a fully resident grid, which constrains your launch geometry; reach for
it only when the algorithm genuinely needs it.

## 4. Streams, async copies, graphs
```cpp
hipStream_t s; hipStreamCreate(&s);
float* h; hipHostMalloc(&h, bytes);                    // pinned memory — required for real async DMA
hipMemcpyAsync(d, h, bytes, hipMemcpyHostToDevice, s);
kernel<<<grid, block, 0, s>>>(d, n);
hipStreamSynchronize(s);
```

Three operational notes:

- Copies overlap compute only across **separate streams**; order them with `hipEventRecord` and
  `hipStreamWaitEvent`.
- For multi-GPU, one process per GPU is the configuration that behaves predictably. Set
  `GPU_MAX_HW_QUEUES=2`, and turn off NUMA balancing for training runs.
- **HIP graphs are the answer to launch overhead in a decode loop**: capture with
  `hipStreamBeginCapture`, then `hipGraphInstantiate` and `hipGraphLaunch`. They are also a measurement
  tool — if a workload is launch-bound, capturing it into a graph is how you find out what its
  GPU-bound time actually is.

## 5. A tiled LDS GEMM, FMA path
```cpp
#define TM 64
#define TN 64
#define TK 16
__global__ void __launch_bounds__(256, 2)              // 4 waves; 2 waves/SIMD caps VGPR at 256
gemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
           float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TK][TM + 1];                   // the +1 is illustrative — see below
    __shared__ float Bs[TK][TN + 1];
    int tx = threadIdx.x, ty = threadIdx.y;            // 16x16
    int row0 = blockIdx.y*TM, col0 = blockIdx.x*TN;
    float acc[4][4] = {{0}};                           // 4x4 register micro-tile

    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int i=ty;i<TM;i+=16) for (int kk=tx;kk<TK;kk+=16) As[kk][i]=A[(row0+i)*K+(k0+kk)];
        for (int kk=ty;kk<TK;kk+=16) for (int j=tx;j<TN;j+=16) Bs[kk][j]=B[(k0+kk)*N+(col0+j)];
        __syncthreads();
        #pragma unroll
        for (int kk=0;kk<TK;++kk) {
            float a[4], b[4];
            #pragma unroll
            for (int i=0;i<4;++i) a[i]=As[kk][ty*4+i];
            #pragma unroll
            for (int j=0;j<4;++j) b[j]=Bs[kk][tx*4+j];
            #pragma unroll
            for (int i=0;i<4;++i)
                #pragma unroll
                for (int j=0;j<4;++j) acc[i][j] += a[i]*b[j];   // maps to FMA
        }
        __syncthreads();
    }
    /* store acc with bounds checking */
}
dim3 block(16,16);                                     // 256 threads = 4 waves
dim3 grid((N+TN-1)/TN, (M+TM-1)/TM);                   // aim for >= 1024 blocks
```

Two decisions in there are load-bearing on AMD. `__launch_bounds__(256, 2)` tells the compiler to fit
two waves per SIMD, which caps VGPR usage at 256 and keeps occupancy predictable. The 4×4 register
micro-tile is what makes the inner loop FMA-bound instead of LDS-bound — without it, every multiply
waits on a shared-memory read.

> **Do not take `+1` as the padding answer.** gfx950 has **64 banks**, so whether a given stride is
> conflict-free depends on `(byte_stride / 4) mod 64` for your element type and tile shape. The `+1`
> here is a placeholder for "some pad goes here." Derive the real one — `hip_lds_staging.md`.

## 6. Upgrading to the MFMA path
Swap the FMA inner loop for `__builtin_amdgcn_mfma_*` and change four things around it:

- Keep the accumulator in a **stable** `vector_size` variable that lives across the whole K-loop, so it
  stays resident in AGPRs rather than being spilled and reloaded.
- **Double-buffer the LDS tiles.** 160 KiB accommodates three or four stages on gfx950.
- Load with **`global_load_lds`** so tiles bypass VGPR staging entirely and free registers.
- Add `sched_group_barrier` only after the ISA has shown you that the default schedule is leaving the
  matrix core idle. Adding it speculatively usually makes things worse.

Full treatment: `hip_builtins.md` §1 and §4, plus `hip_lds_staging.md`.

## Verify
Every template here should produce a hot loop containing `global_load_dwordx4`, `ds_*_b128` wherever
LDS is involved, **no** `v_accvgpr_*` on the MFMA path, and `.private_segment_fixed_size: 0`.

If any of those four is off, the template is not the problem — read `hip_traps.md`, which indexes the
causes by symptom.

## Sources
- 64-bit masks, `__shfl` semantics, mask-shape performance, the half-float restriction:
  https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- Cooperative groups and the tile-size ceiling:
  https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html
- Streams, graphs, multi-GPU settings (`GPU_MAX_HW_QUEUES`), grid sizing:
  https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- The tiled LDS GEMM and its MFMA upgrade path:
  https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
