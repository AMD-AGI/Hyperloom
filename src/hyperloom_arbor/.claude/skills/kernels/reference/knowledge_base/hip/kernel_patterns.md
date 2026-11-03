# HIP Kernel Patterns

## Kernel Structure Template

### Raw HIP Kernel Skeleton
```cpp
#include <hip/hip_runtime.h>

// Tile configuration as compile-time constants
constexpr int BLOCK_M = 256;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 128;
constexpr int WARP_SIZE = 64;
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = WARP_SIZE * NUM_WARPS;

// Shared memory tile with swizzling
struct SmemTile {
    uint8_t data[BLOCK_M * BLOCK_K];
    __device__ uint32_t u32_ptr() const {
        return static_cast<uint32_t>(__builtin_amdgcn_ds_offset(data));
    }
};

__global__ void __launch_bounds__(NUM_THREADS)
my_kernel(const void* __restrict__ A,
          const void* __restrict__ B,
          void* __restrict__ C,
          int M, int N, int K) {
    // 1. Compute workgroup/warp/lane IDs
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // 2. Reserve registers (prevent allocator drift)
    reserve_agpr_range<0, 255>();
    reserve_vgpr_range<PIN_START, 255>();

    // 3. Setup BufferSRD for bounds-checked GMEM access
    BufferSRD srd_a, srd_b;
    setup_srd(&srd_a, A, M * K * sizeof(type));
    setup_srd(&srd_b, B, K * N * sizeof(type));

    // 4. Prologue: preload first tiles into SMEM
    load_gmem_to_smem_srd(srd_a, smem_a[0], ...);
    load_gmem_to_smem_srd(srd_b, smem_b[0], ...);
    __builtin_amdgcn_s_barrier();

    // 5. Load first operands to registers
    ds_read_pinned<16, VGPR_A0, OFFSET_A0>(smem_a[0].u32_ptr());
    ds_read_pinned<16, VGPR_B0, OFFSET_B0>(smem_b[0].u32_ptr());

    // 6. Main loop (shifted-LDG pipeline)
    for (int ki = 0; ki < K_iters - 2; ki++) {
        // Phase 1-4: interleaved MFMA + LDS_READ + GMEM_LOAD
        // (see Software Pipeline section)
    }

    // 7. Epilogue phases (drain pipeline)

    // 8. Store output
    store_output(C, acc_regs, M, N, warp_id, lane_id);
}
```

## BufferSRD Pattern (Production Implementation)

```cpp
struct BufferSRD {
    int32x4_t srd;  // 128-bit packed descriptor

    __device__ __forceinline__ explicit BufferSRD(
        const void* base_ptr,
        uint32_t num_bytes = 0xffffffffu  // Default: full range
    ) {
        struct __attribute__((packed)) {
            const void* p;       // 64-bit base pointer
            uint32_t    r;       // num_records (size in bytes)
            uint32_t    c;       // 0x00020000 = NUM_RECORDS_OOB_SELECT
        } res{base_ptr, num_bytes, 0x00020000u};
        srd = __builtin_bit_cast(int32x4_t, res);
        // SRD MUST be uniform across all lanes
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            srd[i] = __builtin_amdgcn_readfirstlane(srd[i]);
    }
};
```

## MFMA Accumulation Pattern

```cpp
// Zero accumulator in AGPR
zero_agpr_range<0, 255>();

// Main compute: MFMA with builtin + anti-eviction barrier
for (int tile = 0; tile < num_tiles; tile++) {
    c = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a_frag, b_frag, c, 0, 0, 0);
    asm volatile("" : "+v"(c));  // keep c live in registers
}

// Read results from AGPR for store
float32x4 result = read_agpr<float32x4, ACC_START>();
```

## Scaled MFMA Pattern (MXFP8)

```cpp
// Microscaling: FP8 data + E8M0 per-block scale factors
// MX_BLOCK_SIZE = 32 elements per scale factor

// Pre-shuffle scales into MFMA-consumable layout (separate kernel)
__global__ void preshuffle_scales(const uint8_t* scales_in, uint32_t* scales_out,
                                   int rows, int cols) {
    // Reorganize E8M0 scales from row-major to 16x4 blocks
    // Zero-extend uint8 → uint32 for MFMA scale operand
}

// Main kernel: scaled MFMA
__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
    a_data,     // 8 VGPR (128 FP8 elements)
    b_data,     // 8 VGPR (128 FP8 elements)
    c_acc,      // 4 AGPR (16 FP32 results)
    cbsz,       // A format: 0=fp8e4m3, 1=fp8e5m2, etc.
    blgp,       // B format
    0,           // A scale select
    scale_a,    // 1 VGPR: A microscaling factor
    0,           // B scale select
    scale_b     // 1 VGPR: B microscaling factor
);
```

## Workgroup Swizzling Pattern

```cpp
// XCD-aware block mapping for L2 locality
__device__ void compute_tile_indices(int block_id, int grid_m, int grid_n,
                                      int& pid_m, int& pid_n) {
    constexpr int NUM_XCDS = 32;  // gfx950
    int wgm = (grid_n > 32) ? 4 : 8;

    int tiles_per_group = wgm * grid_n;
    int group_id = block_id / tiles_per_group;
    int local_id = block_id % tiles_per_group;

    pid_m = group_id * wgm + local_id % wgm;
    pid_n = local_id / wgm;

    // Clamp for partial groups at grid edge
    if (pid_m >= grid_m) {
        pid_m = grid_m - 1;
    }
}
```

## Four-Phase Pipeline (Production Pattern from MXFP8 GEMM)

Each K-iteration processes 4 subtiles with interleaved MFMA, LDS read, and GMEM load:

```
Phase 1: MFMA(A0×B0) + LDS_READ(B1) + GMEM_LOAD(ki+2 B)
  - 16 MFMA ops (4×4 tile grid)
  - 8 ds_read_b128 for B1 data prefetch
  - 4 ds_read_b32 for B1 scale prefetch
  - 4 buffer_load_lds for next K-iter B data

Phase 2: MFMA(A0×B1) + LDS_READ(A1) + GMEM_LOAD(ki+2 A)
  - 16 MFMA ops
  - LDS prefetch A1 data + scales
  - GMEM prefetch A for ki+2

Phase 3: MFMA(A1×B0) + LDS_READ(A0') + GMEM_LOAD(ki+3 A)
  - 16 MFMA ops
  - LDS prefetch A0 from NEXT buffer (ping-pong swap)
  - GMEM prefetch A for ki+3

Phase 4: MFMA(A1×B1) + LDS_READ(B0') + GMEM_LOAD(ki+3 B)
  - 16 MFMA ops
  - LDS prefetch B0 from NEXT buffer
  - GMEM prefetch B for ki+3
```

**Latency hiding budget:** 64 MFMA × ~8 cycles = ~512 cycles compute per K-iteration,
covers typical GMEM latency (~400-600 cycles).

**Epilogue Phases (drain pipeline after main loop):**
```
Epilogue 1: MFMA + LDS + last GMEM load    (ki = K_iters - 2)
Epilogue 2: MFMA + LDS only, no GMEM       (2 phases)
Epilogue 3: MFMA + LDS only                (2 phases, from last buffer)
Epilogue 4: Pure MFMA, no loads             (2 phases, drain accumulator)
```

**Phase Template Functions:**
```cpp
template<int PIN_A, int PIN_AS, int PIN_B, int PIN_BS,
         int ACC_ROW, int ACC_COL,
         int PIN_NEXT_B, int PIN_NEXT_BS>
__device__ void phase_mfma_lds_ldg(...);   // MFMA + LDS read + GMEM load

template<int PIN_A, int PIN_AS, int PIN_B, int PIN_BS,
         int ACC_ROW, int ACC_COL>
__device__ void phase_mfma_lds(...);       // MFMA + LDS read only

template<int PIN_A, int PIN_AS, int PIN_B, int PIN_BS,
         int ACC_ROW, int ACC_COL>
__device__ void phase_mfma_only(...);      // Pure MFMA only
```

## Shared Memory Double-Buffer Pattern

```cpp
extern __shared__ uint8_t __shm[];

// Layout: [2 buffers] × [subtiles] × [data + scales]
struct SmemLayout {
    SmemTile a_data[2][NUM_SUBTILES];  // Double-buffered A
    SmemTile b_data[2][NUM_SUBTILES];  // Double-buffered B
    uint32_t a_scale[2][NUM_SUBTILES][SCALE_SIZE];
    uint32_t b_scale[2][NUM_SUBTILES][SCALE_SIZE];
};

auto* smem = reinterpret_cast<SmemLayout*>(__shm);

// Ping-pong buffer index
int buf = 0;
for (int ki = 0; ki < K_iters; ki++) {
    // Read from smem->a_data[buf], smem->b_data[buf]
    // Write next tile to smem->a_data[buf ^ 1], smem->b_data[buf ^ 1]
    buf ^= 1;
}
```

## HipKittens Kernel Skeleton

```cpp
#include "kittens.cuh"
using namespace kittens;

constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = kittens::WARP_THREADS * NUM_WARPS;
using G = kittens::group<NUM_WARPS>;

// Tile type definitions
using a_tile = st<bf16, TILE_H, TILE_W, st_16x32_s>;
using b_tile = st<bf16, TILE_H, TILE_W, st_16x32_s>;
using c_tile = rt<float, TILE_H, TILE_W, row_l, rt_16x32_s>;

__global__ void __launch_bounds__(NUM_THREADS)
hk_kernel(gl<bf16, -1, -1, -1, -1> A_gl,
          gl<bf16, -1, -1, -1, -1> B_gl,
          gl<float, -1, -1, -1, -1> C_gl) {

    extern __shared__ int __shm[];
    shared_allocator al((int*)&__shm[0]);  // MUST be first

    // Allocate shared tiles
    a_tile (&a_smem)[2] = al.allocate<a_tile, 2>();  // double-buffered
    b_tile (&b_smem)[2] = al.allocate<b_tile, 2>();

    // Register tiles for accumulation
    c_tile c_reg;
    zero(c_reg);

    // Chiplet-aware block mapping
    auto [block_m, block_n] = chiplet_transform_chunked(blockIdx.x, grid_m, grid_n);

    // Main loop
    for (int k = 0; k < K_tiles; k++) {
        // Load: global → shared
        G::load(a_smem[k & 1], A_gl, {block_m, k});
        G::load(b_smem[k & 1], B_gl, {k, block_n});
        __syncthreads();

        // Compute: shared → register → MMA
        rt<bf16, TILE_H, TILE_W, row_l, rt_16x32_s> a_reg, b_reg;
        load(a_reg, a_smem[k & 1]);
        load(b_reg, b_smem[k & 1]);
        mma(c_reg, a_reg, b_reg);

        __syncthreads();
    }

    // Store: register → global
    G::store(C_gl, c_reg, {block_m, block_n});
}

// Launch
void launch(const bf16* A, const bf16* B, float* C, int M, int N, int K) {
    auto A_gl = gl<bf16, -1, -1, -1, -1>(A, nullptr, nullptr, nullptr, M, K);
    auto B_gl = gl<bf16, -1, -1, -1, -1>(B, nullptr, nullptr, nullptr, K, N);
    auto C_gl = gl<float, -1, -1, -1, -1>(C, nullptr, nullptr, nullptr, M, N);

    int grid = (M / BLOCK_M) * (N / BLOCK_N);
    int smem_size = /* computed from tile sizes */;
    hipFuncSetAttribute(hk_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    hk_kernel<<<grid, NUM_THREADS, smem_size>>>(A_gl, B_gl, C_gl);
}
```

## PyTorch Integration Pattern

```cpp
#include <torch/extension.h>

torch::Tensor my_kernel_wrapper(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous());
    TORCH_CHECK(B.is_cuda() && B.is_contiguous());

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));

    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(NUM_THREADS);
    int smem_size = sizeof(SmemLayout);

    hipFuncSetAttribute(my_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    my_kernel<<<grid, block, smem_size, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_kernel", &my_kernel_wrapper, "My HIP kernel");
}
```

## Reference Code Locations

### Mxfp8 GEMM Example (raw HIP)
- Kernel: `${KA_WORKSPACE}/Primus-Turbo/csrc/kernels/gemm/turbo/turbo_gemm_mxfp8_kernel.h`
- Helpers: `${KA_WORKSPACE}/Primus-Turbo/csrc/include/primus_turbo/device/`
  - `mfma.cuh` — MFMA instruction wrappers
  - `memory.cuh` — BufferSRD, buffer_load_lds, ds_read_pinned
  - `register.cuh` — AGPR/VGPR pinning, read, zero, reserve
  - `dtype.h` — float32x4, int32x8, FP8 type definitions

### HIP AMD Documentation
- API docs: `${KA_WORKSPACE}/hip-amd/docs/`
- Examples: `${KA_WORKSPACE}/hip-amd/docs/tools/example_codes/`
- Low precision types: `${KA_WORKSPACE}/hip-amd/docs/reference/low_fp_types.rst`
- Hardware: `${KA_WORKSPACE}/hip-amd/docs/understand/hardware_implementation.rst`

### HipKittens
- Main header: `${KA_WORKSPACE}/HipKittens/include/kittens.cuh`
- Tile types: `${KA_WORKSPACE}/HipKittens/include/types/`
- Operations: `${KA_WORKSPACE}/HipKittens/include/ops/`
- GEMM kernels: `${KA_WORKSPACE}/HipKittens/kernels/gemm/`
- Attention kernels: `${KA_WORKSPACE}/HipKittens/kernels/attn/`
- Profiling tools: `${KA_WORKSPACE}/HipKittens/docs/profiling/`
