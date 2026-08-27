# AITER ASM Kernel Writing Patterns (gfx950)

## Architecture: Three-Tier Kernel Writing

AITER supports three levels of kernel development, from highest to lowest abstraction:

### Tier 1: Triton JIT (Prototyping)
- Python-level `@triton.jit` kernels
- Auto-tunes via `@triton.autotune`
- Best for: rapid iteration, algorithm validation
- Limitation: cannot control MFMA scheduling, register allocation, LDS layout

### Tier 2: CK-native HIP with Inline MFMA (Production)
- C++ HIP kernels using `__builtin_amdgcn_mfma_*` intrinsics
- Full control over: MFMA tile shape, LDS layout, vectorized loads, register usage
- Build via AITER JIT (`compile_ops` decorator, `ffi_type="ctypes"`)
- Example: `csrc/ck_mla_decode/mla_decode_fwd_kernel.hpp`

### Tier 3: Pre-compiled HSACO (.co) Binaries
- Hand-written AMDGPU ISA assembled offline
- Deployed as `.co` files in `hsa/gfx950/`
- Loaded via `HsacoLauncher` + `hipModuleLaunchKernel`
- Highest performance but not open-source

## Writing CK-native Kernels (Tier 2 — Primary Target)

### MFMA Intrinsics for gfx950

```cpp
// bf16 16x16x32 — the workhorse for MLA
__device__ float4 __builtin_amdgcn_mfma_f32_16x16x32_bf16(
    bf16x8_t a,   // 8 bf16 packed as short8
    bf16x8_t b,   // 8 bf16 packed as short8
    float4 c,     // 4 fp32 accumulator per lane
    int cbsz,     // 0
    int abid,     // 0
    int blgp      // 0
);

// Output layout per lane (64 lanes per warp):
//   C[lane/16 * 4 + row_idx, lane % 16] for row_idx in [0,3]
//   Each lane produces 4 rows in a 16-wide column

// fp8 variants (for quantized attention):
__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(...)
__builtin_amdgcn_mfma_f32_32x32x16_bf16(...)  // wider tile
```

### AGPR Pinning (CRITICAL for correctness)
```cpp
// NEVER use "+a" asm constraints — drops reg_idx=0, causes 21 dB SNR
// ALWAYS use builtins + "+v" barrier:
float4 c = {0.0f, 0.0f, 0.0f, 0.0f};
c = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
asm volatile("" : "+v"(c));  // pin accumulator to AGPR
// Result: 142 dB SNR, 170 AGPR utilization
```

### Vectorized Memory Loads
```cpp
// uint4 = 16 bytes = 8 bf16 values (dwordx4 on GCN)
using uint4 = __attribute__((__vector_size__(16))) unsigned int;

// Global → LDS vectorized load pattern:
const uint4* kv_vec = reinterpret_cast<const uint4*>(kv_ptr);
__shared__ bf16_t lds_kv[BLOCK_N][LDS_STRIDE];
uint4* lds_vec = reinterpret_cast<uint4*>(lds_kv);

for (int vi = tid; vi < TOTAL_VECS; vi += BLOCK_SIZE) {
    int row = vi / VECS_PER_ROW;
    int col = vi % VECS_PER_ROW;
    lds_vec[row * VECS_PER_ROW + col] = kv_vec[pidx * kv_stride + col];
}
__syncthreads();
```

### LDS Layout Design Rules

1. **Padding for bank conflict avoidance**: Add `LDS_PAD = 8` bytes per row
   - LDS has 32 banks × 4 bytes = 128 bytes per cycle
   - Without padding, threads in same warp hit same bank
   - `LDS_STRIDE = HEAD_DIM + LDS_PAD` (e.g., 576 + 8 = 584)

2. **Total LDS budget**: gfx950 has 64 KB LDS per CU
   - Occupancy=2 requires ≤ 32 KB per block
   - Occupancy=1 allows up to 64 KB
   - MLA decode uses 38.4 KB → occupancy=1 (acceptable for decode)

3. **Split LDS into regions**: KV data (large) + intermediate P matrix (small)
   ```
   LDS[0 .. KV_SIZE-1]         → KV tile (BLOCK_N × LDS_STRIDE × sizeof(bf16))
   LDS[KV_SIZE .. KV_SIZE+P_SIZE] → P matrix (MFMA_HEADS × BLOCK_N × sizeof(bf16))
   ```

### Thread-to-MFMA Mapping

```
Block: 256 threads = 4 warps × 64 lanes
Warp ID:  tid / 64        → [0,3]
Lane ID:  tid % 64        → [0,63]
Sub-group: lane_id / 16   → [0,3] (4 groups per warp, 16 lanes each)

MFMA 16x16 output:
  Row assignment: (lane/16)*4 + [0,1,2,3]  → 4 rows per lane
  Col assignment: lane%16                    → 1 column per lane
  Each MFMA produces 16×16 output = 256 elements = 4 per lane × 64 lanes
```

### Online Softmax Pattern (for Attention Kernels)
```cpp
// Per-head accumulators (maintained across KV tiles):
float rmax = -INFINITY;   // running max
float rsum = 0.0f;        // running exp-sum
float4 o_acc = {0};       // output accumulator

// Per KV tile:
float new_max = fmaxf(rmax, tile_max);
float rescale = expf(rmax - new_max);
rsum = rsum * rescale + tile_exp_sum * expf(tile_max - new_max);
rmax = new_max;
// Rescale output accumulator:
for (int i = 0; i < V_TILES; i++)
    o_acc[i] *= rescale;
// Accumulate new PV contribution:
o_acc += tile_pv * expf(tile_max - new_max);
```

### Warp Shuffle for Reduction
```cpp
// Row-max across 16 lanes (for softmax):
float val = score;
for (int offset = 8; offset > 0; offset >>= 1)
    val = fmaxf(val, __shfl_xor(val, offset, 16));
// After: all 16 lanes in a sub-group have the same max
```

## Dispatch Chain: Python → ASM Kernel

```
1. @compile_ops("module_name", ffi_type="ctypes")
2. JIT builds C++ module → .so (cached in aiter/jit/build/)
3. Module loads .co binary from hsa/gfx950/
4. Python wrapper: torch tensors → ctypes void*
5. hipModuleLaunchKernel(grid, block, shared_mem, stream, args)
```

### Key Environment Variables
- `AITER_REBUILD=0|1|2` — cache / rebuild / force-rebuild
- `ENABLE_CK=1` — enable ComposableKernel backend
- `GPU_ARCHS=gfx950` — target architecture
- `AITER_LOG_MORE=1` — verbose dispatch logging

## Performance Tuning Knobs

| Parameter | MLA Decode Value | Notes |
|-----------|-----------------|-------|
| BLOCK_N | 32 | KV sequence tile |
| MFMA_HEADS | 16 | Heads per MFMA tile |
| Q_PASSES | 2 | Q batches per KV tile |
| NUM_WARPS | 4 | Warps per block |
| MFMA shape | 16x16x32 bf16 | Primary MFMA tile |
| LDS_PAD | 8 | Bank conflict avoidance |
| Vectorized loads | uint4 (16B) | Maximum bandwidth |
| VGPR target | ≤256 | Occupancy=2 threshold |
