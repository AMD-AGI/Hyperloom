# HipKittens Development Guide

HipKittens is AMD's port of ThunderKittens — minimal, opinionated C++ embedded
primitives for fast AMD AI kernels. Built from the hardware up.

## Setup

```cpp
#include "kittens.cuh"
using namespace kittens;

// For gfx950: compile with -DKITTENS_CDNA4
// For gfx942: use cdna3 branch, compile without KITTENS_CDNA4
```

## Three-Level Tile Hierarchy

### Global Tiles (gl) — HBM data descriptors
```cpp
// Template: gl<dtype, batch, depth, rows, cols>
// Use -1 for dynamic dimensions
gl<bf16, -1, -1, -1, -1> A_gl(ptr, nullptr, nullptr, nullptr, M, K);
gl<bf16, 1, 1, 128, 64> A_gl_static(ptr);  // fixed-size
```

### Shared Tiles (st) — LDS with bank-conflict-free swizzling
```cpp
// Template: st<dtype, rows, cols, shape>
// Shapes: st_16x16_s, st_16x32_s, st_32x16_s, st_32x32_s, st_8x32_s
st<bf16, 64, 128, st_16x32_s> a_smem;  // 64 rows × 128 cols in LDS

// Allocate via shared_allocator (MUST be first thing in kernel)
extern __shared__ int __shm[];
shared_allocator al((int*)&__shm[0]);
auto (&tiles)[2] = al.allocate<st<bf16, 64, 64, st_16x32_s>, 2>();  // double-buffered
```

### Register Tiles (rt) — VGPR data for computation
```cpp
// Template: rt<dtype, rows, cols, layout, shape>
// Layouts: row_l (row-major), col_l (column-major)
// Shapes: rt_16x16, rt_16x32, rt_32x16, rt_32x32, rt_16x128 (FP8)
rt<float, 64, 64, row_l, rt_16x32_s> c_reg;  // Accumulator in registers
rt<bf16, 64, 64, col_l, rt_16x32_s> a_reg;   // Column-major for MMA input

// Row/Column vectors
rv<float, 64, col_l> max_vec;   // Column vector (one per row)
rv<float, 64, row_l> sum_vec;   // Row vector (one per column)
```

## Valid Tile Shape Combinations

| Data Type | Base Shape | MMA Instruction | Notes |
|-----------|-----------|-----------------|-------|
| bf16/fp16 | rt_16x16 | mfma_f32_16x16x32 | Standard |
| bf16/fp16 | rt_16x32 | mfma_f32_16x16x32 | Asymmetric tiles |
| bf16/fp16 | rt_32x16 | mfma_f32_32x32x16 | Wider M |
| bf16/fp16 | rt_32x32 | mfma_f32_32x32x16 | Large accumulator |
| fp8 | rt_16x128 | mfma_f32_16x16x128 | 128-element wide K |
| fp8 | rt_32x64 | mfma_f32_32x32x64 | Wide FP8 |

Constraint: Tile rows/cols must be divisible by base shape dimensions.

## Memory Transfer Operations

### Global → Shared (group-level, block-wide)
```cpp
using G = kittens::group<NUM_WARPS>;

// Basic load
G::load(shared_tile, global_tile, {row_coord, col_coord});

// With prefilled swizzled offsets (precompute for repeated access)
G::prefill_swizzled_offsets(shared_tile, global_tile, offsets);
G::load(shared_tile, global_tile, offsets);  // Faster repeated loads

// Internal: uses llvm_amdgcn_raw_buffer_load_lds for direct GMEM→LDS
```

### Shared → Register (warp-level)
```cpp
// Row-major: ds_read_b64 (stride=4) or ds_read_b128 (stride=8)
load(register_tile, shared_tile, {row_offset, col_offset});

// Column-major: ds_read_b64_tr_b16 for transposed reads
// (2-3 DS reads per element due to layout rearrangement)
load(register_tile_col, shared_tile);
```

### Register → Shared (warp-level)
```cpp
store(shared_tile, register_tile, {row_offset, col_offset});
// Row-major: Two ds_write_b64 per 8-stride tile
// Column-major: Element-wise ds_write_b64
```

### Register → Global (warp-level)
```cpp
store(global_tile, register_tile, {row_coord, col_coord});
// Uses buffer_store_b64/b128 with automatic type conversion
```

### Global → Register (warp-level, direct)
```cpp
load(register_tile, global_tile, {row_coord, col_coord});
// Uses buffer_load_b64/b128 — bypasses shared memory
// Good for non-reused data (output stores, single-use reads)
```

## Compute Operations

### Matrix Multiply-Accumulate (MMA)
```cpp
// D = A × B + C (accumulate)
mma_AB(d_reg, a_reg, b_reg, c_reg);    // A row, B col
mma_ABt(d_reg, a_reg, b_reg, c_reg);   // A row, B row (B transposed)
mma_AtB(d_reg, a_reg, b_reg, c_reg);   // A col, B col (A transposed)
mma_AtBt(d_reg, a_reg, b_reg, c_reg);  // A col, B row (both transposed)

// Available MFMA intrinsics per shape:
// mfma_f32_16x16x32_bf16:  16×16 output, 32-element reduction
// mfma_f32_32x32x16_bf16:  32×32 output, 16-element reduction
// mfma_f32_32x32x32_bf16:  32×32 output, 32-element reduction (cascaded)
// mfma_f32_16x16x128_f8:   16×16 output, 128-element FP8 reduction
// mfma_f32_32x32x64_f8:    32×32 output, 64-element FP8 reduction
```

### Element-Wise Operations
```cpp
// Unary maps
zero(tile);                    // Set all elements to 0
exp2(dst, src);                // Element-wise exp2
log2(dst, src);                // Element-wise log2
abs(dst, src);                 // Element-wise absolute value
relu(dst, src);                // Element-wise max(0, x)

// Binary maps (tile-tile)
add(dst, a, b);                // dst = a + b
sub(dst, a, b);                // dst = a - b
mul(dst, a, b);                // dst = a * b
div(dst, a, b);                // dst = a / b
max(dst, a, b);                // dst = max(a, b)
min(dst, a, b);                // dst = min(a, b)

// Binary maps (tile-scalar)
mul(dst, src, scalar);         // dst = src * scalar

// Row/Column maps (apply vector along axis)
row_map<base_ops::mul>(dst, src, col_vec);   // Multiply each row by vec
col_map<base_ops::add>(dst, src, row_vec);   // Add vec to each column
sub_row(dst, src, col_vec);                   // Subtract vec from each row
sub_col(dst, src, row_vec);                   // Subtract vec from each col
```

### Reductions
```cpp
// Row reduction: matrix → column vector
row_reduce<base_ops::sum>(col_vec, tile);      // Sum across columns
row_reduce<base_ops::max>(col_vec, tile);      // Max across columns
col_max(col_vec, tile);                         // Shorthand for row_reduce max

// Column reduction: matrix → row vector
col_reduce<base_ops::sum>(row_vec, tile);      // Sum down rows
col_sum(row_vec, tile);                         // Shorthand

// Internal: Uses __shfl_down + __builtin_amdgcn_permlane32_swap for wide reductions
```

### Type Conversions
```cpp
// Between tile types
copy(dst_tile, src_tile);      // Type-converting copy (e.g., bf16 → float)
conv(dst_tile, src_tile);      // Explicit type conversion

// Layout swaps (row ↔ column)
swap_layout(dst_col, src_col);  // Between column-major shapes
// Uses __builtin_amdgcn_permlane16_swap internally

// Transpose
transpose(dst, src);           // Swap layout AND transpose the matrix
```

## Scheduling Patterns

### Scheduling Group Barriers (CDNA4)
```cpp
// Fine-grained instruction scheduling
__builtin_amdgcn_sched_group_barrier(mask, count, group);
// mask: which pipeline — 0x08=MFMA, 0x02=VALU, 0x01=VMEM, 0x400=EXP
// count: number of instructions to schedule from this pipeline
// group: scheduler group ID (0-15)

// Template helper for MMA+VALU interleaving
template<int Pairs, int VALU_CNT, int Group>
__device__ static void sched_barrier_pairs() {
    __builtin_amdgcn_sched_group_barrier(0x08, 1, Group);     // 1 MFMA
    __builtin_amdgcn_sched_group_barrier(0x02, VALU_CNT, Group); // N VALU
    if constexpr (Pairs > 1)
        sched_barrier_pairs<Pairs - 1, VALU_CNT, Group>();
}

// For EXP operations
template<int Pairs, int EXP_CNT, int Group>
__device__ static void sched_barrier_exp_pairs() {
    __builtin_amdgcn_sched_group_barrier(0x08, 1, Group);
    __builtin_amdgcn_sched_group_barrier(0x400, EXP_CNT, Group);
}
```

### Pipeline Patterns

**8-Wave Ping-Pong (GEMM)**
- 8 warps split into 2 groups of 4
- Group A computes while Group B loads; swap each K-step
- Double-buffered shared memory

**4-Wave Interleave (Attention)**
- 4 warps with fine-grained MMA/load/softmax interleaving
- Scheduler barriers enforce instruction ordering
- 8-cluster main loop + 12-cluster epilogue

## Chiplet-Aware Block Mapping

```cpp
// CDNA4: 32 XCDs, keep workgroups in same XCD for L2 locality
auto [block_m, block_n] = chiplet_transform_chunked(blockIdx.x, grid_m, grid_n);

// chiplet_transform_chunked maps linear block IDs to 2D tile coordinates
// while distributing across XCDs to maximize L2 cache reuse
```

## Buffer Resource Descriptors

```cpp
// Create SRSRC for buffer operations (raw_buffer_load, raw_buffer_store)
auto srsrc = make_srsrc(base_ptr, stride);

// Or create buffer resource
auto br = make_buffer_resource(base_ptr);

// Used in: global_to_shared loads (llvm_amdgcn_raw_buffer_load_lds)
//          global_to_register loads (buffer_load_b64/b128)
//          register_to_global stores (buffer_store_b64/b128)

// readfirstlane hoisting: broadcast SRD to all lanes
__builtin_amdgcn_readfirstlane(scalar_val);  // Uniform SRD across warp
```

## Common Constants

```cpp
WARP_THREADS = 64                   // AMD warp size
MAX_SHARED_MEMORY = 96 * 1024       // 96 KB default dynamic shared
KITTENS_DEFAULT_ALIGN = 128         // Byte alignment

// Helper functions
laneid()        // Current lane (0-63)
warpid()        // Current warp index
ceil_div(a, b)  // Safe ceiling division
```

## Reference Kernel Summaries

### GEMM (bf16fp32, 256×256×64)
- 8 warps in 2×4 grid
- Double-buffered `st<bf16>` shared tiles
- `rt<bf16, ..., rt_16x32_s>` register tiles
- `chiplet_transform_chunked` for XCD locality
- `readfirstlane` hoisting for shared address gen
- File: `${KA_WORKSPACE}/HipKittens/kernels/gemm/bf16fp32/`

### GQA Forward Attention
- 8 warps, 8-cluster pipeline in main loop
- Online softmax with rescaling (threshold=8)
- `sched_group_barrier` for MFMA/VALU/EXP interleaving
- Causal and non-causal variants
- File: `${KA_WORKSPACE}/HipKittens/kernels/attn/gqa/`

### LayerNorm
- Memory-bound, reduction-heavy
- Row-reduce for mean/variance
- Fused with scale/bias
- File: `${KA_WORKSPACE}/HipKittens/kernels/layernorm/`

### Rotary Embeddings
- Pure memory-bound operation
- Direct global→register→global
- File: `${KA_WORKSPACE}/HipKittens/kernels/rotary/`
