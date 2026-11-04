# MFMA Instructions & Register Operations (gfx950)

## MFMA Instruction Reference

### Scaled MFMA (FP8/FP6/FP4 with microscaling)

**v_mfma_scale_f32_16x16x128_f8f6f4**
- Output: 4 AGPR (16×16 = 256 fp32, grouped as 4×4 tiles)
- A input: 8 VGPR (128 FP8 elements packed in 32 bits each)
- B input: 8 VGPR (128 FP8 elements)
- Scale A: 1 VGPR (uint32, E8M0 microscaling factor)
- Scale B: 1 VGPR (uint32, E8M0 microscaling factor)
- Latency: ~32 cycles, throughput ~1 per 2 cycles

**Format encoding via cbsz/blgp:**
| cbsz | blgp | A format | B format |
|------|------|----------|----------|
| 0 | 0 | FP8 E4M3 | FP8 E4M3 |
| 1 | 0 | FP8 E5M2 | FP8 E4M3 |
| 0 | 1 | FP8 E4M3 | FP8 E5M2 |
| 1 | 1 | FP8 E5M2 | FP8 E5M2 |

**Inline asm (pinned registers):**
```asm
v_mfma_scale_f32_16x16x128_f8f6f4 a[ACC:ACC+3], v[A:A+7], v[B:B+7], a[ACC:ACC+3], v[SA], v[SB] op_sel_hi:[0,0,0]
; With format: append "cbsz:1" and/or "blgp:1"
```

**Builtin (compiler-managed registers):**
```cpp
__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
    a_data,   // int32x8: 8 VGPR
    b_data,   // int32x8: 8 VGPR
    c_acc,    // float32x4: 4 AGPR (input/output accumulator)
    cbsz,     // int: A format selector
    blgp,     // int: B format selector
    0,        // int: A scale index
    scale_a,  // int: A scale value
    0,        // int: B scale index
    scale_b   // int: B scale value
);
asm volatile("" : "+v"(c_acc));  // MANDATORY: prevent register eviction
```

### Standard MFMA (BF16/FP16)

| Instruction | M×N×K | Input | Output | VGPRs per operand |
|-------------|-------|-------|--------|-------------------|
| mfma_f32_16x16x32_bf16 | 16×16×32 | BF16 | FP32 | 4 VGPR A, 4 VGPR B, 4 AGPR C |
| mfma_f32_32x32x16_bf16 | 32×32×16 | BF16 | FP32 | 4 VGPR A, 4 VGPR B, 16 AGPR C |
| mfma_f32_32x32x32_bf16 | 32×32×32 | BF16 | FP32 | 8 VGPR A, 8 VGPR B, 16 AGPR C |
| mfma_f32_16x16x32_f16 | 16×16×32 | FP16 | FP32 | 4 VGPR A, 4 VGPR B, 4 AGPR C |

### Standard MFMA (FP8)

| Instruction | M×N×K | Input | Output | VGPRs per operand |
|-------------|-------|-------|--------|-------------------|
| mfma_f32_16x16x128_f8 | 16×16×128 | FP8 | FP32 | 8 VGPR A, 8 VGPR B, 4 AGPR C |
| mfma_f32_32x32x64_f8 | 32×32×64 | FP8 | FP32 | 8 VGPR A, 8 VGPR B, 16 AGPR C |

**Builtin pattern (standard MFMA):**
```cpp
c = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c, 0, 0, 0);
asm volatile("" : "+v"(c));  // Keep c live
```

## MFMA Lane Layout (gfx950)

- Output C_col = lane_id % 32 (FIXED across all MFMA variants)
- Output C_row = f(register_index) (varies by MFMA shape)
- For 16×16: 4 elements per lane, packed as float32x4
- For 32×32: 16 elements per lane, packed as float32x16

**CRITICAL:** This layout is architecture-specific. Never assume CUDA-like lane
mappings. Always verify with ISA documentation before coding store patterns.

## Register Reserve Functions

### AGPR Operations
```cpp
// Reserve AGPR range (prevent compiler from using these registers)
template<int START, int END>
__device__ __forceinline__ void reserve_agpr_range();

// Zero AGPR range (initialize accumulator)
template<int START, int END>
__device__ __forceinline__ void zero_agpr_range();
// Generates: v_accvgpr_write_b32 a[i], 0  for each register

// Read from AGPR to VGPR
template<typename T, int AC>
__device__ __forceinline__ T read_agpr() {
    // Generates: v_accvgpr_read_b32 %0, a[AC+i]  for each element
    // Returns: T (e.g., float32x4 for 4 consecutive AGPRs)
}
```

### VGPR Operations
```cpp
// Reserve VGPR range
template<int START, int END>
__device__ __forceinline__ void reserve_vgpr_range();

// Pinned read from LDS to specific VGPRs
template<int Bytes, int VDST, int IMM_OFFSET = 0>
__device__ __forceinline__ void ds_read_pinned(uint32_t lds_addr) {
    // Bytes=16: ds_read_b128 v[VDST:VDST+3], lds_addr offset:IMM_OFFSET
    // Bytes=4:  ds_read_b32 v[VDST], lds_addr offset:IMM_OFFSET
}
```

## Register Pinning Layout (Production Example: 256×256×128 MXFP8 GEMM)

```
VGPR Layout (256 total, all pinned):
  v[0:111]     Compiler-managed: addresses, pointers, loop variables
  v[112:115]   A scale buffer 0 (4 × uint32_t)
  v[116:119]   A scale buffer 1 (4 × uint32_t)
  v[120:123]   B scale buffer 0 (4 × uint32_t)
  v[124:127]   B scale buffer 1 (4 × uint32_t)
  v[128:159]   A data buffer 0 (4 frags × 8 VGPR = 32 VGPR)
  v[160:191]   A data buffer 1 (4 frags × 8 VGPR = 32 VGPR)
  v[192:223]   B data buffer 0 (4 frags × 8 VGPR = 32 VGPR)
  v[224:255]   B data buffer 1 (4 frags × 8 VGPR = 32 VGPR)

AGPR Layout (256 total):
  a[0:255]     C accumulator: 4 subtiles × 16 tiles × 4 fp32 = 256 AGPR
               Index: a[(row*2 + col)*64 + tile*4 + elem]
```

## BufferSRD (Shader Resource Descriptor)

```cpp
struct BufferSRD {
    int32x4_t srd;  // 128-bit descriptor

    __device__ __forceinline__ explicit BufferSRD(
        const void* base_ptr,
        uint32_t num_bytes = 0xffffffffu  // Default: full address range
    ) {
        struct __attribute__((packed)) {
            const void* p;       // Base pointer (64-bit)
            uint32_t    r;       // num_records (size in bytes)
            uint32_t    c;       // Control: 0x00020000 = NUM_RECORDS_OOB_SELECT
        } res{base_ptr, num_bytes, 0x00020000u};
        srd = __builtin_bit_cast(int32x4_t, res);
        // Broadcast to all lanes (SRD must be uniform)
        for (int i = 0; i < 4; ++i)
            srd[i] = __builtin_amdgcn_readfirstlane(srd[i]);
    }
};
```

**Control field values:**
- `0x00020000`: NUM_RECORDS mode, OOB returns 0
- `0x00027000`: Raw buffer mode, OOB returns 0 (alternative)

## GMEM → SMEM Direct Load

```cpp
template<int Bytes>
__device__ __forceinline__ void load_gmem_to_smem_srd(
    const BufferSRD& srd,
    uint32_t ldg_offset,   // Thread-local GMEM byte offset
    uint32_t lds_addr,     // LDS destination byte address
    int32_t soffset        // Scalar offset (SGPR-based)
) {
    // Intrinsic: llvm_amdgcn_raw_buffer_load_lds(srd, lds, Bytes, ldg_offset, soffset, 0, 0)
    // Bypasses VGPRs entirely: GMEM data goes directly to LDS
    // gfx950: Bytes = 4, 12, or 16
    // gfx942: Bytes = 4 only
}
```

## Wait Count Instructions

```cpp
template<int CNT>
__device__ __forceinline__ void wait_vmcnt() {
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(CNT) : "memory");
}

template<int CNT>
__device__ __forceinline__ void wait_lgkmcnt() {
    asm volatile("s_waitcnt lgkmcnt(%0)" : : "n"(CNT) : "memory");
}
```

**Strategy (from production kernel):**
- `wait_vmcnt<0>()` — Prologue: ensure initial GMEM loads complete
- `wait_vmcnt<12>()` — Main loop: allow 12 outstanding GMEM ops (latency hiding)
- `wait_vmcnt<6>()` — Epilogue transition: tighten outstanding ops
- `wait_lgkmcnt<0>()` — Before barrier: ensure all LDS reads complete

## Synchronization & Scheduling

```cpp
// Workgroup barrier
__builtin_amdgcn_s_barrier();

// Scheduling fence (prevent instruction reordering)
__builtin_amdgcn_sched_barrier(0);

// Fine-grained scheduling (CDNA4)
__builtin_amdgcn_sched_group_barrier(mask, count, group);
// mask: 0x08=MFMA, 0x02=VALU, 0x01=VMEM, 0x400=EXP
// count: how many instructions of this type to schedule
// group: scheduler group ID

// Broadcast lane 0 value to all lanes (for uniform descriptors)
__builtin_amdgcn_readfirstlane(value);

// Cross-lane permutation (CDNA4)
__builtin_amdgcn_permlane16_swap(src, offset);   // 16-lane swap
__builtin_amdgcn_permlane32_swap(src, offset);   // 32-lane swap
```

## Microscaling (MX) Scale Pre-Shuffle

Scales must be reorganized from row-major to MFMA-consumable 16×4 blocks before
the main kernel:

```cpp
// Separate pre-shuffle kernel
template<typename InType, typename OutType>
__global__ void preshuffle_scale_16x4_kernel(
    const InType* scales_in,     // E8M0 uint8 scales, row-major
    OutType* scales_out,         // uint32 scales, 16×4 block layout
    int rows, int cols
) {
    // Zero-extend uint8 → uint32 for MFMA scale operand
    // Rearrange from [rows, cols] to [rows/16, cols/4, 16, 4]
}

// Launch: one block per 16 rows, 64 threads
preshuffle_scale_16x4_kernel<<<rows / 16, 64, 0, stream>>>(
    scale_raw, scale_preshuffled, rows, scale_cols);
```

## LDS Data Loading Patterns (Pinned Registers)

### Data Subtile Load (32 VGPR = 4 × 8 VGPR fragments)
```cpp
template<int VSTART>
static void load_data_subtile_pinned(uint32_t subtile_addr, uint32_t (&lds_offsets)[2]) {
    uint32_t addr0 = subtile_addr + lds_offsets[0];
    uint32_t addr1 = subtile_addr + lds_offsets[1];
    // 8 × ds_read_b128 = 8 × 16 bytes = 128 bytes per thread
    ds_read_pinned<16, VSTART + 0,  0>(addr0);     // v[VSTART:VSTART+3]
    ds_read_pinned<16, VSTART + 4,  0>(addr1);     // v[VSTART+4:VSTART+7]
    ds_read_pinned<16, VSTART + 8,  2048>(addr0);  // +2KB offset
    ds_read_pinned<16, VSTART + 12, 2048>(addr1);
    ds_read_pinned<16, VSTART + 16, 4096>(addr0);  // +4KB offset
    ds_read_pinned<16, VSTART + 20, 4096>(addr1);
    ds_read_pinned<16, VSTART + 24, 6144>(addr0);  // +6KB offset
    ds_read_pinned<16, VSTART + 28, 6144>(addr1);
}
```

### Scale Subtile Load (4 VGPR = 4 × uint32)
```cpp
template<int VSTART>
static void load_scale_subtile_pinned(uint32_t addr, uint32_t offset) {
    uint32_t base = addr + offset * sizeof(uint32_t);
    ds_read_pinned<4, VSTART + 0, 0>(base);     // v[VSTART]
    ds_read_pinned<4, VSTART + 1, 256>(base);   // +256B
    ds_read_pinned<4, VSTART + 2, 512>(base);   // +512B
    ds_read_pinned<4, VSTART + 3, 768>(base);   // +768B
}
```

## Swizzle Formulas

### Column Swizzle (bank conflict avoidance)
```cpp
uint32_t swizzle_col(uint32_t row, uint32_t col) {
    return col ^ (row >> 1);
}
```

### LDS Read Offsets (with swizzle)
```cpp
void compute_lds_offsets(uint32_t (&lds_offsets)[2], int lane_id) {
    for (int i = 0; i < 2; ++i) {
        uint32_t lds_row = lane_id % 16;
        uint32_t lds_col = lane_id / 16 + i * 4;
        uint32_t swz_col = swizzle_col(lds_row, lds_col);
        lds_offsets[i] = lds_row * 128 + swz_col * 16;
    }
}
```

### GMEM Load Offsets
```cpp
void compute_ldg_offsets(uint32_t (&ldg_offsets)[2], uint32_t stride, int lane_id) {
    for (int i = 0; i < 2; ++i) {
        uint32_t ldg_row = i * 8 + lane_id / 8;
        uint32_t ldg_col = swizzle_col(ldg_row, lane_id % 8);
        ldg_offsets[i] = ldg_row * stride + ldg_col * 16;
    }
}
```
