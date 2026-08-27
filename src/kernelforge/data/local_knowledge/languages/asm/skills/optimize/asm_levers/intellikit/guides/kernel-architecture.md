---
guide: kernel-architecture
category: architecture
architecture: gfx950
tags: [GEMM, attention, GEMV, grouped-GEMM, register-budget, LDS-layout, software-pipeline, uber-kernel]
---

# Kernel Architecture Patterns for gfx950 (MI355X / CDNA4)

How different kernel types are structured in hand-written AMDGCN assembly on MI355X.
Covers prologue, main loop, epilogue, register budgets, LDS layouts, and scheduling
decisions for each kernel family. Read this before diving into the assembly.
All patterns empirically validated on MI355X silicon.

---

## Table of Contents

1. [BF16 GEMM (128x128 tile)](#1-bf16-gemm-128x128-tile)
2. [FP8 Grouped GEMM -- Variable-K Wgrad](#2-fp8-grouped-gemm----variable-k-wgrad)
3. [FP8 Grouped GEMM -- Persistent FWD/Dgrad](#3-fp8-grouped-gemm----persistent-fwddgrad)
4. [Forward Attention](#4-forward-attention)
5. [Backward Attention](#5-backward-attention)
6. [GEMV (Decode)](#6-gemv-decode)
7. [Uber-Kernel / Mega-Kernel Fusion](#7-uber-kernel--mega-kernel-fusion)
8. [Cross-Kernel Comparison Table](#8-cross-kernel-comparison-table)

---

## 1. BF16 GEMM (128x128 Tile)

MFMA: `v_mfma_f32_16x16x32_bf16` (16 cycles, 16384 FLOPs/inst).

### Block Diagram

```
                     PROLOGUE
                        |
        +---------------+---------------+
        |                               |
  Load kernargs              Compute thread identity
  (s_load_dwordx4 x3)       (wave_id, lane, mfma_row, kgrp)
        |                               |
        +---------------+---------------+
                        |
              Precompute LDS addrs (v14-v21, persistent)
              Precompute global row offsets (v32-v35)
              Zero accumulators (a[0:63])
                        |
                   K-LOOP (.Lloop)
                        |
        +-----------------------------------------------+
        |                                               |
   Global loads (4x global_load_dwordx4)      Compute block
   Issue at loop start, drain after compute   (16 MFMAs, B-reuse)
        |                                               |
        |    +------ ds_read B0, B1 ------+             |
        |    | ds_read A[BI=0..3]         |             |
        |    | MFMA a[BI*16+0:3]  x B0    |  Pass 1     |
        |    | MFMA a[BI*16+4:7]  x B1    |             |
        |    +----------------------------+             |
        |    +------ ds_read B2, B3 ------+             |
        |    | ds_read A[BI=0..3]         |             |
        |    | MFMA a[BI*16+8:11] x B2    |  Pass 2     |
        |    | MFMA a[BI*16+12:15] x B3   |             |
        |    +----------------------------+             |
        +-----------------------------------------------+
                        |
              s_waitcnt vmcnt(0)  -- drain global loads
              s_barrier           -- wait for all waves to finish reading
              ds_write new tile data to LDS
              s_barrier           -- wait for all waves to finish writing
              K offset += 64 bytes (32 bf16 elements)
              branch back to .Lloop
                        |
                   EPILOGUE
                        |
              MFMA drain (s_nop 15)
              v_accvgpr_read_b32 (read 64 AGPRs)
              Per-block: address compute + bounds check
              16x STORE_BLK (global_store_dword or global_atomic_add_f32)
              s_waitcnt vmcnt(0)
              s_endpgm
```

### Main Loop Structure (per K-iteration)

Each iteration processes K_STEP=32 bf16 elements:

1. **Issue global loads** (4 loads: A_lo, A_hi, B_lo, B_hi) at loop top
2. **Compute block** (16 MFMAs using B-reuse pattern, 12 LDS reads)
3. **Drain global loads** (s_waitcnt vmcnt(0))
4. **Barrier** (ensure all waves done reading LDS)
5. **Write new tile to LDS** (ds_write_b128, 4 writes per matrix)
6. **Barrier** (ensure all waves see new data)
7. **Advance K offset** (SGPR s24 += 64)

### Register Budget

```
VGPRs:   60 (accum_offset = 60)
AGPRs:   64 (16 MFMA blocks x 4 AGPRs each)
Total:   124 (next_free_vgpr = 124)
Waves/SIMD: 4 (512/124 = 4)
WGs/CU:  1 (4 waves x 64 threads = 256 = 1 WG)

Allocation:
  v[0:6]     Thread identity (tid, wave_id, lane, mfma_row, kgrp, ...)
  v[7:8]     Cooperative load decomposition (row_local, col_quarter)
  v[9:10]    LDS write offsets (persistent)
  v[14:17]   A LDS read addresses (4 persistent, one per BI)
  v[18:21]   B LDS read addresses (4 persistent, one per BJ)
  v[28:31]   B operand buffer (alternate)
  v[32:35]   Persistent global row offsets
  v[40:43]   A operand buffer
  v[44:47]   B operand buffer (primary)
  v[48:55]   Global load destinations (A_lo, A_hi, B_lo, B_hi)
  a[0:63]    MFMA accumulators
```

### LDS Footprint

```
Region A:  0..8191     (128 rows x 64 bytes/row = 8 KB)
Region B:  8192..16383 (128 rows x 64 bytes/row = 8 KB)
Total:     16 KB (single-buffered)
```

Row-major contiguous storage. Each row = K_STEP x sizeof(bf16) = 32 x 2 = 64 bytes.
ds_write_b128 stores 8 bf16 per thread. ds_read_b128 reads 8 consecutive K values,
matching the MFMA's 8-bf16-per-kgrp requirement.

Double-buffering was tested (32 KB) but showed no benefit because global loads were
fast enough that the single-buffer approach with barriers was not bottlenecked.

### Key Scheduling Decisions

- **B-reuse pattern**: Load each B column once, reuse across all 4 A rows. Reduces
  LDS reads from 20 to 12 per K-step (40% reduction, +7% performance).
- **Global loads at loop start**: All 4 loads issued before any compute. During ~128
  cycles of MFMA work, loads complete in background. Interleaving loads between MFMA
  pairs was tested and found slightly worse.
- **Persistent addresses**: LDS read addresses (v14-v21) and global row offsets (v32-v35)
  computed once before K-loop. Only the K offset (SGPR s24) advances each iteration.
  Saves ~16 VALU instructions per K-step.
- **B pre-transposed**: Caller provides B.T.contiguous(). Eliminates 32x ds_write_b16
  scatter per thread that caused 91% of all LDS stall cycles.

### Performance

| Property | Value |
|----------|-------|
| Compute bound? | Yes for large shapes, memory-bound for small M*N |
| Roofline % | 74-76% of optimized GEMM libraries on large shapes |
| MFMAs per K-step | 16 |
| Cycles per K-step (theoretical) | 128 (16 x 8 issue rate) |
| Cycles per K-step (measured) | ~1500-3000 (barriers, LDS, address overhead) |
| Compute density | 92.4% of active cycles in MFMA after B-scatter fix |

### Tile Size vs Arithmetic Intensity

| Tile | VGPRs | AGPRs | Total | Waves/SIMD | AI (FLOP/byte) | Status |
|------|-------|-------|-------|------------|-----------------|--------|
| 64x64 | ~40 | 16 | ~56 | 8 | 16 | Memory-bound |
| 128x128 | 60 | 64 | 124 | 4 | 64 | Sweet spot |
| 256x128 | 60 | 128 | 188 | 2 | 85 | Slower (LDS overhead) |
| 256x256 | ~128 | 256 | ~384 | 1 | 128 | Requires deep pipelining |

The 128x128 tile is the practical sweet spot for hand-written assembly. Larger tiles
need deeper software pipelining to amortize overhead, which is extremely hard to
replicate by hand without code generation.

---

## 2. FP8 Grouped GEMM -- Variable-K Wgrad

MFMA: `v_mfma_f32_16x16x32_fp8_bf8` (legacy, 28 cycles) or
`v_mfma_f32_32x32x64_f8f6f4` (dot_scaled, 32 cycles, 8x FLOPs).

### Block Diagram

```
                     PROLOGUE
                        |
        +---------------+---------------+
        |                               |
  Load group offsets           Compute thread identity
  from pointer table           (tid -> LDS addrs via bitop3+XOR)
        |                               |
        +---------------+---------------+
                        |
              Zero 128 accumulators (v[0:127])
              Issue initial buffer_loads
              Write to LDS (ds_write2st64_b64)
              Branch if single K-iteration
                        |
               INNER LOOP (.L4)
        +-----------------------------------------------+
        | 1. Compute next-iteration addresses (VALU)    |
        | 2. Issue 2 buffer_load_dwordx4 (LHS prefetch) |
        | 3. s_waitcnt lgkmcnt(0) + s_barrier            |
        | 4. s_setprio 3                                 |
        | 5. 9 ds_read_b64_tr_b8 (initial LDS reads)    |
        |                                                |
        |    +---- 128 MFMAs (4 K-steps x 32 tiles) ----+
        |    | K-step 0: 32 MFMAs + 3 ds_reads          |
        |    | K-step 1: 32 MFMAs + 3 ds_reads          |
        |    | K-step 2: 32 MFMAs + 3 ds_reads          |
        |    | K-step 3: 32 MFMAs + 3 ds_reads          |
        |    +------------------------------------------+
        |                                                |
        | 6. Issue 2 buffer_load_dwordx4 (RHS prefetch)  |
        | 7. s_waitcnt lgkmcnt(0) + s_barrier            |
        | 8. 4 ds_write2st64_b64 (write data to LDS)    |
        | 9. Last 12 MFMAs during ds_write co-execution  |
        | 10. s_setprio 0                                |
        | 11. s_cbranch_scc1 .L4                         |
        +-----------------------------------------------+
                        |
                   EPILOGUE (.L2)
                        |
              Final K-iteration MFMAs (no prefetch)
              Scale: v_pk_mul_f32 (a_scale * b_scale)
              Convert: v_cvt_pk_bf16_f32
              Store: buffer_store_dwordx2
              s_waitcnt vmcnt(0)
              s_endpgm
```

### Variable-K Handling

The wgrad kernel computes `dW = A^T @ B` where K = number of tokens assigned to
each expert. K varies per expert group (32 groups, ~4096 tokens mean).

```
Expert group lookup:
  1. Load offs[g] and offs[g+1] from pointer table
  2. K_g = offs[g+1] - offs[g]
  3. num_iterations = K_g / K_step
  4. Set up buffer descriptors for LHS[offs[g]:offs[g+1], :]
  5. Set up buffer descriptors for RHS[offs[g]:offs[g+1], :]
```

### Persistent Scheduling

One workgroup per CU (256 WGs = 256 CUs). Each WG iterates over tiles:

```
global_tile_id = workgroup_id_x
while global_tile_id < total_tiles:
    process_tile(global_tile_id)
    global_tile_id += num_workgroups  // stride = 256
```

XCD-aware ordering with swizzle group size 4 distributes tiles across the 8 XCDs
for L2 cache locality.

### Register Budget

```
VGPRs:   224 (accum_offset = 224)
AGPRs:   0
Total:   224 (next_free_vgpr = 224)
Waves/SIMD: 2 (512/224 = 2)

Allocation:
  v[0:127]     MFMA accumulators (32 tiles x 4 VGPRs)
  v[128:135]   Buffer load destinations (LHS)
  v[136:139]   Scale constants
  v[140:161]   LDS read results (rotating, A/B operands)
  v[162:169]   LDS write addresses (persistent, live across loop)
  v[176:181]   LDS read base addresses (via bitop3 + XOR)
  v[196:197]   LDS write base pointers
  v[198:204]   K offset, RHS bases, bounds check
  v[216:223]   Dead in inner loop -- freed by MOV elimination,
               used for buffer_load hoisting
```

### LDS Layout

```
Total: 64 KB (dynamically allocated)
Double-buffered, XOR-swizzled addressing

6 base addresses (XOR-related):
  v176 = base0 (from bitop3)
  v177 = base0 XOR 32
  v178 = base0 XOR 64
  v179 = base0 XOR 96
  v180 = base1 (different bitop3)
  v181 = base1 XOR 64

4 K-phases per iteration:
  Phase 0: offset 0x0000, 0x0080
  Phase 1: offset 0x2000, 0x2080
  Phase 2: offset 0x4000, 0x4080
  Phase 3: offset 0x6000, 0x6080

Write pattern: 4x ds_write2st64_b64 per iteration
Read pattern: 21x ds_read_b64_tr_b8 per iteration (9 initial + 12 interleaved)
```

### Key Scheduling Decisions

- **128 MFMAs per iteration**: 32 accumulator tiles x 4 K-steps. Each K-step
  reads B operands and A operands from LDS, issues 32 MFMAs covering all tiles.
- **ds_read_b64_tr_b8 (transpose reads)**: Hardware transpose for `A^T @ B`
  without explicit LDS transposition. 2 VGPRs output per read.
- **RHS buffer_load hoisting** (+9%): The biggest ASM optimization. RHS loads
  were only 8 MFMAs (~224 cycles) before drain; HBM latency is ~400 cycles.
  Hoisting to loop top using dead v[216:223] gives ~3584 cycles of cover.
- **lgkmcnt(2) pipelining**: Drain 1-of-3 outstanding LDS reads per MFMA batch.
  Overlaps LDS read latency with MFMA execution.

### Performance

| Property | Legacy MFMA | dot_scaled |
|----------|-------------|------------|
| MFMA instruction | 16x16x32_fp8_bf8 | 32x32x64_f8f6f4 |
| MFMAs per iteration | 128 | 16 |
| Compute per inst (FLOPs) | 16,384 | 131,072 |
| Best TFLOPS | 1,252 (mega v13) | 2,150 |
| % of FP8 peak | 35% | 60% |
| Bottleneck | LDS read latency | Structural (barriers, scheduling) |

The dot_scaled upgrade (1.76x) dwarfs all ASM-level optimizations combined (+13.2%).
At 16 warps (dot_scaled), hardware hides latency via wave interleaving, making
manual scheduling tricks ineffective.

---

## 3. FP8 Grouped GEMM -- Persistent FWD/Dgrad

MFMA: `v_mfma_f32_16x16x128_f8f6f4` (MI355X-native, 32 cycles, 65536 FLOPs/inst).

### Block Diagram

```
                     PROLOGUE
                        |
              56 s_nop 0 (alignment, branched over)
              Persistent tile index from workgroup_id_x
              Expert group lookup via pointer table
              Division/modulo for M/N tile indices
                        |
               FIRST TILE (.L4)
        +-----------------------------------------------+
        | 8 buffer_load_dwordx4 (4 A + 4 B)            |
        | s_barrier + s_waitcnt lgkmcnt(0)               |
        | vmcnt drain (7..0) -> ds_write A tiles         |
        | s_barrier + ds_read 16 LDS blocks              |
        | s_barrier + ds_write B tiles                   |
        | s_barrier + ds_read B                          |
        | 32 MFMAs + next-group address computation      |
        | Scale / convert / store                        |
        +-----------------------------------------------+
                        |
               K-LOOP (.L8, steady-state)
        +-----------------------------------------------+
        | 1. Compute A-side address offsets              |
        | 2. 4 buffer_load_dwordx4 (A tiles)            |
        | 3. s_waitcnt lgkmcnt(0) + s_barrier            |
        | 4. 24 ds_read_b128 (12 A + 12 B)              |
        | 5. 32 v_mfma_f32_16x16x128_f8f6f4             |
        |    (interleaved with ds_reads for B rotation)  |
        | 6. 4 buffer_load_dwordx4 (B tiles)            |
        | 7. s_barrier + 8 ds_write_b128                 |
        | 8. 3 tail MFMAs                                |
        | 9. s_cbranch_scc1 .L8                          |
        +-----------------------------------------------+
                        |
               GROUP TRANSITION
        +-----------------------------------------------+
        | Expert lookup, accumulator zero (128 v_mov_b32)|
        | Address setup, initial buffer_loads + LDS write|
        | ~400 VALU/SALU instructions, NO MFMA overlap   |
        +-----------------------------------------------+
                        |
                   EPILOGUE
                        |
              Scale: v_pk_mul_f32
              Convert: v_cvt_pk_bf16_f32
              Store: 32x buffer_store_dwordx2
              Next-tile prefetch + 4 barriers
```

### MFMA Scheduling Pattern

32 MFMAs organized in 4 phases of 8. Each phase uses one B-operand pair (16 VGPRs)
with all 4 A-operand groups (32 VGPRs):

```
Phase 1: B = v[222:229]
  MFMA 1-4: v[0:7], v[8:15], v[24:31], v[16:23]  x  v[222:229]

Phase 2: B = v[230:237]
  MFMA 5-8: same A groups  x  v[230:237]

  [ds_reads reload B operands]

Phase 3-4: repeat with new B data

  [ds_reads reload again]

Phase 5-8: repeat for remaining 16 MFMAs
```

A-side operands loaded ONCE per K-iteration (8:1 reuse). B-side rotated 4 times
(2.67:1 reuse).

### Two Variants

**Variant A (Standard, ds_write pipeline):**
- HBM -> VGPRs -> ds_write -> LDS -> ds_read -> VGPRs -> MFMA
- 2 barriers per K-iteration
- Amenable to ASM optimization (+3.5-5.2% combined)

**Variant B (Pingpong, direct-to-LDS):**
- HBM -> LDS directly (buffer_load ... offen lds)
- Bypasses VGPRs entirely for the load path
- 3 barriers per K-iteration
- 15-20% faster baseline, but immune to ASM optimization

### Register Budget

```
Variant A:
  VGPRs:   248 (accum_offset = 248, 0 AGPRs)
  Total:   248
  Waves/SIMD: 2 (512/248 = 2)

  Allocation:
    v[0:31]     A-side MFMA SrcA operands (from LDS)
    v[32:55]    A-side buffer_load destinations
    v[40:47]    MFMA accumulators (tiles 0-1)
    v[56:63]    Per-tile base addresses (persistent)
    v[72:191]   MFMA accumulators (tiles 2-31, 128 VGPRs)
    v[200:211]  Scale factors, constants
    v[218]      ds_write base address
    v[222:237]  B-side MFMA SrcB operands (rotated)
    v[238:247]  Dead in .L8 -- target for buffer_load hoisting

Variant B:
  v[0:127]    MFMA accumulators (32 tiles x 4 VGPRs)
  v[128:167]  Address VGPRs (bpermute, buffer_load addrs)
  v[168:199]  A-side MFMA SrcA operands
  v[200:239]  B-side MFMA SrcB operands
```

### LDS Layout

```
Double-buffered:
  Bank 0: 0x0000 - 0x7FFF (32 KB)
  Bank 1: 0x8000 - 0xFFFF (32 KB)
  Total: 64 KB

Each bank holds one K-tile:
  A-side: 4 tiles at offsets 0, 8192, 16384, 24576
  B-side: 4 tiles at offsets 32768, 40960, 49152, 57344

Write pointer alternates between banks each K-iteration.
```

### Key Scheduling Decisions

- **XCD-aware tile ordering**: Workgroups on the same XCD process adjacent tiles
  to maximize L2 cache locality.
- **Partial B-side buffer_load hoist** (+2%): Only 2 of 4 loads hoisted. Full
  hoisting causes -9 to -11% regression from memory queue saturation.
- **Tail MFMA interleaving** (+0.5-1.5%): Fill vmcnt drain bubbles with MFMAs
  that have already consumed their operands.
- **Group transition overhead**: ~400 instructions with zero MFMA overlap.
  Occurs 32 times per kernel dispatch (~3% of total time).

### Performance

```
Per K-iteration cycle budget:
  32 MFMAs x 32 cycles   = 1024 cycles  (73.5% compute efficiency)
  2 barriers x ~100 cycles =  200 cycles  (14.3%)
  s_waitcnt stalls         =  100 cycles   (7.2%)
  VALU address computation =   60 cycles   (4.3%)
  SALU + branch            =   10 cycles   (0.7%)
  Total                    = 1394 cycles

Best measured: 2376 TFLOPS on gate_up_dgrad (Variant B)
              = 91% of roofline, 67% of FP8 peak
```

The 74-78% roofline ceiling is structural. No amount of instruction scheduling
can eliminate the barriers. Breaking through requires triple-buffer LDS (needs
96 KB, only 64 KB available) or larger MFMA tiles (32x32x128).

---

## 4. Forward Attention

MFMA: `v_mfma_f32_32x32x16_bf16` (32 cycles, 32768 FLOPs/inst).

### Block Diagram

```
                     PROLOGUE
                        |
        Load Q to LDS (buffer_load_dwordx4, 4 loads)
        Load Q to VGPRs (v[66:81], 16 VGPRs)
        Precompute constants (log2e * scale, -inf, column indices)
        Zero O accumulators (v[2:33])
        Initialize running max (v121 = -inf), running sum (v96 = 0)
                        |
               MAIN LOOP (iterate over K-columns, BLOCK_N=64 per iter)
        +-----------------------------------------------+
        |                                               |
        |  +----- GEMM 1: QK^T (8 MFMAs) ------+      |
        |  | Sub-tile 0: v[50:65] += Q x K       |      |
        |  |   D0: 2 MFMAs (K=32, 2 steps x 16) |      |
        |  |   D1: 2 MFMAs                       |      |
        |  | Sub-tile 1: v[34:49] += Q x K       |      |
        |  |   D0: 2 MFMAs                       |      |
        |  |   D1: 2 MFMAs                       |      |
        |  +-------------------------------------+      |
        |                                               |
        |  +----- CAUSAL MASK (~192 instructions) -+    |
        |  | Block-level skip (s_cbranch if fully   |    |
        |  |   above diagonal)                     |    |
        |  | Per-element: v_cndmask_b32 with -inf  |    |
        |  |   for masked positions (32 elements)  |    |
        |  +---------------------------------------+    |
        |                                               |
        |  +----- ONLINE SOFTMAX (~117 instructions) -+ |
        |  | Row max: 16x v_max3_f32 chain (lane-local)||
        |  | Half-wave exchange: 1x ds_bpermute        ||
        |  | New max: v_max3_f32(prev, this, other)    ||
        |  | Scale+exp: 33x v_fma_f32 + 33x v_exp_f32 ||
        |  | Row sum: 31x v_add_f32 chain              ||
        |  | O rescale: v_pk_mul_f32 x 16 (packed)     ||
        |  +-------------------------------------------+|
        |                                               |
        |  +----- GEMM 2: PV (8 MFMAs) --------+      |
        |  | P (packed BF16) x V (from LDS)      |      |
        |  | D0 half: 4 MFMAs -> v[18:33]        |      |
        |  | D1 half: 4 MFMAs -> v[2:17]         |      |
        |  +-------------------------------------+      |
        |                                               |
        |  +----- PREFETCH K[N+1] + V[N+1] ----+      |
        |  | 4 buffer_load_dwordx2 (K/V tiles)   |      |
        |  | v_perm_b32 byte shuffle (V layout)  |      |
        |  | ds_write2_b32 (K/V to LDS)          |      |
        |  | buffer_load_dwordx4...lds (Q async) |      |
        |  +-------------------------------------+      |
        |                                               |
        |  2x s_barrier (LDS sync)                      |
        |  branch back to loop top                      |
        +-----------------------------------------------+
                        |
                   EPILOGUE
                        |
              O normalization: v_div_scale/fmas/fixup (1/l_acc)
              O scaling: v_mul_f32 x 32
              BF16 conversion: v_cvt_pk_bf16_f32 x 40
              Byte shuffle: v_perm_b32 x 16
              Store: 8x buffer_store_dwordx2 (exec-masked)
```

### Two-GEMM Structure

Each loop iteration computes two GEMMs:

1. **QK^T** (score matrix S): `S[128x64] = Q[128xD] x K[DxN]`
   - 8 MFMAs (2 sub-tiles x 2 D-halves x 2 K-steps)
   - Q from VGPRs (persistent), K from LDS
   - Output to S accumulators (v[34:65], 32 VGPRs)

2. **PV** (output accumulation): `O[128xD] += P[128x64] x V[64xD]`
   - 8 MFMAs
   - P from VGPRs (packed BF16 from softmax), V from LDS
   - Output to O accumulators (v[2:33], 32 VGPRs)

Between the two GEMMs: causal masking + online softmax + O rescaling.

### Online Softmax (The Critical Section)

Key insight: swap A/B operands so K=A, Q=B. This places all K-column scores
for a given Q-row in a single lane's VGPRs, enabling lane-local reduction.

```
Per iteration:
  1. Row max:  16x v_max3_f32 (reduce 32 S values to 1 per lane)
              + 1x ds_bpermute (exchange max between half-waves)
              + 1x v_max3_f32 (combine)
              = 18 instructions total

  2. Rescale: alpha = exp2(old_max_scaled - new_max_scaled)
              O_acc *= alpha  (16x v_pk_mul_f32, packed 2-wide)

  3. Exp:     33x v_fma_f32 (scale + shift)
              33x v_exp_f32 (P = exp2(S_scaled - max_scaled))

  4. Row sum: 31x v_add_f32 (serial chain, lane-local)
              + 1x ds_bpermute (exchange sum between half-waves)
              + 1x v_fmac_f32 (l_acc = sum + prev_l_acc * alpha)

  Total: ~117 instructions / ~269 cycles
```

Compare to naive butterfly bpermute: 96 bpermutes per iteration, consuming 94% of
loop time with MFMA utilization at only 7.5%.

**BF16 constraint**: Tile-wide max fails (cos <= 0.977 from underflow). Per-row max
is mandatory for BF16 correctness.

### Register Budget

```
VGPRs:   134 (can be compressed to 128 for occupancy bump)
AGPRs:   0 (all MFMA results kept in VGPRs to avoid accvgpr_read hazards)
Waves/SIMD: 3 at 134 VGPRs, 4 at 128 VGPRs

Allocation:
  v[2:17]     O accumulator D1 (FP32, 16 VGPRs)
  v[18:33]    O accumulator D0 (FP32, 16 VGPRs)
  v[34:49]    S accumulator sub-tile 1 (FP32, 16 VGPRs)
  v[50:65]    S accumulator sub-tile 0 (FP32, 16 VGPRs)
  v[66:81]    Q tile data (BF16 packed, 16 VGPRs)
  v82-86      LDS bases, K/V address temps
  v87         -inf constant (0xff800000)
  v88-93      K/V load data + perm intermediates
  v96         Running sum (l_acc)
  v97-98      Causal mask column indices
  v100-107    PV LDS read base addresses
  v121        Running max
  v[128:133]  Exp intermediates (dead outside softmax window)

SGPRs: ~51
  s13         log2e * softmax_scale (precomputed constant)
  s[8:11]     Buffer descriptor for Q
  s[20:23]    Buffer descriptor for K
  s24         LDS stride constant
  s28-29      Sequence length bounds
```

### LDS Footprint

```
Estimated: ~40-48 KB

Regions:
  V data D0:     0x0000-0x0FFF  (4 KB)
  V data D1:     0x1000+         (4 KB)
  Q per wave:    0x1100 + wave_id * 0x410  (~1 KB each, 4 waves)
  K staging:     via v_perm offsets  (4-8 KB)
```

No explicit V transpose. Inline v_perm_b32 byte shuffling (4 buffer_load_dwordx2
+ 8 v_perm_b32 + 4 ds_write2_b32) replaces a separate transpose pass.

### 3-Stage Software Pipeline

```
Iteration N:
  [COMPUTE]  QK^T MFMAs using K[N] from LDS
  [LOAD]     buffer_load K[N+1] from global        <-- issued during softmax
  [COMPUTE]  Causal mask + Softmax
  [STORE]    K[N+1] arrives, perm+write to LDS      <-- during softmax exp chain
  [COMPUTE]  O rescale
  [COMPUTE]  PV MFMAs using V[N] from LDS + P
  [LOAD]     buffer_load V[N+1] + async Q reload    <-- during PV MFMAs
  [SYNC]     2x s_barrier
```

Memory latency is fully hidden: ~400+ instructions between issuing K[N+1] load
and needing the result for QK^T[N+1] MFMAs.

### Performance

| Property | Value |
|----------|-------|
| Compute bound? | Yes (1498 TFLOPS achieved) |
| MFMA utilization | High (VALU:MFMA ratio = 18:1) |
| Total NOPs | 4 (minimal hazard management) |
| s_setprio | Zero (natural instruction mix suffices) |
| Best achieved | 2.824 ms / 1558 TFLOPS (1.08x CK, 3-tier s_setprio) |
| VGPR liveness optimization | 135->128 VGPRs, 3->4 waves, +3% |

---

## 5. Backward Attention

MFMA: `v_mfma_f32_16x16x16_bf16` (16 cycles) or `v_mfma_f32_16x16x32_bf16` (16 cycles).

### Block Diagram

```
        KERNEL 1: ODO (precompute)
        +---------------------------+
        | D[i] = rowsum(dO[i] * O[i])|
        | Per-row dot product        |
        +---------------------------+

        KERNEL 2: MAIN (dQ/dK/dV)
                     |
                PROLOGUE
                     |
        Load K/V to LDS (via buffer_load)
        Construct buffer descriptors for Q, dO, LSE
        Compute batch/head offsets
        Zero dK/dV accumulators (a[112:159])
                     |
        OUTER LOOP (over Q-blocks, 4 blocks per tile)
        +-----------------------------------------------+
        |                                               |
        |  Block N (e.g., Block 1 at label_05B5):       |
        |                                               |
        |  +--- GEMM0: S = Q x K^T (24 MFMAs) ----+   |
        |  | 2 Q-tiles x 3 KV-tiles x 4 MFMAs      |   |
        |  | Q from LDS (via v16), K from AGPRs     |   |
        |  | Output: S scores in v[52:75]            |   |
        |  +----------------------------------------+   |
        |                                               |
        |  +--- CAUSAL MASK -------------------------+  |
        |  | Block-level skip (s_cbranch if K > Q)   |  |
        |  | Element-level: set masked to -inf       |  |
        |  +----------------------------------------+   |
        |                                               |
        |  +--- SOFTMAX RECOMPUTE ------------------+   |
        |  | exp2(scale * S[i,j] - LSE[i])          |   |
        |  | 48 transcendental ops (v_exp_f32)       |   |
        |  | 88 DPP quad_perm shuffles               |   |
        |  +----------------------------------------+   |
        |                                               |
        |  +--- GEMM1: dV += P^T x dO (MFMAs) -----+  |
        |  | P (packed BF16 via v_perm_b32) as A     |  |
        |  | dO from LDS as B                        |  |
        |  | Accumulate into a[136:159] (dV AGPRs)   |  |
        |  +----------------------------------------+   |
        |                                               |
        |  +--- GEMM2: dP = dO x V^T (MFMAs) ------+  |
        |  | dO as A, V from AGPRs as B              |  |
        |  +----------------------------------------+   |
        |                                               |
        |  +--- Compute dS = P * (dP - D) ----------+  |
        |  | Element-wise, uses precomputed D[i]     |  |
        |  +----------------------------------------+   |
        |                                               |
        |  +--- GEMM3: dK += dS^T x Q (MFMAs) -----+  |
        |  | Accumulate into a[112:135] (dK AGPRs)   |  |
        |  +----------------------------------------+   |
        |                                               |
        |  +--- GEMM4: dQ = dS x K (MFMAs) --------+  |
        |  | Store via global_atomic_pk_add_bf16     |  |
        |  | (a16 path: BF16 atomics, no dq_convert) |  |
        |  +----------------------------------------+   |
        |                                               |
        |  s_barrier (between blocks)                   |
        |  Repeat for blocks 2, 3, 4                    |
        +-----------------------------------------------+
                     |
                EPILOGUE
                     |
        v_accvgpr_read_b32 (read dK/dV from AGPRs)
        v_cvt_pk_bf16_f32 (convert to BF16)
        buffer_store (write dK, dV to global)
        s_waitcnt vmcnt(0)
        s_endpgm
```

### 5-GEMM Decomposition

```
GEMM0: S  = Q  x K^T    (score matrix)
GEMM1: dV = P^T x dO    (value gradient, accumulated across Q-blocks)
GEMM2: dP = dO x V^T    (probability gradient)
GEMM3: dK = dS^T x Q    (key gradient, accumulated across Q-blocks)
GEMM4: dQ = dS  x K     (query gradient, BF16 atomic accumulation)
```

### 2-Kernel Pipeline (a16 Mode)

The a16 mode (hd64 + gfx950) uses BF16 atomic `global_atomic_pk_add_bf16` for dQ
accumulation directly, eliminating the need for a separate `dq_convert` kernel.
Result: 2-kernel pipeline instead of 3-kernel.

### Register Budget

```
VGPRs:   232 (accum_offset = 256)
AGPRs:   160 (a[0:159])
Total:   416 (next_free_vgpr = 512 declared, 416 actual minimum)
Waves/SIMD: 1 (need <= 256 for 2 waves -- not achievable)
SGPRs:   102

AGPR dual-purpose architecture:
  a[0:23]     K data tiles (loaded from LDS via ds_read_b128)
  a[24:95]    Other tile data (Q, dO)
  a[96:111]   Dual-use: Q data (GEMM0) / P data (dK/dV GEMM)
              (works because GEMM0 is complete before dK/dV begins)
  a[112:135]  dK accumulators (accumulate-in-place)
  a[136:159]  dV accumulators (accumulate-in-place)

Key VGPRs:
  v[16]       Q LDS address (swizzled)
  v[18]       P-data LDS address
  v[28]       K LDS address (= v16 + s60)
  v[52:75]    GEMM0 output (S scores)
  v[160:175]  K data moved from AGPRs for 16x16x32 conversion
  v[228]      Constant: only truly free VGPR
  v[229:231]  Constants for BF16 packing (0xffff0000, 0x7fff0000, 0x7fff)
```

### LDS Layout (Swizzled)

```
Total: 64 KB (statically allocated)

Swizzle formula for Q base address (v16):
  v16 = (t0*264 + t1*4 + t2*32 + t3*64 + t4*16 + t5*8) * 4
  where t_i = bit i of thread_id
  264 = 0x108 chosen because 264 mod 32 = 8 (avoids 2-way bank conflicts)

Regions:
  Q tile:  v16 + offsets (0, 512, 2176, 2688)
  K tile:  v28 = v16 + s60
  P tile:  v18 + offsets (4352+)
  dO tile: v16 + offsets (8704+)  -- shares v16 base with Q!
```

### Key Scheduling Decisions

- **480 MFMAs total** per kernel invocation (v_mfma_f32_16x16x16_bf16)
- **Softmax recomputation**: BWD recomputes exp2(S - LSE) from stored LSE values.
  48 transcendental ops per inner loop iteration.
- **GQA handling**: `kv_head = q_head / ratio` (ratio = HQ/HKV = 8). dK/dV
  accumulation uses BF16 atomics across the 8 Q heads sharing one KV head.
- **BF16 native conversion**: Replace 10-instruction software F32->BF16 rounding with
  single `v_cvt_pk_bf16_f32` on gfx950 (10:1 instruction reduction).
- **AGPR dual-purpose**: a[96:111] serves as Q data for GEMM0 and P data for dK/dV
  GEMM (temporally disjoint).

### Performance

| Property | Value |
|----------|-------|
| Compute bound? | No -- latency-bound at 1 wave/SIMD |
| MFMA utilization | 39% of peak (740 TF / 1890 TF) |
| Wait fraction | 31% of active cycles stalled on memory |
| HBM utilization | 11.4% (NOT bandwidth-limited) |
| L1 hit rate | 88.8% (tiling works well) |
| Instruction mix | 79% VALU, 16% LDS |
| Bottleneck | CU-level latency, not bandwidth |

The kernel is fundamentally limited by occupancy. At 1 wave/SIMD, no second wave
can hide memory latency. BF16 native conversion saved 432 VALU instructions but
they were hidden behind memory stalls. Only crossing the 256-VGPR threshold
for 2-wave occupancy would help, but the 160-AGPR gap makes this impossible
via register remapping alone.

---

## 6. GEMV (Decode)

MFMA: `v_mfma_f32_16x16x32_f16` (16 cycles, K=32 per instruction).
Used in decode-time inference (e.g., Llama 3.1 8B on MI355X).

### Block Diagram

```
                     PROLOGUE
                        |
        Load kernargs (W, hidden, y, M, K, residual, w_norm, eps)
        Compute tile assignment: WG handles one 16-row output tile
        Load input x to LDS (K * 2 bytes)
        s_barrier (ensure all threads see x)
                        |
               K-LOOP (iterate over K dimension)
        +-----------------------------------------------+
        |                                               |
        | Load weight tile: global_load_dwordx4         |
        |   W[M/16][K/32][16][32] layout = MFMA-native  |
        |                                               |
        | Load x slice: ds_read_b128 from LDS           |
        |                                               |
        | s_waitcnt vmcnt(0) lgkmcnt(0)                 |
        |                                               |
        | v_mfma_f32_16x16x32_f16 a[0:3], W, x, a[0:3] |
        |   (accumulate partial dot products)            |
        |                                               |
        | Advance W pointer, x pointer                  |
        | Loop back                                     |
        +-----------------------------------------------+
                        |
               REDUCTION
                        |
        Write partial sums to LDS scratch (1024 bytes)
        s_barrier
        Lane 0 of each wave reduces across 4 waves
        Store final 16 output values
                        |
              s_waitcnt vmcnt(0)
              s_endpgm
```

### Tiled Weight Layout

```
W[M/16][K/32][16][32] FP16

Standard: W[row, col] at offset (row * K + col) * 2
Tiled:    W[tile_m, tile_k, local_m, local_k] at offset
          ((tile_m * (K/32) + tile_k) * 16 * 32 + local_m * 32 + local_k) * 2
```

The weight matrix is repacked from standard [M, K] layout to this tiled format
for direct MFMA consumption. Each 16x32 tile maps directly to one MFMA's A operand.
Repacking is done once at model load time (weights are static for inference).

### Register Budget

```
VGPRs:   ~64
AGPRs:   4 (a[0:3] for MFMA output)
accum_offset: 256 (standard for gfx950)
next_free_vgpr: 512 (metadata requirement for ROCm 7.2)

Key allocations:
  Input x data:     loaded via global_load_dwordx4
  Weight tile:      loaded via global_load_dwordx4
  MFMA accumulators: a[0:3] (4 FP32 partial sums per thread)
  LDS reduction:    ds_write/ds_read for cross-wave reduce
```

### LDS Layout

```
Region 1: 0 .. K*2-1           Input x vector (broadcast to all threads)
Region 2: K*2 .. K*2+1023      256-thread reduction scratch (1024 bytes)
Total: K*2 + 1024 bytes
```

### Key Scheduling Decisions

- **MFMA-based, not dot product**: rocBLAS GEMV uses MFMA for a 2x bandwidth
  advantage on small matrices vs vector dot product instructions.
- **1 WG per output tile**: Grid = ((M+15)/16, 1, 1). Each WG = 256 threads = 4 waves.
- **LDS for x broadcast**: Input vector loaded to LDS once, read by all threads.
  Avoids redundant global loads.
- **Cross-wave reduction via LDS**: 4 waves produce partial sums in AGPRs, then
  reduce through LDS scratch space.

### Fused Variants

The 12 production kernels fuse GEMV with adjacent operations:

| Kernel | Fused operations | Kernarg size |
|--------|-----------------|-------------|
| gemv_mfma_rnc | rmsnorm_copy + QKV GEMV | 56 bytes |
| gemv_mfma_rnr | rmsnorm_residual + gate_up GEMV | 64 bytes |
| gemv_mfma_silu_down | SiLU + down GEMV + residual add | -- |
| gemv_mfma_add | GEMV + residual add | -- |

Key data flow constraint for rnr_gateup: `hidden += residual` is NOT idempotent
across WGs, so a SEPARATE `hidden_out` buffer is used.

### Performance

| Property | Value |
|----------|-------|
| Compute bound? | No -- memory-bandwidth-bound (loading 16 GB weights/step) |
| Roofline at 5.3 TB/s | 2831 us / 353 tok/s |
| Best achieved | 4073 us / 246 tok/s (70% of roofline) |
| Gap analysis | Dispatch overhead + non-weight memory traffic |

---

## 7. Uber-Kernel / Mega-Kernel Fusion

Concept: fuse multiple operations into a single GPU dispatch to eliminate inter-kernel overhead.

### Evolution of Dispatch Count

```
Unfused (9 dispatches/layer):
  rmsnorm_copy -> QKV GEMV -> RoPE + KV cache -> attention ->
  O proj GEMV -> rmsnorm_residual -> gate+up GEMV -> SiLU*mul -> down GEMV+add

Fused v1 (7 dispatches/layer):
  rnc_qkv -> rope_kvcache -> attention -> O proj ->
  rnr_gateup -> SiLU*mul -> down+add

Fused v2 (6 dispatches/layer):
  rnc_qkv -> rope_kvcache -> attention -> O proj ->
  rnr_gateup -> silu_down

Uber-kernel (1 dispatch/layer, NOT IMPLEMENTED):
  All operations fused into single dispatch

Mega-kernel (1 dispatch total, NOT IMPLEMENTED):
  Entire model in one dispatch
```

### Inter-WG Barrier Design

For uber-kernel fusion, operations within a layer must synchronize without
returning to the host. The wave-0-only barrier spin pattern:

```asm
; Only wave 0 of each WG participates in the barrier
; Other waves wait at s_barrier
v_cmp_eq_u32 vcc, v[wave_id], 0
s_cbranch_vccz .Lskip_barrier

; Wave 0: atomic increment counter, spin until all WGs arrive
flat_atomic_add v[result], v[counter_addr], v[one] sc0
s_waitcnt vmcnt(0)
.Lspin:
  flat_load_dword v[val], v[counter_addr] sc0 sc1
  s_waitcnt vmcnt(0)
  v_cmp_ge_u32 vcc, v[val], s[num_wgs]
  s_cbranch_vccz .Lspin

.Lskip_barrier:
s_barrier    ; local WG sync -- wave 0 done spinning, wake other waves
```

Key coherence requirements on gfx950:
- `flat_atomic_add ... sc0` for atomic with return
- `flat_load ... sc0 sc1` for spin-loads (bypass L1/L0 cache)
- `flat_store ... sc0 sc1` for cross-CU visible stores
- `sc0` alone on loads deadlocks (stale cached value)
- `buffer_gl0_inv` is NOT available on gfx950

### Performance Impact

- Barrier latency: ~50 ns per WG
- 32 WGs = ~1.6 us, 256 WGs = ~12.5 us
- Uber-kernel with wave-0-only barrier: **6% faster** than separate kernel dispatches
- The speedup comes from eliminating HIP dispatch overhead (~4 us per dispatch)
  and avoiding host-device round-trips

### HSA AQL Direct Dispatch

For maximum dispatch throughput, bypass HIP entirely:

```
AQL packet (64 bytes):
  header, workgroup_size, grid_size,
  group_segment_size (LDS), kernel_object,
  kernarg_address, completion_signal

Dispatch latency: ~100 ns (vs ~4 us via HIP)
Caveat: Queue wrapping causes SEGVs on ROCm 7.2 (known issue)
```

### Dispatch Count Summary

| Approach | Dispatches/layer | Total (32 layers) | Notes |
|----------|-----------------|-------------------|-------|
| Unfused | 9 | 290 | Baseline |
| Fused v1 | 7 | 226 | Incremental fusion |
| Fused v2 | 6 | 194 | Best implemented |
| Uber-kernel | 1 | 34 | Requires inter-WG barrier |
| Mega-kernel | - | 1 | Entire model, one dispatch |

---

## 8. Cross-Kernel Comparison Table

| Property | BF16 GEMM | FP8 Wgrad | FP8 FWD | FWD Attn | BWD Attn | GEMV |
|----------|-----------|-----------|---------|----------|----------|------|
| **MFMA** | 16x16x32 bf16 | 16x16x32 fp8 / 32x32x64 | 16x16x128 f8f6f4 | 32x32x16 bf16 | 16x16x16 bf16 | 16x16x32 f16 |
| **Tile** | 128x128 | 256x256 | 256x256 | 128x64 | 16x16 tiles | 16-row output |
| **MFMAs/iter** | 16 | 128 / 16 | 32 | 16 (8+8) | 480 total | 1 per K-step |
| **VGPRs** | 60 | 224 | 248 | 134 | 232 | ~64 |
| **AGPRs** | 64 | 0 | 0 | 0 | 160 | 4 |
| **Total regs** | 124 | 224 | 248 | 134 | 416 | ~68 |
| **Waves/SIMD** | 4 | 2 | 2 | 3-4 | 1 | 4+ |
| **LDS (KB)** | 16 | 64 | 64 | ~40-48 | 64 | K*2+1K |
| **Barriers/iter** | 2 | 2 | 2-3 | 2 | varies | 1 |
| **K-loop?** | Yes | Yes | Yes | Yes (over N) | Yes (over Q) | Yes |
| **Software pipeline** | Single-buf | Double-buf | Double-buf | 3-stage | Interleaved | Pre-load |
| **Bottleneck** | Compute | LDS reads | Barriers | Compute | Latency (occ) | HBM BW |
| **Roofline %** | 74-76% | 35-60% | 74-78% | ~80% | 39% | 70% |
| **Best TFLOPS** | 440 | 2150 | 2376 | 1558 | 248 | -- |

### When to Use Which Pattern

- **BF16 GEMM**: General matmul, inference weight computation. Simple structure,
  good starting point for learning assembly.
- **FP8 Grouped GEMM**: MoE training (fwd, dgrad, wgrad). Persistent scheduling
  critical for multi-expert efficiency.
- **FWD Attention**: Flash attention forward pass. Two-GEMM + inline softmax.
  Requires deep understanding of MFMA output layout for efficient softmax.
- **BWD Attention**: Flash attention backward pass. 5-GEMM + softmax recomputation.
  Register pressure makes 2-wave occupancy nearly impossible.
- **GEMV**: Decode-time inference. Memory-bandwidth-bound. Fuse with adjacent
  operations (rmsnorm, activation) to reduce dispatch overhead.
- **Uber-kernel**: When dispatch overhead dominates (many small operations per layer).
  Requires inter-WG barrier infrastructure.

### Universal Architecture Rules

These apply to ALL kernel types on gfx950:

1. `next_free_vgpr = accum_offset + num_agprs` (not just num_vgprs)
2. `s_waitcnt vmcnt(0)` before `s_endpgm` (or stores leak)
3. `.args` metadata required in kernel descriptor (or error 701)
4. vmcnt/lgkmcnt drain oldest-first (FIFO)
5. s_waitcnt -> MFMA needs 0 NOPs (waitcnt provides sufficient gap)
6. MFMA -> ACCVGPR_READ needs s_nop 15 x 2-4 (no HW interlock)
7. Transcendentals (v_exp, v_rcp) need s_nop 3 before reading result
8. VGPRs >= accum_offset alias AGPRs (ds_read there clobbers accumulators)
9. v_cvt_pk_bf16_f32 required for BF16 packing (manual shift produces inf/NaN)
10. Check MFMA opcode first -- instruction upgrade > scheduling optimization
