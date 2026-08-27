---
guide: lds-patterns
category: memory
architecture: gfx950
tags: [LDS, bank-conflict, double-buffer, triple-buffer, direct-to-LDS, ds_read, ds_write, swizzle, addressing]
---

# LDS Usage Patterns for gfx950 Assembly Kernels

Comprehensive reference for Local Data Share (LDS) usage in hand-written AMDGCN assembly kernels targeting MI355X (gfx950/CDNA4). Covers hardware model, bank conflict avoidance, multi-buffering, direct-to-LDS loads, layout patterns, ds_read/ds_write instruction selection, addressing for different MFMA widths, common bugs, and size constraints.

Covers GEMM, grouped GEMM, and attention kernel patterns on MI355X hardware.

---

## Table of Contents

1. [Hardware Model](#1-hardware-model)
2. [Bank Conflict Avoidance](#2-bank-conflict-avoidance)
3. [Double-Buffering](#3-double-buffering)
4. [Triple-Buffering](#4-triple-buffering)
5. [Direct-to-LDS (buffer_load ... lds)](#5-direct-to-lds)
6. [Async LDS DMA Pipeline](#6-async-lds-dma-pipeline)
7. [LDS Layout Patterns](#7-lds-layout-patterns)
8. [B Transpose Elimination](#8-b-transpose-elimination)
9. [ds_read Instruction Selection](#9-ds_read-instruction-selection)
10. [Addressing for Different MFMA Widths](#10-addressing-for-different-mfma-widths)
11. [Common Bugs](#11-common-bugs)
12. [Size Constraints and Occupancy](#12-size-constraints-and-occupancy)

---

## 1. Hardware Model

### Physical Parameters

| Parameter | Value |
|-----------|-------|
| Banks | 64 |
| Bank width | 4 bytes |
| Address period | 256 bytes (64 banks x 4 bytes) |
| Bandwidth per SIMD | 128 bytes/cycle (read), 128 bytes/cycle (write) |
| Standard capacity | 64 KB per CU |
| Extended capacity | 160 KB per CU (confirmed via `rocminfo` and `hipDeviceProp`) |
| Bank formula | `bank = (byte_addr / 4) % 64` |
| Latency | ~20-40 cycles for ds_read |

### 4-Phase Execution Model

LDS processes a wave64 as 4 sequential phases of 16 threads each:

```
Phase 0: threads  0-15
Phase 1: threads 16-31
Phase 2: threads 32-47
Phase 3: threads 48-63
```

Within each phase, 16 threads issue simultaneously to the 64-bank LDS. Conflicts are evaluated per-phase -- two threads in different phases never conflict with each other, even if they access the same bank. Two threads in the same phase accessing the same bank but different addresses cause a bank conflict. Two threads in the same phase accessing the same address get a free broadcast (hardware shortcut).

This 4-phase model is critical: the bank conflict analysis must consider which 16 threads execute together, not the full 64-thread wave.

### Measured Throughput (ds_read_b32, No Conflicts)

Microbenchmark results on MI355X:

| Instruction | CPI (cycles per issue) | Notes |
|-------------|------------------------|-------|
| `ds_read_b32` | ~12.2 | 4 bytes per thread |
| `ds_read_b64` | ~6.1 | 8 bytes per thread, must use even-aligned dest VGPR |
| `ds_read_b128` | ~3.3 | 16 bytes per thread, 4-VGPR destination |
| `ds_write_b32` | ~13.6 | 4 bytes per thread |
| `ds_write_b128` | ~3.3 | 16 bytes per thread, offset field up to 65535 bytes |

Wider reads have better throughput because the hardware amortizes fixed overhead across more data.

---

## 2. Bank Conflict Avoidance

### Measured Conflict Costs

Bank conflicts add approximately 4 CPI per doubling of conflict degree, measured on MI355X with `ds_read_b32`:

| Access Pattern | Stride (bytes/thread) | CPI | Conflict Degree |
|----------------|-----------------------|-----|-----------------|
| Linear (stride 4) | `tid * 4` | 11.9 | None (1-way) |
| Broadcast (all same addr) | `0` | 11.9 | None (free broadcast) |
| 2-way | `tid * 32` | 15.9 | 2-way |
| 4-way | `tid * 64` | 19.8 | 4-way |
| 8-way | `tid * 128` | 23.8 (est.) | 8-way |
| 16-way | `tid * 256` | 27.7 | 16-way |

**Key insight**: Broadcast is free. When all 16 threads in a phase access the same address, hardware broadcasts the single read to all requesters at no extra cost. This is exploited by attention kernels for shared scale-factor reads.

### Stride Padding for Column Reads

When a matrix is stored row-major in LDS and the MFMA requires column-major reads, adjacent threads access elements separated by `row_stride` bytes. If `row_stride` is a multiple of 256 (the bank period), all 16 threads in a phase hit the same bank -- 16-way conflict.

**Fix: pad the row stride** to break the alignment:

```
# BWD attention GEMM1/GEMM3 column reads
# Before: 128-byte row stride -> (128/4) % 64 = 0 -> 32-way conflict!
# After:  132-byte row stride -> (132/4) % 64 = 33 -> no conflict

row_stride = K_tile * sizeof(bf16) + PADDING
# K_tile=64, sizeof(bf16)=2: 64*2 = 128 -> pad to 132
```

Padding from 128 to 132 bytes eliminates the 32-way bank conflict on column reads of K/V tiles.

### Swizzle Formula

Backward attention uses a sophisticated swizzle that avoids bank conflicts entirely for `ds_read_b128` access. The LDS write address is computed as:

```
v16 = (t0*264 + t1*4 + t2*32 + t3*64 + t4*16 + t5*8) * 4
```

Where `t0..t5` are bit fields of `threadIdx.x`:
- `t0 = tid[5:0] >> 5` (or similar bit decomposition)
- `264 mod 32 = 8`, which guarantees that consecutive `t0` values map to banks 8 apart, avoiding 2-way conflicts

This swizzle is applied to all Q, K, V data written to LDS, enabling exclusively `ds_read_b128` reads (no column reads needed because data is pre-transposed via the swizzle).

### Inherent Conflicts in ds_read_b128

When 16 threads issue `ds_read_b128` with stride 16 bytes/thread (the natural contiguous layout), each thread reads 16 bytes spanning 4 banks. With 16 threads reading 4 banks each from a 64-bank space, some bank overlap is unavoidable. This produces inherent 4-way conflicts in certain layouts. The measured CPI of ~3.3 for `ds_read_b128` already accounts for this.

---

## 3. Double-Buffering

### Concept

Double-buffered LDS uses two regions (ping/pong). While MFMAs read from one region (consuming), global loads fill the other region (producing). A barrier separates the phases to prevent read-after-write races.

### Layout

Two equally-sized LDS banks at fixed offsets:

```
LDS Region 0: offset 0x0000   (A tile + B tile for even iterations)
LDS Region 1: offset 0x4000   (A tile + B tile for odd iterations)
```

For a 16KB-per-side tile pair, total double-buffer requirement is 32KB. For the persistent FWD GEMM, each region holds both A and B tiles, and the two regions alternate via SGPR pointers `s62`/`s63`.

### Toggle Mechanism

**XOR toggle** (variable-K wgrad kernel):

```asm
v_xor_b32_e32 v196, 0x4000, v196   ; toggle LDS write base between 0x0000 and 0x4000
```

The VGPR `v196` holds the current LDS write base. After each iteration's `ds_write` completes, the XOR flips to the other bank. The read addresses use a separate register with the complementary offset.

**K-loop unroll 2x** (alternative): unrolling the K-loop by 2 eliminates the toggle entirely -- even iterations hardcode region 0, odd iterations hardcode region 1. This saves 10 `v_xor` toggle instructions + 1 `s_movk` per iteration pair.

### Barrier Protocol

The double-buffer requires exactly 2 barriers per K-iteration:

```asm
; --- Phase 1: Compute from current LDS bank ---
ds_read ...                         ; read tiles from bank A
s_waitcnt lgkmcnt(0)                ; drain reads
v_mfma ...                          ; compute on read data

; --- Phase 2: Prefetch into other LDS bank ---
buffer_load_dwordx4 ...             ; load from HBM into VGPRs
s_waitcnt vmcnt(0)                  ; drain HBM loads
ds_write ...                        ; write to bank B

; --- Barrier 1: Wait for all waves to finish writing ---
s_waitcnt lgkmcnt(0)                ; drain LDS writes (REQUIRED before barrier)
s_barrier                           ; all waves done writing to bank B

; --- Next iteration: compute from bank B, prefetch into bank A ---
```

**Critical rule**: `s_waitcnt lgkmcnt(0)` MUST precede every `s_barrier`. The barrier synchronizes waves but does NOT guarantee LDS write visibility. Without the lgkmcnt drain, a wave may pass the barrier while its LDS writes are still in flight, and other waves read stale data.

### Persistent FWD GEMM: 4 Barriers Per Iteration

The persistent FWD kernel uses 4 barriers per K-iteration because A-side and B-side tiles are loaded separately:

```
Barrier 1: sync B-side LDS write from previous iteration
  -> safe to read B from LDS
Barrier 2: sync B-side LDS reads complete
  -> safe to write new B data
Barrier 3: sync A-side LDS write
  -> safe to read A from LDS
Barrier 4: sync A-side LDS reads complete
  -> safe to write new A data
```

All 4 barriers are load-bearing. Removing any single barrier causes inter-wave data corruption (measured at cos=0.988 when barrier 3 was removed).

### Direct-to-LDS Variant: 3 Barriers

The `use_block_pingpong=True` kernel variant uses `buffer_load ... lds` (direct-to-LDS) and requires 3 barriers:

```
Barrier 1: All waves done reading from LDS bank A -> safe to write new data
Barrier 2: All buffer_loads to LDS complete -> safe to read from LDS bank B
Barrier 3: All waves done with reads from bank B -> safe for next iteration
```

Barrier 3 prevents the next iteration's direct-to-LDS buffer_loads from overwriting data that slow waves are still reading.

### Barrier Removal: Dead End

Removing barriers to increase throughput was attempted repeatedly and consistently failed:

- Removing double-buffer barriers: -4.4% to -5.4% regression (occupancy drops from 2 WG/CU to 1 WG/CU because LDS requirement doubles)
- Every barrier in a ping-pong scheme protects a phase transition in the double-buffer lifecycle

---

## 4. Triple-Buffering

### Concept

Triple-buffering uses 3 LDS regions. While MFMAs read from region A, global loads write to region B, and region C holds data for the next compute phase. This eliminates 2 of 4 barriers per iteration because the read and write regions are always distinct -- no synchronization needed between readers and writers of different regions.

### Pipeline Efficiency

Attention kernels achieve 92-97% efficiency with triple-buffering. The extra LDS region eliminates most barrier stalls at the cost of higher LDS usage.

### LDS Requirement

Triple-buffering requires 3x the per-tile LDS allocation:

```
Example: 32KB per tile pair (A + B)
Double-buffer: 32KB x 2 = 64KB  (fits in standard 64KB LDS)
Triple-buffer: 32KB x 3 = 96KB  (EXCEEDS standard 64KB, needs extended 160KB)
```

On MI355X, the extended LDS mode provides 160KB per CU, which comfortably fits triple-buffered tiles. However, extended LDS reduces the available shared memory for other purposes and may affect overall CU resource allocation.

### Status

Triple-buffering was identified as a potential architectural improvement but was not implemented in any of the optimized kernels. The complexity of managing 3 LDS regions in hand-written assembly (3 sets of read/write address SGPRs, cyclic rotation logic) makes it a high-effort change.

---

## 5. Direct-to-LDS

### Mechanism

`buffer_load_dwordx4 vaddr, s[desc], offset offen lds` loads data from HBM directly into LDS, bypassing VGPRs entirely. The LDS write offset comes from the `m0` special register.

```asm
s_mov_b32 m0, s8                                    ; set LDS write offset
buffer_load_dwordx4 v98, s[40:43], 0 offen lds      ; HBM -> LDS directly
```

### Key Properties

1. **Counts in vmcnt, NOT lgkmcnt**: Unlike `ds_write` (which counts in lgkmcnt), direct-to-LDS loads are tracked by the vmcnt counter. Drain with `s_waitcnt vmcnt(0)`, not `lgkmcnt(0)`.

2. **No VGPR data consumption**: The VGPR operand (`v98` above) provides only the per-lane offset added to `m0` for the LDS destination address. The loaded data never touches VGPRs.

3. **m0 write hazard**: On gfx950, there is a 1-cycle write-to-read hazard on the `m0` register. A `buffer_load ... lds` must not issue in the cycle immediately after `m0` is written. Insert `s_nop 0` or another scalar instruction between the `s_mov_b32 m0` and the buffer_load:

```asm
s_mov_b32 m0, s8           ; write m0
s_nop 0                    ; 1-cycle hazard delay (REQUIRED)
buffer_load_dwordx4 v98, s[40:43], 0 offen lds
```

Removing the `s_nop 0` between `s_add_i32 m0, ...` and `buffer_load ... lds` caused cos=0.992-0.995 (subtle correctness failure).

### Advantage Over VGPR-Staged Pattern

```asm
; TRADITIONAL (2-step):
buffer_load_dwordx4 v[128:131], v128, s[28:31], 0 offen  ; HBM -> VGPR (vmcnt)
s_waitcnt vmcnt(0)                                         ; wait for HBM data
ds_write2st64_b64 v196, v[128:129], v[130:131] ...        ; VGPR -> LDS (lgkmcnt)

; DIRECT-TO-LDS (1-step):
s_mov_b32 m0, lds_offset
buffer_load_dwordx4 v_addr, s[desc], 0 offen lds          ; HBM -> LDS (vmcnt only)
```

Direct-to-LDS eliminates:
- 4 VGPRs per load (data never touches registers)
- The ds_write instruction and its lgkmcnt tracking
- The pipeline bubble between vmcnt drain and ds_write

### LDS Layout Differences

The LDS address for direct-to-LDS comes from `m0 + per-lane offset`. The per-lane offset is derived from the VGPR operand, creating a data layout in LDS that differs from naive expectations. Agent 3 in the FWD swarm found correctness failures (cos=0.905) due to misunderstood LDS layout from direct-to-LDS loads, requiring a revert to VGPR-staged loads.

### Impact on Buffer_load Hoisting

The direct-to-LDS kernel variant (`use_block_pingpong=True`) runs 15-20% faster than the VGPR-staged variant, but **cannot benefit from buffer_load hoisting** because there are no VGPR destinations to decouple. Different optimization techniques are needed for each variant.

---

## 6. Async LDS DMA Pipeline

### 3-Stage Pipeline

Forward attention uses a 3-stage software pipeline where K/V loads for iteration N+1 start during the softmax computation of iteration N, and K/V permute+write to LDS happens during the PV MFMAs:

```
Stage 1 (GEMM0 MFMAs):  Compute S = Q*K^T
                         Simultaneously: K/V loads for N+1 in flight (vmcnt)
Stage 2 (Softmax):       Compute P = softmax(S)
                         Simultaneously: K/V loads arriving
Stage 3 (PV MFMAs):      Compute O += P*V
                         Simultaneously: K/V perm+write to LDS for N+1
```

This pipeline achieves near-full utilization with only 4 `s_nop` total in the entire kernel and zero AGPRs.

### VGPR-Staged DMA Pattern (GGEMM Wgrad)

The variable-K wgrad kernel uses a simpler pattern where buffer_loads and ds_writes are interleaved with MFMA compute:

```asm
.L4:  ; K-tile loop
  ; --- Compute phase ---
  ds_read_b64_tr_b8 x9            ; 9 transpose reads for A operands
  s_waitcnt lgkmcnt(7)
  v_mfma x64                      ; 64 MFMAs using first batch
  ds_read_b64_tr_b8 x9            ; 9 transpose reads for second batch
  v_mfma x64                      ; 64 MFMAs using second batch

  ; --- Prefetch phase ---
  buffer_load_dwordx4 x2          ; LHS loads for next iteration
  buffer_load_dwordx4 x2          ; RHS loads for next iteration
  s_waitcnt vmcnt(0)
  ds_write2st64_b64 x4            ; write next iteration's data to LDS

  ; --- Barrier ---
  s_waitcnt lgkmcnt(0)
  s_barrier
  s_cbranch .L4
```

### Buffer_load Hoisting

The dominant optimization across all kernel families: moving buffer_loads earlier in the loop to increase HBM latency cover.

**Wgrad (+9-10%)**: RHS buffer_loads moved from ~8 MFMAs before drain (128 cycles cover) to loop top (~128 MFMAs = 2048+ cycles cover). Dead registers `v[216:223]` used as alternate destinations:

```asm
; BEFORE: 128 cycles of HBM latency cover
... MFMA x120 ...
buffer_load_dwordx4 v[144:147], v144, s[4:7], 0 offen   ; RHS
... MFMA x8 ...
s_waitcnt vmcnt(0)   ; STALLS ~170 cycles (300 - 128 = 172 cycles exposed)

; AFTER: 2048+ cycles of HBM latency cover
buffer_load_dwordx4 v[216:219], v216, s[4:7], 0 offen   ; RHS (hoisted, dead regs)
buffer_load_dwordx4 v[220:223], v217, s[4:7], 0 offen   ; RHS (hoisted, dead regs)
... MFMA x128 ...
s_waitcnt vmcnt(0)   ; FREE (data arrived hundreds of cycles ago)
```

**Persistent FWD (+2-4%)**: Only partial hoisting (2 of 4 B-side loads) works. Full hoisting of all 8 loads causes -9% to -11% regression due to memory queue saturation at low occupancy. The memory controller has finite queue depth, and front-loading all loads creates contention.

---

## 7. LDS Layout Patterns

### Row-Major Contiguous (Standard GEMM)

The simplest layout -- each matrix tile stored row-major in LDS:

```
Element[row, col] at LDS offset: base + row * row_stride + col * element_size

Example (128x32 BF16 tile, no padding):
  row_stride = 32 * 2 = 64 bytes
  Total size = 128 * 64 = 8192 bytes (8 KB)

Two tiles (A + B): 16 KB per double-buffer bank
Double-buffered total: 32 KB
```

The basic 128x128 GEMM uses 128 rows x 64 bytes/row = 8192 bytes per tile, with A and B tiles totaling 16KB per bank.

### Interleaved Write Pattern (ds_write2st64_b64)

The wgrad kernel uses `ds_write2st64_b64` which writes two 8-byte values at stride-64 offsets (64 dwords = 256 bytes apart). This interleaved layout matches the MFMA operand access pattern for FP8 transpose reads:

```asm
ds_write2st64_b64 v196, v[128:129], v[132:133] offset0:0 offset1:1
; Writes v[128:129] at v196 + 0*256 = v196
; Writes v[132:133] at v196 + 1*256 = v196 + 256
```

### Contiguous 128-bit Write Pattern (ds_write_b128)

The persistent FWD kernel uses `ds_write_b128` for contiguous 128-bit writes. This non-transpose layout is simpler because the FWD kernel does not need matrix transposition:

```asm
ds_write_b128 v20, v[32:35] offset:0       ; 16 contiguous bytes at v20+0
ds_write_b128 v20, v[36:39] offset:16384   ; next bank
```

The `ds_write_b128` offset field supports values up to 65535 bytes, which eliminates VALU address computation in many cases. This is significant because VALU instructions during the ds_write drain phase add to the critical path.

### Swizzled Layout (BWD Attention)

The kernel uses a swizzled LDS layout where each thread computes its write address using a custom formula that guarantees bank-conflict-free `ds_read_b128` access:

```
v16 = (t0*264 + t1*4 + t2*32 + t3*64 + t4*16 + t5*8) * 4

Where t0..t5 decompose threadIdx.x:
  264 = 0x108 (key: 264 mod 32 = 8, avoids 2-way bank conflicts)
```

LDS regions in BWD attention:
- Q region: at `v16 + offsets`, per-wave
- K region: at `v28 = v16 + s60`
- P region: at `v18 + offsets`
- dO region: shares `v16` base

Optimized kernels exclusively use `ds_read_b128` and avoid column reads entirely via pre-transposition in the swizzle.

### V Transpose Avoidance via v_perm_b32

Optimized kernels avoid an explicit LDS transpose pass for V data by using `v_perm_b32` to shuffle data in VGPRs before writing to LDS:

```asm
; V data arrives from HBM as buffer_load_dwordx2 (2 bf16 pairs)
; Need V_T[D_col][K_row] layout but have V[K_row][D_col]
; Solution: VGPR shuffle via v_perm_b32 before ds_write

buffer_load_dwordx2 ...             ; 4 instructions (load V chunk)
v_perm_b32 ...                      ; 8 instructions (shuffle for transpose)
ds_write2_b32 ...                   ; 4 instructions (write transposed)
; Total: 16 instructions vs 386 for naive LDS transpose pass
```

---

## 8. B Transpose Elimination

### The Problem

When computing `C = A * B` with B stored row-major, the MFMA needs column-major access to B. A naive approach stores B row-major in LDS and uses `ds_write_b16` scatter writes to transpose each element individually. This creates massive bank conflicts.

### Measured Impact

Hardware performance counters on the GEMM kernel:

| Metric | B Scatter (row-major B) | B Pre-Transposed | Improvement |
|--------|------------------------|-------------------|-------------|
| SQ_WAIT_INST_LDS | 178,755 cycles | 14,715 cycles | **12.1x reduction** |
| ds_write_b16 wait | 11,172 cycles/wave | N/A | Eliminated |
| Per-instruction wait | 9.2 CPI | N/A | Eliminated |
| Kernel time | ~2x baseline | baseline | **2x speedup** |

The B transpose scatter was responsible for **91% of all LDS wait cycles**. Pre-transposing B (storing it column-major from the host side) eliminated the scatter entirely.

### B-Reuse Pattern

With B pre-transposed, the GEMM can exploit B-reuse across multiple A-tile rows:

```
Without B-reuse: 20 LDS reads per K-step (10 A reads + 10 B reads)
With B-reuse:    12 LDS reads per K-step (8 A reads + 4 B reads)
Reduction: 40%
```

The B tile data is shared across multiple MFMA rows, so reading it once and reusing it across MFMAs reduces total LDS traffic.

### Decision Framework

Pre-transpose B when:
- B is accessed column-major by the MFMA (standard `C = A * B`)
- The kernel is LDS-bound (high SQ_WAIT_INST_LDS)
- Host-side transpose is acceptable (one-time cost amortized over many GEMM calls)

Use in-LDS transpose when:
- The kernel handles both `A*B` and `A^T*B` (wgrad needs transpose, FWD does not)
- Hardware transpose reads are available (`ds_read_b64_tr_b8` for FP8)
- The data format supports transpose reads (FP8/BF16 only)

---

## 9. ds_read Instruction Selection

### ds_read_b128 (16 bytes, 4 VGPRs)

**Throughput**: ~3.3 CPI
**Use case**: Contiguous 128-bit reads for non-transpose data (FWD GEMM A/B tiles, swizzled layouts)

```asm
ds_read_b128 v[16:19], v20 offset:0       ; read 16 bytes into v[16:19]
```

**Feeding MFMAs**:
- For 16x16x16 MFMA: one `ds_read_b128` feeds 2 consecutive MFMAs (8 bf16 values = 2 MFMA k-groups)
- For 16x16x32 MFMA: one `ds_read_b128` feeds 1 MFMA (16 bf16 values = 1 MFMA's full K)

**Clobber hazard**: Writes 4 consecutive VGPRs. Any live data in the destination range is destroyed. See [Section 11](#11-common-bugs) for details.

### ds_read2_b64 (2x 8 bytes, non-contiguous)

**Use case**: Reading two non-contiguous 8-byte values in a single instruction. Offsets are specified in qwords (8-byte units), max offset = 255 = 2040 bytes.

```asm
ds_read2_b64 v[12:15], v16 offset0:0 offset1:128
; Reads 8 bytes from v16 + 0*8 = v16 into v[12:13]
; Reads 8 bytes from v16 + 128*8 = v16 + 1024 into v[14:15]
```

Used when accessing data with non-power-of-2 strides or when two reads from different regions can be combined.

### ds_read_b64_tr_b8 (Hardware Transpose Read, FP8)

**Use case**: Reading FP8 data with automatic hardware transpose for `A^T * B` patterns (wgrad kernels).

```asm
ds_read_b64_tr_b8 v[152:153], v180 offset:16384
; Reads 8 bytes with FP8 transpose, producing 2 VGPRs
```

**Key properties**:
- Produces 2 VGPRs (64 bits) of transposed FP8 data
- Counts in lgkmcnt (same as regular ds_read)
- Destination must be an even-numbered VGPR pair (alignment requirement)
- 48 of these per inner loop iteration in the wgrad kernel
- Semantics match what the MFMA expects for transposed operands

The wgrad kernel uses 9 `ds_read_b64_tr_b8` per MFMA phase (4 for A operands, 4 for B operands, 1 for scale factor), totaling 18 per half-iteration and 36 per full K-tile step.

### ds_read_b64_tr_b16 (Hardware Transpose Read, BF16)

`ds_read_b64_tr_b16` assembles successfully on gfx950 but its semantics are undocumented. Production kernels do not use it, preferring the swizzled layout with `ds_read_b128` instead.

### ds_read_b64 (8 bytes, 2 VGPRs)

**Throughput**: ~6.1 CPI
**Constraint**: Destination VGPR must be even-aligned (e.g., v[146:147], not v[147:148]).

Confirmed alignment requirement for `ds_read_b64` destination registers.

### Instruction Selection Decision Tree

```
Need contiguous 16+ bytes?
  -> ds_read_b128 (best throughput, 4-VGPR clobber)

Need transposed FP8 access?
  -> ds_read_b64_tr_b8 (hardware transpose)

Need two non-contiguous 8-byte reads?
  -> ds_read2_b64 (dual offset, qword granularity)

Need 8 contiguous bytes?
  -> ds_read_b64 (even-aligned dest only)

Need 4 bytes or less?
  -> ds_read_b32 (worst throughput, use only if necessary)
```

---

## 10. Addressing for Different MFMA Widths

### 16x16x16 MFMA (v_mfma_f32_16x16x16_bf16)

Each thread provides data for one M-row (determined by `lane_id % 16`) and one K-group (determined by `lane_id / 16`, 4 groups of 16 threads).

**LDS read address formula**:

```
offset = (wave_offset + block_index*16 + mfma_row) * row_stride + kgrp * kgrp_stride

Where:
  mfma_row  = lane_id % 16          ; which M/N row this thread handles
  kgrp      = lane_id / 16          ; which K-group (0..3)
  row_stride = K_tile * sizeof(bf16) ; bytes per row (e.g., 32*2 = 64)
  kgrp_stride = 8 * sizeof(bf16)    ; bytes per K-group (8 bf16 = 16 bytes)
  wave_offset = wave_id * (M_per_wave) ; rows handled by this wave
  block_index = which 16x16 block within the tile

Example: 128x128 tile with K=32 bf16
  row_stride = 64 bytes
  kgrp_stride = 16 bytes
  Thread 0  (mfma_row=0, kgrp=0): base + 0*64 + 0*16 = base
  Thread 16 (mfma_row=0, kgrp=1): base + 0*64 + 1*16 = base + 16
  Thread 1  (mfma_row=1, kgrp=0): base + 1*64 + 0*16 = base + 64
```

One `ds_read_b128` (16 bytes) provides one full K-group of 8 bf16 values for one row. Two `ds_read_b128` reads feed 2 consecutive MFMAs in a 16x16x16 configuration.

### 16x16x32 MFMA (v_mfma_f32_16x16x32_fp8_bf8)

The thread-to-K mapping differs from 16x16x16. Each thread provides data for a different slice of the K=32 dimension. **You cannot simply swap the MFMA opcode from 16x16x16 to 16x16x32 without recomputing all LDS addresses.**

Confirmed empirically:

> "Can't just swap MFMA opcodes, thread-to-K mapping differs, need new addresses"

**Key difference**: for 16x16x32, each thread contributes to a larger K slice (32 vs 16), which changes how many bytes each thread needs to read and from which LDS offset. The LDS read pattern must match the MFMA's expected operand layout.

One `ds_read_b128` provides the full K=32 bytes of FP8 data for one MFMA (16 fp8 values per K-group x 2 K-groups = 32 values = 32 bytes). So one `ds_read_b128` feeds exactly 1 MFMA in the 16x16x32 configuration (vs 2 MFMAs in 16x16x16).

### 16x16x128 MFMA (v_mfma_f32_16x16x128_f8f6f4)

The MI355X-native FP8 MFMA reads 8 VGPRs for SrcA and 8 VGPRs for SrcB (128 FP8 values each):

```
Per MFMA: 8 VGPRs SrcA (64 bytes) + 8 VGPRs SrcB (64 bytes)
LDS reads: 2x ds_read_b128 per operand side (minimum)
```

The persistent FWD kernel uses 12 `ds_read_b128` total per K-tile step: reads for both A and B tiles across all MFMA operands.

### 32x32x64 MFMA (v_mfma_f32_32x32x64_f8f6f4)

Uses 8 VGPRs per source operand (same as 16x16x128), but the output tile is 32x32 instead of 16x16. This instruction does 131,072 FLOPs per instruction (4x more than 16x16x32_fp8_bf8), making it the preferred choice when LDS bandwidth permits.

The LDS layout must accommodate 32-row tiles instead of 16-row tiles, doubling the per-block LDS footprint but halving the number of blocks needed to cover the same output area.

---

## 11. Common Bugs

### Bug 1: ds_read_b128 Destination Clobber

`ds_read_b128` writes 4 consecutive VGPRs. If ANY of those VGPRs hold live data (MFMA operands, address pointers, scale factors), the live data is silently destroyed.

**Example**:
```asm
; v[138:139] holds FP8 scale factor, set in prologue
; Inner loop optimization uses v[138:139] as ds_read temp:
ds_read_b64_tr_b8 v[138:139], v205 offset:8192   ; CLOBBERS scale factor!
; ... later in epilogue ...
v_pk_mul_f32 v[0:1], v[138:139], v[0:1]           ; reads garbage scale
```

**Symptom**: cos_sim drops to 0.003 (random output) or specific systematic error depending on what was clobbered.

**Fix**: Trace register liveness across the ENTIRE kernel (prologue, inner loop, epilogue, outer loop). Registers set once in the prologue and consumed in the epilogue are invisible hazards when editing only the inner loop.

**Restore pattern**: if a register must be temporarily reused, restore it before the consuming code:
```asm
; At .L2 label (epilogue entry):
v_mov_b32_e32 v138, v136   ; restore scale factor
v_mov_b32_e32 v139, v136
```

### Bug 2: lgkmcnt FIFO Mistracking

`s_waitcnt lgkmcnt(N)` drains the N oldest LDS operations. Adding a new `ds_read` anywhere in the sequence shifts the counter for ALL subsequent instructions.

**Example**:
```asm
; 4 ds_reads outstanding:
ds_read v[152:153]    ; #1 (oldest)
ds_read v[154:155]    ; #2
ds_read v[222:223]    ; #3
ds_read v[138:139]    ; #4 (newest)

s_waitcnt lgkmcnt(2)  ; drains #1 and #2 (oldest 2)
; v[152:153] READY, v[154:155] READY
; v[222:223] STILL IN FLIGHT (3rd in FIFO)
; v[138:139] STILL IN FLIGHT (4th in FIFO)

USE v[222:223]        ; BUG! Data not ready!
```

**Fix**: Count EVERY ds_read and ds_write from the last `lgkmcnt(0)`. Map each `lgkmcnt(N)` to which specific operations have completed (oldest first) and which remain in flight. The FIFO is based on issue order, not register number.

**Tracing template**:
```
After lgkmcnt(0):          counter = 0
ds_read v[A]:              counter = 1  (oldest)
ds_read v[B]:              counter = 2
ds_read v[C]:              counter = 3  (newest)
s_waitcnt lgkmcnt(1):      drains A and B, C still pending
USE v[A]: OK
USE v[B]: OK
USE v[C]: BUG! (still in flight, lgkmcnt=1 means 1 remains)
```

### Bug 3: Inter-Wave LDS Race (Shared Region)

When multiple waves share an LDS region (e.g., for inter-wave communication or shared prefetch buffers), a single barrier per iteration is insufficient.

**Problem**: wave 0 writes prefetch data to shared LDS before wave 1 finishes reading the previous iteration's data:

```
Wave 0: ds_read (read old data) -> compute -> ds_write (write new data)
Wave 1: ds_read (read old data) -> compute -> ...
                                                  ^-- Wave 1 still reading!
                              ^-- Wave 0 already writing!
```

**Symptom**: ~3% error accumulating per iteration. The corruption is subtle because only some threads in some waves get stale data.

**Fix**: TWO barriers per loop iteration -- one before the LDS writes (to ensure all waves finished reading) and one after (to ensure all waves finished writing before any wave reads):

```asm
; All waves must finish reading before any wave writes
s_waitcnt lgkmcnt(0)
s_barrier                  ; BARRIER 1: all reads complete

; Now safe to write
ds_write ...
s_waitcnt lgkmcnt(0)
s_barrier                  ; BARRIER 2: all writes complete

; Now safe to read
ds_read ...
```

### Bug 4: accum_offset AGPR Aliasing

VGPRs at index >= `accum_offset` alias Accumulator GPRs (AGPRs). A `ds_read` targeting VGPRs in the AGPR range silently writes to accumulators instead.

**Example**:
```asm
; accum_offset = 64
; ds_read to v[64:67] clobbers AGPR a[0:3] (the MFMA accumulators!)
ds_read_b128 v[64:67], v20 offset:0   ; WRITES TO AGPRs, NOT VGPRs!
```

**Fix**: Set `.amdhsa_accum_offset` to at least the highest VGPR index used by any ds_read, ds_write, buffer_load, or VALU instruction. For kernels with 0 AGPRs, set `accum_offset = next_free_vgpr`.

### Bug 5: Barrier Removal Regression

Removing double-buffer barriers to "eliminate stalls" consistently causes either correctness failure or performance regression:

- **Correctness failure**: removing any barrier in the ping-pong scheme causes inter-wave data corruption
- **Performance regression** (-4.4% to -5.4%): if the removal "works" by doubling LDS usage (no ping-pong), occupancy drops from 2 WG/CU to 1 WG/CU

### Bug 6: Prefetch Drain to Shared LDS

When prefetch loads drain (`vmcnt(0)`) and their data is written to LDS via `ds_write`, the writes can corrupt data that other waves are still reading from the previous iteration. This is a specific instance of Bug 3, triggered by the prefetch drain completing before all waves have consumed the current iteration's data.

### Bug 7: buffer_load WAW with ds_read Destinations

Buffer_load writes VGPRs asynchronously (~300 cycles after issue). If a ds_read writes to the same VGPRs between the buffer_load issue and its data arrival, the buffer_load completion overwrites the ds_read data:

```asm
buffer_load_dwordx4 v[220:223], ...   ; issued now, data arrives in ~300 cycles
; ... 200 cycles later ...
ds_read_b64 v[222:223], ...           ; writes v[222:223] with LDS data
; ... 100 cycles later ...
; buffer_load DATA ARRIVES, overwrites v[222:223] with HBM data!
; The ds_read value is LOST
```

**Fix**: Keep buffer_load destination registers completely disjoint from all ds_read destinations within the load's flight window.

---

## 12. Size Constraints and Occupancy

### LDS Capacity

| Mode | Capacity per CU |
|------|-----------------|
| Standard | 64 KB |
| Extended | 160 KB |

MI355X supports the extended 160KB mode, confirmed via `rocminfo` and `hipDeviceProp_t.sharedMemPerBlock`.

### LDS vs Occupancy

On gfx950, **LDS size does NOT usually limit occupancy** -- VGPRs are the binding constraint:

```
VGPR file: 512 VGPRs per SIMD
MFMA kernels typically use 224-248 VGPRs
  224 VGPRs -> 512/224 = 2 waves/SIMD
  248 VGPRs -> 512/248 = 2 waves/SIMD
  256 VGPRs -> 512/256 = 2 waves/SIMD

LDS: 64 KB standard
  32 KB double-buffer -> 64/32 = 2 WGs/CU (matches VGPR limit)
  Even with 48 KB LDS, VGPRs limit first
```

LDS size does NOT affect occupancy when VGPRs are the limit. The 512 VGPR/SIMD ceiling means most MFMA-heavy kernels hit the VGPR limit at 2 waves/SIMD before LDS becomes a constraint.

**Exception**: Kernels using fewer than ~128 VGPRs (4+ waves/SIMD) may find LDS limits occupancy if each wave group uses >16KB of LDS.

### LDS Allocation Granularity

LDS is statically allocated. Optimized kernels declare `shared_mem=0` in the kernel descriptor because they use the statically allocated 65536 bytes. The allocation is per-workgroup and shared among all waves in the workgroup.

### Typical LDS Budgets by Kernel Type

| Kernel | LDS per WG | Buffering | Notes |
|--------|-----------|-----------|-------|
| GEMM 128x128 (bf16) | 32 KB | Double | 16 KB per bank (8 KB A + 8 KB B) |
| GGEMM Wgrad (FP8) | ~32 KB | Double | `ds_write2st64_b64` interleaved layout |
| GGEMM FWD Persistent | ~64 KB | Double | 32 KB per bank, 4 barriers/iter |
| FWD Attention | ~40-48 KB | Triple | Q + K + V staging areas |
| BWD Attention | ~48 KB | Various | Q, K, P, dO regions with swizzle |

### Size Optimization: ds_write_b128 Offset Field

`ds_write_b128` has an offset field supporting values up to 65535 bytes. This means a single base VGPR can address the entire 64KB LDS without any VALU offset computation:

```asm
; No VALU needed for offset calculation:
ds_write_b128 v_base, v[0:3] offset:0        ; write at base + 0
ds_write_b128 v_base, v[4:7] offset:16384    ; write at base + 16384
ds_write_b128 v_base, v[8:11] offset:32768   ; write at base + 32768
ds_write_b128 v_base, v[12:15] offset:57344  ; write at base + 57344
```

This eliminates 4+ VALU instructions per ds_write group, reducing critical-path latency in the write drain phase.

