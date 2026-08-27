---
guide: register-allocation
category: register-management
architecture: gfx950
tags: [VGPR, AGPR, accum_offset, occupancy, register-budget, register-lifetime, register-clobber]
---

# Register Allocation and Occupancy Management for gfx950 (MI355X / CDNA4)

A comprehensive guide to planning, implementing, and debugging register allocation in hand-written AMDGCN assembly kernels for MI355X. Covers GEMM, attention FWD/BWD, GEMV, and grouped FP8 GEMM kernels.

---

## 1. VGPR/AGPR Budget and Occupancy Breakpoints

### Hardware Resources

MI355X (CDNA4) per SIMD:
- **512 VGPRs** (unified VGPR+AGPR file)
- **800 SGPRs** (practical limit ~102-106 per wave with system SGPRs)
- **160 KB LDS per CU** (shared across workgroups on the CU)
- **4 SIMDs per CU**, max 8 waves per SIMD

### Occupancy Breakpoints

The occupancy table is determined solely by `next_free_vgpr` (which includes both VGPRs and AGPRs):

| next_free_vgpr | Waves/SIMD | VGPRs per Wave | Notes |
|----------------|------------|----------------|-------|
| 1-64           | 8          | 64             | Maximum occupancy |
| 65-73          | 7          | 73             | |
| 74-85          | 6          | 85             | |
| 86-102         | 5          | 102            | |
| 103-128        | 4          | 128            | Sweet spot for compute-heavy kernels |
| 129-170        | 3          | 170            | |
| 171-256        | 2          | 256            | Minimum viable for latency-bound kernels |
| 257-512        | 1          | 512            | Avoid unless absolutely necessary |

**Key thresholds to memorize:**
- **128 VGPRs -> 4 waves** (optimal for most kernels)
- **170 VGPRs -> 3 waves**
- **256 VGPRs -> 2 waves**

The occupancy formula is: `waves_per_simd = floor(512 / next_free_vgpr)`, capped at 8.

### Real-World Occupancy Examples

| Kernel | next_free_vgpr | Waves/SIMD | Bottleneck |
|--------|---------------|------------|------------|
| FWD attention (optimized) | 128 | 4 | None (optimal) |
| FWD attention (original) | 136 | 3 | 7 unnecessary VGPRs |
| 128x128 GEMM | 124 | 4 | None (60 VGPRs + 64 AGPRs) |
| 256x128 GEMM | 188 | 2 | AGPRs (128 accumulators) |
| BWD attention | 512 (declared) / 416 (actual) | 1 | AGPRs (160 accumulators) |
| Variable-K wgrad | 224 | 2 | accum_offset at 224 |
| Persistent FWD GEMM | 248 | 2 | 32 tiles x 4 VGPRs accumulators |

### The 135 -> 128 Victory

The forward attention kernel initially used 134 VGPRs (rounded to 136 for alignment), yielding 3 waves/SIMD. Analysis found 7 VGPRs (v128-v134) held intermediate exp2 results live only during the softmax window (about 65 lines of code). By remapping these intermediates to registers that were dead during softmax (v94 during softmax, v116-v121 after causal mask evaluation), the kernel dropped to exactly 128 VGPRs = 4 waves/SIMD. Result: **3% speedup** (2.936ms -> 2.871ms / 1532 TFLOPS). A single VGPR over the threshold costs 33% occupancy.

---

## 2. accum_offset Semantics and Encoding

### What accum_offset Does

`accum_offset` divides the 512-VGPR register file into two regions:

```
v[0]  ...  v[accum_offset-1]   = Architectural VGPRs (general purpose)
v[accum_offset] ... v[511]      = AGPRs (aliased as a[0], a[1], ...)
```

The mapping is: `v[accum_offset + N]` physically aliases `a[N]`.

### Rules

1. **accum_offset must be a multiple of 4.** Values like 66 cause assembly errors. Use 68 instead.
2. **accum_offset range:** 4 to 256 (gfx950 constraint).
3. **next_free_vgpr = accum_offset + num_agprs.** This is the single most important rule in gfx950 register allocation. Violating it is the #1 assembly bug in practice (see Section 9).

### Kernel Descriptor Declaration

```asm
.amdhsa_kernel my_kernel
  .amdhsa_next_free_vgpr  124    ; = accum_offset + num_agprs = 60 + 64
  .amdhsa_accum_offset     60    ; first 60 VGPRs are general purpose
  .amdhsa_next_free_sgpr   52
  ; ...
.end_amdhsa_kernel
```

### accum_offset Field Location in KD Binary (gfx950)

On gfx950, `accum_offset` is stored in **bits [3:0] of compute_pgm_rsrc3** (the RSRC3 word of the kernel descriptor). This is different from gfx90a/gfx940 where it was in bits [19:14].

The encoding formula:
```
rsrc3_field = accum_offset / 4 - 1
```

To decode: `accum_offset = (rsrc3_field + 1) * 4`.

Example: `rsrc3 = 0x00000008` means `(8 + 1) * 4 = 36` VGPRs before accumulators.

### When accum_offset = next_free_vgpr (Zero AGPRs)

Setting `next_free_vgpr == accum_offset` allocates **zero AGPRs**. This is valid for kernels that use VGPRs as MFMA destinations (FWD attention does this). But if any MFMA instruction writes to AGPRs (a[N] operands), those writes go to unallocated registers and reads return garbage.

The persistent grouped GEMM kernels use accum_offset=224 with next_free_vgpr=248, meaning 24 AGPRs are available. The variable-K wgrad kernel uses accum_offset=224 with next_free_vgpr=224, meaning 0 AGPRs -- it stores all MFMA results in VGPRs v[0:127].

---

## 3. AGPR Aliasing: The Unified Register File

### The Physical Reality

On gfx950 (CDNA4), VGPRs and AGPRs share a unified physical register file. VGPRs at or above `accum_offset` are the **same physical storage** as AGPRs:

```
v[accum_offset + 0]  <==>  a[0]
v[accum_offset + 1]  <==>  a[1]
v[accum_offset + 2]  <==>  a[2]
...
v[accum_offset + N]  <==>  a[N]
```

### Why This Matters

Any instruction that writes to a VGPR in the AGPR-aliased range silently clobbers accumulator state. This includes:

- `ds_read_b128 v[N:N+3]` where N >= accum_offset
- `global_load_dwordx4 v[N:N+3]` where N >= accum_offset  
- `buffer_load_dwordx4 v[N:N+3]` where N >= accum_offset
- `v_mov_b32 v[N]` where N >= accum_offset

### Real Example: ds_read Clobbers Accumulators

With accum_offset = 64:
```asm
; a[0:3] hold MFMA accumulator results
ds_read_b128 v[64:67], v_addr   ; CLOBBERS a[0:3]!
; a[0:3] now contain LDS data, not MFMA results
```

This was discovered empirically when an attention kernel wrote LDS data to v[64+] and wiped out accumulated QK^T scores. The symptom was cos_sim near 0 with correct magnitudes.

### Defensive Rule

**ALL compute VGPRs must have register numbers below accum_offset.** If you need more working VGPRs, you must increase accum_offset -- and correspondingly increase next_free_vgpr to `accum_offset + num_agprs`.

---

## 4. Register Clobber Patterns (The #1 Bug Class)

Register clobber is the single most common class of bugs in hand-written gfx950 assembly. Every kernel project has encountered at least one. They are ranked here by frequency and severity.

### Pattern 1: ds_read_b128 Destination Clobber

**Frequency:** Common.

`ds_read_b128 v[N:N+3]` writes 4 consecutive VGPRs. If any of v[N], v[N+1], v[N+2], v[N+3] hold live values, they are silently overwritten.

**Trap #1 (pointer clobber):** `ds_read_b128 v[16:19], v8` clobbered v17:v18 which held a scale base pointer. Worked for K=128 (1 inner loop iteration) but faulted at K>=256 (2+ iterations where the clobbered address was reused).

**Trap #2 (cross-phase liveness):** v[138:139] appeared dead inside the inner MFMA loop but was alive across the loop exit -> store path. A ds_read clobbered the FP8 scale factor, producing cos=0.003.

**Trap #3 (prologue address clobber):** v[162:163] held LDS base addresses computed in the prologue. Using them as ds_read_b64_tr_b8 destinations corrupted addresses in all subsequent iterations. cos=0.931.

**Rule:** When using multi-dword LDS/global loads, map out VGPR lifetimes across ALL paths -- prologue, inner loop, epilogue, store sections, outer loop, and all branch targets.

### Pattern 2: MFMA Output VGPR Overlap with Scratch

**Frequency:** Common.

When MFMA output VGPRs overlap with registers used as scratch temporaries:

```asm
v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[4:7], a[0:3]
; ... later ...
v_add_f32 v8, v9, v10   ; scratch computation
v_mul_f32 v9, v8, v11   ; uses v[8:23] as scratch
; BUT if accum_offset=0 and MFMA writes to v[8:23]... clobber!
```

**Real case (BWD attention):** Scratch temps in v[8:23] silently overwrote live MFMA accumulator outputs. cos_sim=0.92 for dK. Only found by mapping register lifetimes line-by-line.

### Pattern 3: buffer_load WAW Hazard

**Frequency:** Common.

`buffer_load_dwordx4 v[88:91]` before an MFMA that reads v[88:91] as source creates a WAW race. The buffer_load has ~400-cycle latency. If it returns before the MFMA reads its source operands, the MFMA gets the buffer_load data instead of the intended values.

**Rule:** When prefetching via buffer_load, always target VGPRs that have NO reads between the load issue and the corresponding `s_waitcnt vmcnt(0)`. Verified: v[124:127] (dead between QK MFMA and exp2 sections) worked; v[88:91] (read by MFMA at line 453) produced cos=0.83.

### Pattern 4: global_load Clobbers LDS Read Address

**Frequency:** Common.

`global_load_dwordx4 v[36:39]` was issued before `ds_read_u16 v24, v38`. v38 held the LDS address but was in the global_load destination range. Under memory pressure (many concurrent WGs), the load data arrived and overwrote v38 before the ds_read used it.

**Symptom:** Non-deterministic NaN (0xFFC0) at specific lanes, only under concurrent WG execution.

### Pattern 5: Cross-Phase Register Dual-Use

**Frequency:** Common.

The same VGPR range serves different purposes in different kernel phases, and a phase transition destroys live data.

**Real case (BWD attention):** v[160:163] was used as both K N-tile 2 data for GEMM0 AND GEMM2 accumulator. Zero-initializing accumulators for GEMM2 destroyed K data needed by GEMM0. Fix: reload from LDS via ds_read_b128 before each GEMM0 block.

**Real case (uber-kernel):** Phase 3 attention `flat_load_dword v4` destroyed the `lane/16` value needed by Phase 4 MFMA weight addressing.

### Pattern 6: Scale Factor Clobber in Epilogue

**Frequency:** Common.

v[138:139] held FP8 scale factors across the inner loop. A ds_read in the inner loop body used v[138:139] as a rotating temp, destroying the scale factor. The scale was needed at loop exit for the store epilogue. Symptom: correct accumulation but wrong output magnitude (cos near 0).

### Pattern 7: A-Operand Clobber During MFMA Reordering

**Frequency:** Rare.

When reordering MFMAs for better scheduling, v[146:147] (A-operand for later MFMAs) was overwritten by an earlier ds_read that used the same register range as a destination. The reordering moved the ds_read before the MFMA that consumed v[146:147].

---

## 5. Register Lifetime Analysis Methodology

### Step 1: Map All Definitions and Uses

For every instruction in the kernel:
1. Record the **destination** register(s) -- the DEF point
2. Record the **source** register(s) -- the USE points
3. For read-modify-write instructions like `v_and_b32 v0, 0x3ff, v0`, v0 is BOTH a source and destination at the same instruction. The source read needs the OLD value.

Special cases for multi-dword operations:
- `ds_read_b128 v[N:N+3]`: DEF = {v[N], v[N+1], v[N+2], v[N+3]}
- `ds_read_b64 v[N:N+1]`: DEF = {v[N], v[N+1]}
- `buffer_load_dwordx4 v[N:N+3]`: DEF = {v[N], v[N+1], v[N+2], v[N+3]}
- `v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[4:7], a[0:3]`: DEF = {a[0:3]}, USE = {v[0:3], v[4:7], a[0:3]}

### Step 2: Build Def-Use Chains

For each register, find all (def, use) pairs. A register is live from its def to its last use before the next def.

For loop bodies, the analysis must consider:
- Values defined before the loop and used inside (live throughout)
- Values defined inside the loop and used in the same iteration (intra-iteration liveness)
- Values defined inside the loop and used in the next iteration (cross-iteration liveness -- adds 1 iteration of live range)
- Values defined inside the loop and used after the loop (live until loop exit)

### Step 3: Group-Aware Analysis

For `ds_read_b128`, the hardware allocates all 4 destination registers atomically. Even if only v[N] is consumed, v[N+1:N+3] are live from the ds_read until v[N] is consumed (the hardware writes all 4).

This matters for pressure analysis: naive per-register analysis undercounts peak pressure because it ignores that register groups must be allocated as units.

### Step 4: Dead Register Discovery Protocol

To find registers available for reuse:

1. Start from the full register map (Section 8)
2. For each inner loop iteration, mark registers as DEAD at the point where their last USE occurs and before their next DEF
3. Look for registers that are dead for a span long enough to hold your data
4. **Verify across ALL phases:** A register dead in the inner loop may be live in the prologue, epilogue, store path, or outer loop. You must check every path.

**Example from grouped GEMM:** v[216:223] were originally copies of loop-invariant LDS base addresses (redundant MOVs from v180/v181/v177/v178/v179). After eliminating the MOVs, all 8 registers became dead in the inner loop -- perfect targets for RHS buffer_load hoisting. This single optimization gave +10% speedup.

**Example from BWD attention:** v[232:255] were globally dead (24 VGPRs). Used for K data promotion from AGPRs when converting 16x16x16 to 16x16x32 MFMA.

### Step 5: Pressure Analysis Progression

Rigorous pressure analysis showed the following progression:

| Method | Peak Live | Description |
|--------|-----------|-------------|
| Interval-based (naive) | 319 | Assumes registers live for entire first-to-last-use span. Overcounts. |
| Proper def-use chains | 174 | Only counts registers between their def and last use. |
| Group-aware | 208 | Accounts for 4-reg groups from ds_read_b128. |
| Per-register disjoint | 275 | Final accurate count considering all constraints. |

The gap between def-use (174) and group-aware (208) shows the cost of multi-dword load destinations. The gap between group-aware (208) and per-register (275) shows the cost of cross-iteration liveness and scheduling constraints.

---

## 6. SGPR Layout

### Without Kernarg Preloading (Standard)

When `enable_sgpr_kernarg_segment_ptr` is set and `enable_sgpr_workgroup_id_x/y/z` are set, the system SGPRs are allocated as:

```
s[0:1]  = kernarg_segment_ptr (64-bit pointer to kernel arguments)
s2      = workgroup_id_x
s3      = workgroup_id_y
s4      = workgroup_id_z
```

**Critical hazard:** `s_load_dwordx4 s[4:7], s[0:1], 0x0` overwrites s4 with kernarg data. You must save workgroup_id_z before the first kernarg load completes:

```asm
s_mov_b32 s29, s4          ; save wg_id_z BEFORE kernarg loads land
s_load_dwordx4 s[4:7], s[0:1], 0x0
s_load_dwordx4 s[8:11], s[0:1], 0x10
s_waitcnt lgkmcnt(0)       ; now s[4:17] = kernarg data, s29 = wg_id_z
```

### With Kernarg Preloading (Triton/Persistent GEMM Pattern)

The Triton compiler uses kernarg preloading via `USER_SGPR_COUNT` in the kernel descriptor. With `USER_SGPR_COUNT=16`:

```
s[0:1]   = kernarg_segment_ptr
s[2:15]  = Preloaded kernarg data (14 dwords = 56 bytes)
             s[2:3]   = kernarg[0:7]    (first pointer)
             s[4:5]   = kernarg[8:15]   (second pointer)
             s[6:7]   = kernarg[16:23]  (third pointer)
             s[8:9]   = kernarg[24:31]  (fourth pointer)
             s[10:11] = kernarg[32:39]
             s[12:13] = kernarg[40:47]
             s[14]    = kernarg[48:51]
             s[15]    = kernarg[52:55]
s[16]    = workgroup_id_x  (placed AFTER the 16 user SGPRs)
```

**Critical:** The workgroup_id system SGPRs are pushed to `s[USER_SGPR_COUNT]`, not s2. With USER_SGPR_COUNT=16, workgroup_id_x is at s16. Getting this wrong means the kernel reads preloaded kernarg data as a workgroup ID.

### SGPR Budget

Practical SGPR usage across real kernels:

| Kernel | SGPRs Used | Notable Allocations |
|--------|-----------|---------------------|
| FWD attention | ~51 | s[8:11] Q bufdesc, s[20:23] K bufdesc, s[48:49] exec mask |
| 128x128 GEMM | ~50 | s[4:17] kernargs, s[20:23] buffer desc, s26+ loop state |
| BWD attention | 92 | s[0:1] karg, s[8:9] Q ptr, s[32:91] all tensor descriptors |
| Persistent grouped GEMM | ~50 | s[4:7] RHS bufdesc, s[24:27] LHS bufdesc, s45 K-loop iter |

SGPRs are almost never the occupancy bottleneck (800/wave vs typical usage of 50-100).

### Buffer Resource Descriptor Construction

Standard raw buffer descriptor from a 64-bit base pointer:

```asm
s[N]   = base_addr_low
s[N+1] = (base_addr_high & 0xFFFF) | 0x40000
s[N+2] = 0x80000000       ; num_records (unclamped)
s[N+3] = 0x20000          ; format/stride fields
```

This pattern is used for buffer_load/buffer_store with the `offen` addressing mode.

---

## 7. Occupancy Analysis

### The Framework

Occupancy on gfx950 is limited by whichever resource is most constrained:

```
VGPR limit:  floor(512 / next_free_vgpr)
SGPR limit:  floor(800 / next_free_sgpr)    [rarely binding]
LDS limit:   floor(160KB / lds_per_workgroup) * waves_per_workgroup
Thread limit: floor(1024 / threads_per_block)  [for waves per WG]
```

In practice, **VGPRs are almost always the sole bottleneck**. SGPR, LDS, and thread limits rarely bind.

### Case Study: BWD Attention 2-Wave Infeasibility Proof

The most rigorous occupancy analysis performed aimed to reduce the BWD attention kernel from 1 wave/SIMD to 2 waves/SIMD by reducing total VGPR allocation from 416 to <= 256.

**Kernel stats:**
- 232 VGPRs used (v0-v231)
- 111 peak simultaneously live VGPRs
- 112 AGPRs as MFMA tile sources (a[0:111])
- 48 AGPRs as MFMA accumulators (a[112:159])
- accum_offset = 256, next_free_vgpr = 512 (declared), 416 (actual minimum)

**Strategy:** Promote the 112 tile AGPRs (a[0:111]) to VGPRs, keeping only 48 accumulator AGPRs. This would give accum_offset = 208 (the 208 VGPRs needed for compute + promoted tiles) + 48 AGPRs = 256 total. Exactly at the 2-wave threshold.

**Pressure analysis results:**

| Analysis Method | Peak Live | + 48 Accum AGPRs | vs 256 Target |
|----------------|-----------|-------------------|---------------|
| Def-use chains | 174 | 222 | 34 under -- looks feasible! |
| Group-aware | 208 | 256 | Exactly at limit |
| Per-register (final) | 275 | 323 | **67 over -- infeasible** |

The gap of 67 registers proved that 2-wave occupancy is impossible via register remapping alone. The kernel would need architectural restructuring -- splitting into multiple passes, reducing tile size, or eliminating pipeline stages.

**Why the analyses diverge:**
- Def-use chains count only 16 tile AGPRs live at once (only the ones actively being consumed by the current MFMA). But ds_read_b128 loads arrive in groups of 4, and the scheduling constraints force more to be live simultaneously.
- Per-register analysis revealed that at the peak pressure point (middle of the MFMA chain), 275 distinct register slots are simultaneously needed.

### Case Study: FWD Attention 4-Wave Success

FWD attention used 134 VGPRs with 0 AGPRs (all MFMA destinations in VGPRs), giving 3 waves/SIMD. Analysis found:

- v128-v134 (7 VGPRs) held exp2 intermediate results, live only during a 65-line softmax window
- v94 was dead during the softmax window (contains lane-within-wave, not needed during exp)
- v116-v121 were dead after causal mask evaluation completed

Remapping exp2 intermediates to these dead registers dropped the kernel to exactly 128 VGPRs = 4 waves/SIMD. **3% speedup** -- pure occupancy gain.

### Occupancy vs Performance: When 1 Wave Is OK

Not all kernels benefit from higher occupancy:

| Kernel | Occupancy | Performance | Why |
|--------|-----------|-------------|-----|
| BWD attention | 1 wave | 113 TFLOPS | Memory-latency-bound (31% wait fraction). More waves wouldn't help because the bottleneck is per-wave latency, not parallelism. |
| Persistent grouped GEMM | 2 waves | 1540-1912 TFLOPS | Moderate benefit. 2 waves hide some memory latency but barriers serialize. |
| FWD attention | 4 waves | 1532 TFLOPS | Maximum benefit. Compute-bound kernel where more waves directly increase throughput. |

**Rule of thumb:** Occupancy helps most when the kernel is compute-bound or has independent waves. For memory-latency-bound kernels at 1 wave, the gains from 2 waves are often in the noise (s_setprio experiments showed only ~0.7% at 1 wave).

---

## 8. Practical Register Maps from Real Kernels

### Map A: 128x128 BF16 GEMM (60 VGPRs + 64 AGPRs)

The simplest production-quality GEMM register allocation. accum_offset=60, next_free_vgpr=124.

```
VGPRs (v0-v59):
  v0       tid (thread ID within workgroup, persistent)
  v1       wave_id (v0 >> 6)
  v2       lane (v0 & 63)
  v3       wave_row (v1 >> 1)
  v4       wave_col (v1 & 1)
  v5       mfma_row (v2 & 15)
  v6       mfma_kgrp (v2 >> 4)
  v7       row_local (tid/4, for cooperative loads)
  v8       col_quarter (tid%4, for cooperative loads)
  v9       A LDS write offset (persistent across K-loop)
  v10      B LDS write offset (persistent across K-loop)
  v11-v13  row_local_hi and derivatives
  v14-v17  A LDS read addresses (4 persistent, one per BI=0..3)
  v18-v21  B LDS read addresses (4 persistent, one per BJ=0..3)
  v22-v23  temp (address computation scratch)
  v24-v27  B pass 1 global load destination
  v28-v31  B operand buffer (MFMA B input, alternate)
  v32-v35  persistent global row offsets (A_lo, A_hi, B_lo, B_hi)
  v36-v37  address pair scratch
  v38-v39  free during compute
  v40-v43  A operand buffer (MFMA A input)
  v44-v47  B operand buffer (MFMA B input, primary)
  v48-v51  A pass 0 global load destination
  v52-v55  A pass 1 global load destination
  v56-v59  B pass 0 global load destination

AGPRs (a0-a63):
  a[BI*16 + BJ*4 .. BI*16 + BJ*4 + 3]  for MFMA block (BI, BJ)
  16 blocks x 4 AGPRs = 64 total
  BI=0..3 (4 blocks along M), BJ=0..3 (4 blocks along N)
```

**Key design choices:**
- LDS read addresses (v14-v21) are precomputed once before the K-loop -- saves ~16 VALU per K iteration
- Global row offsets (v32-v35) are persistent -- only the K offset (SGPR) advances
- Zero slack: 60 VGPRs leaves no room for further register-based optimizations

### Map B: FWD Attention (134 VGPRs, 0 AGPRs)

No AGPRs used -- all MFMA destinations are VGPRs. accum_offset=128 (after optimization), next_free_vgpr=128.

```
VGPRs (v2-v133):
  v[2:17]    O accumulator D0 (FP32, MFMA destination)
  v[18:33]   O accumulator D1 (FP32, MFMA destination)
  v[34:49]   S accumulator sub-tile 1 (FP32, MFMA destination)
  v[50:65]   S accumulator sub-tile 0 (FP32, MFMA destination)
  v[66:81]   Q tile data (BF16 packed in VGPRs, MFMA source)
  v82        Q-row LDS base offset
  v83        Thread ID (mbcnt result)
  v84-v85    K/V global address temps
  v86        V LDS write base
  v87        -inf constant (0xff800000) for causal masking
  v[88:93]   K/V load data + perm intermediates
  v94        Lane-within-wave (dead during softmax -> available for exp2 temps)
  v95        Wave ID (v83 >> 6)
  v96        l_acc running sum
  v97-v98    Causal mask column indices
  v99        ds_bpermute partner address
  v[100:107] LDS read base addresses for PV (4 D0, 4 D1)
  v[108:112] K/V global offset computation
  v[113:114] Q LDS read offsets
  v[115:120] Causal mask column indices (+1..+6)
             (Dead after mask eval -> available for exp2 temps)
  v121       log2e-scaled softmax value / running max
  v[122:127] Temporary: exp results, P packed, K/V load
             (Remapped from v[128:133] in optimization pass)
```

**Why no AGPRs:** One approach avoids AGPRs because reading AGPR results requires `v_accvgpr_read_b32` with 4x `s_nop 15` (64 NOP cycles on MI355X XDL2x 16x16 opcodes). By keeping MFMA results in VGPRs, this eliminates AGPR read hazards entirely.

### Map C: BWD Attention (256 VGPRs + 160 AGPRs)

The most complex register allocation. accum_offset=256, next_free_vgpr=416 (actual).

```
VGPRs (v0-v231, 232 used):
  v[0:3]     MFMA source operands (Q tile, from LDS)
  v[4:7]     MFMA source operands (K tile, from LDS)
  v[8:23]    DANGER ZONE: scratch temps that overlap with potential MFMA output range
  v[24:31]   Global address computation, V tile data
  v[32:63]   S matrix values (32 output VGPRs from QK^T MFMAs)
  v[64:75]   Softmax intermediates (exp2, perm results)
  v[76:91]   Dead during GEMM0 -> available for Q AGPR promotion (16x16x32 conversion)
  v[92:111]  K/V global load destinations, LDS write addresses
  v[112:135] dQ/dK/dV store path, address computation
  v[136]     -inf constant (0xff800000)
  v[137:139] FP8 scale factors (CAREFUL: v[138:139] clobbered by ds_read!)
  v[140:163] LDS read results, rotating temps, phase-dependent dual-use
  v[164:175] v_perm_b32 results for BF16 packing (dK/dV MFMA inputs)
  v[176:231] Additional MFMA source operands, address computation, loop state

AGPRs (a0-a159):
  a[0:111]   MFMA tile data sources (Q/K loaded from LDS via ds_read_b128)
             60 ds_read_b128 instructions load tile data here
  a[112:159] MFMA accumulators for dK/dV (accumulate-in-place across loop iterations)
```

**Key hazard:** v[160:163] is dual-use: K N-tile 2 data (GEMM0) AND GEMM2 accumulator. Zero-init for GEMM2 destroys K data. Must reload from LDS.

### Map D: Variable-K Wgrad Grouped GEMM (224 VGPRs, 0 AGPRs)

accum_offset=224, next_free_vgpr=224.

```
VGPRs (v0-v223):
  v[0:127]     MFMA accumulators (32 tiles x 4 VGPRs, zeroed per expert group)
  v[128:135]   Buffer load destinations (LHS: v[128:131], RHS: v[132:135])
  v[136:139]   Scale constants for epilogue (v_pk_mul_f32)
  v[140:161]   LDS read results (A/B operands), rotating temps
  v[162:169]   LDS write addresses (live across loop and epilogue -- DO NOT USE AS SCRATCH)
  v[170:175]   Free / temp
  v[176:181]   LDS read base addresses (XOR-swizzled, computed in prologue)
  v[196:197]   LDS write base pointers
  v[198]       K offset (incremented each iteration)
  v[199:204]   RHS address bases, bounds check
  v[205:215]   Originally redundant copies (eliminated -> freed for optimization)
  v[216:223]   Dead after MOV elimination -> used for RHS buffer_load hoisting (+10%)
```

### Map E: Persistent FWD Grouped GEMM (248 VGPRs, 0 AGPRs)

accum_offset=248, next_free_vgpr=248 (some additional operands above accum_offset in v[224:247] may be handled by the assembler separately).

```
VGPRs (v0-v247):
  v[0:31]      A-side MFMA SrcA operands (from LDS via ds_read_b128)
               Also reused as B-side buffer_load destinations LATE in the loop
  v[32:55]     A-side buffer_load destinations (HBM -> VGPR -> LDS)
  v[40:47]     MFMA accumulators tiles 0-1 (v[44:47]=tile0, v[40:43]=tile1)
  v[56:63]     Per-tile base addresses (persistent across K-loop)
  v[64:70]     A-side load address bases (persistent)
  v[65,67]     B-side ds_read base addresses
  v[72:191]    MFMA accumulators (tiles 2-31, 120 VGPRs)
  v[193:196]   B-side load address bases (persistent)
  v[200:211]   Scale factors, constants for epilogue
  v[212:215]   ds_read base addresses (source, copied to working regs)
  v[218]       ds_write base address
  v[222:237]   B-side MFMA SrcB operands (rotated across 4 phases)
  v[238:247]   Dead in .L8 K-loop -> primary target for buffer_load hoisting
```

### Map F: GEMV for DeepSeek-V4 Inference

Much simpler than GEMM -- no tile blocking, no LDS transpose:

```
  accum_offset = 256 (conservative, allows space for MFMA accumulators)
  v[0:3]       MFMA accumulators (a[0:3])
  v[4:7]       Weight data (from global_load)
  v[8:11]      Activation data (from ds_read_b128 in LDS)
  v[12:15]     Dequantized weight (after FP8->FP16 conversion)
  v[16:19]     FP8 scale byte + dequant temps
  v[20]        Tile K offset (hazard: dual-use with load destination in gemv_fp4_dual)
```

---

## 9. Common Mistakes Ranked by Frequency

### Rank 1: next_free_vgpr == accum_offset (Zero AGPR Bug)

**Frequency:** Very common in AGPR-using kernels.

**What happens:** Setting `next_free_vgpr = accum_offset` allocates zero AGPRs. All MFMA writes to AGPRs (a[N]) go to unallocated registers. Reads return garbage. Two consecutive MFMAs with the same accumulator produce identical output (both read uninitialized state).

**Fix:**
```asm
.amdhsa_next_free_vgpr  (accum_offset + num_agprs)  ; NOT just accum_offset
```

**Detection:** If MFMA output is garbage but the MFMA source operands are correct (verified via store-after-load debug), suspect this bug first.

### Rank 2: Register Clobber (See Section 4)

**Frequency:** At least one instance per kernel project.

**Detection heuristic:** If cos_sim is between 0.5 and 0.99 (partially correct), or if correctness degrades with more loop iterations, register clobber is the most likely cause.

### Rank 3: accum_offset Not Multiple of 4

**Frequency:** Hit in early development. Assembly error is immediate and clear, but confusing the first time.

**Fix:** Round up: `accum_offset = ceil(num_vgprs / 4) * 4`.

### Rank 4: MFMA Column-Major Layout Confusion

**Frequency:** Every agent in the 8-agent swarm hit this independently.

For `v_mfma_f32_16x16x32_bf16`: `a[k] = C[col_group*4+k, lane_row]` where `lane_row = lane_id % 16`, `col_group = lane_id / 16`. Four AGPRs hold 4 consecutive ROWS in the same COLUMN, not 4 columns in the same row. The store path must stride by `ldc` for each AGPR value.

For `v_mfma_f32_32x32x16_bf16`: M (row) = VGPR-derived, N (col) = lane-derived. Must use `v_cvt_pk_bf16_f32` for BF16 packing (manual shift/or produces garbage).

### Rank 5: s_movk_i32 Sign Extension

**Frequency:** Occasional.

`s_movk_i32 s0, 32768` sets s0 to **-32768** (0xFFFF8000), not 32768. The 16-bit immediate is sign-extended. No assembler warning.

**Rule:** Never use `s_movk_i32` for values >= 32768. Use `s_mov_b32` with a 32-bit literal.

### Rank 6: WG_ID Overwritten by Kernarg Load

**Frequency:** Hit in early kernel scaffold development.

s4 (workgroup_id_z) is overwritten by `s_load_dwordx4 s[4:7], s[0:1], 0`. Must save it before the load completes.

### Rank 7: .args Metadata Missing

**Frequency:** Hit once per new kernel, causes error 701 on launch.

ROCm 7.2 requires complete `.args` metadata in the `.amdgpu_metadata` section. An empty `.args: []` causes kernel launch failure.

### Rank 8: Stale .co File After Reassembly

**Frequency:** Hit frequently during iterative development.

After modifying a `.s` file, the `.co` on the remote cluster may be stale. Always re-assemble from the current `.s` before testing. Compare instruction counts between original and reassembled binaries as a sanity check.

### Rank 9: Expanding VGPR Count Across Occupancy Threshold

**Frequency:** Hit in grouped GEMM optimization.

Going from 248 to 256 VGPRs crosses the 2-wave -> 2-wave boundary (no change), but going from 248 to 257 would cross to 1 wave. In the persistent FWD GEMM, expanding to 256 VGPRs for deeper software pipelining caused a **9-11% regression** due to increased register pressure without any occupancy benefit, plus the additional instructions needed to manage the wider register allocation.

### Rank 10: Mismatched flat/global Memory Instructions

Using `flat_store` in one kernel and `global_load` in a consumer kernel on the same buffer causes stale reads on gfx950. Always use consistent memory instruction types across kernels sharing data.

---

## Appendix: Register Allocation Planning Checklist

When planning a register allocation from scratch:

1. **Count MFMA accumulators.** Each 16x16 output tile needs 4 VGPRs/AGPRs. 128x128 tile = 64 accumulators. 256x256 tile = 128 accumulators. Decide: VGPRs or AGPRs?

2. **Count MFMA source operands.** ds_read_b128 loads 4 VGPRs. Each MFMA consumes 4-8 VGPRs per operand. Double-buffer if software pipelining.

3. **Count address registers.** LDS read bases (persistent across K-loop), global load address pairs, LDS write bases. These are live throughout the kernel.

4. **Count temporary registers.** Address computation scratch, dequantization intermediates, softmax intermediates. These have short lifetimes.

5. **Sum the peak.** Add accumulators + max simultaneously-live source operands + addresses + temps. If using AGPRs, set accum_offset >= (sources + addresses + temps), and next_free_vgpr = accum_offset + num_agprs.

6. **Check occupancy.** Compare next_free_vgpr against breakpoints (128/170/256). If over a threshold, look for dead register ranges to reuse (Section 5).

7. **Verify no aliasing.** Every VGPR used as a ds_read/buffer_load destination must be checked against ALL live registers at that point. Draw the lifetime map.

8. **Stress test at scale.** Correctness at K=128 (1 iteration) does not prove correctness at K=4096 (32 iterations). Register clobber bugs often manifest only with multiple loop iterations.
