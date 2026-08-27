---
guide: memory-coherence-formats
category: memory
architecture: gfx950
tags: [memory-ordering, waitcnt, vmcnt, lgkmcnt, coherence, FP8, BF16, toolchain, sc0, sc1]
---

# Memory Ordering, Waitcnt Rules, and Numeric Formats on gfx950

Guide for gfx950 (CDNA4 / MI355X) assembly programming. Covers memory ordering and
coherence, waitcnt counter management, FP8 format handling, BF16 format handling, and the
assembly toolchain. Every rule was discovered empirically on real MI355X silicon.

---

## Table of Contents

1. [Memory Ordering and Coherence](#1-memory-ordering-and-coherence)
2. [Waitcnt Rules Quick Reference](#2-waitcnt-rules-quick-reference)
3. [FP8 Format Handling](#3-fp8-format-handling)
4. [BF16 Format Handling](#4-bf16-format-handling)
5. [Toolchain and Build](#5-toolchain-and-build)

---

## 1. Memory Ordering and Coherence

### 1.1 vmcnt FIFO Ordering

`s_waitcnt vmcnt(N)` drains the **oldest** outstanding VMEM loads first, in strict FIFO
order. The issue order of global_load / buffer_load instructions determines which load
completes at which vmcnt threshold. N means "wait until at most N loads remain outstanding."

```asm
global_load_dwordx4 v[48:51], v[0:1], off   ; load 0 (oldest)
global_load_dwordx4 v[52:55], v[2:3], off   ; load 1
global_load_dwordx4 v[56:59], v[4:5], off   ; load 2
global_load_dwordx4 v[60:63], v[6:7], off   ; load 3 (newest)

s_waitcnt vmcnt(3)   ; load 0 done, 3 remain
; safe to use v[48:51]

s_waitcnt vmcnt(2)   ; loads 0-1 done, 2 remain
; safe to use v[52:55]

s_waitcnt vmcnt(1)   ; loads 0-2 done, 1 remains
; safe to use v[56:59]

s_waitcnt vmcnt(0)   ; all done
; safe to use v[60:63]
```

**The prefetch ordering trap.** If you issue prefetch loads BEFORE data loads, the data
loads become the newest (last to drain). A vmcnt(N) intended for the data loads would
instead drain the prefetch first, potentially stalling longer than expected or consuming
the wrong data entirely.

```
; WRONG: prefetch first, data second
global_load_dwordx4 v[80:83], v[40:41], off   ; prefetch (oldest)
global_load_dwordx4 v[48:51], v[0:1], off     ; data A (newer)
global_load_dwordx4 v[52:55], v[2:3], off     ; data B (newest)

s_waitcnt vmcnt(2)   ; drains PREFETCH, not data A
; v[48:51] may not be ready
```

```
; CORRECT: data first, prefetch last
global_load_dwordx4 v[48:51], v[0:1], off     ; data A (oldest)
global_load_dwordx4 v[52:55], v[2:3], off     ; data B
global_load_dwordx4 v[80:83], v[40:41], off   ; prefetch (newest)

s_waitcnt vmcnt(2)   ; drains data A (oldest)
; v[48:51] is ready
```

**Rule:** Always issue loads in the order you need their results. Data loads that require
immediate consumption must be issued BEFORE prefetch loads.


### 1.2 lgkmcnt FIFO Ordering

`s_waitcnt lgkmcnt(N)` follows the same FIFO principle as vmcnt. It drains the oldest
pending LDS (ds_read, ds_write) and SMEM (s_load) operations first, leaving N newest
operations outstanding.

```asm
ds_read_b128 v[40:43], v14                   ; read 0 (oldest)
ds_read_b128 v[44:47], v15                   ; read 1 (newest)

s_waitcnt lgkmcnt(1)   ; read 0 done, 1 remains
; safe to use v[40:43] as MFMA source

v_mfma_f32_16x16x32_bf16 a[0:3], v[40:43], v[28:31], a[0:3]

s_waitcnt lgkmcnt(0)   ; both done
; safe to use v[44:47]
```

**The lgkmcnt(1) pipelining pattern.** Issue the NEXT ds_read before draining the CURRENT
one. Use lgkmcnt(1) to wait for only the older read while the newer overlaps with MFMA
execution. This is the standard technique for hiding LDS latency in MFMA-heavy loops.

```asm
; Pipelined LDS reads with lgkmcnt(1)
ds_read_b128 v[40:43], v14            ; B col 0 (oldest)
ds_read_b128 v[44:47], v15            ; B col 1 (next, prefetched)

s_waitcnt lgkmcnt(1)                  ; B col 0 ready (1 pending)
v_mfma ... v[40:43] ...               ; consume B col 0

; issue next read while MFMA runs
ds_read_b128 v[40:43], v16            ; B col 2 (prefetch)

s_waitcnt lgkmcnt(1)                  ; B col 1 ready
v_mfma ... v[44:47] ...               ; consume B col 1
```

**Caveat:** On MFMA-heavy loops where the MFMA chain is much longer than ds_read latency,
tightening lgkmcnt from (0) to (1) has near-zero impact because the MFMA pipeline already
fully hides the LDS latency. The benefit is measurable (+0.3%) only when LDS latency is
a meaningful fraction of the MFMA compute window.


### 1.3 Counter Independence

vmcnt and lgkmcnt are completely independent counters tracking separate hardware queues:

| Counter | Tracks | Operations |
|---------|--------|------------|
| vmcnt | VMEM queue | global_load, global_store, buffer_load, buffer_store, global_atomic |
| lgkmcnt | LDS + SMEM queue | ds_read, ds_write, s_load |

`s_waitcnt lgkmcnt(0)` does NOT affect vmcnt. Waiting on one counter has zero effect on
the other. They can be combined in a single instruction:

```asm
s_waitcnt vmcnt(0) lgkmcnt(0)    ; drain both queues
```

Or waited independently:

```asm
s_waitcnt lgkmcnt(0)             ; only drain LDS/SMEM
; vmcnt unchanged, global loads still in flight
```

**Note on buffer_load ... lds:** When using `buffer_load_dwordx4 ... offen lds` (direct-to-
LDS DMA), the operation counts in **vmcnt**, not lgkmcnt, because it goes through the VMEM
path even though the destination is LDS.


### 1.4 flat vs global Coherence Mismatch

`flat_load/store` and `global_load/store` use different cache paths on gfx950. Mixing them
across kernels that share the same buffers causes stale reads due to cache coherence
domain mismatch.

**Evidence:** In a DeepSeek-V4-Flash inference pipeline,
`hc_pre_rmsnorm.s` and `rmsnorm.s` used `flat_` instructions while the downstream
`gemv_fp8.s` used `global_` instructions. Converting just the two `flat_` kernels to
`global_` pushed NaN onset from layer 5 to layer 8.

**The selective fix rule.** Converting ALL kernels indiscriminately to `global_` made things
WORSE (NaN at layer 1 instead of layer 5). The correct approach is to identify specific
producer-consumer kernel pairs that share buffers and ensure they use consistent memory
instruction types.

```
; WRONG: blanket conversion of all kernels
; This can break kernels that depend on flat address space semantics

; CORRECT: selective conversion of producer-consumer pairs
; Producer kernel: uses global_store for buffer X
; Consumer kernel: uses global_load for buffer X (matching)
```

**Rule:** Use consistent memory instruction types (`global_` or `flat_`) across all kernels
that share a given buffer. Prefer `global_load/store` for global memory. Reserve `flat_` for
cases where the kernel genuinely needs the flat address space (e.g., code that operates on
both LDS and global memory through the same pointer).


### 1.5 Inter-Workgroup Barrier Coherence

Cross-CU synchronization via flat atomics on gfx950 requires specific coherence flags.
The memory ordering model on gfx950 uses `sc0`/`sc1` scope flags (NOT `glc`/`slc` from
older generations).

| Operation | Required Flags | Purpose |
|-----------|---------------|---------|
| `flat_atomic_add v_dst, v[addr], v_val` | `sc0` | Atomic with return, L1 scope |
| `flat_store_dword v[addr], v_data` | `sc0 sc1` | Cross-CU visible store (both flags) |
| `flat_load_dword v_dst, v[addr]` | `sc0 sc1` | Spin-load bypassing L1/L0 cache |

**Critical:** `sc0` alone on loads is NOT sufficient. It will deadlock because the load
returns a stale cached value and the spin-loop never sees the update.

**`buffer_gl0_inv` is NOT available on gfx950.** Do not attempt to use it for cache
invalidation.

**Pattern: wave-0-only barrier spin.**

Having all threads (256 WGs x 4 waves x 64 lanes) spin on system-coherent loads wastes
memory controller bandwidth. The validated pattern is: only wave 0 spins on the atomic
counter, then `s_barrier` synchronizes all waves within the workgroup.

```asm
; --- Inter-WG barrier (wave 0 only) ---
; Save exec, mask to wave_id == 0
v_cmp_eq_u32 s[48:49], v_wave_id, 0
s_and_saveexec_b64 s[50:51], s[48:49]

; Wave 0: atomic increment
flat_atomic_add v_cnt, v[barrier_addr:barrier_addr+1], v_one sc0
s_waitcnt vmcnt(0)

; Wave 0: spin until all WGs have incremented
.Lbarrier_spin:
    s_sleep 2
    flat_load_dword v_cnt, v[barrier_addr:barrier_addr+1] sc0 sc1
    s_waitcnt vmcnt(0)
    v_cmp_ge_u32 vcc, v_cnt, s_target
    s_cbranch_vccz .Lbarrier_spin

; Restore exec
s_mov_b64 exec, s[50:51]

; Sync all waves in this WG
s_barrier
```

**Barrier latency:** approximately 50 ns per workgroup (32 WGs = ~1.6 us, 256 WGs =
~12.5 us). The wave-0-only pattern cuts barrier overhead from ~56 us to ~6 us.

**Monotonic barrier base.** Instead of resetting GPU VRAM barrier counters between dispatch
steps (which requires hipMemset + sync), increment a `barrier_base` value each step:
`step_base = step * num_layers * num_WGs`. Counters accumulate monotonically, eliminating
any host-side reset.


### 1.6 s_waitcnt vmcnt(0) Before s_endpgm

**MANDATORY.** Every gfx950 kernel must issue `s_waitcnt vmcnt(0)` before `s_endpgm`.
Without it, in-flight global stores are silently dropped. The wave terminates before the
stores commit to memory.

```asm
; CORRECT kernel exit
s_waitcnt vmcnt(0)
s_endpgm

; WRONG — stores may leak
s_endpgm
```

**Evidence:** Adding `s_waitcnt vmcnt(0)` before `s_endpgm` in all
26 kernels of a DeepSeek-V4-Flash inference pipeline pushed NaN onset from layer 12 to
layer 23 (11 more layers clean). The leaked stores from producer kernels left stale data
for downstream consumers.

**Check early-exit branches.** If the kernel has branch-to-exit paths (e.g., `.Lexit:`),
ALL paths must flow through the waitcnt:

```asm
.Lexit:
    s_waitcnt vmcnt(0)
    s_endpgm
```

This rule was independently discovered and confirmed across multiple validation campaigns.


### 1.7 buffer_inv Considered Harmful

`buffer_inv` (vL1 TCP cache invalidate) at the start of consumer kernels is actively
harmful on gfx950 for single-stream inference pipelines. It invalidates L1 cache entries
that the previous kernel's stores just populated. If the L1-to-L2 writeback has not
completed, the consumer reads stale data from L2.

**Evidence:**
- With buffer_inv in all 13 kernels: NaN at L2-L25, every run.
- Without buffer_inv: 5/5 runs produce 0 NaN across all 43 layers.

The `s_waitcnt vmcnt(0)` before `s_endpgm` in the producer kernel is sufficient to
guarantee store visibility for single-stream (same HIP stream) pipelines. The buffer_inv
breaks this guarantee by discarding valid L1 data.

**Rule:** Do NOT use `buffer_inv` at kernel start on gfx950 for single-stream pipelines.

**When buffer_inv might be needed:** Multi-stream dispatch where the producer is on a
different HIP stream. But this pattern is rare in practice, and even then `buffer_inv sc1`
was observed to WORSEN NaN propagation.


### 1.8 global_load Address Captured at Issue Time

`global_load_dwordx4 v[24:27], v[24:25], off` -- where the destination v[24:27] overlaps
with the address v[24:25] -- is ISA-legal and works correctly.

The `global_load` instruction reads the address register pair at **issue time** before the
destination registers are written (which happens asynchronously when data arrives from
memory). The address is consumed at issue; the destination is written later.

**Rule:** Address/destination overlap in `global_load` is safe. However, avoid this pattern
when possible for clarity. The address pair is consumed immediately at issue; the destination
is written only when the load completes.

**Caveat:** This does NOT mean you can freely use the address registers before the load
completes. The address is consumed at issue time, but any OTHER instruction that writes to
those registers between issue and vmcnt drain would overwrite the loaded data.


### 1.9 global_load Destination WAW Hazard

If a `global_load` destination register range overlaps with a subsequent `ds_read` address
VGPR, the global_load data can arrive asynchronously and overwrite the address before the
ds_read uses it. This produces non-deterministic NaN at specific lanes.

```asm
; HAZARD: v38 is both global_load dest and ds_read address
global_load_dwordx4 v[36:39], v[0:1], off    ; v38 will be clobbered
ds_read_u16 v24, v38                          ; v38 may already be overwritten
```

**Fix:** Ensure no global_load destination register overlaps with any subsequent LDS address
operand before the vmcnt drain. Move global loads after LDS reads, or use non-overlapping
registers.


### 1.10 glc/slc vs sc0/sc1

gfx950 uses `sc0 sc1` memory ordering flags, NOT `glc slc` from MI300X and older
architectures. The old syntax either fails to assemble or produces incorrect memory ordering
behavior.

| Old Flag (pre-gfx950) | gfx950 Equivalent | Scope |
|------------------------|-------------------|-------|
| `glc` | `sc0` | L1 cache scope |
| `slc` | `sc1` | Agent/system scope |

---

## 2. Waitcnt Rules Quick Reference

### 2.1 Complete Counter Table

| Counter | Queue | Operations That Increment |
|---------|-------|--------------------------|
| **vmcnt** | VMEM | `global_load_*`, `global_store_*`, `buffer_load_*`, `buffer_store_*`, `global_atomic_*`, `buffer_load_* ... lds` |
| **lgkmcnt** | LDS + SMEM | `ds_read_*`, `ds_write_*`, `s_load_*` |

**Notes:**
- `buffer_load_dwordx4 ... offen lds` (direct-to-LDS DMA) increments **vmcnt**, not
  lgkmcnt, because it goes through the VMEM path.
- vmcnt and lgkmcnt are completely independent. Waiting on one does not affect the other.
- Both drain in FIFO order (oldest first).
- `s_waitcnt` can combine both: `s_waitcnt vmcnt(0) lgkmcnt(0)`.


### 2.2 Safe Waitcnt Values for Common Patterns

| Pattern | Waitcnt | Rationale |
|---------|---------|-----------|
| Before consuming global_load data | `vmcnt(N)` where N = loads issued after this one | FIFO drain |
| Before consuming ds_read data | `lgkmcnt(N)` where N = LDS/SMEM ops issued after this one | FIFO drain |
| Before ds_write following ds_read from same LDS region | `lgkmcnt(0)` | Ensure read completes before overwrite |
| Before s_barrier (double-buffered LDS) | `lgkmcnt(0)` | All pending ds_writes must be visible |
| Before s_endpgm | `vmcnt(0)` | Mandatory: stores leak without it |
| Before MFMA consuming ds_read result | `lgkmcnt(N)` followed by 0 NOPs | No extra NOPs needed after waitcnt (validated) |
| Before global_atomic | `vmcnt(0)` (if prior stores must be visible) | Depends on use case |
| Before reading SMEM (s_load result) | `lgkmcnt(N)` | s_load shares lgkmcnt with ds_read/write |


### 2.3 lgkmcnt(0) Required Before s_barrier

Before any `s_barrier` that synchronizes LDS double-buffering across waves, ALL pending
`ds_write` operations from the current wavefront must be drained. The barrier synchronizes
waves (ensures all waves reach the barrier) but does NOT guarantee that LDS writes from
wave 0 are visible to wave 1.

```asm
; CORRECT: drain before barrier
ds_write_b128 v_addr, v[48:51]        ; write next-iter data
s_waitcnt lgkmcnt(0)                  ; ensure write completes
s_barrier                             ; now other waves can see the data

; WRONG: missing drain
ds_write_b128 v_addr, v[48:51]
s_barrier                             ; other waves may read stale data
```

**Double-barrier pattern for multi-wave kernels.** When multiple waves share LDS regions
with software pipelining, you need TWO barriers per loop iteration: one before writes
(to protect readers of the current data) and one after writes (to protect writers from
overwriting data still being read):

```asm
; --- End of compute phase (all waves reading LDS) ---
s_barrier                             ; barrier 1: all waves done reading

; --- Write phase (refill LDS for next iteration) ---
ds_write_b128 v_addr, v[data:data+3]
s_waitcnt lgkmcnt(0)
s_barrier                             ; barrier 2: all waves see new data

; --- Next compute phase ---
ds_read_b128 v[src:src+3], v_addr     ; reads new data safely
```


### 2.4 vmcnt Before MFMA Source VGPRs

When global_load destinations are used as MFMA source operands, drain with the appropriate
vmcnt before the MFMA:

```asm
global_load_dwordx4 v[48:51], v[0:1], off     ; data A
global_load_dwordx4 v[52:55], v[2:3], off     ; data B

s_waitcnt vmcnt(1)                              ; A ready (B still in flight)
; ds_write A to LDS, barrier, ds_read back ...

s_waitcnt lgkmcnt(0)                            ; LDS reads done
v_mfma_f32_16x16x32_bf16 a[0:3], v[40:43], v[44:47], a[0:3]
; NO NOPS needed between lgkmcnt and MFMA
```

**0 NOPs between s_waitcnt and v_mfma.** This was validated across 15+ GEMM shapes. The waitcnt instruction itself provides sufficient pipeline gap for ds_read
result VGPRs to be readable by the MFMA. Removing the 2 NOPs (s_nop 1 x 2) that the
compiler conservatively inserts saves 2 cycles per MFMA, yielding ~5% speedup on deep-K
shapes.

**Note:** This rule applies only to the lgkmcnt->MFMA path. The VALU->MFMA path still
requires 2 NOPs (see NOP cheat sheet).


### 2.5 Interaction with NOP Hazards

The waitcnt counter is orthogonal to NOP-based hazards. Waitcnt resolves data availability
(ensuring load/store operations have completed). NOPs resolve pipeline timing hazards
(ensuring register writes have committed to the register file). Both must be satisfied
independently.

| Hazard | Resolution | Note |
|--------|-----------|------|
| Data not arrived from memory | `s_waitcnt vmcnt(N)` or `lgkmcnt(N)` | Counter-based |
| VALU write not committed before MFMA read | `s_nop 1` x 2 (8 cycles) | Timing-based |
| MFMA result not committed before ACCVGPR_READ | `s_nop 15` x 4 (64 cycles) | Timing-based |
| Transcendental result not committed before consumer | `s_nop 3` (16 cycles) | Timing-based |
| s_waitcnt followed by MFMA | 0 NOPs | waitcnt provides sufficient gap |


### 2.6 Progressive lgkmcnt Draining (Triton Pattern)

The Triton compiler uses progressive lgkmcnt draining to overlap LDS read latency with
compute. Instead of a single lgkmcnt(0), it issues multiple waitcnts at different depths:

```asm
; Triton's progressive drain pattern
ds_read_b64_tr_b8 v[138:139], v93 offset:4096   ; read 0
ds_read_b64_tr_b8 v[140:141], v93 offset:4608   ; read 1
; ... more reads ...
ds_read_b64_tr_b8 v[160:161], v93 offset:12288  ; read N

s_waitcnt lgkmcnt(6)   ; drain oldest reads, keep 6 pending
v_mfma ...              ; consume drained reads

s_waitcnt lgkmcnt(4)   ; drain more, keep 4 pending
v_mfma ...

s_waitcnt lgkmcnt(1)   ; almost all done, keep 1 pending
v_mfma ...

s_waitcnt lgkmcnt(0)   ; final drain
v_mfma ...              ; consume last read
```

This enables maximum overlap between LDS read latency and MFMA compute. The pattern is
effective when the number of ds_reads per iteration is large (16-48 reads as in grouped
GEMM kernels).

---

## 3. FP8 Format Handling

### 3.1 OCP vs FNUZ: Format Definitions

gfx950 supports two FP8 encoding families with different exponent biases:

| Property | OCP (E4M3FN) | FNUZ (E4M3FNUZ) |
|----------|-------------|-----------------|
| Exponent bias | 7 | 8 |
| Zero encoding | 0x00 | 0x00 and 0x80 |
| NaN encoding | 0x7F | None (no NaN) |
| Infinity | None | None |
| Max representable value | 448.0 | 240.0 |
| torch dtype | `torch.float8_e4m3fn` | `torch.float8_e4m3fnuz` |

**The distinction matters for MFMA instructions.** The native gfx950 FP8 MFMA instructions
(`v_mfma_f32_16x16x128_f8f6f4` and `v_mfma_f32_32x32x64_f8f6f4`) interpret source data
as **OCP** format (bias=7). If the actual data is FNUZ (bias=8), a scale correction is
required.



### 3.2 The FNUZ-to-OCP Correction (0.25x)

When using `tl.dot_scaled` (which generates native `v_mfma_f32_32x32x64_f8f6f4`) with
FNUZ-encoded data, the MFMA interprets each value with OCP's bias=7 instead of FNUZ's
bias=8. This makes each value 2x larger than intended (one exponent bit difference).
With BOTH operands A and B affected, the product is 4x too large.

**Correction:** Multiply the output scale by 0.25:

```python
# In host code (Python/C++)
combined_scale = scale_A * scale_B * 0.25   # 0.5 per operand, squared
```

```asm
; In assembly, apply after MFMA accumulation:
v_mul_f32 v_result, 0x3e800000, v_result    ; multiply by 0.25
; Or fold into the existing scale multiplication:
; scale_factor = scale_A * scale_B * 0.25
```

**Evidence:** Without the 0.25x correction, the dot_scaled kernel
produced output 4x too large. Applying `combined_scale *= 0.25` restored cos=1.000 against
the reference output.

**Rule:** When using native gfx950 FP8 MFMA instructions with FNUZ data, apply a 0.25x
scale correction (0.5x per operand). When data is OCP format, no correction is needed.


### 3.3 CBSZ/BLGP Encoding for Format Selection

The `v_mfma_f32_16x16x128_f8f6f4` and `v_mfma_f32_32x32x64_f8f6f4` instructions use
CBSZ and BLGP modifier fields to select the data format of each operand:

| Modifier | Bits | Controls | Values |
|----------|------|----------|--------|
| CBSZ[2:0] | instruction encoding | Matrix A format | 0=E4M3, 1=E5M2, 4=E2M1 (FP4) |
| BLGP[2:0] | instruction encoding | Matrix B format | 0=E4M3, 1=E5M2 |

**Common combinations:**

```asm
; FP8 x FP8 (E4M3 x E4M3):
v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3] cbsz:0 blgp:0

; FP8 x BF8 (E4M3 x E5M2):
v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3] cbsz:0 blgp:1

; FP4 x FP4 (E2M1 x E2M1):
v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3] cbsz:4 blgp:4
```

**For legacy FP8 MFMA** (`v_mfma_f32_16x16x32_fp8_bf8`), `blgp:1` selects FP8/BF8 format
(A=FP8 E4M3, B=BF8 E5M2). This instruction is MI300-compatible but 4x lower throughput
than the native f8f6f4 variants.


### 3.4 FP8 MFMA Instruction Variants

| Opcode | Tile | K/inst | Cycles | FLOPs/inst | Src VGPRs | Generation |
|--------|------|--------|--------|------------|-----------|------------|
| `v_mfma_f32_16x16x32_fp8_bf8` | 16x16 | 32 | 16 | 16,384 | 4+4 | MI300 legacy |
| `v_mfma_f32_16x16x128_f8f6f4` | 16x16 | 128 | 32 | 65,536 | 8+8 | MI355X native |
| `v_mfma_f32_32x32x64_f8f6f4` | 32x32 | 64 | 32 | 131,072 | 8+8 | MI355X native |
| `v_mfma_scale_f32_16x16x128_f8f6f4` | 16x16 | 128 | ~32 | 65,536 | 8+8 | MI355X (built-in scaling) |

**The MFMA upgrade is the single biggest optimization for FP8 workloads.** Switching from
legacy `v_mfma_f32_16x16x32_fp8_bf8` to native `v_mfma_f32_32x32x64_f8f6f4` (via
`tl.dot_scaled` in Triton) gives **1.74x-1.76x** speedup on wgrad shapes. This dwarfs
all ASM scheduling optimizations combined (+13.2% best case).


### 3.5 E8M0 Scaling (FP8 Dequantization)

The E8M0 (8-bit exponent, 0-bit mantissa) format is used as a per-tensor or per-block
scale factor for FP8 data. An E8M0 byte represents a power of 2 via the exponent encoding
of IEEE FP32.

**Converting E8M0 byte to FP32 scale:**

```asm
; E8M0 byte -> FP32: shift byte into exponent position
v_lshlrev_b32 v_scale, 23, v_byte    ; (byte << 23) = float32 with that exponent
```

The result is `2^(byte - 127)` as an IEEE FP32 value (sign=0, exponent=byte, mantissa=0).



### 3.6 v_cvt_scalef32_pk_f16_fp8: FP8 Dequantization

The native gfx950 instruction for FP8-to-FP16 conversion with simultaneous scaling:

```asm
; Dequantize FP8 data with E8M0 scale
v_lshlrev_b32 v20, 23, v20                           ; E8M0 -> float32 scale
v_cvt_scalef32_pk_f16_fp8 v12, v10, v20               ; low 4 bytes -> 2 FP16 values
v_cvt_scalef32_pk_f16_fp8 v13, v10, v20 op_sel:[1,0,0]  ; high 4 bytes -> 2 FP16 values
v_cvt_scalef32_pk_f16_fp8 v14, v11, v20               ; next dword
v_cvt_scalef32_pk_f16_fp8 v15, v11, v20 op_sel:[1,0,0]
```

**op_sel:[1,0,0]** selects the upper 2 bytes from the source dword (bits [31:16]) instead
of the lower 2 bytes (bits [15:0]). This allows processing an entire 4-byte dword with
two instructions: one for the low half, one for the high half.

**Pipeline pattern for GEMV kernels:**

```asm
; Load FP8 weight tile and scale byte
global_load_dwordx2 v[10:11], v[6:7], off     ; 8 bytes = 8 FP8 values
global_load_ubyte v20, v[22:23], off           ; 1 E8M0 scale byte

s_waitcnt vmcnt(0)

; Dequant: 8 FP8 -> 4 pairs of FP16
v_lshlrev_b32 v20, 23, v20                    ; scale byte -> float32
v_cvt_scalef32_pk_f16_fp8 v12, v10, v20
v_cvt_scalef32_pk_f16_fp8 v13, v10, v20 op_sel:[1,0,0]
v_cvt_scalef32_pk_f16_fp8 v14, v11, v20
v_cvt_scalef32_pk_f16_fp8 v15, v11, v20 op_sel:[1,0,0]

; Feed dequantized FP16 to MFMA
v_mfma_f32_16x16x32_f16 a[0:3], v[12:15], v[16:19], a[0:3]
```


### 3.7 FLOAT_ROUND_MODE Mismatch

The backward attention kernel labels its BF16 conversion as "rtz" (round-to-zero,
`bf16_cvt=2` in the CSV config), but the actual assembly implements **rtne** (round-to-
nearest-even) in software. The `v_bfe_u32 + v_add3_u32 + v_cndmask_b32` pattern extracts
the rounding bit and adds bias -- that is rtne, not truncation.

Real rtz would be a single `v_lshrrev_b32 v, 16, v` (just truncate the low 16 bits).

The native `v_cvt_pk_bf16_f32` instruction on gfx950 also performs rtne, so replacing
software BF16 rounding with the native instruction is semantically correct despite
the misleading label.


### 3.8 FNUZ Gets Software-Emulated

On gfx950, FNUZ FP8 data (`fp8e4b8`, `torch.float8_e4m3fnuz`) passed to standard MFMA
instructions gets **software-emulated** via upcasting to FP16. The native MFMA hardware
interprets FP8 as OCP format.

To use native FP8 MFMA with FNUZ data, you must apply the 0.25x correction (Section 3.2)
and treat the data as OCP at the instruction level.

---

## 4. BF16 Format Handling

### 4.1 v_cvt_pk_bf16_f32 is Mandatory

**This is the single most frequently rediscovered gfx950 gotcha in practice.**

Manual BF16 packing via `v_lshrrev_b32 v_dst, 16, v_src` to extract the upper 16 bits of
an FP32 value as BF16 produces **inf/NaN** on gfx950. The hardware requires the native
conversion instruction.

```asm
; WRONG: produces inf/NaN on gfx950
v_lshrrev_b32 v_lo, 16, v_f32_lo
v_lshrrev_b32 v_hi, 16, v_f32_hi
v_and_or_b32 v_packed, v_hi, 0xffff0000, v_lo

; CORRECT: native conversion
v_cvt_pk_bf16_f32 v_packed, v_f32_lo, v_f32_hi
```

**Packing order:** `v_cvt_pk_bf16_f32 dst, S0, S1` puts S0 in the **low** 16 bits and
S1 in the **high** 16 bits. This is the opposite of what most people assume.

```asm
; v_cvt_pk_bf16_f32 dst, src_lo, src_hi
; dst[15:0]  = BF16(src_lo)   -- low half
; dst[31:16] = BF16(src_hi)   -- high half
```

**The rounding mode** is rtne (round-to-nearest-even), matching the software BF16 rounding
pattern.

**Instruction reduction.** Replacing software BF16 rounding with native
`v_cvt_pk_bf16_f32` eliminates 10 instructions per pair of F32->BF16 conversions:
`v_cmp_u_f32 + v_bfe_u32 + v_add3_u32 + v_cndmask_b32 + v_lshrrev_b32` x2 + `v_and_or_b32`
becomes a single instruction. This freed 3 constant VGPRs (v229=0xffff0000, v230=0x7fff0000,
v231=0x7fff) and reduced total instructions by 10% in the BWD attention kernel.
Independently discovered across multiple validation campaigns.


### 4.2 v_perm_b32 for BF16/FP32 Output Formatting

`v_perm_b32` is used for byte-level shuffling of BF16 packed data. Optimized kernels use it extensively
for V data reordering before writing to LDS:

```asm
; V data shuffle pattern:
; Constants: s16 = 0x01000504, s37 = 0x03020706
v_perm_b32 v_out0, v_in0, v_in1, s16    ; select bytes [1,0,5,4]
v_perm_b32 v_out1, v_in0, v_in1, s37    ; select bytes [3,2,7,6]
```

The selector constant encodes which byte from the two source dwords goes into each byte
of the destination. This avoids explicit shift/mask/or sequences for data reformatting.

**Scheduling note:** Optimized kernels precompute all perm results before
any MFMA consumption, creating a register pressure spike. Computing perm results just-in-
time (interleaved with MFMA consumption) reduces peak register pressure by 8 VGPRs.


### 4.3 BF16 Atomic Precision Limits

`global_atomic_pk_add_bf16` (used for atomically accumulating BF16 dQ in backward
attention) has an inherent precision limit of approximately cos_sim = 0.986 versus FP32
reference.

This is NOT a bug. It is the hardware precision limit of BF16 atomic accumulation across
multiple workgroups. Both the original kernel and reassembled kernels achieve identical
cos = 0.986542 for dQ, while dK and dV (which use FP32 accumulation) achieve cos = 0.999990.

**Implication:** When validating backward attention kernels that use BF16 atomics for dQ,
do NOT expect cos_sim > 0.99 for the dQ output. The precision floor is architectural.

For higher precision, use the a32 path (FP32 atomic accumulation) with a separate
dQ reduction kernel, at the cost of an additional kernel launch.


### 4.4 FP16 Intermediate Overflow at Deep Layers

In inference pipelines using FP16 intermediate buffers (e.g., `x_buf` between kernels),
values at layer 5+ can overflow the FP16 range (max = 65504), producing inf and then NaN
that propagates through all subsequent layers.

```
Layer 4: max(x_buf) = 42000 (within FP16 range)
Layer 5: max(x_buf) = 73000 (EXCEEDS FP16 max -> inf -> NaN)
```

**Mitigation strategies:**
1. Use BF16 intermediates (range up to ~3.39e38, matching FP32 exponent range).
2. Recompute from FP32 state instead of accumulating in FP16.
3. Apply clipping / renormalization between layers.



### 4.5 BF16 Store Path Template

The standard store path for MFMA FP32 accumulators to BF16 output:

```asm
; --- MFMA drain ---
s_nop 15
s_nop 15
s_nop 15
s_nop 15                               ; 64 NOPs for MFMA pipeline drain

; --- Read accumulators ---
v_accvgpr_read_b32 v8, a0              ; row 0, col 0
v_accvgpr_read_b32 v9, a1              ; row 1, col 0
v_accvgpr_read_b32 v10, a2             ; row 2, col 0
v_accvgpr_read_b32 v11, a3             ; row 3, col 0

; --- Wait for ACCVGPR_READ ---
s_nop 1
s_nop 1                                ; 8 cycles for read to commit

; --- Pack FP32 pairs to BF16 ---
v_cvt_pk_bf16_f32 v12, v8, v9          ; pack rows 0,1
v_cvt_pk_bf16_f32 v13, v10, v11        ; pack rows 2,3

; --- Bounds-checked store ---
v_cmp_lt_u32 vcc, v_col, s_N
s_mov_b64 s[24:25], vcc                ; save column mask

v_cmp_lt_u32 vcc, v_row, s_M
s_and_b64 exec, s[24:25], vcc          ; col AND row mask
global_store_dword v[addr:addr+1], v12, off
s_mov_b64 exec, -1                     ; restore exec

v_add_co_u32 v_row, vcc, v_row, s_ldc  ; advance to next row pair
; ... repeat for v13 ...
```

**Note on MFMA drain disagreement.** There is a contradiction in findings on the
required number of s_nop 15 instructions between the last MFMA and first v_accvgpr_read_b32:

| Source | Claim | Context | Hardware-Validated? |
|--------|-------|---------|---------------------|
| Standalone GEMM | 1x s_nop 15 (16 cycles) | Standalone GEMM kernel, 16x16x32 | Yes (15+ shapes) |
| Inference pipeline | 2x s_nop 15 (32 cycles) INSUFFICIENT | Deep layers | Yes (NaN at L6-L20) |
| BWD attention (A) | 2x s_nop 15 sufficient | BWD attention kernel | Partial |
| BWD attention (B) | 2x s_nop 15 (30 cycles) | BWD attention kernel | Referenced, not independently validated |
| Conservative rule | 4x s_nop 15 (64 cycles) | Compilation of all findings | Recommended default |

**Resolution:** The 1x-sufficient finding was on standalone GEMM kernels where
the MFMA pipeline has no interactions with other kernel stages. The 2x-insufficient
finding was in a multi-kernel inference pipeline where timing interactions between
consecutive kernel launches and deeper pipeline depth exposed the hazard. The
**conservative rule is 4x s_nop 15 (64 cycles)** unless you have empirically validated a
shorter drain for your specific kernel on the specific shapes and batch sizes you will
deploy.

---

## 5. Toolchain and Build

### 5.1 Assembly Build Pipeline

```
                  +----------+
                  |  .s file |  (hand-written or disassembled)
                  +----+-----+
                       |
         +-------------+-------------+
         |                           |
    [Full build]              [Patch workflow]
         |                           |
         v                           v
  clang -x assembler          llvm-mc -> .o
  -target amdgcn-amd-amdhsa   ld.lld -> new.co
  -mcpu=gfx950                patch_co.py --ref ref.co
  -o kernel.co                         -> patched.co
         |                           |
         v                           v
      kernel.co                 patched.co
   (standalone)            (Triton KD preserved)
```

**Full build (standalone kernel):**

```bash
/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa \
    -mcpu=gfx950 -o kernel.co kernel.s
```

This produces a shared object (ELF Type 0x3) that can be loaded with `hipModuleLoad`.

**Critical:** Using `-c` produces a relocatable object (ELF Type 0x1) that causes
`hipModuleLoad error 209 (InvalidKernelFile)`. You must produce a shared object, either
via `clang` without `-c`, or via separate assemble + link:

```bash
# Two-step: assemble then link
/opt/rocm/llvm/bin/llvm-mc --triple=amdgcn-amd-amdhsa --mcpu=gfx950 \
    --filetype=obj kernel.s -o kernel.o
/opt/rocm/llvm/bin/ld.lld -shared kernel.o -o kernel.co
```

**Container path note:** Inside `rocm/pytorch-nightly` containers, the tools may be at
`/opt/rocm-7.2.0/lib/llvm/bin/` rather than `/opt/rocm/llvm/bin/`.



### 5.2 Patch Workflow (patch_co.py)

When modifying Triton-compiled kernels, do NOT build standalone .co files. Triton's kernel
descriptor (KD) contains critical metadata (kernarg preloading offsets, accum_offset,
USER_SGPR_COUNT, float_mode) that standalone assembly cannot easily reproduce.

**The patch workflow:**

1. Disassemble reference .co:
   ```bash
   llvm-objdump -d --mcpu=gfx950 ref.co > ref_disasm.s
   ```

2. Clean up disassembly (via `disasm_to_asm.py` or manual editing -- these are utility scripts included in the toolkit):
   - Strip hex addresses and raw encoding comments
   - Convert numeric branch targets to labels
   - Fix DPP instructions (may be dropped by llvm-objdump)

3. Edit the .s file (apply optimizations).

4. Reassemble:
   ```bash
   llvm-mc --triple=amdgcn-amd-amdhsa --mcpu=gfx950 \
       --filetype=obj edited.s -o edited.o
   ld.lld -shared edited.o -o edited.co
   ```

5. Patch ONLY the .text section into the reference .co:
   ```bash
   python3 patch_co.py --ref ref.co --new edited.co -o patched.co
   ```

`patch_co.py` (a utility script included in the toolkit) splices the new .text section into the reference .co's ELF structure,
preserving the original KD, metadata, and .args section.

**Hazards of patching:**
- If the new .text is LARGER than the reference, `patch_co.py`'s Python bytearray slice
  assignment shifts the rest of the ELF, corrupting it. Verify text sizes match or are
  smaller.
- If smaller, NOP padding is added after s_endpgm (harmless).
- Same instruction count can produce different .text sizes due to encoding differences
  (some instructions are 4 bytes, others 8 bytes).

**Round-trip validation:** Always verify that an unmodified reassembly produces bit-
identical output (cos=1.000000) and identical timing before applying optimizations. This
proves the clean .s IS the Triton binary.


### 5.3 Disassembly Round-Trip Issues

Several instruction types cause problems during disassembly and reassembly:

| Issue | Symptom | Fix |
|-------|---------|-----|
| DPP `quad_perm` dropped | 184 instructions silently missing from reassembly | Count instructions before/after; fix disassembly parser |
| Branch targets as raw offsets | Labels not generated; branch targets are numeric | Use `disasm_to_asm.py` to convert offsets to labels |
| Encoding size mismatch | Same instructions produce different .text sizes | Verify .text size; use patch_co.py for size-safe patching |
| `.note` section bloat | A 10,836-byte .note pushes .text across I-cache boundary | Reassembly strips .note to ~456 bytes |
| `.amdhsa_kernel` directives | Not all llvm-mc versions support them | Use MessagePack metadata in .note section as fallback |

**Evidence:** A reassembly that dropped 608 instructions (184 DPP
shuffles + additional MFMAs) produced a "28% faster" kernel that was actually broken --
it skipped half the softmax reductions and produced zeros for dK/dV.

**Rule:** ALWAYS validate correctness before measuring performance on reassembled kernels.
Compare instruction count between original and reassembled binaries.


### 5.4 .args Metadata Required (ROCm 7.2)

ROCm 7.2 on gfx950 requires `.args` in the `.amdgpu_metadata` YAML section. Without it,
`hipModuleLaunchKernel` returns error 701 ("too many resources requested for launch") even
when VGPRs, SGPRs, and LDS are well within limits.

Every assembly kernel must include a complete `.args` list:

```yaml
.amdgpu_metadata:
  amdhsa.version: [1, 2]
  amdhsa.kernels:
    - .name: my_kernel
      .symbol: my_kernel.kd
      .kernarg_segment_size: 128
      .group_segment_fixed_size: 65536
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .sgpr_count: 52
      .vgpr_count: 124
      .max_flat_workgroup_size: 256
      .args:
        - { .offset: 0,  .size: 8, .value_kind: global_buffer }   # ptr A
        - { .offset: 8,  .size: 8, .value_kind: global_buffer }   # ptr B
        - { .offset: 16, .size: 8, .value_kind: global_buffer }   # ptr C
        - { .offset: 24, .size: 4, .value_kind: by_value }        # M
        - { .offset: 28, .size: 4, .value_kind: by_value }        # N
        - { .offset: 32, .size: 4, .value_kind: by_value }        # K
.end_amdgpu_metadata
```

**Cost of missing .args:** 2+ hours of debugging a trivial `s_endpgm` kernel that
failed to launch.

**Triton hidden args:** Triton appends 2 hidden arguments after user args:
`global_scratch` (NULL) and `profile_scratch` (NULL). Total args = user args + 2 NULL
pointers. Without these, the kernel reads garbage.


### 5.5 Kernel Descriptor Specifics (gfx950)

**accum_offset field location changed.** On gfx950, `accum_offset` is in bits [3:0] of
`compute_pgm_rsrc3` (NOT bits [19:14] as on gfx940/gfx90a).

| Field | gfx940/gfx90a | gfx950 |
|-------|---------------|--------|
| accum_offset | rsrc3 bits [19:14] | rsrc3 bits [3:0] |
| Encoding formula | `accum_offset / 4 - 1` | `accum_offset / 4 - 1` |
| Decode formula | `(field + 1) * 4` | `(field + 1) * 4` |

**Key descriptor fields:**

```
.amdhsa_next_free_vgpr:   accum_offset + num_agprs  (MUST include AGPRs)
.amdhsa_accum_offset:     must be multiple of 4
.amdhsa_next_free_sgpr:   total SGPRs used (rounded up to allocation granularity)
.amdhsa_float_round_mode: controls rounding for VALU FP operations
.amdhsa_group_segment_fixed_size: LDS allocation in bytes
.amdhsa_user_sgpr_kernarg_segment_ptr: 1 (enable kernarg pointer in s[0:1])
.amdhsa_system_sgpr_workgroup_id_x: 1
.amdhsa_system_sgpr_workgroup_id_y: 1
.amdhsa_system_sgpr_workgroup_id_z: 1
```

**System SGPR layout:**
- With `USER_SGPR_COUNT=2` (standalone assembly): `s[0:1]` = kernarg_ptr, `s2` = WG_ID_X,
  `s3` = WG_ID_Y, `s4` = WG_ID_Z.
- With kernarg preloading (Triton, `USER_SGPR_COUNT=16`): `s[0:1]` = kernarg_ptr,
  `s[2:15]` = preloaded kernargs, `s16` = WG_ID_X.

**Mismatching USER_SGPR_COUNT** between the kernel descriptor and the code causes WG_ID_X
to be read from the wrong SGPR, producing crashes or wrong results.


### 5.6 Docker Build Environment

The standard container for gfx950 assembly work:

```bash
docker run --rm --network=host --device=/dev/kfd --device=/dev/dri \
    --group-add video --ipc=host --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v /home/$USER/workspace:/workspace \
    --entrypoint /bin/bash \
    rocm/pytorch-nightly:latest -c "
        # Tools are at /opt/rocm/llvm/bin/ or /opt/rocm-7.2.0/lib/llvm/bin/
        which clang    # verify tool location
        clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 \
            -o /workspace/kernel.co /workspace/kernel.s
    "
```

**Key gotcha:** Docker images may not be shared across compute nodes. Use
`rocm/pytorch-nightly` which is commonly pre-pulled, or ensure your image is available
on the target node before use.

**Force rebuild after rsync:**

```bash
rm -f build/kernel.co && make
```

rsync preserves timestamps by default. If the remote .s file has the same mtime as the
local one, make sees no change and skips the rebuild. Multiple debugging attempts were
wasted testing stale binaries because of this.


### 5.7 Validation Harness Pattern

```python
import ctypes, torch, torch.nn.functional as F

# Load kernel
lib = ctypes.CDLL("libamdhip64.so")
mod = ctypes.c_void_p()
func = ctypes.c_void_p()
lib.hipModuleLoad(ctypes.byref(mod), b"kernel.co")
lib.hipModuleGetFunction(ctypes.byref(func), mod, b"my_kernel")

# Pack kernargs (HOST memory, NOT GPU)
args = (ctypes.c_uint8 * 128)()
# ... pack pointers and scalars at correct offsets ...

# Launch
extra = (ctypes.c_void_p * 5)(
    1,                         # HIP_LAUNCH_PARAM_BUFFER_POINTER
    ctypes.cast(args, ctypes.c_void_p),
    2,                         # HIP_LAUNCH_PARAM_BUFFER_SIZE
    ctypes.cast(ctypes.pointer(ctypes.c_size_t(128)), ctypes.c_void_p),
    0                          # HIP_LAUNCH_PARAM_END
)
lib.hipModuleLaunchKernel(func, gdx, gdy, gdz, bdx, bdy, bdz,
                          shared_mem, None, None,
                          ctypes.cast(extra, ctypes.c_void_p))

# Validate
cos_sim = F.cosine_similarity(
    output.flatten().unsqueeze(0).float(),
    reference.flatten().unsqueeze(0).float()
).item()

# Interpretation:
#   cos_sim ~0.06:  random correlation (totally wrong)
#   cos_sim ~0.93:  register clobber (partial corruption)
#   cos_sim ~0.986: BF16 atomic precision floor (dQ with a16 path)
#   cos_sim >= 0.999: correct within BF16 rounding
#   cos_sim = 1.000: bit-identical
```

**Kernarg buffer must be host memory.** Using `torch.empty(N, device='cuda')` as the
kernarg buffer causes SIGSEGV because HIP tries to dereference a GPU address from the host
side. Always use ctypes arrays or numpy arrays.

**shared_mem parameter:** For kernels with statically allocated LDS
(`group_segment_fixed_size` in the descriptor), pass `shared_mem=0`. Passing 65536 causes
double-allocation or errors. For Triton-generated kernels, pass the correct dynamic shared
memory size (65536 or 135104 for dot_scaled).

