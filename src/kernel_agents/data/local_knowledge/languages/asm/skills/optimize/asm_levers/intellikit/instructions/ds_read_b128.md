---
instruction: ds_read_b128
category: memory
architecture: gfx950
tags: [ds_read_b128, lgkmcnt, LDS, VGPR-clobber, bank-conflict]
---

# ds_read_b128

## Opcode

`ds_read_b128 v[D:D+3], vAddr offset:N`

Reads 128 bits (16 bytes, 4 dwords) from LDS at address `vAddr + offset` into 4 consecutive VGPRs.

## Cycle Counts (Measured on MI355X)

| Measurement | CPI | Notes |
|-------------|-----|-------|
| db.json (stride1) | ~3.0 | 256 iters, latency mode |
| Earlier measurement | ~3.3 | Older methodology |
| empirical measurement | ~20-39 cycles (data ready) | Includes pipeline + LDS access |

The ~3 CPI from db.json represents issue-to-issue throughput. Actual data availability (when the register is safe to read) is ~20-39 cycles, but this latency is hidden during MFMA co-execution.

## Counter

**lgkmcnt**. ds_read_b128 increments lgkmcnt at issue and decrements when data arrives in the destination VGPRs.

## FIFO Ordering

lgkmcnt is FIFO -- oldest issued LDS operation drains first. `lgkmcnt(N)` means "wait until at most N LDS/SMEM operations remain outstanding."

```
After lgkmcnt(0):           counter = 0
ds_read_b128 v[A:A+3]:     counter = 1  (oldest)
ds_read_b128 v[B:B+3]:     counter = 2
ds_read_b128 v[C:C+3]:     counter = 3  (newest)
s_waitcnt lgkmcnt(1):      drains A and B, C still pending
USE v[A:A+3]: OK (completed)
USE v[B:B+3]: OK (completed)
USE v[C:C+3]: BUG! (still in flight)
```

**Critical rule**: If you need data from the NEWEST pending read, you MUST use lgkmcnt(0).

## Coherence Flags

None applicable. LDS is CU-local, no cross-CU coherence needed. Coherence between wavefronts within the same workgroup is provided by `s_barrier`.

## Known Hazards

### 1. Destination VGPR Clobber
ds_read_b128 writes to 4 consecutive VGPRs. If any of those VGPRs hold live data (MFMA accumulators, address registers, scale factors), the arriving data will overwrite them.

**Most dangerous**: When v[D:D+3] overlaps with live MFMA A/B operands that are consumed by later MFMAs in the same iteration. The ds_read is asynchronous -- it writes when data arrives from LDS, which can happen during MFMA co-execution.

### 2. ds_read Clobbers Its Own Address Register
`ds_read_b128 v[16:19], v16 offset:N` uses v16 as the address and v16 is also the first destination. The address is read first (safe for this read), but v16 is overwritten with LDS data. On the NEXT loop iteration, v16 no longer holds the LDS base address.

**Fix**: Either restore the address register with `v_add_u32_e32 v16, 0, v214` (MOV from backup), or change the address operand to use the backup register directly: `ds_read_b128 v[16:19], v214 offset:N`.

### 3. lgkmcnt(0) Before s_barrier
gfx950 may auto-insert lgkmcnt(0) before s_barrier (empirically confirmed), but the safe practice is to always explicitly place `s_waitcnt lgkmcnt(0)` before any `s_barrier` in a double-buffered loop to ensure all pending ds_writes are visible.

## Known Bugs

None specific to ds_read_b128 beyond the general clobber patterns.

## Alignment Requirements

- Destination: even-numbered VGPR base (e.g., v[4:7], v[16:19], not v[5:8])
- LDS address: the offset field supports values 0-65535 bytes
- The effective LDS address (vAddr + offset) should be 16-byte aligned for optimal performance

## LDS Variant

This IS an LDS instruction. No global memory variant.

## Performance Notes

- **Hidden by MFMA co-execution**: The 20-39 cycle data latency is fully hidden during 32-cycle MFMAs. In MFMA-heavy loops, lgkmcnt tightening has ZERO measurable impact.
- **Bank conflicts**: gfx950 has 64 LDS banks (NOT 32 as on MI300X). Bank conflicts add ~4 CPI per doubling of conflict degree. 16-way conflicts add ~6 CPI. Broadcasts (all lanes same address) are free.
- **LDS capacity**: MI355X has 128 KB LDS per CU (doubled from MI300X's 64 KB).
- **Batching vs serial**: Serial reads to the same VGPR pair (read, wait, consume, read, wait, consume) waste pipeline slots. Batching reads into separate destination VGPRs and using progressive lgkmcnt drain is +0.3% on GEMM kernels.

## Common Patterns

### GEMM A-Operand Load (Persistent FWD)
```asm
; Load 32 VGPRs of A-side MFMA operands from LDS
ds_read_b128 v[0:3],   v214 offset:0        ; A group 0
ds_read_b128 v[4:7],   v214 offset:2048     ; A group 0 continued
ds_read_b128 v[8:11],  v214 offset:4096     ; A group 1
ds_read_b128 v[12:15], v214 offset:6144     ; A group 1 continued
ds_read_b128 v[16:19], v214 offset:8192     ; A group 2
ds_read_b128 v[20:23], v214 offset:10240    ; A group 2 continued
ds_read_b128 v[24:27], v214 offset:12288    ; A group 3
ds_read_b128 v[28:31], v214 offset:14336    ; A group 3 continued
s_waitcnt lgkmcnt(0)
; Now all v[0:31] hold A-side data for 32 MFMAs
```

### GEMM B-Operand Load with Rotation
```asm
; Phase 1: load B-side operands
ds_read_b128 v[222:225], v212 offset:32768  ; B group 0
ds_read_b128 v[226:229], v212 offset:34816  ; B group 0 continued
s_waitcnt lgkmcnt(0)
; 8 MFMAs using v[222:229]...
; Phase 2: reload B-side
ds_read_b128 v[230:233], v213 offset:40960  ; B group 1
ds_read_b128 v[234:237], v213 offset:43008  ; B group 1 continued
s_waitcnt lgkmcnt(0)
; 8 MFMAs using v[230:237]...
```

### Attention Q/K/V Load
```asm
; Q-tile from LDS
ds_read_b128 v[0:3],   v_lds_q offset:0
ds_read_b128 v[4:7],   v_lds_q offset:512
; K-tile from LDS
ds_read_b128 v[8:11],  v_lds_k offset:0
ds_read_b128 v[12:15], v_lds_k offset:512
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_fp8_bf8 ...
```
