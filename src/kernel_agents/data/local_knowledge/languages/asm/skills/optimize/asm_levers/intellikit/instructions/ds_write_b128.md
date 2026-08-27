---
instruction: ds_write_b128
category: memory
architecture: gfx950
tags: [ds_write_b128, lgkmcnt, LDS, s_barrier, double-buffer]
---

# ds_write_b128

## Opcode

`ds_write_b128 vAddr, v[S:S+3] offset:N`

Writes 128 bits (16 bytes, 4 dwords) from 4 consecutive VGPRs to LDS at address `vAddr + offset`.

## Cycle Counts (Measured on MI355X)

| Measurement | CPI | Notes |
|-------------|-----|-------|
| db.json (stride1) | ~4.9 | 512 iters, latency mode |

## Counter

**lgkmcnt**. ds_write_b128 increments lgkmcnt at issue and decrements when the write completes in LDS.

## FIFO Ordering

lgkmcnt FIFO -- same rules as ds_read. ds_write and ds_read operations share the SAME lgkmcnt FIFO. When mixing reads and writes, track all operations together.

## Coherence Flags

None (LDS is CU-local). Cross-wavefront visibility within a workgroup requires `s_barrier` AFTER the lgkmcnt drain.

## Known Hazards

### 1. lgkmcnt(0) Required Before s_barrier
In double-buffered loops, all pending ds_writes must be drained before `s_barrier`. Without `lgkmcnt(0)`, other wavefronts may read stale LDS data:

```asm
; CORRECT:
s_waitcnt lgkmcnt(0)    ; drain all LDS writes
s_barrier               ; now other waves see our writes

; WRONG:
s_barrier               ; other waves might read before our writes land
```

gfx950 MAY auto-insert lgkmcnt(0) before s_barrier (empirically observed), but relying on this is unsafe. Always be explicit.

### 2. Write-Read Ordering
Within a single wavefront, LDS writes followed by LDS reads to the same address are ordered by lgkmcnt. But between different wavefronts, the write is only visible after both `lgkmcnt(0)` AND `s_barrier`.

### 3. LDS Shared Region Race (Inter-Wave)
When using shared LDS regions (not per-wave private LDS), prefetching/draining data to shared LDS before all waves have finished reading from it causes corruption. Need s_barrier between the read phase and the write phase.

Empirically validated on MI355X.

## Known Bugs

None specific.

## Alignment Requirements

- Source VGPRs: even-numbered VGPR base (e.g., v[128:131])
- Address VGPR (vAddr): single VGPR, no alignment constraint
- Offset field: 0-65535 bytes
- Effective LDS address should be 16-byte aligned for optimal throughput

## LDS Variant

This IS an LDS instruction.

## Performance Notes

- **Serialized on vmcnt drain in GEMM kernels**: In the standard GEMM inner loop, 8 ds_write_b128 instructions are serialized on progressive vmcnt drain (vmcnt(7) through vmcnt(0)), each writing data from buffer_loads that have just completed. This serialization is a significant source of stall cycles.
- **Eliminated by direct-to-LDS loads**: The `buffer_load ... lds` variant bypasses the buffer_load -> ds_write pipeline entirely, providing 15-20% speedup. When using this variant, ds_write_b128 is not needed for the data staging path.
- **Tail MFMA interleaving**: Interleaving MFMAs between ds_write instructions during the vmcnt drain fills stall bubbles with useful compute (+0.5-1.7% on persistent GEMM).

## Common Patterns

### GEMM LDS Staging (Standard Pipeline)
```asm
; After buffer_loads complete, stage data to LDS
s_waitcnt vmcnt(7)                               ; drain oldest load
ds_write_b128 v218, v[32:35] offset:0            ; A-tile to LDS bank
s_waitcnt vmcnt(6)
ds_write_b128 v218, v[36:39] offset:8192         ; A-tile continued
s_waitcnt vmcnt(5)
ds_write_b128 v218, v[48:51] offset:16384        ; A-tile continued
s_waitcnt vmcnt(4)
ds_write_b128 v218, v[52:55] offset:24576        ; A-tile continued
s_waitcnt vmcnt(3)
ds_write_b128 v218, v[238:241] offset:32768      ; B-tile to LDS bank
s_waitcnt vmcnt(2)
ds_write_b128 v218, v[242:245] offset:40960      ; B-tile continued
s_waitcnt vmcnt(1)
ds_write_b128 v218, v[0:3] offset:49152          ; B-tile continued
s_waitcnt vmcnt(0)
ds_write_b128 v218, v[4:7] offset:57344          ; B-tile continued
```

### Double-Buffer LDS Write
```asm
; Write to ping-pong bank (v218 alternates between bank 0 and bank 1)
ds_write_b128 v218, v[128:131] offset:0           ; bank A (offset 0x0000)
; Next iteration:
ds_write_b128 v218, v[128:131] offset:0           ; bank B (v218 now points to 0x8000)
```
