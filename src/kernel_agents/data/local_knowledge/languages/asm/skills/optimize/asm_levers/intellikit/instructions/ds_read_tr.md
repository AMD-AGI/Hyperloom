---
instruction: ds_read_tr
category: memory
architecture: gfx950
tags: [ds_read_tr, lgkmcnt, LDS, transpose, MFMA-operand, VGPR-clobber]
---

# ds_read Transpose Family

## Variants

| Opcode | Element Width | Output | Counter |
|--------|---------------|--------|---------|
| `ds_read_b64_tr_b8` | 8-bit elements | 2 VGPRs (64-bit) | lgkmcnt |
| `ds_read_b64_tr_b16` | 16-bit elements | 2 VGPRs (64-bit) | lgkmcnt |

These are hardware transpose reads -- they read data from LDS and rearrange (transpose) the byte/halfword layout across lanes. Used for loading MFMA operands from LDS when the data is stored in a different layout than what the MFMA instruction expects.

## Syntax

```asm
ds_read_b64_tr_b8  v[D:D+1], vAddr offset:N
ds_read_b64_tr_b16 v[D:D+1], vAddr offset:N
```

## Cycle Counts (Measured on MI355X)

| Variant | CPI | Notes |
|---------|-----|-------|
| `ds_read_b64_tr_b16` | ~3.5 | 1024 iters, latency mode (db.json) |

No separate db.json measurement for `ds_read_b64_tr_b8`, but empirical measurements report ~4-8 cycles issue, ~20-39 cycles data ready. Expect similar CPI to tr_b16.

## Counter

**lgkmcnt**. Same behavior as regular ds_read -- one lgkmcnt increment per instruction.

## FIFO Ordering

lgkmcnt FIFO -- same rules as all ds_read variants. Oldest operation drains first.

In a typical GEMM kernel, 9 ds_read_b64_tr_b8 instructions are batched per inner loop iteration:
```
ds_read 1 (oldest):  lgkmcnt position 9
ds_read 2:           lgkmcnt position 8
...
ds_read 9 (newest):  lgkmcnt position 1

s_waitcnt lgkmcnt(7): drains reads 1-2
s_waitcnt lgkmcnt(2): drains reads 1-7
s_waitcnt lgkmcnt(0): drains all
```

## Coherence Flags

None (LDS is CU-local).

## Known Hazards

### 1. Even-Aligned VGPR Pair Required
Destination MUST be an even-numbered VGPR pair (e.g., v[146:147], v[152:153]). Odd-aligned pairs (e.g., v[205:206]) cause silent failure -- the read completes but writes garbage.

### 2. Destination Clobber
Same as all ds_read variants. The 2-VGPR destination range will be overwritten on completion. If the destination holds an MFMA A-operand that is consumed by later MFMAs in the iteration, the ds_read will silently corrupt the operand.

**Example**: ds_read into v[146:147] was issued BEFORE 4 MFMAs that read v[146:147] as SrcA. The ds_read data arrived during the MFMA sequence, corrupting the A-operand for 3 of the 4 MFMAs. Fix: defer the ds_read until AFTER all consuming MFMAs.

### 3. Address Self-Clobber
If vAddr falls within the destination range (v[D:D+1]), the address is consumed at issue time (safe for this read) but destroyed for future iterations.

## Known Bugs

None specific beyond alignment requirements.

## Alignment Requirements

- Destination: MUST be even-numbered VGPR pair
- Offset field: 0-65535 bytes
- LDS address alignment requirements depend on the transpose width (8-byte aligned for tr_b8, 16-byte aligned for tr_b16 recommended)

## LDS Variant

This IS an LDS instruction. The transpose behavior is a hardware feature of the LDS read path.

## Performance Notes

- **Primary MFMA operand loader for wgrad kernels**: The GGEMM wgrad kernel uses 9 ds_read_b64_tr_b8 per inner loop iteration to load A and B operands for 128 MFMAs.
- **Latency hidden by MFMA**: The ~20-39 cycle data latency is fully hidden during MFMA co-execution. Tightening lgkmcnt from 0 to higher values has ZERO measurable impact on GGEMM performance.
- **Register pressure for batching**: Batching multiple tr reads into separate VGPR pairs (instead of serializing through one pair) gives +0.3% improvement. Requires finding dead even-aligned VGPR pairs.

## Common Patterns

### GGEMM Wgrad A/B Operand Load
```asm
; 9 batched transpose reads for one K-iteration
ds_read_b64_tr_b8 v[152:153], v180 offset:16384   ; A-side, group 0
ds_read_b64_tr_b8 v[150:151], v180 offset:0        ; A-side, group 0
ds_read_b64_tr_b8 v[148:149], v181 offset:16384    ; A-side, group 1
ds_read_b64_tr_b8 v[154:155], v181 offset:0        ; A-side, group 1
ds_read_b64_tr_b8 v[140:141], v176 offset:16384    ; B-side, group 0
ds_read_b64_tr_b8 v[142:143], v176 offset:0        ; B-side, group 0
ds_read_b64_tr_b8 v[144:145], v177 offset:16384    ; B-side, group 1
ds_read_b64_tr_b8 v[156:157], v178 offset:16384    ; B-side, group 2
ds_read_b64_tr_b8 v[158:159], v179 offset:16384    ; B-side, group 3

s_waitcnt lgkmcnt(7)  ; drain oldest 2 (A groups ready)
v_mfma_f32_16x16x32_fp8_bf8 ...  ; use v[152:153] and v[140:141]
; progressive drain continues...
```

### Batched Reads with Progressive Drain
```asm
; Issue 4 reads into separate pairs
ds_read_b64_tr_b8 v[146:147], v176 offset:8192
ds_read_b64_tr_b8 v[138:139], v177 offset:8192   ; alternate dest
ds_read_b64_tr_b8 v[222:223], v178 offset:8192   ; alternate dest
ds_read_b64_tr_b8 v[160:161], v179 offset:8192   ; alternate dest

s_waitcnt lgkmcnt(3)   ; drain oldest (v[146:147] ready)
; 4 MFMAs using v[146:147]
s_waitcnt lgkmcnt(2)   ; drain next (v[138:139] ready)
; 4 MFMAs using v[138:139]
; ...
```

### LDS Addressing for 16x16x32 MFMA
For `v_mfma_f32_16x16x32_fp8_bf8`, the LDS address must use the FULL row stride (not tile width). The thread-to-K mapping differs from 16x16x16 MFMAs, requiring address remapping when switching between MFMA sizes.

Empirically validated on MI355X.
