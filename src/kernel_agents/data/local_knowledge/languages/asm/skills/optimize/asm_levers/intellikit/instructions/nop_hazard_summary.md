---
instruction: nop_hazard_summary
category: sync
architecture: gfx950
tags: [s_nop, hazard, MFMA, VALU, ACCVGPR, transcendental, readlane, interlock, pipeline]
---

# NOP Hazard Summary for gfx950

## Overview

gfx950 has several missing hardware interlocks that require software NOP insertion. These are the most common source of non-deterministic bugs in hand-written assembly.

## Complete NOP Table

| Transition | NOPs Required | Symptom If Missing |
|-----------|---------------|-------------------|
| VALU write → MFMA read (same VGPR) | s_nop 1; s_nop 1 (2 NOPs) | MFMA reads stale data (values ~17-30 instead of expected) |
| MFMA → v_accvgpr_read_b32 | 2x s_nop 15 minimum (32 cycles) | Returns in-flight partial results, clips to FP16 max |
| v_accvgpr_read_b32 → VALU/store | s_nop 1; s_nop 1 (2 NOPs) | Reads stale ACCVGPR copy |
| v_accvgpr_write_b32 → MFMA (same AGPR) | s_nop 0 (1 NOP minimum) | MFMA reads pre-write value |
| Transcendental → read result | s_nop 3 (4 cycles) | Reads pre-computation value |
| VALU write → v_readlane_b32 (same VGPR) | s_nop 1 (2 cycles) | Returns pre-VALU value |
| s_waitcnt → MFMA | 0 NOPs | No hazard — waitcnt provides sufficient gap |
| ds_read → MFMA (via waitcnt) | 0 NOPs | Waitcnt gap is sufficient |

## Critical: MFMA Drain Is Context-Dependent

The MFMA→ACCVGPR_READ drain varies by kernel context:

| Context | Required NOPs | Validated |
|---------|---------------|-----------|
| Standalone GEMM (16x16x32) | 1x s_nop 15 (16 cycles) | Yes (15+ shapes) |
| Inference pipeline (deep layers) | 2x s_nop 15 (32 cycles) | Yes (NaN at L6-L20 without) |
| Multi-stage pipelines | 4x s_nop 15 (64 cycles) | Conservative safe default |

**When in doubt, use 4x s_nop 15.** The cost is 64 cycles — negligible compared to the days of debugging non-deterministic NaN.

## Transcendental Instructions Affected

All transcendental VALU ops need s_nop 3 before reading their result:
- `v_exp_f32`
- `v_rcp_f32`
- `v_rsq_f32`
- `v_log_f32`
- `v_sqrt_f32`

The hardware does NOT interlock — you get the pre-computation value.

## Scheduling Around NOPs

During NOP windows, schedule independent work:
```asm
v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[4:7], a[0:3]
; --- MFMA executing, co-execute during 32-cycle window ---
s_load_dwordx4 s[12:15], s[0:1], 0x40     ; prefetch next descriptor
ds_read_b128 v[20:23], v16                  ; read next tile
s_waitcnt lgkmcnt(0)                        ; drain LDS read
; --- now safe to read MFMA results ---
s_nop 15
s_nop 15
v_accvgpr_read_b32 v8, a0
```
