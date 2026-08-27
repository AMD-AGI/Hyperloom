---
instruction: inter_wg_barrier
category: sync
architecture: gfx950
tags: [barrier, cross-workgroup, sc0, sc1, flat_atomic, coherence, sense-reversing]
---

# Inter-Workgroup Barrier (Cross-CU Synchronization)

## Overview

`s_barrier` only synchronizes waves within a single workgroup. Cross-workgroup synchronization requires a software barrier using global atomics with correct cache coherence flags.

## Sense-Reversing Barrier Pattern

```asm
; --- Atomic increment barrier counter ---
flat_atomic_add v1, v[24:25], v1 sc0      ; atomic with return value needs sc0

; --- Signal completion (sense flag store) ---
flat_store_dword v[26:27], v2 sc0 sc1      ; BOTH sc0 AND sc1 required

; --- Spin-wait on sense flag ---
.Lspin:
  flat_load_dword v3, v[28:29] sc0 sc1    ; BOTH sc0 AND sc1 required
  s_waitcnt vmcnt(0) lgkmcnt(0)
  v_cmp_ne_u32 vcc, v3, v4
  s_cbranch_vccnz .Lspin
```

## Coherence Flags (Critical)

| Instruction | Flags | Why |
|-------------|-------|-----|
| `flat_atomic_add` (with return) | `sc0` | Return value coherence |
| `flat_store_dword` (signal) | `sc0 sc1` | Cross-CU visibility |
| `flat_load_dword` (spin) | `sc0 sc1` | Bypass L1/L0 cache |

**`sc0` alone on loads is NOT sufficient.** Without `sc1`, flat_load reads from L1 cache which is NOT coherent across CUs. The spin loop never sees the updated flag and **deadlocks**.

## Measured Latency

Barrier latency scales linearly with workgroup count (~50ns per atomic add):

| Workgroups | Latency |
|-----------|---------|
| 32 | ~1.6 us |
| 64 | ~3.1 us |
| 128 | ~6.2 us |
| 256 | ~12.5 us |

## Optimization: Wave-0-Only Spin

Only wave 0 of each workgroup participates in the global barrier spin. Other waves wait on `s_barrier` after wave 0 clears:

```asm
; Only wave 0 spins on global barrier
v_cmp_eq_u32 vcc, v[wave_id], 0
s_cbranch_vccz .Lskip_global
  ; ... global barrier spin ...
.Lskip_global:
s_barrier                                   ; all waves sync locally
```

This reduces contention on the atomic counter and was measured at 6% faster in uber-kernel configurations.

## Alternatives

- `s_dcache_inv_vol` works but is unnecessary if using `sc0 sc1` on both loads and stores
- `buffer_gl0_inv` is NOT available on gfx950
