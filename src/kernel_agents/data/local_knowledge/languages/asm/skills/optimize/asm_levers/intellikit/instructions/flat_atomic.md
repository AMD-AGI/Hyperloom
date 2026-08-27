---
instruction: flat_atomic
category: sync
architecture: gfx950
tags: [flat_atomic, vmcnt, coherence, inter-WG-barrier, sc0]
---

# flat_atomic Family

## Variants

| Opcode | Operation | Return | Counter |
|--------|-----------|--------|---------|
| `flat_atomic_add_f32` | FP32 atomic add | optional | vmcnt |
| `flat_atomic_cmpswap` | Compare-and-swap | always | vmcnt |
| `flat_atomic_add` | Integer atomic add | optional | vmcnt |

## Cycle Counts (Measured on MI355X)

| Variant | Mode | CPI | Notes |
|---------|------|-----|-------|
| `flat_atomic_add_f32` | latency | ~104.4 | 256 iters, 64-way contended |
| `flat_atomic_cmpswap` | latency | ~122.1 | 256 iters, 64-way contended |

For comparison, `global_atomic_add_f32`:
- 64-way contended: ~104.6 CPI
- Uncontended: ~26.1 CPI

Atomics are expensive -- 100-120x slower than non-atomic loads. Plan accordingly.

## Counter

**vmcnt**. Atomics count against the same vmcnt FIFO as loads and stores.

## FIFO Ordering

vmcnt FIFO. Atomics enter the FIFO in issue order alongside loads and stores.

## Coherence Flags

### flat_atomic_add_f32: NO sc0/sc1 support (CRITICAL BUG/LIMITATION)

`flat_atomic_add_f32` does **NOT** support `sc0` or `sc1` modifiers on gfx950. The assembler may accept the syntax, but the hardware ignores the flags. This means flat_atomic_add_f32 always operates with default coherence.

Empirically validated on MI355X.

### flat_atomic_cmpswap: Standard sc0/sc1 support

`flat_atomic_cmpswap` and integer atomics (`flat_atomic_add`) support `sc0` and `sc1` normally.

### Inter-Workgroup Barrier Pattern

For inter-WG barriers, use `flat_atomic_add sc0` (integer add, not f32) for the arrival counter, since flat_atomic_add_f32 cannot provide system coherence:

```asm
; Inter-WG barrier (only wave 0 participates):
flat_atomic_add v_result, v[barrier_addr], v_one sc0   ; increment counter
s_waitcnt vmcnt(0)
; Poll until all WGs have arrived:
.L_spin:
flat_load_dword v_count, v[barrier_addr] sc0 sc1
s_waitcnt vmcnt(0)
v_cmp_ge_u32 vcc, v_count, s_num_wgs
s_cbranch_vccz .L_spin
```

This pattern achieves ~50ns per WG per barrier. Empirically validated on MI355X.

## Known Hazards

### 1. High Latency
Atomics are extremely slow (100-120 CPI for 64-way contended). Avoid in hot loops.

### 2. flat_atomic_add_f32 Coherence Limitation
Cannot guarantee cross-CU visibility. Use integer flat_atomic_add with sc0 for inter-WG synchronization.

## Known Bugs

### flat_atomic_add_f32 sc0/sc1 Ignored
The hardware does not honor sc0/sc1 modifiers on this specific instruction. This is unique to the f32 variant -- integer atomics support coherence flags normally.

## Alignment Requirements

- Address: 64-bit VGPR pair (even-aligned)
- Data: single VGPR (f32) or VGPR pair (cmpswap: expected + new)
- Memory address must be naturally aligned (4-byte for dword operations)

## LDS Variant

flat_atomic can target LDS if the address falls in the LDS range. For explicit LDS atomics, use `ds_add_f32`, `ds_cmpst_b32`, etc.

## Performance Notes

- Contention is the dominant factor. Uncontended atomics (~26 CPI) are 4x faster than 64-way contended (~104 CPI).
- For reductions, prefer wave-level reduction (v_add_f32 across lanes via DPP/readlane) followed by a single atomic per wave, rather than per-lane atomics.
- The inter-WG barrier using atomics costs ~50ns per WG. Only wave 0 in each WG should participate in the barrier spin loop, with the remaining waves joining via `s_barrier` after wave 0 signals completion.

## Common Patterns

### Inter-Workgroup Barrier (Wave-0 Only)
```asm
; Only wave 0 participates (other waves wait at s_barrier)
s_cbranch_execnz .L_not_wave0
; ... wave 0 barrier logic with flat_atomic_add sc0 ...
.L_not_wave0:
s_barrier    ; all waves in WG synchronize
```

### Global Reduction
```asm
; Per-wave reduction first (in registers), then one atomic per wave:
; ... v_add_f32 reduction across lanes ...
v_readlane_b32 s_result, v_sum, 0
v_mov_b32 v_data, s_result
flat_atomic_add_f32 v[addr], v_data
s_waitcnt vmcnt(0)
```
