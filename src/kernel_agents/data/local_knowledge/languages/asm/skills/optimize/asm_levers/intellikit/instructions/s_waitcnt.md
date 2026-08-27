---
instruction: s_waitcnt
category: sync
architecture: gfx950
tags: [s_waitcnt, vmcnt, lgkmcnt, FIFO, s_endpgm, s_barrier]
---

# s_waitcnt

Wait for outstanding memory operations to complete. Controls vmcnt (global/buffer loads), lgkmcnt (LDS/SMEM), and expcnt (exports).

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | SOPP |
| Overhead CPI | 0.168 ref-clk (median 172 ticks / 1024 iters) |
| Normalized overhead | ~1 shader cycle (when counter already at 0) |
| Category | Scalar flow control |

## Hazards

### No NOP needed between s_waitcnt and MFMA

On gfx950, `s_waitcnt lgkmcnt(0)` followed immediately by an MFMA that reads the waited-on data requires zero NOPs. The waitcnt provides the necessary pipeline delay:

```asm
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[0:3]  ; OK, no NOP
```

## Known Bugs / Gotchas

### vmcnt is FIFO -- drains oldest first

`vmcnt(N)` means "wait until at most N loads remain outstanding." Loads drain in FIFO order (oldest issued first). This is critical when reordering buffer_loads:

```
Load issue order:          FIFO position:
buffer_load A0  (oldest)   vmcnt(3) drains this
buffer_load A1             vmcnt(2) drains this
buffer_load B0             vmcnt(1) drains this
buffer_load B1  (newest)   vmcnt(0) drains this

After vmcnt(2): A0 and A1 are complete. B0 and B1 still pending.
```

When hoisting buffer_loads to an earlier position (e.g., moving B-side loads before A-side loads), the FIFO order changes and ALL downstream vmcnt values must be recalculated. Getting this wrong causes silent data corruption because you consume data from an in-flight load.

### lgkmcnt is also FIFO

Same FIFO semantics for LDS operations. `lgkmcnt(N)` drains the oldest outstanding ds_read/ds_write first:

```asm
ds_read v[A]     ; lgkmcnt position 3 (oldest)
ds_read v[B]     ; lgkmcnt position 2
ds_read v[C]     ; lgkmcnt position 1 (newest)

s_waitcnt lgkmcnt(1)   ; drains A and B, C still pending
; USE v[A]: OK
; USE v[B]: OK
; USE v[C]: BUG -- still in flight!
```

### Prefetch before data loads breaks vmcnt

If a prefetch load is issued before data loads, the prefetch occupies the oldest FIFO position. `vmcnt(N)` drains the prefetch first, not the data load you actually need:

```asm
; WRONG (prefetch is oldest):
buffer_load PREFETCH     ; vmcnt=2 (oldest, drains first)
buffer_load DATA         ; vmcnt=1 (newest)
vmcnt(1)                 ; drains PREFETCH, DATA still pending!

; RIGHT (data is oldest):
buffer_load DATA         ; vmcnt=2 (oldest, drains first)
buffer_load PREFETCH     ; vmcnt=1 (newest)
vmcnt(1)                 ; drains DATA, PREFETCH still pending
```

### MANDATORY: vmcnt(0) before s_endpgm

Every gfx950 kernel MUST have `s_waitcnt vmcnt(0)` before `s_endpgm`. Without it, outstanding buffer_store operations may not commit to memory ("store leak"). The hardware does NOT guarantee stores complete at program termination.

```asm
s_waitcnt vmcnt(0) lgkmcnt(0)   ; MANDATORY
s_endpgm
```

### MANDATORY: lgkmcnt(0) before s_barrier

Before any `s_barrier` in a double-buffered LDS loop, all pending LDS writes must be drained. The barrier synchronizes waves but does NOT guarantee LDS write visibility:

```asm
s_waitcnt lgkmcnt(0)    ; drain all LDS writes
s_barrier               ; NOW synchronize waves
```

Removing the lgkmcnt(0) causes memory access faults or stale LDS reads.

### s_barrier auto-inserts lgkmcnt(0) on gfx950

Empirically confirmed: gfx950 auto-inserts `lgkmcnt(0)` before `s_barrier`. However, explicitly writing it is recommended for clarity and portability. The auto-insertion covers LDS writes but the explicit form is the documented guarantee.

## Common Usage Patterns

### Double-buffered LDS loop
```asm
.L8:
  buffer_load_dwordx4 v[A], ...        ; issue loads
  s_waitcnt lgkmcnt(0)                 ; drain previous iteration's writes
  s_barrier                            ; sync all waves
  ds_read_b128 v[src], ...             ; read current bank
  ; ... MFMAs ...
  s_waitcnt vmcnt(0)                   ; drain buffer loads
  ds_write_b128 v[dst], v[A], ...      ; write to other bank
  s_cbranch .L8
```

### Serialized vmcnt drain for ds_write
```asm
s_waitcnt vmcnt(7)                     ; drain load 0 (oldest)
ds_write_b128 v_base, v[load0], ...
s_waitcnt vmcnt(6)                     ; drain load 1
ds_write_b128 v_base, v[load1], ...
; ... continue for all loads
```

## Sources

- vmcnt FIFO ordering discovery
- lgkmcnt FIFO ordering, s_barrier auto-waitcnt
- Empirically validated on MI355X: vmcnt/lgkmcnt FIFO detailed analysis
- lgkmcnt miscounting case studies
- vmcnt recount protocol for load hoisting
- isa-bench: s_waitcnt_overhead latency kernel
