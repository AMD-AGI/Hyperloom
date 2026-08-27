---
instruction: s_barrier
category: sync
architecture: gfx950
tags: [s_barrier, lgkmcnt, LDS, double-buffer, inter-WG-barrier]
---

# s_barrier

Workgroup barrier: synchronizes all waves within a workgroup. All waves must reach the barrier before any can proceed.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | SOPP |
| Latency (256 threads) | 0.344 ref-clk CPI (median 88 ticks / 256 iters) |
| Latency (512 threads) | 0.531 ref-clk CPI (median 68 ticks / 128 iters) |
| Latency (64 threads) | 0.172 ref-clk CPI (median 44 ticks / 256 iters) |
| Estimated cost | ~50-100 shader cycles per barrier |
| Category | Scalar flow control |

## Hazards

### Auto-inserts lgkmcnt(0) on gfx950

gfx950 auto-inserts `s_waitcnt lgkmcnt(0)` before `s_barrier`. This ensures all pending LDS writes from the current wave are visible to other waves when they pass the barrier. However, writing explicit `s_waitcnt lgkmcnt(0)` before the barrier is still recommended for clarity and portability.

## Known Bugs / Gotchas

### All barriers in double-buffered loops are load-bearing

Every `s_barrier` in a double-buffered LDS loop is structurally required. Removing any one causes inter-wave data corruption:

- **Barrier 1 (before reads):** All waves finished writing to LDS bank A from previous iteration. Safe to read.
- **Barrier 2 (before writes):** All waves finished reading from LDS bank A. Safe to overwrite with new data.
- **Barrier 3 (ping-pong variant):** All waves finished reading from bank B. Safe for next iteration to write to bank B.

Attempting to remove barrier 3 from the direct-to-LDS kernel variant caused `cos=0.988` (deterministic corruption). All 3 barriers were confirmed load-bearing.

### Barrier overhead is the dominant structural ceiling

In GGEMM persistent FWD kernel, 2-4 barriers per K-iteration contribute ~14% of total cycle time (~200 cycles out of ~1394 per iteration). This is the primary reason the kernel plateaus at 74% roofline efficiency. No amount of instruction scheduling can eliminate this overhead -- it requires algorithmic changes (triple-buffering, larger tiles, or async copy).

### When s_barrier is mandatory

1. **LDS double-buffer swap:** Between reading from one bank and writing to the other
2. **After direct-to-LDS buffer_load:** Between `buffer_load ... lds` completion and `ds_read` from the loaded data
3. **Inter-wave synchronization:** When one wave's LDS writes must be visible to other waves before they proceed

### Inter-workgroup barriers (NOT s_barrier)

For inter-workgroup synchronization (e.g., uber-kernel design), `s_barrier` is insufficient -- it only synchronizes within a workgroup. Inter-WG barriers require `flat_atomic sc0` + `flat_load/store sc0 sc1` for coherence, costing ~50ns per WG per barrier step. Only wave 0 should spin; other waves wait via `s_barrier` after wave 0 signals.

## Common Usage Patterns

### LDS double-buffer loop
```asm
.L8:
  s_waitcnt lgkmcnt(0)
  s_barrier               ; all waves done writing to LDS
  ds_read_b128 ...        ; read from current bank
  ; ... MFMAs ...
  s_waitcnt vmcnt(0)
  ds_write_b128 ...       ; write to other bank
  s_barrier               ; all waves done reading
  s_cbranch .L8
```

### Forward attention (5 barriers per iteration)
```asm
; Only 5 s_barrier total in the entire kernel
; All at LDS sync points for Q/K/V data exchange
```

## Sources

- Empirically validated on MI355X
- Barrier overhead can reach ~14% of kernel cycles in compute-bound kernels
- Removing barriers without proper synchronization causes silent data corruption
- isa-bench: s_barrier_256, s_barrier_512, s_barrier_64 latency kernels
