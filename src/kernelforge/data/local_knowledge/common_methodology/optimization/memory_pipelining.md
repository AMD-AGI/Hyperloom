---
title: memory pipelining (global_load_lds, software pipelining, prefetch)
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, training, both]
updated: 2026-08-23
---

# memory pipelining

## TL;DR
A matrix-core GEMM is fast only if the next operand tile is already on its way while the current tile
computes. CDNA's mechanism is **`global_load_lds` (buffer/global load directly into LDS)** — the
async-copy analogue that skips the round trip through VGPRs — combined with **software pipelining**
(triton `num_stages`, manual double/triple-buffer). On **CDNA4** the direct path widens to **128-bit
`GLOBAL_LOAD_LDS`** and adds an L1→LDS load path, so each instruction stages more per issue. The goal
is full overlap of `global_load` → `ds_write` → `ds_read` → `v_mfma`. **Both mechanisms are gated on the
loop's form**: a trip count or an address that comes from a tensor *load* rather than a tensor *shape*
turns both of them off silently — read `[[optimization/loop_form_and_pipelining.md]]` before tuning
`num_stages` on a sparse or gather kernel. See
`[[optimization/lds_and_bank_conflicts.md]]`, `[[optimization/mfma_scheduling.md]]`,
`[[hardware/cdna3_mi300/memory_hierarchy.md]]`, `[[hardware/cdna4_mi350/memory.md]]`.

## Concepts (the hardware)
- **`global_load_lds` / `buffer_load_*_lds`**: a load instruction whose destination is **LDS**, not a
  VGPR. It frees the register file (no staging VGPRs ⇒ better occupancy,
  `[[optimization/occupancy_and_registers.md]]`) and overlaps with compute. This is CDNA's equivalent
  of NVIDIA `cp.async`.
- **CDNA4 widening**: CDNA4 supports **128-bit `GLOBAL_LOAD_LDS`** (vs narrower on CDNA3) and a direct
  **L1→LDS** path, raising staging throughput per instruction and cutting latency for MFMA operand
  feeds.
- **`ds_read` / `ds_write`**: LDS read/write (conflict-sensitive, `[[optimization/lds_and_bank_conflicts.md]]`).
  `ds_read` feeds MFMA; `ds_write` lands the staged global data.

## Software pipelining (the structure)
**Precondition — the loop must be schedulable.** The pipeliner works on a `for`/`tl.range` loop whose
bound is loop-invariant and whose staged loads have addresses affine in the induction variable. A
runtime bound is fine (`tl.cdiv(K, BLOCK_K)` with `K` a kernel argument is the normal GEMM K-loop); a
bound read out of a tensor, or a load addressed by another load in the same iteration, is not — see
`[[optimization/loop_form_and_pipelining.md]]`. Everything below assumes the precondition holds.

A K-loop pipelined to `S` stages keeps `S` tiles in flight:
1. **Prologue**: issue `global_load_lds` for tiles `0..S-1`.
2. **Steady state** per K-step `k`: `v_mfma` on tile `k` (from LDS) **while** `global_load_lds` for tile
   `k+S` is in flight **while** `ds_read` for `k+1` overlaps.
3. **Epilogue**: drain remaining MFMAs.

Levers:
- **`num_stages` (triton)**: number of pipeline stages. `2` = classic double-buffer; raise to `3`–`4`
  for **K-deep prefill** GEMMs to hide longer global latency; lower it if LDS/registers run out
  (`[[operators/dense_gemm/tuning.md]]`). Each stage costs another LDS tile.
- **Double-buffer ping-pong** (CK/asm): two LDS buffers alternate load/compute
  (`[[optimization/lds_and_bank_conflicts.md]]` §double-buffer).
- **Prefetch distance**: issue the load far enough ahead to cover HBM latency; too far wastes LDS, too
  near exposes latency. Tune with `num_stages` / manual unroll depth.
- **ds_read ↔ ds_write overlap**: schedule the consumer `ds_read` of the current tile against the
  producer `ds_write` of the next so the LDS port stays busy without conflicts.

## CDNA3 vs CDNA4
| | CDNA3 (MI300X) | CDNA4 (MI350X) |
|---|---|---|
| direct-to-LDS load | `global_load_lds` (narrower) | **128-bit `GLOBAL_LOAD_LDS`** |
| L1→LDS path | via registers | **direct L1→LDS** |
| LDS capacity for stages | 64 KB ⇒ fewer stages | 160 KB ⇒ more stages / bigger tiles |

So on CDNA4 you can afford deeper pipelines and larger staged tiles; re-tune `num_stages` per gen.

## Pitfalls
- Loading through VGPRs (plain `global_load` → `ds_write`) when `global_load_lds` is available —
  wastes registers and serializes load/compute.
- `num_stages` too high ⇒ LDS overflow drops occupancy below the latency-hiding threshold (net slower).
- Prefetch that creates bank conflicts on `ds_write` (see swizzle/padding).
- Assuming CDNA3 has the CDNA4 128-bit GLOBAL_LOAD_LDS / L1→LDS path — re-check ISA per target.
- No barrier discipline between stages ⇒ races on the shared LDS buffer.
- **Tuning `num_stages` on a loop that cannot be pipelined at all.** A data-dependent `while`, or a
  gather whose address comes out of a load in the same iteration, forfeits both mechanisms with no
  diagnostic; the sweep then reads flat and looks like "pipelining does not help here". Fix the loop
  form first (`[[optimization/loop_form_and_pipelining.md]]`).

## Verify
- ISA dump: presence of `buffer_load_*_lds` / `global_load_*_lds`, pipeline unroll, `s_waitcnt`
  placement (`[[languages/triton_amd/isa_verify.md]]`).
- Omniperf: HBM read BW, `ds_*` stalls, MFMA busy — overlap shows as high MFMA busy with HBM near peak
  but no MFMA stall (`[[profiling/]]`, `[[optimization/roofline_and_bottlenecks.md]]`).
- A/B: sweep `num_stages ∈ {1,2,3,4}`; the latency curve dips then rises when LDS overflows. A curve
  that is *flat* across all four is not a tuning result — it says the loop is never being pipelined
  (`[[optimization/loop_form_and_pipelining.md]]`).
