---
title: FlyDSL attention on gfx950 — package map
kind: index
scope: languages/flydsl/skills/attention
updated: 2026-10-26
pinned_source: ROCm/FlyDSL@dd8373349 (v0.3.2; kernels/attention)
---

# FlyDSL attention — package map

How to make a FlyDSL attention kernel fast and correct on gfx950 (MI350/MI355X): flash forward, flash
backward (dQ, dK, dV) and decode with split KV. The cards hold measured lessons — what moved wall time, what
did not, and how to tell which is which before spending a week on it. They are mechanisms, diagnostics and
decision rules, not a catalogue of kernels or configurations.

- FlyDSL facts are stated for v0.3.2. Instruction rates and MFMA timings were measured on gfx950 with
  microbenchmarks; API behaviour was checked against the v0.3.2 source and its LLVM.
- Numbers marked *(field)* come from measured attention work (A/B on real kernels). Read them as
  magnitudes: they show how large an effect was where it was measured, not what it will be on your kernel.
- Signed percentages of wall time, throughput or a benchmark score are speed changes (+ = faster) unless the
  sentence says faster or slower or the card states otherwise; a sign next to a named quantity (cycles, sclk, a
  counter, bytes, registers, an error norm) is the change of that quantity.
- *Occupancy N* means N co-resident workgroups per CU; waves per SIMD are stated as such. An 8-wave workgroup
  alone on a CU is occupancy 1 at 2 waves per SIMD.
- Most attention bodies are latency- and rendezvous-bound, not throughput-bound. Several lessons below
  exist because a throughput explanation looked right and was wrong.

## Start here

| You are… | Read |
|---|---|
| starting on an attention kernel, or deciding what to try next | `flydsl_attn_workflow.md` |
| finding out where the time goes (counters, probes, clocks) | `flydsl_attn_attribution.md` (with `../gemm/flydsl_gemm_attribution.md`) |
| mapping the grid: XCD order, dispatch order, GQA head grouping | `flydsl_attn_grid_dispatch.md` |
| choosing the MFMA atom, laying out fragments, using S^T | `flydsl_attn_mfma_layout.md` |
| writing the softmax: exp, LSE, online rescale, numerics | `flydsl_attn_softmax.md` |
| working on LDS: transpose reads, layouts, bank conflicts | `flydsl_attn_lds.md` (with `../optimize/lds-optimization.md`) |
| fighting registers or occupancy, or overlapping MFMA with VALU | `flydsl_attn_registers_occupancy.md` |
| placing barriers, waits, scheduling hints, prefetch | `flydsl_attn_schedule_sync.md` (with `../gemm/flydsl_gemm_schedule_and_locality.md`) |
| writing the backward pass: dQ/dK/dV accumulation, determinism, fusion | `flydsl_attn_backward.md` |
| writing decode or split-KV with a combine step | `flydsl_attn_decode_split_kv.md` |
| measuring an A/B on attention | `flydsl_attn_measurement.md` (with `../gemm/flydsl_gemm_measurement.md`) |
| writing or trusting a correctness gate | `flydsl_attn_correctness.md` (with `../bottleneck/debug-flydsl-kernel.md`) |

## FlyDSL's own attention code (v0.3.2, `kernels/attention/`)

Anchors for the API and patterns the cards refer to; read the source for anything else.

| File | What it is |
|---|---|
| `flash_attn_gfx950.py`, `flash_attn_utils.py`, `flash_attn_interface.py` | bf16 flash forward for gfx950: 8 waves in two phase-staggered groups (load/compute ping-pong), 32x32x16 MFMA, QK computed as S^T, causal loop stopping at the diagonal, per-wave lazy rescale |
| `flash_attn_fp8_gfx950.py` | fp8 flash forward with the same dual-wave structure |
| `flash_attn_generic.py` | f16/bf16 flash-forward builder keeping Q, P and the O accumulators in 32x32 MFMA registers |
| `swa_gfx950.py` | GQA sliding-window forward |
| `mla_fwd_decode.py`, `mla_fwd_decode_m16x8_fp8_fp8.py` | MLA decode launcher and a 128-head fp8-Q/fp8-KV decode kernel |
| `pa_decode_fp8.py`, `pa_decode_swa.py`, `pa_decode_tile.py`, `pa_common.py`, `pa_metadata*.py` | paged-attention decode: fp8 with persistent scheduling, sliding window, and a readable tile-programming reference |

There is no attention backward kernel in v0.3.2; the backward lessons come from measured FlyDSL backward
kernels written outside the FlyDSL repository.

## Cards

| Card | One line |
|---|---|
| `flydsl_attn_workflow.md` | decision order: pin the bound, audit the grid, count redundant MFMA, then hide latency; what is usually dead |
| `flydsl_attn_attribution.md` | real clock, counter blind spots, which counter ranks which change, stall pools, subtraction and addition probes, power caps |
| `flydsl_attn_grid_dispatch.md` | XCD-major mapping, dispatch order as list scheduling, padding, GQA sharer merging |
| `flydsl_attn_mfma_layout.md` | MFMA timing on gfx950, atom choice for forward and backward, S^T, row sums through MFMA, fragment layouts, fp8 atoms |
| `flydsl_attn_softmax.md` | v_exp_f32 cost, writing exp in FlyDSL, LSE formats, online rescale, why a fixed reference max is not a general lever |
| `flydsl_attn_lds.md` | gfx950 transpose reads and their lane maps, 16-element blocks, row alignment, bank-conflict counters |
| `flydsl_attn_registers_occupancy.md` | the 512-register pool, occupancy steps, spills, MFMA–VALU co-execution and when dual-wave pays |
| `flydsl_attn_schedule_sync.md` | sched_barrier and sched_group_barrier (TRANS mask for exp), setprio, LLVM scheduler options, hazards, waits, prefetch, barriers |
| `flydsl_attn_backward.md` | determinism by construction, the D/A law for dQ partials, atomics, workspace layout, fusion forms that win and lose, masked loops |
| `flydsl_attn_decode_split_kv.md` | combine as a fixed cost, cooperative rendezvous, the CTA/CU cliff, split-K rules, gather locality |
| `flydsl_attn_measurement.md` | attention-specific A/B protocol, FLOP accounting, host-bound small shapes, coverage traps |
| `flydsl_attn_correctness.md` | SNR + determinism gates, spike inputs, shared-LSE end-to-end gates, window conventions, probes |
