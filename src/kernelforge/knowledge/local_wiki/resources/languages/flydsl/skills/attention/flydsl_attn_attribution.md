---
title: FlyDSL attention — attribution: the true clock, which counter ranks which change, stall pools and probes
kind: language
lever: flydsl_attn_attribution
gens: [gfx950]
dtypes: [bf16]
regimes: [prefill, backward, decode, occupancy-1, power-capped]
updated: 2026-10-26
---

# Attributing an attention kernel's time

> Counters propose; probes decide. This card adds what attention bodies taught on top of
> `../gemm/flydsl_gemm_attribution.md` (compute-additivity and byte-rate probes, drop/alias/identity/illegal
> arms, additive margins, the wave-cycle budget, readings you cannot trust), which it does not repeat. Trace
> capture is in `../profile/`. Numbers marked *(field)* are measured values; use them for direction and order of
> magnitude, not as targets.

## Route here when
- You are about to rank candidates for an attention body and need the counter that can see the change.
- A counter reads "LDS-bound", "latency-bound" or "X% waiting" and you are about to spend a round on it.
- A probe priced a pool and the built change did not deliver it.

## 1. Measure the real clock first
- Derive sclk from `GRBM_GUI_ACTIVE / 8 XCDs / dispatch duration` (`../gemm/flydsl_gemm_attribution.md` §4).
  A `rocm-smi` sample taken while the benchmark runs can land after a cached kernel has already finished and
  report the idle clock *(field: 158 MHz)*.
- Profiling changes the clock it measures: *(field)* ~1.82 GHz derived under counter collection against ~2.0 GHz
  without. Compare arms under the same collection mode.
- A roofline needs the true clock. The bf16 MFMA roofline is `256 CU × 4 SIMD × 1024 FLOP/cycle × sclk`
  (`v_mfma_f32_16x16x32_bf16`: 16384 FLOP in 16 cycles). Utilisation computed at the nominal clock reads ~20%
  low *(field)* and turns "energy-bound" into "room left".

## 2. Which counter ranks which change

| Change | Rank it with | Blind counters |
|---|---|---|
| grid, dispatch order, XCD placement, bytes | `SQ_WAIT_ANY`, `SQ_VALU_MFMA_COEXEC_CYCLES` | L2 hit rate (can rise with zero wall effect) |
| schedule of an occupancy-1 body (anchors, fences, hoists) | `SQ_WAVE_CYCLES`, `SQ_WAIT_INST_LDS` | MFMA busy |
| the same instructions in a new order | `SQ_VALU_MFMA_COEXEC_CYCLES` with `GRBM_GUI_ACTIVE` | `SQ_VALU_MFMA_BUSY_CYCLES`, `SQ_ACTIVE_INST_VALU` |
| MFMA atom at the power cap | wall, sclk and power together | MFMA busy, MFMA utilisation |
| an `iglp_opt` strategy or a software-pipelining choice | co-execution | instruction and register counts |

- **`SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA` is a constant**, the pipe cycles of the instruction mix (16.000
  for `v_mfma_f32_16x16x32_bf16`). Nine arms whose wall time spanned 76% read 16.000 bit-identically *(field)*.
  It cannot see the atom's energy, the instruction order, or how well the MFMAs are covered.
- **Rulers for occupancy-1 body scheduling.** *(field)* `SQ_WAVE_CYCLES` repeated within 0.20–0.64% and
  `SQ_WAIT_INST_LDS` within 0.1%, against 0.62–2.0% for the benchmark score on the same arms; both exclude the
  host and the neighbours. Rank with them, then close with the benchmark.
- **Co-execution is the only counter that ranked three `iglp_opt` strategies in wall order** *(field)*: MFMA
  busy was bit-identical across them, and −351 instructions elsewhere on the same body was worth 0.0%.
- **Normalise before comparing.** `SQ_VALU_MFMA_BUSY_CYCLES` and `SQ_VALU_MFMA_COEXEC_CYCLES` are summed over
  the 4 SIMDs, `SQ_ACTIVE_INST_VALU` and `SQ_WAVE_CYCLES` are not: ratios such as coexec / MFMA busy compare
  directly, absolute shares need ÷4.
- **Do not copy a co-execution target from another kernel.** *(field)* A dkdv body read coexec / MFMA busy
  19.2% and a forward on the same machine 48.1%, but `SQ_WAVE_CYCLES / SQ_BUSY_CYCLES` (a resident-wave proxy)
  was 7.7 against 31.5: most of the forward's overlap came from sibling waves. Derive the reachable value from
  the pipe arithmetic of your body.

## 3. The stall pool
MFMA busy and VALU busy are fixed work. Split the timeline into `busy_union = MFMA% + VALU% − coexec%` and
`neither = 1 − busy_union`. For a target speed-up X with co-execution unchanged:
```
stall' = 1/X − (MFMA% + VALU% − coexec%)    # as a share of today's timeline
```
- *(field)* A dkdv body at 44.0 / 36.4 / 8.5 (neither 28.1%) needed stall cut to 14.7% for 1.155×, with no
  co-execution gain; doubling co-execution alone still required stall to fall 17%. Every route crosses the
  stall pool; price it first.
- The overlap headroom to report is `busy_union` shrinking to the larger pipe with waits unchanged (1.39× on
  that body *(field)*), not `(MFMA% + VALU%) / max` (1.83× there), which assumes zero waiting and no existing
  overlap.

## 4. Wait counters are not the bound
- `SQ_WAIT_ANY` counts wave-cycles, not idle SIMD cycles: at ≥2 waves per SIMD another wave issues while one
  waits. *(field)* A forward read `SQ_WAIT_ANY` 22%; deleting every `vmcnt` wait, every `lgkmcnt(0)` or every
  `s_barrier`, one class per arm, moved wall time ≤2.6% in total. Use
  the `WAIT_*` family to name the dependency, never as a recoverable percentage; `MfmaUtil` 66% is not "34%
  left" either.
- `SQ_VALU_MFMA_BUSY_CYCLES` at least 4–5× `SQ_LDS_IDX_ACTIVE` means the LDS port is not the limit, whatever
  `SQ_WAIT_INST_LDS` says (`../gemm/flydsl_gemm_attribution.md` §6).

## 5. Probe discipline
Subtractive arms give a ceiling; additive arms give a marginal price; the two can disagree in sign
(`../gemm/flydsl_gemm_attribution.md` §2–§3). What attention bodies add:
- **Do all three layers in one go**: instructions issued with addresses pointed out of bounds (memory only) →
  instructions and their consumers removed → the next stage removed. *(field)* A "skip the partial store" arm
  also removed the GEMM whose accumulator the store read (dead-code elimination): 6.78 → 5.70 → 4.91 → 4.18 ms.
  The middle arm alone would have charged the GEMM to bytes and overstated them by 72%.
- **Keep the consumer chain alive.** *(field)* Deleting 32 exps also deleted 4 MFMAs and 16 converts and inflated
  the reading from +8% to +21.6%. Check an opcode histogram for every subtractive arm.
- **A deletion ceiling is not a hiding gain.** *(field)* Skipping the per-tile K/V DMA showed 8.7%; the built
  double-buffered prefetch lost 4.3–4.7%, because the deleted cost had a side effect (LDS write/read contention)
  the prefetch kept. Split a cost into throughput and latency before building a hider: a deletion probe prices both
  together, and only latency can be hidden.
- **Illegal subtraction overstates.** *(field)* Replacing every barrier with a scheduling fence read +3.77%;
  opening one redundant barrier priced 0.10% each; the legal reduction from 16 to 9 barriers lost 1.2–1.7%,
  because it moved an LDS write burst into a transposed-read burst (`flydsl_attn_lds.md` §9). The illegal arm
  also bought freedom from wave drift, which no legal version gets.
- **Price a synchronisation family at N, N/2 and 0.** *(field)* Two barriers per iteration → one gained
  0.12–0.18%, → zero lost 0.25%: the remaining barrier pinned the dual-wave phase. Measuring only 0 hides both
  the sign and the optimum. Both 2 → 1 arms broke a real write-after-read dependency (SNR 51.7 → 23.0 dB): they
  are bounds, not edits; the legal form doubles the LDS buffers (2 → 4) so one rendezvous covers two KV tiles.
- **A price expires when the code around it changes.** *(field)* Deleting one barrier priced 0.97% while
  `sched_barrier(0)` fences wrapped the barriers; after they were relaxed the whole family priced within ±0.25%
  (above). Re-measure rendezvous prices on the current tree.
- **Pick a probe direction that does not spill.** *(field)* Six arms that removed or replaced a row-sum MFMA run
  all spilled; running it twice on the same accumulator was register-neutral and priced it at 4.4% of wall,
  not the 11% its pipe share suggested.

## 6. The stall census
One arm per wait source: remove exactly one source, keep the consumer graph and the MFMA and exp counts, pass an
ISA gate (opcode histogram, spill, VGPR) before benchmarking, bracket the arms with same-binary controls.
- The pools overlap; do not add them.
- *(field)* One forward ranked: exp/transcendental 7.8% > LDS reads 5.8% > K/V DMA 3.95% > per-workgroup fixed
  cost 1.4% > grid and tail ≤0.2% > rendezvous ≤0.25%; no counter had suggested that order.
- An arm whose opcode sequence hashes identical to the base is inert: drop it without benchmarking.

## 7. Cycles to wall
- **The conversion rate depends on direction.** *(field)* A controlled pair that added 4.04% cycles cost 3.69%
  wall (91%); a change that removed 1.4% cycles without removing work returned 0.12–0.24% wall (8–17%). Measure
  the rate in the improving direction on your own kernel before pricing a cycle-only candidate.
- Do not apply a cycle→wall discount to a number already measured in wall time. *(field)* That double discount
  turned an 8.7% pool into 2.5% and closed the largest axis of a kernel.

## 8. At the power cap
- **At ≥99% TBP, wall ≈ total energy / TBP.** A pure reorder does not change the energy; it trades power per cycle
  against cycles, so sclk and `GRBM_GUI_ACTIVE` move in opposite directions and wall barely moves. *(field)* Fourteen
  reorder arms, from maximum to minimum interleave, all fell 0 to 3.4% below the deployed order, which sat at the top
  of the band. One instruction-identical pair: co-execution −9.86%, cycles +4.39%, sclk +3.6%, 1.87% slower.
- **Three price classes:** deleting work pays ~100%; deleting cycles without work pays ~15%; reordering pays
  0 ± 2% *(field, on a forward whose co-execution covered 44% of SIMD cycles, 52% of MFMA busy)*; on a backward at
  17% coexec / MFMA busy pure reorders still gained
  0.3–1.3% *(field)*. Report coexec / MFMA busy with the class, and name the class before proposing a change.
- **A slower arm clocks higher.** *(field)* Running a row-sum MFMA twice added 11.1% MFMA-pipe work: 4.4% slower,
  sclk +3.9%. Count a change as an energy win only when wall and clock improve together; a clock rise alone is
  not one.
- **Transcendentals are the power-heavy instructions on these bodies.** *(field)* Removing the 64 `v_exp_f32` of a
  loop gained 8.7% wall with sclk +3.1%. Rank by trans instruction count, not by MFMA cycles; whether a byte change
  is a cycle or an energy change must be measured (`flydsl_attn_grid_dispatch.md` §9).
- **An occupancy gain is partly paid back as clock.** *(field)* 2.985 → 3.978 waves per SIMD: power unchanged,
  sclk −3.3%, 0.5% faster from 3.9% fewer cycles. Report sclk next to every occupancy candidate; below the
  cap, measure whether the cycle gain survives before assuming it does.

## 9. Per-kernel additive probes
Run the compute-additivity probe (`../gemm/flydsl_gemm_attribution.md` §1) on each kernel of an operator, not once for
the operator. *(field)* Adding MFMA work to a sparse-MLA prefill changed wall time by 0.1% (fully hidden); quadrupling
the PV MFMAs in its decode producer made it 7% slower (~6 wall cycles per 16-cycle `16x16x32` MFMA, about a third of
its pipe time). One kernel of the family is fabric-bound, the other is not.

## 10. Budget arithmetic for "hide it under the MFMAs"
- Pipe cycles, issue block and VALU shadow per MFMA atom are in `flydsl_attn_mfma_layout.md` §1; exp and VALU
  rates in `flydsl_attn_softmax.md` §1. In short: 8 shadow cycles per 16x16x32 and 24 per 32x32x16, shared by
  every wave on the SIMD; an exp uses them twice as fast as a multiply and never runs beside other VALU work.
- Check a per-class model against `SQ_ACTIVE_INST_VALU` before trusting a VALU budget: *(field)* only the
  8-cycle exp fit a backward body's counter within 1%.

## Related
`../gemm/flydsl_gemm_attribution.md` · `flydsl_attn_workflow.md` · `flydsl_attn_measurement.md` ·
`flydsl_attn_schedule_sync.md` · `../profile/kernel-trace-analysis/SKILL.md` ·
`../../../../hardware/mi350_clocks.md` · `../../../../hardware/mi350_execution.md`
