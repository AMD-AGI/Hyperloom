---
title: FlyDSL GEMM — attribution: price where the time goes before changing anything
kind: language
lever: flydsl_gemm_attribution
gens: [gfx950]
dtypes: [fp8, mxfp4, mxfp8, bf16]
regimes: [compute-bound, feed-bound, power-capped]
updated: 2026-10-19
---

# Attributing a FlyDSL GEMM's time

> Counters propose; probes decide. Every number here marked *(field)* was measured on MI355X (gfx950,
> 1400 W cap) on FlyDSL kernels: GEMMs unless the note says attention. Use them for direction and order of magnitude, not as targets.
> Trace capture and ATT tooling are in `../profile/`; this card is about turning readings into a price.

## Route here when
- You need to know which part of a kernel a change could save before building it.
- A counter says "LDS-bound" or "latency-bound" and you are about to act on it.
- A change moved wall time but not the counter you expected, or the other way round.

## 1. Two probes that classify the regime without a profiler

**Compute additivity.** Add compute with no extra memory traffic and see whether wall time moves. For
example, run the MFMA loop twice over the same operands (the result is wrong; this probe is for timing
only). Check in the ISA that the instructions really were added and not removed as dead code.
- Wall time does not move → compute is fully hidden. Drop every "fewer instructions / less VALU / fewer
  LDS issues / different occupancy" candidate at once. *(field)* +76% MFMAs cost +0.1%.
- Wall time grows in proportion → compute is on the critical path, and removing instructions is a real
  lever.
- Do it per kernel; two kernels of one operator can sit in different regimes.
- The same idea for synchronization: add two `gpu.barrier()` per tile. If that is free, barrier count is
  not the problem; look at what the barrier protects.

**Apparent byte rate.** Divide the kernel's bytes by its wall time and compare with the stream bandwidth
you measured on the same box. *(field)* MI355X: `copy_` 4.98 TB/s, `add(out=)` 6.25 TB/s, `sum` 3.74,
`amax` 2.01 (a reduction is not a bandwidth ruler). A rate above stream bandwidth means caches are serving
part of the traffic; the kernel is not at the HBM limit.

## 2. Subtractive probes: remove a part, read the time

Build each arm as a compile-time switch on the real kernel. Prove each arm compiled in (an output checksum
that changes). Keep probe switches out of the delivered code.

| Arm | What it does | What it prices |
|---|---|---|
| **drop** | do not issue the fills (or reads, stores) of the main body | the whole cost of that stream |
| **alias** | issue the same instructions, but point every address at one block that stays in L2 | instruction issue + LDS write, without the data movement |
| **identity** | replace a new cross-lane helper with an identity that keeps the same return shape | the request-path gain of a change, without its VALU cost |
| **illegal** | remove a wait or a step entirely (wrong results) | the ceiling of any change that shortens it (`flydsl_gemm_grid_pricing.md` §7) |

- Touch only the main loop body. Keep the prologue and K-tail as they are, so LDS always holds legal data
  and the kernel cannot hang.
- *(field)* On an fp8 GEMM, dropping the main body's fills saved 155 µs (18.6%); aliasing them saved
  roughly half of that. The other half was issue and LDS-write cost, which deeper prefetch cannot reach.
  Without the alias arm, prefetch would have been credited with twice what it could deliver.
- **A probe gives a bound, not an achievable gain, and the bound can be wrong in either direction.**
  *(field, attention)* Several "store is 30–47% of wall" and "half the MFMAs" ceilings turned into net
  losses when built, because the removed part was doing other work (cross-wave sharing, latency hiding). Other
  changes beat their bound because they also removed a serial dependency or a spill. Decide only on the
  real implementation.
- **Instruction counts predict exactly; time does not.** *(field)* A paper count of 1.043478× MFMAs matched
  `SQ_INSTS_MFMA` to five digits, while the time saved was 50–60% of the proportional estimate.
- **An isolated micro-benchmark can flip the sign.** *(field, attention)* A reduce kernel's best workgroup shape in
  isolation lost 0.4% in the full pass. Use micro-benchmarks to discard bad options, and pick the winner
  in the full pipeline.

## 3. Additive probes: do it twice, read the margin

When a part cannot be removed safely, run it one extra time and fold the result back exactly.
- Fold back with an exact operation (e.g. `× 1/N` with N a power of two after a cross-wave sum) so the
  correctness gate still passes.
- Write the extra copy to its own scratch so it adds no write-after-read hazard, and place it after the
  original so the original's waits are already satisfied.
- Make the probe a compile-time parameter that appears in the kernel name, and time it against the base
  in the same process. *(field)* Cross-process spread reached 12.9%, which hides a 1–4% margin.
- **The margin is an upper bound on the saving only when the kernel has no idle issue slots.**
  *(field, attention)* A rendezvous that cost +1–4% as an extra copy was *slower* to remove (+1.2–5.9%), because its reads were
  covering another read's latency.
- The same trick prices a whole launch. *(field)* `rocprofv3 --kernel-trace` start→end overstated a small
  kernel 2–3× against its marginal cost in the stream.

## 4. At the power cap, price in watts, not instructions

A dense GEMM at the power cap trades cycles for clock. Two changes with the same cycle saving can be
worth very different wall time.
- **Build an all-removed arm and sample the clock under sustained load.** *(field)* With every load, LDS
  read, barrier and scale operand removed, wall time fell 25.7%, but cycles fell only 12.6%: the clock rose
  from 1702 to 2034 MHz. The kernel was energy-bound, not issue-bound.
- **Energy-bound pools add up.** *(field)* Removing the A DMA, the B loads and the A LDS reads separately
  saved 8.09 + 5.50 + 6.16 = 19.75%; together, 20.98%. In an issue-bound kernel the pools would overlap.
- **Rank candidates by what an instruction moves, not by how many you delete.** *(field)* Deleting 7.4% of the
  hot loop's instructions, all SALU and `s_nop`, gained 0%; cutting global bytes per MAC by 20% gained
  5.4%. One `ds_read_b128` was worth an order of magnitude more than one SALU.
- **Cycles convert to wall at a shape-dependent rate.** *(field)* 5.3% fewer cycles gave 2.2% wall on one
  shape (42%), and 7.1% fewer gave 1.4% on another (20%). Discount cycle-only gains accordingly.
- **Read the clock to classify a win.** If the clock moved, the change saved energy; if wall time moved
  and the clock did not, the change overlapped a stall.
- **Sample the clock correctly.** Use one shape under sustained load for at least 45 s and average only
  the samples taken under load. The mode of a whole benchmark run is the idle clock. Alternatively use
  `GRBM_GUI_ACTIVE / 8 / dispatch time`: the counter is summed over the 8 XCDs, and only long dispatches
  give sane values.

## 5. The wave-cycle budget

`SQ_ACTIVE_INST_ANY + SQ_WAIT_INST_ANY + SQ_WAIT_ANY = SQ_WAVE_CYCLES` (all in quad-cycles): issuing,
wanting to issue but blocked by a busy pipe, and waiting on `s_waitcnt` or `s_barrier`.
- In an MFMA-bound loop a high `SQ_WAIT_INST_ANY` is healthy. Idle time is `SQ_WAIT_ANY`; read it next to
  MFMA busy. If `SQ_WAIT_ANY` rises while MFMA busy is unchanged, the change is slower. *(field)* 31.7% →
  50.0% came with −26%. The sign is reliable; the slope is not.
- **This budget is the only locator for changes that keep bytes and instructions identical**, such as a
  different wave → tile assignment. *(field)* A wave-grid change lost 26% with every memory, LDS and MFMA
  counter bit-identical.
- **`SQ_ACTIVE_INST_ANY / SQ_WAVE_CYCLES` gates "remove instructions" candidates, per kernel.** *(field)* 19%
  on a two-waves-per-SIMD kernel left 5× headroom, so removing instructions was worth nothing there; 57–60%
  on one occupancy-1 kernel made it the top lever; another occupancy-1 kernel read 36–41%. Measure it on
  your kernel. Region markers (`s_setprio`, `sched_barrier`) do not count as instructions saved.
- `SQ_WAIT_INST_LDS` near 1% of wave cycles rules out the whole "tune `ds_read` waits / LDS phase" family.
  It says nothing about LDS energy (§4).

## 6. LDS: latency, bandwidth, or neither

`SQ_WAIT_INST_LDS` counts dependent waits. Co-resident waves can hide them, so a high value is not proof that
LDS bounds the kernel. *(field, attention)* A kernel with 81% of its cycle budget in LDS waits gained 0.0% when each of
three LDS read streams was halved. Decide with a halve-the-reads probe and ISA counts. Compare
`SQ_LDS_IDX_ACTIVE` with `SQ_VALU_MFMA_BUSY_CYCLES` to see whether the LDS port is anywhere near busy.

## 7. Per-tile fixed cost

- **Same FLOPs, different tile counts.** Swap N and K (e.g. 8192×12288×3072 and 8192×3072×12288). MFMA
  counters come out bit-identical, so the whole time difference is the extra tiles' fixed cost, read from
  one dispatch each. *(field)* That gave about 16k cycles per tile, an epilogue-dominated serial VALU stream.
- **Trip-count additive probe.** Add 2 and 4 iterations to the loop count in the same process, with the
  rest of the ISA unchanged. Three points give the per-phase cost P and the fixed cost F directly.
- **A K-sweep fit (`flydsl_gemm_grid_pricing.md` §6) has five systematic errors:**
  1. Using the nominal clock instead of the measured one.
  2. Counting a neighbour kernel whose time also grows with K into P. Take the target kernel's row from a
     kernel trace.
  3. Different cells running at different clocks at the cap. F is latency, so it is constant in time,
     not in cycles.
  4. Tile count or tiles per workgroup changing with K.
  5. F is a residual and an order of magnitude more fragile than P.

## 8. Upper bounds for "hide it under the MFMAs"

- **Each MFMA offers 12 cycles of shadow on gfx950.** `SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA` reads 16
  cycles per 16x16 MFMA, and issue takes 4. *(field)* The same 12 was read on three different kernels. The
  ceiling for interleaving work is `12 × MFMAs in the phase`.
- **The register allocator is the constraint.** *(field)* An epilogue interleaved into the last chunk's MFMAs
  did interleave (one MFMA per 29 instructions), but spills went 0 → 16 and wall time lost 2–4%. Dump the
  ISA and check spills before timing.
- **At two or more waves per SIMD, moving VALU work between clusters is throughput-invariant.** Other waves
  already fill the gaps, so the expected gain is about zero. *(field, attention)* Moving a row sum gave −3.1% and −0.1%;
  deleting it gave +6.4%. Only changes that reduce total pipe cycles pay.
- **Find the bottleneck pipe first.** If a "bare" VALU chain sits on the busiest pipe, covering it with
  MFMAs changes nothing. *(field, attention)* A loop with 1308 VALU cycles against 1152 MFMA cycles was VALU-bound;
  covering the chain lost 1.5%, and removing 128 VALU cycles saved 11.25%.
- `v_exp_f32` and other transcendentals issue at half rate (8 cycles) on gfx950. Check a per-class model
  against `SQ_ACTIVE_INST_VALU`.

## 9. Readings you cannot trust

- rocprof's `VGPR_Count` reports half the true value, and `Accum_VGPR_Count` reports 0. Read registers from
  the ISA (`flydsl_gemm_mainloop.md` §10).
- `MfmaUtil` accumulates per wave: a kernel with two waves per SIMD reads twice the value of a one-wave
  kernel doing the same work.
- `SQ_LEVEL_WAVES` and `SQ_ACCUM_PREV_HIRES` read 0 on rocprofv3 1.1.0 for gfx950. Wait counters cannot
  separate `vmcnt` waits from barrier waits there; use probes.
- In an ATT trace, an MFMA waiting for its operand is charged to the MFMA, not to an `s_waitcnt`. *(field)*
  1.8% VMEM wait next to 89.5% MFMA stall was hidden memory latency.
- **Static ISA size is nearly free; a spill slot is not.** *(field)* Halving a whole-loop kernel's static
  instructions gained 0.2%, and one extra spill slot cost about 0.45%. Removing code can also create a
  spill, because the allocator's input changed.
- Do not time with `FLYDSL_DUMP_IR=1` or a single repetition. *(field)* A dumped run read +15%; three
  normal runs read ±0.2%.

## Related
`flydsl_gemm_measurement.md` (rulers, null controls, A/B design) · `flydsl_gemm_grid_pricing.md` (rounds,
fixed cost, illegal arm) · `flydsl_gemm_mainloop.md` · `../profile/kernel-trace-analysis/SKILL.md` ·
`../../../../hardware/mi350_execution.md`
