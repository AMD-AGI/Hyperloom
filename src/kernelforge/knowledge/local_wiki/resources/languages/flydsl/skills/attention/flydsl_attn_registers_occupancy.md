---
title: FlyDSL attention — register budget, occupancy steps and MFMA–VALU overlap
kind: language
lever: flydsl_attn_registers_occupancy
gens: [gfx950]
dtypes: [bf16]
regimes: [prefill, backward, causal, gqa]
updated: 2026-10-26
sources:
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_gfx950.py:73
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_gfx950.py:427
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:1036
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:1614
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/backends/rocm.py:52
---

# Registers, occupancy and MFMA–VALU overlap in a FlyDSL attention kernel

> Attention bodies carry an O accumulator, Q packs, a score tile and softmax state at once, so the
> register file sets the occupancy and the occupancy decides which overlap tricks can pay. The register
> model is in `../gemm/flydsl_gemm_mainloop.md` §7 (one 512-entry pool per SIMD shared by arch VGPRs and
> AGPRs, granule 8, arch rounded to 4 before AGPRs are added), workgroup-quantized occupancy in
> `../gemm/flydsl_gemm_grid_pricing.md` §4, and the tier table in `../../../../hardware/mi350_execution.md`.
> This card adds the attention rules.
>
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as targets.

## Route here when
- You want one more wave per SIMD, or a change moved `.vgpr_count`, spills or the scratch size.
- You plan dual-wave, ping-pong, warp specialization or "hide exp under the MFMAs".
- Freed registers or AGPR pinning did not pay.

## 1. Read the budget from the ISA
- Waves per SIMD = `min(8, floor(512 / roundup8(align4(arch) + agpr)))`, then rounded down to whole workgroups
  (steps: `../../../../hardware/mi350_execution.md`). In these cards *occupancy N* means N co-resident
  workgroups per CU, which is what decides whether sibling waves can cover each other. A wave
  addresses at most 256 arch VGPRs; AGPRs come out of the same pool.
- Read `.vgpr_count`, `.agpr_count`, `vgpr_spill_count` and `private_segment_fixed_size` from the
  `FLYDSL_DUMP_IR=1` dump. rocprof's register columns are wrong (`../gemm/flydsl_gemm_attribution.md` §9).
  *(field)* A dQ kernel reported as 172 VGPR / 0 AGPR was 256 + 81 = 337: one wave per SIMD.
- For an 8-wave workgroup only 2/4/6/8 waves per SIMD are reachable. *(field)* An 8-wave forward at 4
  waves/SIMD had to stay at ≤ 128 registers; at 3 waves/SIMD the second workgroup does not fit, so the
  third wave slot buys nothing.
- FlyDSL v0.3.2: the occupancy target is the `waves_per_eu` compile hint (`maxnreg` also exists).
  FlyDSL's gfx950 forward defaults to `waves_per_eu=2`; its D=128 causal kernel compiles to 246 VGPRs,
  0 AGPRs, no spills. Kernel attributes through `value_attrs` go under `"llvm.passthrough"`
  (`../gemm/INDEX.md` § FlyDSL version).

## 2. Force occupancy with `waves_per_eu` only when the code already nearly fits
- *(field)* Allocation 130 → 128 registers with 2 spills: +8.5%, the deployed setting. Beyond that the allocator
  meets the target by spilling. *(field)* Forward, 3 → 4 waves/SIMD: 128 registers, spills
  46 → 213, hot loop 244 → 343 instructions, 5.47× slower. A dQ kernel forced to `waves_per_eu` 2: MFMA
  utilization 18% → 9%, 2× slower; to 3: 4× slower.
- To climb a step, delete live values. The reverse trade is often cheap: *(field)* dropping a forward from
  3 to 2 waves/SIMD to clear all spills cost 3.2–3.9%.

## 3. Price occupancy in steps
- Price a step before building for it. *(field)* +8.5% on a forward whose code already nearly fit the next step
  (130 → 128 registers, 2 spills) when only `waves_per_eu` changed (§2). Padding LDS with a dead
  buffer to force one workgroup per CU, everything else unchanged: *(field)* −9.9%. *(field)* The same dead-LDS
  probe on a decode producer that halved its occupancy cost 0.1–0.2%, which closed a whole register-squeeze family
  without writing it.
- At the power cap part of an occupancy gain returns as clock. *(field)* 3 → 4 waves/SIMD at 96.6% of TBP:
  sclk −3.3%, 3.9% fewer cycles, wall 0.5% faster. Report sclk with every occupancy arm (`flydsl_attn_measurement.md`).
- A change that drops a step loses even when it removes traffic: GQA merging past the step is in
  `flydsl_attn_grid_dispatch.md` §9.
- **Registers saved without crossing a step are worth 0 or less.** *(field)* At occupancy 1, dropping an
  AGPR tie that freed 14 registers: −0.80%. Sharing VOP3P splat constants (20 → 4 VGPRs, `.vgpr_count`
  unchanged): −0.9% at 3 waves/SIMD, −0.61% at 2. The 72+72-register example is in
  `../gemm/flydsl_gemm_mainloop.md` §4.
- Exception: while spills or a private segment exist, moving a value out of the loop carry pays without a
  step. *(field)* A 16-dword loop-carried softmax seed rebuilt by scalar broadcast at each use: private
  segment 188 → 16 B, +0.40%; LICM hoists the broadcast, and the saving lands in prologue/epilogue spills.

## 4. Before squeezing, find where the peak is
- Print the highest VGPR index referenced in each basic block. *(field)* Nine blocks peaked at 142–149
  with `.vgpr_count` 150; of three liveness cuts, two left the allocation unchanged and one raised it
  to 154. Only
  a block clearly above the others is worth local work; otherwise change the resident set.
- Liveness follows the scheduled order, not the text. *(field)* Reordering to end the coexistence of two
  64-score tiles freed 4 registers.
- The budget is the shape of the live intervals, not a dword count. *(field)* Freeing 16 loop-carried
  dwords did not make room for a knob that needed 8 (still 12 spills).
- MFMAs that consume a value anchor its allocation in place. *(field)* Deleting 8 row-sum MFMAs: spill
  118; replacing half with VALU: 93; moving 2 of them: spill 4 → 8.

## 5. Spill scale
- *(field)* At 128 VGPRs, 4 waves/SIMD: 5 spills −0.3%, 10 spills −0.2%, 35 spills −19%. Single-digit
  spills cost 0.2–0.4%; 30+ is a cliff.
- Spill 0 is not a goal. *(field)* Reaching 0 by dropping a K prefetch that covered the first QK MFMAs:
  −1.1%. Re-sweep "keep a little more resident" knobs at the current spill level.
- Gate every arm on the ISA spill count before timing; a dump takes seconds, a bench round does not.

## 6. Code shapes that spill
- **A loop remainder is a third body instance.** Register pressure scales with body instances. *(field)* 2× unroll
  plus a tail loop: spill 2 → 35, −19%; removing the tail: 35 → 7 (10 → 0 with a shallower K prefetch). Write the
  remainder as a predicated second body: a runtime `if j + 1 < end:` on a workgroup-uniform condition (the rewriter
  lowers it to `scf_if_dispatch`), yielding the accumulators from both branches. Wrapping one body instance in a
  workgroup-uniform `if` whose `else` yields the accumulators is harmless; a branch holding its own body copy is not
  (below); a conditional barrier inside a branch spilled 97.
- **Specialize with sequential ifs, not nested ones.** Each runtime `if` returns its accumulators and the `else`
  yields them unchanged. *(field)* Nested: 476 registers (arch + 219 AGPR), above the 464 that kept a co-resident
  reduce kernel resident; sequential: 460,
  spill 0, 468 fewer instructions in the masked loop.
- **`scf.if` around a large region keeps loop invariants (Q packs, LDS bases) live across it.** *(field)*
  A per-wave phase-reversal vehicle was 10.6% slower by itself; duplicating the main loop per branch, 27% slower.
- Runtime conditional syntax: `../bottleneck/debug-flydsl-kernel.md` § Runtime versus compile-time
  conditionals.

## 7. AGPRs are not free space
- *(field)* Forcing `amdgpu-agpr-alloc` to 32/64 to absorb 31 spilled dwords raised spills to 52/77 (the
  second also added 26 `v_accvgpr` moves).
- Pinning a value in AGPRs (tied asm `"=a,0"`) is zero-sum when the register budget is already spent: the allocator
  evicts another AGPR resident instead of growing the AGPR count. *(field)* Pinning a 32-dword pack: `.agpr_count`
  223 → 219, another operand group pushed out, shuttle traffic up, −0.49%. Check that `.agpr_count` rose before
  trusting a pin.
- Pin only values an MFMA reads directly and that stay fixed for a kv block. *(field)* Two such operand
  groups: +0.79% and +0.53%; two per-head-step groups: −3.4% and −2.6%.
- VALU cannot read AGPRs, so every VALU use costs a `v_accvgpr_read`. Judge pins by the
  `v_accvgpr_*` count and wall time, not `accum_offset`; untied pins delete the operand
  (`../gemm/flydsl_gemm_correctness.md` §7).

## 8. MFMA–VALU co-execution
What the rules below rest on (`flydsl_attn_mfma_layout.md` §1, `flydsl_attn_softmax.md` §1): a `16x16x32` bf16
MFMA leaves an 8-cycle VALU shadow and a `32x32x16` 24, shared by every wave on the SIMD; an exp uses it twice as
fast as a multiply and gains nothing over other VALU work from MFMA overlap.
- Read coexec / MFMA busy and the stall pool as in `flydsl_attn_attribution.md` §2–§3.

**Two co-resident workgroups per CU (backward): the overlap is already free.** Two independent workgroups sharing
each SIMD fill each other's barrier and drain bubbles with no barrier between them. In-workgroup overlap trades that
for a barrier tax. *(field)* 8-wave single workgroup: MFMA utilization 53 → 40%, −11%; adding a phase stagger
recovered only to 1029 against the 4-wave baseline's 1116; 2-producer/2-consumer role split: spills at occupancy 2,
half the GEMM throughput at occupancy 1; cross-head double buffering: 31% and 42% slower. Check whether the
baseline already co-hosts two workgroups per CU before building any of these.

**Hiding exp under MFMAs with two co-resident workgroups is worth ~0.** *(field)* `sched_barrier(0)`-forced
interleave 0.00%; counted `sched_group_barrier` MFMA:TRANS 1:2, 2:4, 4:8: +0.26%, −0.04%, −0.07%. Only removing
instructions moves wall there (`../gemm/flydsl_gemm_attribution.md` §8).

**One workgroup per CU (forward) is the only regime where in-workgroup dual-wave scheduling can pay.** FlyDSL
v0.3.2's gfx950 forward is the reference: 8 waves, split into two groups by `wave_id // 4`; group B issues one extra
`s_barrier` at entry and group A one at exit, so the groups run one cluster apart, one loading while the other
computes. Each loop iteration is 8 clusters separated by `sched_barrier(0); s_barrier; sched_barrier(0)`,
alternating memory and compute, with compute clusters wrapped in `s_setprio(1)`/`s_setprio(0)`. It is one answer,
not the only one: *(field)* on 8-wave forwards running two workgroups per CU (4 waves/SIMD), turning the stagger on
was 4.8% and 6.3% slower; price it on your own kernel at one workgroup per CU.

**Count what the shadow already hides before moving VALU into it.** *(field)* Moving 16 more exps into a P·V
cluster whose 8 MFMAs already covered 16 `ds_read_b128` and 2 DMAs cost 2.3%, with a perfect weave in the ISA;
moving that cluster's own 16 exps to the QK cluster, which had free VALU slots, was neutral (−0.27%). At two or
more waves per SIMD the clusters are still not interchangeable.

**Independent MFMAs placed in a VALU window get hoisted out.** *(field)* Splitting a GEMM so 16 independent MFMAs
faced the softmax window: coexec 19.2% → 15.9%, −3.0%. Pin the region with `sched_barrier(0)` placed after the
`gpu.barrier()` that owns the LDS write and outside `range_constexpr` loops (`flydsl_attn_schedule_sync.md`).

**Static shadow accounting is a zero-GPU prefilter.** Price each instruction class against the shadow budget above
to discard candidates; rank survivors by measurement. *(field)* Coverage distance, waitcnt and `s_nop` counts, MFMA
run length and pipe/slack models each failed to rank arms on at least one body.

## Verify
- `.vgpr_count`, `.agpr_count`, spills and private segment read from the ISA for every arm.
- The occupancy target is a reachable workgroup multiple, and the step was priced with a dead-LDS or
  `waves_per_eu` probe.
- Overlap work starts from the ISA and the measured residency: co-resident waves and no bare VALU between MFMAs →
  remove instructions; otherwise → schedule.

## Related
`flydsl_attn_schedule_sync.md` · `flydsl_attn_lds.md` · `flydsl_attn_attribution.md` ·
`flydsl_attn_softmax.md` · `flydsl_attn_backward.md` · `../gemm/flydsl_gemm_mainloop.md` ·
`../gemm/flydsl_gemm_grid_pricing.md` · `../../../../hardware/mi350_execution.md`
