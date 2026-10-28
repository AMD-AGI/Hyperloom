---
title: FlyDSL grouped GEMM — measuring: distributions, the deployment's operating point, trace FLOPs, per-cell noise
kind: language
lever: flydsl_grouped_measurement
gens: [gfx950]
dtypes: [bf16, fp8, mxfp8, mxfp4]
regimes: [forward, dgrad, wgrad, skew, kernel-A/B]
updated: 2026-10-27
sources:
  - AMD-AGI/Primus-Turbo@ca1fdd67:benchmark/ops/training/config.py:487-497
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:2453-2467
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:2483-2484
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:2622
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:4772-4777
---

# Measuring a FlyDSL grouped GEMM

> The shared protocol — ruler choice, arm order, warm-up, noise floors, statistics, closing a series of changes —
> is in `../gemm/flydsl_gemm_measurement.md`; probes and power-cap pricing are in
> `../gemm/flydsl_gemm_attribution.md`. This card is what the group dimension adds: the token distribution as an
> axis of the table, the deployment's operating point, FLOP accounting from training traces, and floors that
> differ per cell. Numbers marked *(field)* are measured values; use them for direction and order of magnitude.

## Route here when
- Building or inheriting a grouped-GEMM benchmark, scorer or autotune race.
- A kernel-only gain does not show up in the training step, or the bench and a trace disagree.
- A delta sits near 1% on skewed cells, or two runs of identical code disagree.

## 1. The table: geometries × projections × passes × distributions

- **Several model geometries, both projections, all three passes.** Use the gate_up projection
  (N = 2 · intermediate, K = hidden) and the down projection (N = hidden, K = intermediate) of at least three
  models, for forward, dgrad and wgrad, at the per-call tokens per group the deployment runs plus one size on each
  side. Score the geometric mean across geometries: a single-geometry table lets NT changes overfit without any
  warning (`flydsl_grouped_workflow.md` §8). Take the dimensions from the model configurations; this card fixes
  none.
- **Every distribution is a column.** Per cell:
  - balanced;
  - mild random skew (Turbo's generator: `flydsl_grouped_correctness.md` §2);
  - heavy skew, with one group holding about half of the tokens;
  - one-hot or near-degenerate;
  - empty groups (`flydsl_grouped_correctness.md` §2);
  - the deployment's measured histogram.
- **The extreme cells are real.** *(field)* A one-hot bench cell reproduced the slowest wgrad call of a training
  trace within 1.4%.
- Report each pass's skew-to-balanced ratio and every cell, not only the geometric mean
  (`../gemm/flydsl_gemm_measurement.md` §4). The heavy-skew cells are usually the minimum, and they are where a
  skew lever is accepted or rejected.

## 2. Lock the deployment's operating point

A clean ruler at the wrong operating point optimizes a kernel the training run never launches. Copy these from
the deployment, not from a previous bench:

| Condition | What goes wrong when it drifts |
|---|---|
| Groups per call (local experts = experts / EP, the expert-parallel degree) | bands, tiles per group and skew change with G; tunings do not transfer |
| Per-call tokens per group | the step aggregate is several calls; a bench at the aggregate size reads high |
| Histogram of tokens per group | balanced-only benches miss every skew effect |
| Which axis is padded, and to what | an unpadded contraction can take a slow path the deployment never runs |
| beta and the accumulator dtype | the C read-back of beta=1 scales with the accumulator width |
| Eager or graph replay | host costs are present in one and hidden in the other |
| Every argument the call site passes | one optional argument can select another body |

- **Copy the call site's arguments, not only its shapes.** *(field)* A bench passed a CU count and ran the
  persistent body; the deployment's call site passed none and ran the non-persistent one. Over a long series
  of changes, every change the bench scored as a win was a loss on the deployed body: the bench's 5.2% time saving
  was 0.84% in training.
- **Per call, not per step** (`../gemm/flydsl_gemm_measurement.md` §7). *(field)* A step ran four
  micro-batches, so each call saw a quarter of the step's tokens per group; at that size the wide projection
  reached 23% (forward) and 36% (wgrad) less throughput than at the aggregate size. Confirm the size by
  comparing each kernel's launched grid in a trace with the bench's: the grid fields match exactly when the size
  is right, which is a harder check than durations.
- **Pad the contraction axis the way the deployment does** (`flydsl_grouped_lowprec.md` §6: padding only the
  output axis left a forward on its misaligned slow path). Count useful FLOPs from the unpadded sizes.
- **Accumulator dtype.** *(field)* A wgrad timed with an fp32 accumulator where the deployment used bf16 took
  8–14% more time on one shape (beta=0 and beta=1). The beta=1 read-back itself is priced in
  `flydsl_grouped_wgrad.md` §10.
- **Graph replay hides host enqueue, not kernel boundaries.** *(field)* One extra small kernel on the dependency
  chain still cost a full launch on every call under graph replay (`flydsl_grouped_moe_inference.md` §7). Measure
  launch-count changes on the scoring bench, in the mode the deployment runs.
- **Screen with kernel-only, decide in the step.** *(field)* Kernel time saved converted to step time at 81% over
  one series of changes; a probe 0.5% faster on the deployed grid was 0.71% slower in training.

## 3. Reading a training trace: per-call sizes and FLOPs

- **Read sizes, do not reconstruct them.** Pair each GPU kernel with its CPU op through the trace's
  `External id` and read the op's `Input Dims`: `A = [M_total, K]` and `B = [G, N, K]` give G and the mean tokens
  per group (`M_total / G`) exactly; the per-group histogram is in the `group_lens` contents, not in the dims.
  Rebuilding them from batch size, sequence length and EP is how operating points
  got mixed up.
- **Do not divide by the layer count.** Every token is routed again in every layer: assignments per layer are
  tokens × top-k. A FLOP count divided by the layers reads an order of magnitude low.
- **One kernel name, several logical GEMMs.** The gate_up and down projections of a pass launch the same kernel,
  so a per-name aggregate merges two GEMMs with different N. Split the calls by per-call duration clusters, compute
  each GEMM's rate, and check that the parts sum back to the aggregate.
- **Training traces carry contention.** *(field)* The median of one profiled step read 5–9% low from overlap with
  other work, and the long-tail calls were the ones overlapping collectives. Compare with a bench using the
  minimum or the calls without overlap.
- A trace's per-kernel rate must land on the bench at the same operating point. A miss of an order of magnitude
  is FLOP accounting, not the kernel.

## 4. Timing hygiene

- **Compile once and call the compiled object; compile beta=1 builds on a scratch output**
  (`../attention/flydsl_attn_measurement.md` §6, `flydsl_grouped_host_autotune.md` §2).
- **Keep the host out of the GPU window** (`../gemm/flydsl_gemm_measurement.md` §6), and assert it: time the host
  enqueue of the same R calls with a wall clock and require `t_host < 0.9 · t_gpu`. A slow wrapper looks normal
  until it is compared with a fast one.
- **Give beta=1 a fresh accumulator every repetition.** *(field)* Reusing one bf16 accumulator for 30 repetitions
  overflowed it and read 3200 µs for a 1814 µs kernel.
- **Freeze the denominator.** When a score compares two operating points of the same kernel (two group counts,
  or a wgrad against its own forward), a live denominator rewards slowing it down. Use a copy of the base kernel
  that the work cannot touch, measured in the same interleaved window, plus a live guard on the other point.
  Set the guard's tolerance from its measured spread: *(field)* a min-of-four guard at 0.99 failed the baseline
  itself (its own spread reached 0.984); a geometric mean at 0.98 passed.
- **Version A/B in one process** (`../gemm/flydsl_gemm_measurement.md` §1): load the old kernel as its own module.
  Two versions that define closures of the same name can collide in the JIT cache; bit-identical outputs with a
  ratio of ~1.0 mean "suspect one binary served both arms", not "no effect".
- **A probe tree goes stale when the tree under test is refactored.** *(field)* +4.8% on a probe tree copied
  before a dispatch refactor read ~0 with one of three changed files synced and +0.34% (worst cell +1.05%) with
  all three: re-measure on the tree that will land, with every file it changed.

## 5. Noise per cell and arm order

- **The floor belongs to the cell.** *(field)* In one run of one binary, the A-vs-A reading was −0.02% on a
  balanced cell and +1.39% on a skewed cell, whose histogram was redrawn every run so its tile counts and
  imbalance changed. Set acceptance thresholds per cell; to resolve under ~1.5% on a skewed cell, add samples or
  detrend, and keep the draw identical across arms when ranking a kernel change.
- **Single cells swing more than aggregates.** *(field)* One wgrad cell moved ±2.5% run to run on identical code.
  Decide on paired differences, not on one cell's reading.
- **Detrend with an in-run control arm.** *(field)* A control arm drifted 1.8% over six back-to-back runs; an
  order that looks balanced does not cancel a monotone drift. Normalize each run by the control arm of that run.
- **Arm order** follows `../gemm/flydsl_gemm_measurement.md` §2: one process per arm runs `A B` (or `A B A B`);
  palindromes are only for two module instances interleaved inside one process. The first arm of a batch runs
  cold (`../gemm/flydsl_gemm_measurement.md` §3). *(field)* On grouped tables, cells a knob could not touch (the
  NT cells under a wgrad-only change) read ~0.5 pp low in whichever arm ran first. Use the untouched pass's
  cells as the batch's drift ruler (`../gemm/flydsl_gemm_measurement.md` §4) next to an inert arm (§2 there).

## 6. Small kernels and ISA statistics

- **Price small kernels additively** — group-offset kernels, scale preshuffle, tail kernels, the split-K reduce —
  by launching them 0, 1, 2 and 4 extra times in one process, alternating arm order, and fitting the slope
  (`../gemm/flydsl_gemm_attribution.md` §3). *(field)* `rocprofv3` start→end gave an idle tail kernel 5.10 µs
  against a marginal 1.58 µs, and a working preshuffle 20.3 µs against ~10.2; a candidate priced at +0.374% from
  the trace was worth +0.121% (+0.107% measured).
- **Split ISA dumps per kernel.** A stage dump concatenates a module's kernels with no separator. *(field)* Four
  stores read as "the GEMM's prologue draining" belonged to the preshuffle kernel in front of it. Cut at the
  `.amdhsa_kernel` and label lines first.
- **Count over a steady-state window.** Boundary-body variants copy the tile body in the binary while each
  execution runs one copy. *(field)* A wgrad with four boundary variants read 6.6 instructions per MFMA over the
  whole main-loop region and 3.2 inside an MFMA steady-state window; its NT sibling read 4.1 and 3.4.
- **Say which MFMA utilization you report**, wall or steady state (`flydsl_grouped_workflow.md` §5).

## 7. Power, clocks, and comparisons across precisions

- **Read power per pass before ranking cycle-saving levers.** At ≥99% of board power, part of a cycle saving is
  lost to a lower clock (`../gemm/flydsl_gemm_attribution.md` §4, whose 20% conversion case was the skewed worst
  cell of a grouped MXFP8 NT kernel *(field)*). *(field)* A variable-K wgrad that ran at its maximum clock, far below
  the cap, did not hand cycle savings back as clock. Prefer levers that remove bytes or requests on capped passes
  and skewed cells.
- **Compare precisions on equal terms** (`flydsl_grouped_lowprec.md` §8; quantization inside the timed region:
  `../attention/flydsl_attn_measurement.md` §5).

## 8. The race is a ruler too

Turbo's fp8 NT/NN race scores candidates on balanced synthetic offsets at two canonical sizes, 1024 and 8192
tokens per group (`grouped_gemm_fp8_kernel.py:2453-2467`, `:2483-2484`), and its wgrad race on a balanced load
only (`:4772-4777`). *(field)* An NN dgrad arm that won at the deployed size could not pass the race's 1.5%
adoption margin (`_NP_RACE_MARGIN`, `:2484`, `:2622`) at the two race points
(`../gemm/flydsl_gemm_autotune.md` §2). Check what the race picks at the deployed size and under skew; the
remedies are in `flydsl_grouped_host_autotune.md`.

## Verify
- [ ] The table covers ≥3 geometries × both projections × three passes × balanced, heavy-skew, one-hot and
      empty-group columns.
- [ ] G, per-call tokens per group, the histogram, padding, beta, accumulator dtype, graph mode and the call
      site's arguments are copied from the deployment, and the launched grids match a trace.
- [ ] beta=1 timing uses a fresh accumulator per repetition and a scratch output for compilation.
- [ ] Each cell has its own A-vs-A floor from this session; the decision uses paired differences.
- [ ] Small-kernel prices come from additive launches, not from trace durations.

## Related
`../gemm/flydsl_gemm_measurement.md` · `../gemm/flydsl_gemm_attribution.md` ·
`../attention/flydsl_attn_measurement.md` · `flydsl_grouped_correctness.md` · `flydsl_grouped_host_autotune.md` ·
`../../../../common_methodology/profiling/measure_protocol.md` · `../../../../hardware/mi350_clocks.md`
