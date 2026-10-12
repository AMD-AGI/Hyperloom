# AITER all-reduce: 1-stage / 2-stage crossover

## Objective

One dispatch decision is under optimization, scored as `raw_dispatch`.

| Metric group | Where the decision is made | Current condition |
|---|---|---|
| `raw_dispatch` | `csrc/include/custom_all_reduce.cuh`, `CustomAllreduce::allreduce` | 1-stage when `(world_size_ <= 4 && bytes < 160*1024) \|\| (world_size_ <= 8 && bytes < 80*1024)` |

Which clause is live depends on the rank count this run was launched with, and
only the live one is worth editing — changing the other changes nothing and
wastes an iteration:

| ranks | live clause | effective 1-stage cut | crossover at hidden=7168, BF16 |
|---|---|---|---|
| <= 4 | `world_size_ <= 4 && bytes < 160 KiB` | 160 KiB | 11.4 rows |
| 5..8 | `world_size_ <= 8 && bytes < 80 KiB` | 80 KiB | 5.71 rows |

The default case suite derives its rows from that cut, so it brackets the
crossover at whatever rank count is in use. The named suites (`tp4_wide`,
`tp8_k3`) pin the configurations that have a measured baseline below.

The reference workload is Kimi-K3, `hidden = 7168`, BF16, where one row is
`7168 * 2 = 14 KiB`.

`fused_rmsnorm` is measured for diagnostics but excluded from the KEEP score
because this task targets raw dispatch.

## Measured baseline (8 ranks)

Taken on that configuration (8x MI355X, gfx950, AITER at `36c421f7f`,
`hidden = 7168`, BF16, graph-based timing, warmup 20 / iters 50):

| rows | payload | median_ms | dispatched path |
|---|---|---|---|
| 1 | 14.0 KiB | 0.007043 | 1-stage |
| 2 | 28.0 KiB | 0.007220 | 1-stage |
| 3 | 42.0 KiB | 0.007382 | 1-stage |
| 4 | 56.0 KiB | 0.007528 | 1-stage |
| 5 | 70.0 KiB | **0.007925** | 1-stage — worst case in the sweep |
| 6 | 84.0 KiB | **0.006864** | 2-stage — 13.4% faster than rows=5 |
| 7 | 98.0 KiB | 0.006939 | 2-stage |
| 8 | 112.0 KiB | 0.006897 | 2-stage |
| 12 | 168.0 KiB | 0.006934 | 2-stage |
| 16 | 224.0 KiB | 0.007062 | 2-stage |
| 64 | 896.0 KiB | 0.010431 | 2-stage — production decode payload at conc=64 |

The diagnostic `raw_dispatch` group aggregate at baseline is **0.007421 ms**.

Two facts follow directly, and they are the reason this task exists:

1. **The 1-stage band is monotonically getting worse** as payload grows
   (0.007043 -> 0.007925 across rows 1..5), while 2-stage is flat around
   0.0069 from 84 KiB to 168 KiB.
2. **2-stage at rows=6 is faster than 1-stage at every single row below the
   cut**, including rows=1 at 14 KiB. There is no payload in this range where
   the current 1-stage dispatch is the faster choice.

## Important: this contradicts the TP4 conclusion

A previous campaign of this task on **4x MI325X (gfx942)** concluded that
1-stage was "the path that pays" — it measured 20-35% kernel headroom on
1-stage versus 3-7% on 2-stage, and end-to-end decode gains only materialised
below the crossover. **Do not carry that conclusion over.** On this
configuration the measurement says the opposite: 1-stage loses to 2-stage
across its entire dispatch range.

Treat the TP4 hypotheses about 1-stage kernel headroom as unverified here. The
first question to settle is whether the 80 KiB cut should exist at all at
`world_size_ == 8`, not how to make 1-stage faster.

## Hard rules

- **Never edit `driver.py`.** It is the correctness oracle and the timer. The
  loop blocks edits to it; working around that invalidates every number.
- Keep `use_new = true` and `full_nvlink_ = true`. This task optimizes only the
  new-kernel path.
- **Do not touch the `VLLM_REDUCE_CASE` naive path** (the 512/256 KiB branch).
- **Do not touch the gfx942 `write_mode` branch** (`bytes > 4 MiB`). At
  `hidden = 7168` that is `rows > 293`; the scored set stops at rows=64
  (896 KiB), well below it, so any edit there is unmeasured by this task's
  score. Kimi-K3 decode runs at conc=64, which is the regime that matters.
- **Do not raise fused shapes above `hidden = 8192`.** C++ silently re-clamps
  with `use_1stage && (n % pack_size == 0) && (n / pack_size <= 1024)`; for BF16
  that caps 1-stage at `hidden = 8192` even though the Python gate allows 16384.
- `world_size` is restricted to `{2, 4, 6, 8}` (`world_size_ == 2`
  short-circuits to 1-stage earlier).

## Measurement notes

Read these before interpreting any number.

- **Timing is graph-based on purpose.** The driver captures a chain of
  collectives into a CUDA graph and times replays. Two simpler methods were
  measured and rejected on the TP4 campaign: one call per `barrier` gave 6-10%
  run-to-run spread (barrier exit jitter lands in the sample), and an eager
  burst was stable but CPU-bound — every size reported the same ~21.7 us, i.e.
  Python dispatch cost rather than the kernel.
- **The four `fused_*` cases are measured but not scored.** They remain visible
  diagnostics but do not affect KEEP.
- **`AITER_QUICK_REDUCE_QUANTIZATION` must stay unset.** QuickReduce sits ahead
  of custom all-reduce in the dispatch chain and would silently take over for
  large payloads. The driver refuses to run if it is set.
- **`AITER_AR_1STAGE` is read at import time** (a class attribute on
  `CudaCommunicator`). Changing it inside a running process has no effect;
  comparing modes requires separate launches.
- Each of three candidate measurements is scored independently against the fixed
  pristine baseline; their mean must beat the current best by at least
  `t * sigma / sqrt(3)`, a one-sided 95% Student-t test on those three scores,
  floored at 0.1% of the current best. All-reduce is a noisy measurement, so
  expect the bar to sit well above that floor here.

## What has already been tested (on gfx942 / TP4)

These results are about the kernels themselves rather than the dispatch
threshold, so they most likely still hold. They cost several campaigns to
establish; do not re-derive them.

Both 2-stage kernels look under-occupied — 24-32 workgroups on 304 CUs, ~22 GB/s
against an interconnect ceiling one to two orders of magnitude higher, stage-1
reduce-scatter running on one quarter of each block's threads. **That appearance
was tested and it is not the lever.** Three attempts to convert the idle
capacity into speed were built, verified correct and measured:

| Attempt | Result |
|---|---|
| Split stage-1 across all thread groups (raw) | raw score **+0.87%** (worse) |
| Replace `end_sync` with a per-peer readiness barrier | raw score **+7.73%** (worse) |
| Sweep the block-count cap (24/32 -> higher) | inside noise, repeatedly |

The consistent direction says the low occupancy is not waste: it is what keeps
the p2p request queues from thrashing. More lanes issuing peer traffic, or
looser synchronization, both cost more than they save.

What did work was the opposite move — **making a barrier narrower rather than
the work wider.** Giving each thread two packs instead of one halved the block
participating in the fused epilogue's `__syncthreads`, cut `SQ_WAIT_INST_ANY` by
58% and took wait/VALU from 1.01 to 0.49. Both fused kernels already carry that
mapping. Note what made it work, because it generalizes: **when narrowing a
block, check what else depended on its width.**

One asymmetry is load-bearing and settled: raw reads peers in both stages, while
fused reads peers AND broadcasts its reduced packs to every peer's tmp buffer in
stage 1. Replacing that broadcast with a local write plus a remote gather looks
like a clear win and passes a casual correctness run, but it is a data race —
`end_sync` synchronizes the SAME block index across ranks, so it cannot order a
write against a read issued by a different block, and at these shapes 97% of
gathers read a pack some other block reduced. It surfaced as an 8.4 dB SNR drop
under stability inputs while ordinary inputs still looked fine. **Do not undo
it.**

## Suggested hypotheses

Framed as things to measure, not as instructions. Ordered by what the baseline
data supports.

1. **Does the 80 KiB cut earn its keep at `world_size_ == 8`?** The baseline
   says 2-stage beats 1-stage at every measured row below the cut. Sweeping the
   cut downward — including to zero, i.e. always 2-stage at TP8 — is the most
   direct reading of the data. Verify it rather than assuming it: rows=1 at
   14 KiB is the smallest payload in the score and the one most likely to
   behave differently from the trend.
2. **Why does 1-stage degrade with payload while 2-stage stays flat?** 1-stage
   reads all peers and reduces in one pass with no reduce-scatter phase and no
   tmp round-trip. If its cost grows with payload while 2-stage's does not, the
   limit is likely per-peer read bandwidth or request-queue depth rather than
   compute. Profile it — on this hardware nobody has.
3. **rows=64 (896 KiB) costs 0.010431 ms, well above the 0.0069 plateau.** It
   is the production decode payload, and it is the single most expensive scored
   case. Whatever makes 2-stage flat from 84 KiB to 168 KiB stops working
   somewhere before 896 KiB. Finding that transition may be worth more
   end-to-end than anything in the crossover region.
4. **Barrier width and count, per kernel.** For each `__syncthreads` and each
   `start/end_sync`, ask what data dependency it actually protects and whether
   the threads it stalls all participate in that dependency. The raw path has
   not been examined this way; the fused one has.
5. **A kernel change moves the optimal crossover.** After any kernel edit,
   re-check the threshold instead of assuming an earlier sweep still holds.

## Multi-step changes

Restructuring a barrier or a thread-to-data mapping cannot be done one verified
edit at a time: the kernel is slower part-way through than it was before, and
only pays off once the whole change is in place. The session gate reserves its
opening edits for exactly this — inside that window correctness is still
enforced and the speed is still reported to you, but a slower measurement does
not end the attempt. Carry such a change through to completion before judging
it, and abandon it on evidence that the approach is wrong rather than on one
intermediate number.

## Bar to clear

The scored range spans 14 KiB to 896 KiB. A change confined to one end will
barely move the equal-weight mean case speedup. Individual case deltas remain
useful diagnostics, but only the complete-suite mean decides KEEP.

Correctness must stay above the SNR threshold the loop enforces. The baseline
smoke case measures 47.98 dB.

**Clearing the SNR floor is necessary, not sufficient.** A change that still
passes but lands near the floor has usually broken something, because
reduction-order differences move SNR by a fraction of a dB, not by several.
Treat a multi-dB drop as a defect to explain — a synchronization hole shows up
exactly this way, and only under the stability inputs, while ordinary inputs
keep passing.
