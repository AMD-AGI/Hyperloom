---
title: FlyDSL GEMM at small M (decode, skinny) — what sets the time and what to tune
kind: language
lever: flydsl_gemm_small_m
gens: [gfx950]
dtypes: [bf16]
regimes: [decode, small-M, skinny, split-k]
updated: 2026-10-19
sources:
  - ROCm/aiter@599a01c8:aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py
  - ROCm/aiter@599a01c8:aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py
  - ROCm/aiter@599a01c8:aiter/ops/flydsl/gemm_a16w16_policy.py
  - ROCm/aiter@599a01c8:aiter/tuned_gemm.py
  - ROCm/FlyDSL@dd8373349:kernels/gemm/gemm_a16w16_gfx950.py
---

# FlyDSL GEMM at small M

> Scope: dense GEMM with M ≈ 1–256 (decode steps, speculative-decode verify), mostly bf16. MoE expert GEMMs at
> decode are in `../grouped_gemm/flydsl_grouped_moe_inference.md`. Primus-Turbo's FlyDSL
> GEMMs (the exemplar elsewhere in this package) are built for training shapes and do not target small M.
> aiter's FlyDSL a16w16 backend is FlyDSL's `gemm_a16w16_gfx950` (aiter #5145 onwards). aiter's earlier bf16
> kernels (`splitk_hgemm.py`, `small_m_hgemm.py`) are what `../optimize/flydsl_levers/flydsl_kernel_library.md`
> and `../optimize/flydsl_levers/flydsl_knob_space.md` still describe.
>
> Numbers marked *(field)* are measured values on aiter's earlier small-M kernel. The mechanisms carry over to
> the current kernel; the configurations do not. Each technique marked **Not in the current kernel** is a porting
> candidate to re-measure.

## Route here when
- A decode or speculative-verify step spends visible time in dense GEMMs with M ≤ 256.
- You are tuning, adding tuned rows for, or changing aiter's FlyDSL a16w16 GEMM at small M.
- A small-M result looks good in a micro-benchmark and does not show up in the step.

## 1. Measure it the way decode runs it

- **B is cold in decode.** A step touches each weight once and streams tens of GB between visits, while a
  benchmark that replays one weight keeps it in the 256 MB MALL/L2. *(field)* The same kernel read 30.7 µs
  replayed and 45.1 µs in the real step, and the ranking of configurations flipped. Configurations chosen on a
  replayed weight were later measured to cost the step 3.3%.
- **Rotate enough distinct copies of B to exceed the MALL, with an explicit count.** *(field)* aiter's
  `run_perftest` with the automatic rotation count picked 201 copies and read 41.56 µs for a kernel that read
  15.9–16.1 µs at every explicit count from 8 to 128.
- **Time a replayed CUDA graph, and know the floor.** *(field)* An eager aiter call cost 16.7 µs of host time
  against kernels of 4.6–8 µs. One graph node cost 1.75–1.95 µs even for a 1-element op, so at
  M ≤ 16 on a small weight more than half the measured time is the node.
- **Price candidates through the deployed entry** (`aiter.tuned_gemm.gemm_a16w16`), not a direct kernel call.
  *(field)* A row that a direct call measured 0.76 µs faster was 0.34 µs slower through the entry, and in deployment
  that row alone cost about 4.2 percentage points of the step score.
- **A cold micro-benchmark shortlists; the step decides.** *(field)* Kernel time converted to step time about
  1:1 (ratio 1.04), but 1.2% of the step score landed in other kernels, and a +10% uniformly-weighted gain was
  −1% when weighted by call counts.
- **A single-shape tuner on an idle GPU overvalues split-K.** *(field)* Fifty tuned rows whose split-K only
  paid on an idle device made the step 18% slower (5742 vs 4853 µs). Deeper split-K buys parallelism from CUs
  that are idle only in the micro-benchmark.
- **Allocate fresh tensors for each repetition.** *(field)* Eight allocations of one shape and configuration
  read 3.73–6.19 µs, bimodally: an address (L2-set) effect, not noise (§6, XCD placement).
- **Per-shape noise is large.** *(field)* Per-shape drift of 3–7% while aggregates reproduced to 0.1–1.5%;
  the first arm of a probe read 1–2% slow after warm-up. Adopt a per-shape change only above that shape's
  measured spread. Within one process, run two arms as `A B B A` per rep and shuffle the order every rep
  when there are three or more (`flydsl_gemm_measurement.md` §2).
- **A memory-side headroom estimate needs a measured streaming ceiling**, e.g. a plain read of the same B
  buffer, not the datasheet bandwidth. *(field)* 5.8–6.0 TB/s for a cold read of a 476 MB weight.

## 2. Classify each call site first: streaming or latency-bound

The lessons below flip sign between two regimes. Decide which one a call site is in before reusing a
"what worked" list:

| | Streaming row | Latency-bound row |
|---|---|---|
| Example | o_proj / q_b / qkv projections, B tens of MB, ~100+ N-blocks | a narrow-N projection, B ≈ 1.5–3 MB |
| Bound by | HBM read of B | launch, reduction and memory latency |
| Split-K, more blocks | lose *(field: +17–19% on an o_proj row already at ~7.7 TB/s with 192 blocks)* | win *(field: −40% and −16% on two such rows with a 4× larger grid and split-K 8)* |
| `nt` on B | wins on a long read-once stream | loses: L2 is reducing latency, not only reusing |

## 3. Grid size at small M

- **Count workgroups against the 256-CU dispatch round before anything else.** *(field)* Padding one grid
  from 246 to 264 workgroups (empty blocks exit at once) made the kernel 50% slower: one round became two. Cold,
  through the deployed entry, every winning configuration sat at exactly 256 workgroups and 288 lost by 1.16×.
  `flydsl_gemm_grid_pricing.md` §1 has the arithmetic.
- **"Fewer tiles than CUs" is necessary for starvation, not sufficient.** A cold, read-once B stream can be
  HBM-bound with ~200 blocks; then every way to more blocks loses (table above).
- **Cover M with one M-block: `tile_m = align_up(M, 16)`.** Each M-block re-streams all of B. *(field)* At M=24
  on a 38720-wide weight, two M-blocks of 16 took 127.7 µs and one block of 32 took 90.4 µs; three or more
  M-blocks were 2–4× slower. Make sure the tuner's `tile_m` filter can produce `align_up(M, 16)` for every M it
  covers.
- **Odd tile widths can be the only way to reach a full round** (for N=6144 at small M, only `tile_n` 48 and 96
  give 256 workgroups), but they lost on other rows in deployment. Race them; do not adopt them by arithmetic.
- **Wider tiles lose at small M even when they move fewer bytes.** *(field)* A tile chosen as line-cost-optimal
  by a traffic model was the slowest by 40% on a latency-bound row. Do not prune candidates with a traffic model.

## 4. Split-K

**When it pays:** a single N-block (narrow N), where split-K is the only parallelism *(field: 34.1 → 5.0 µs at
1×32×6144 with split-K 12)*, and latency-bound rows (§2). **When it does not:** streaming rows that already have
~100+ N-blocks, short K (*field:* splitting K=512 to fill a 0.44-wave grid cost 15–18%), and anything chosen on
an idle-device tuner (§1).

**The reduction is a fixed cost that dominates at M=1.**
- The current kernel pre-zeroes C, takes a ticket, publishes it, spins until the C initialization is visible,
  and then adds each split's partial into C with buffer atomics (`splitk_protocol.zero_c` / `take_ticket` /
  `wait_until_initialized`, `gemm_a16w16_gfx950.py:730-764`; `BufferAtomicPkAdd` / `BufferAtomicAdd`,
  :441-461).
- *(field, older kernel of the same design)* That protocol cost 1.5–2.5 µs per launch at M=1. Replacing it with
  per-split workspace planes, a last-arriving block that sums the planes in f32, and **one arrival counter per
  128 B line** brought a 2048×2048, M=1, split-K 8 kernel from 4.66 to 3.85 µs (with a replayed, warm weight;
  the cost is latency and coherence, not bandwidth). Packing the per-tile counters as adjacent dwords put a whole
  launch on two cache lines and cost 1.1–2.5 µs. **Not in the current kernel.**
- A per-split marginal cost does not show the reduction: price the whole kernel.

**Constraints and correctness**
- Split-K must divide K. The current policy enumerates divisors of K from 2 to 9
  (`gemm_a16w16_policy.py:221`).
- **A 256-tile cap refuses split-K on larger grids** (`num_pid_m * num_pid_n > SPLIT_K_SEMAPHORE_MAX_LEN`,
  `gemm_a16w16_gfx950.py:1385`; `SPLIT_K_SEMAPHORE_MAX_LEN = 256`, `_utils.py:20`).
- *(field)* bf16 partials exceeded a 0.02 relative-error gate at split-K ≈ 12 (0.0241); f32 partials read 0.
  Gate every deep split-K row on accuracy, not only the first one tuned.
- Coherence scope of the partials on gfx950: `sc0 sc1` is system scope (past the MALL). *(field)* Using it for
  every partial was 5–9% slower than the narrowest correct scope; agent scope suffices across XCDs, and XCD-local
  scope suffices only when all splits of a tile run on one XCD (workgroup ids `t + s·grid_x` with `grid_x % 8 == 0`).
- **A fully co-resident grid hides a broken split-K hand-off.** *(field)* Racy variants reported error 0 on grids
  that fit the device at once. Test the reduction on a grid larger than one dispatch round.
- If you pad a grid (for XCD alignment or a full round), padding blocks must exit before they touch the
  semaphore, workspace or swizzle.

## 5. Tile, warps and pipeline

- **At M ≤ 16–32, a narrow `tile_n` (16 or 32) with K split inside the block and split-K 1** beat cross-block
  split-K on the rows compared; it needs a predicated C-store tail. *(field)* On another shape, `tile_n` 64 was
  faster than 16 and 32, so race the width.
- **Splitting the K dimension across warps needs an LDS reduction; splitting N across warps does not.** The
  number of K-warps is per shape and tracks `block_k` (at most `block_k / 32`, where 32 is the K of the bf16 MFMA atom).
  *(field)* Moving warps from K to N cut kernel time 1.0% and raised SNR from 50.8 to 94.8 dB.
- **For a row-major (N, K) B, `block_k` is the DRAM burst length of each row segment.** *(field)* 256 was best
  by 9–40% cold on long-K rows and 512 was worse; at K=2048, 128 won because 8 iterations of 256 made the
  pipeline fill too large a share.
- **Stages: deeper prefetch pays until LDS per workgroup crosses an occupancy step.** *(field)* With 2 stages
  the loop barrier drained `vmcnt(0)` every iteration; 2 → 4 stages was 1.43× at the winning tile. At a fixed
  tile, time jumped where LDS per workgroup crossed 81920 B (2 → 1 workgroup per CU). Sweep stages together
  with the tile; judging a tile at fixed 2 stages misranks it.
- **Keep B on the LDS path.** *(field)* Loading B straight to registers was 2.6× slower.
- **The MFMA side is not the lever at decode M.** *(field)* Deleting the whole hot-loop `ds_read` + MFMA moved an
  o_proj row by less than its noise; the time is the B stream.

## 6. Memory-side knobs

- **B's cache policy must be a raced, per-configuration axis.** *(field)*
  - `nt` on B was −4.4 to −9.5% on long read-once streams.
  - On small or shared B it cost 6–15%.
  - It couples with `block_k`: the same switch took one row from 98 to 159 µs at one `block_k` and another from
    153 to 83 µs at a different `block_k`.
  - A wants to stay cached: `nt` on A cost 4.5%.

  **Not in the current kernel**: it has no cache-policy axis on the global→LDS copy
  (`BufferLoadAsyncLDS128b`).
- **Stagger each block's first K-tile** (`k_start = (block_n · R) mod k_iters`, R coprime with `k_iters`).
  - With all blocks starting at k-tile 0, concurrent reads hit the same low address bits.
  - *(field)* −2.9 to −8.6% per shape on cold read-once B.
  - An R sharing a factor with `k_iters` (8 with 64 iterations) collapsed the blocks onto 8 starts and lost 55%.
  - It interacts with the XCD band below; re-sweep both after any tile change.
  - **Not in the current kernel.**
- **XCD placement at small grids.**
  - The hardware deals workgroup i to XCD `i % 8`, so with plain N-fastest order each XCD gets N-tiles 8 apart.
  - A band-cyclic remap (runs of adjacent N-tiles per XCD) removed the bimodal timings in §1 and was −2 to −3% on
    some rows *(field)*, neutral or harmful on others; race it.
  - The current kernel's `BlockSwizzle` falls back to the plain order whenever the grid has fewer than 256
    workgroups or a count not divisible by 8 (`use_simple`, `_utils.py:82`), which is exactly the small-M case.
- **Preload the leading kernel arguments into SGPRs** (`llvm.inreg` argument attributes).
  - This removes the `s_load` → `lgkmcnt(0)` round trip in front of the first B load.
  - *(field)* Kernel time −0.5 to −1.0% per shape; the decode step got about 0.45% faster.
  - **Not in the current kernel.**
- **Fetch only the real A rows when M < 16**, but only where A is a large share of the bytes. *(field)* −1.5% on
  one row, +11% on another where it made the A load wave-predicated and shortened the pipeline. Never predicate
  an LDS DMA per lane: a DMA under a partial exec mask does not land where the reader expects.

## 7. aiter's tuned table at small M

The lookup itself (exact M, then M rounded up to 16/32/64/128 by range, then the next power of two; N and K
exact) is in `../../../../framework/aiter/overall/tuning_db.md`. What it means at small M:
- **A row serves its whole padded-M interval.** *(field)* A row tuned for M=96 also served M 65–96, including a
  verify batch at M=72, and lost 2.6–3.4% of the step there. Validate a row across its interval, including M
  below the target, and add it only if it beats what the pad fallback already gives.
- **A row whose kernel name the current catalog cannot parse silently falls through** to the next candidate or
  the default. Rows written for the pre-#5145 kernel (`flydsl_gemm{stages}_..._split_k...`) do not resolve on
  current aiter (`flydsl_hgemm_..._ks{k}_w{m}x{n}x{k}_...`); re-tune instead of copying them.
- **Merging model tables keeps the row with the lowest `us`** across files written by different rulers, rewrites
  the source CSVs and raises. Edit the owning file in place.
- **Tuner filters can delete the winner silently**: a `tile_n` that must divide N, a `tile_m` cap below
  `align_up(M,16)`, a static cap on split-K tile counts. Read the catalog size for a missing shape before
  concluding "no candidate". The current policy prunes with a static model (`GemmConfigPruner`); treat it like any
  model that replaces a race (`flydsl_gemm_autotune.md` §1).
- **Tuning is compile-bound.** Compile candidates in parallel on the CPU, then time them.
- **Graph-captured decode does not see an in-process table change.** Capture after the table is final.

## 8. Correctness traps specific to small M

- **N tail:** a store vector must be wholly in or out of N (keep `n % 8 == 0` for 16 B bf16 stores), and every
  launch-shape calculation must use `ceil(N / tile_n)`.
- **Wave-predicated loads change the graded `vmcnt`:** recompute the wait from the minimum per-wave issue count.
- **An LDS-size estimator that misses a field** (a 4-byte split-K flag) lets an over-limit configuration compile
  and fail at launch with `hipErrorIllegalState`.
- **An empty error string is not a measurement:** some configurations fail to compile with an empty exception;
  record them as failures, not as slow.
- Subtractive probes must keep the accumulators live and re-derive the pipeline waits, or they also remove the
  throttle they did not mean to remove (`flydsl_gemm_attribution.md` §2).

## Related
`flydsl_gemm_grid_pricing.md` · `flydsl_gemm_epilogue.md` §5 (store cache policy) · `flydsl_gemm_autotune.md` ·
`flydsl_gemm_measurement.md` · `../../../../framework/aiter/overall/tuning_db.md` ·
`../../../../framework/aiter/skills/optimize/aiter_levers/aiter_flydsl_libtype.md`
