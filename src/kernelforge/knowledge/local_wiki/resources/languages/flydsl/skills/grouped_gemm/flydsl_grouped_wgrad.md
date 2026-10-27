---
title: FlyDSL grouped GEMM — variable-K wgrad
kind: language
lever: flydsl_grouped_wgrad
gens: [gfx950]
dtypes: [bf16, fp8, mxfp8, mxfp4]
regimes: [wgrad, variable-k, skew, split-k, beta-accumulate]
updated: 2026-10-27
sources:
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp8_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/utils/gemm_helper.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/protocol.py
---

# Variable-K wgrad in FlyDSL

> The weight gradient of a grouped GEMM is `out[g] = A_gᵀ · B_g`, where `A_g` and `B_g` are the rows
> `[offs[g], offs[g+1])` of `A [M_total, OUT_M]` and `B [M_total, OUT_N]`. Every group has the same number of
> output tiles; the contraction length of a tile is its group's token count. This card covers what that changes:
> the loop, the K tail, MX scale timing, dispatch order, short and deep groups, split-K, beta=1 accumulation and
> the output layout. "FP8", "BF16", "MX8", "MX4" are Primus-Turbo's `grouped_gemm_{fp8,bf16,mxfp8,mxfp4}_kernel.py`
> and "GH" its `utils/gemm_helper.py`, under `primus_turbo/flydsl/` at main `ca1fdd67`. FlyDSL facts are v0.3.2.
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as targets.

## Route here when
- wgrad is the slowest pass of a grouped backward, or the only one that degrades under token skew.
- You are writing a K loop whose trip count is a run-time per-group value.
- One deep group, many tiny groups or empty groups dominate a skewed launch.
- wgrad accumulates into an existing gradient buffer (beta=1), or its output layout differs from its operands'.

**Somewhere else if:** XCD order and longest-first across all passes → `flydsl_grouped_scheduling.md`; tile and
boundary bodies → `flydsl_grouped_tile_pricing.md`; scale packing and quantization → `flydsl_grouped_lowprec.md`;
race and cache mechanics → `flydsl_grouped_host_autotune.md`; gates → `flydsl_grouped_correctness.md`.

## 1. Why the backward's gap sits in wgrad
- Forward and dgrad parallelize over M: total M is fixed, so skew moves tile counts, not the cost of a tile.
  wgrad is the one pass whose per-tile cost follows the token distribution. *(field)* With forward and dgrad
  byte-identical between two builds and robust to skew (skewed/balanced 0.92–0.99), a tensorwise fp8 wgrad ran a
  one-hot distribution at 0.34 (down projection; 0.66 on gate_up) of its balanced speed.
- *(field)* On MXFP8 at 32768 total tokens, wgrad took 1.16–1.35× the time of the dense wgrad with the same FLOPs
  (+200–270 µs), the largest structural gap of the backward. Memory stalls were ≈0, occupancy, VGPR and LDS were
  the same (one workgroup per CU at 128 KB LDS); wall-averaged MFMA utilization was 45.5% against 61%. The gap is
  MFMA idle time at group boundaries (per-group prologue and epilogue, pipeline restarts), not bandwidth.
- Layout. The token axis is the contraction axis, so the tensorwise/bf16 operands `[M_total, OUT]` are K-row on
  both sides and both go through transpose reads, which cost more LDS instructions per MAC than K-contiguous
  reads (`../gemm/flydsl_gemm_datapath.md` §9). MX wgrad instead takes column-wise quantized operands
  `[OUT, M_total]`, because the E8M0 blocks run along the tokens; its loads are K-contiguous (MX8:1459-1460), at
  the price of §3.

## 2. A K loop with a run-time trip count
Two forms are in use.
- **Runtime chunk loop.** An outer `range(n_chunks)` becomes an `scf.for` with a run-time bound; the inner
  `range_constexpr(CHUNK)` unrolls the distance-2, 4-buffer pipeline body. Keep CHUNK even so the Python-level
  cur/next buffer names are back in their starting roles at every chunk boundary (MX8:1428 asserts it; BF16
  uses CHUNK=4, BF16:277-343; FP8's masked kernel 8, FP8:2264-2300). The remainder either runs on zeros from the
  descriptor (§3) or, in MX8, as a `range_constexpr(chunk)` tail whose steps sit under a wave-uniform
  `if k_abs < k_iters` (MX8:1574-1615). *(field)* Chunk sizes 8, 4 and 2 were all neutral on a tensorwise masked
  wgrad: rounding up to a chunk is not where skew costs.
- **Whole loop with a run-time count.** One inline-asm region (`../gemm/flydsl_gemm_mainloop.md` §3) takes the
  trip count in an SGPR. FP8's 4-wave wgrad runs the largest multiple of 6 K blocks (one pass of its pool
  depths) and the 0–5 remainder as a tail fused into the same asm block; a group shorter than one pass runs only
  the tail (`nval_can_be_zero`, FP8:2797-2798, 3434-3437). MX4 passes the raw 256-block count (MX4:1887).

**What a runtime loop body may contain (v0.3.2).** The rewriter turns a runtime `range` in a `@flyc.kernel`
body into a body function plus `scf.for` and infers the carried state: every pre-loop name the body rebinds *and*
every pre-loop name it calls a method on (`_collect_assigned_vars`, `compiler/ast_rewriter.py:156-222`). Each
carried name must flatten to IR values, i.e. be a DSL value or a list/tuple/dict/SimpleNamespace of them
(`compiler/protocol.py:97-108`); a helper object such as a loader raises `TypeError: Cannot extract IR values`.
- Keep the body to calls of plain functions plus rebinding of DSL values. MX8's chunk body is one module-level
  function that returns the eight swapped LDS buffer handles, and the loop carries only those (MX8:1571-1606);
  BF16 hoists its tile bodies into free functions "for the same ast-rewriter reason" (BF16:429).
- Accumulators in register memory (`fx.make_rmem_tensor`) are updated in place and are not carried.
- BF16 emits its chunk loop through a helper, `emit_for`, that calls the rewriter's `scf_for_dispatch`
  (`compiler/ast_rewriter.py:1006`) with no carried state (GH:2134-2143); the body is then a plain Python closure,
  and its Python-level buffer rotation is resolved at trace time.
- The three loop forms and what each carries: `../bottleneck/debug-flydsl-kernel.md` § The three loop forms.

## 3. K tails: let the descriptor zero them, except where it cannot
- **Token-outer operands (tensorwise, bf16).** Fold `m_start · OUT` into the 64-bit descriptor base and bound
  `num_records` to `max(M_g · OUT − tile_offset, 0)`: reads past the group's last row return zero, so the last
  chunk or tail needs no predicate (`_wgrad_rebase`, FP8:2072-2085; FP8:3443-3451). The clamp at zero is what
  keeps an empty group legal (`flydsl_grouped_group_mapping.md` §8).
- **Token-inner operands (MX, `[OUT, M_total]`).** A record count bounds the whole row band, not a row's token
  range, so a tile that runs past its group reads the next group's real tokens and accumulates them. Pad each
  group to the K block (128 tokens in MX8) so the step count is exact (MX8:1481), and guard the last chunk at
  run time (`k_abs < k_iters`). Only an unbalanced gate sees a missing guard: tile-aligned groups never over-run
  (`flydsl_grouped_correctness.md` §1).
- Over-reading past K is legal only when every term it adds is exactly zero: `../gemm/flydsl_gemm_mainloop.md` §6.

## 4. MX: the lever is when the scales are loaded
- *(field)* MXFP8 variable-K wgrad loaded the A/B scales in the same step as the MFMAs that use them, which put
  the scale `ds_read` latency on the MFMA path. Prefetching them one step ahead, as the dense NT kernel does, took
  four shapes from 930/1047/530/307 to 851/936/478/274 µs (about 9%). Keeping the accumulators in SSA alone gained
  0; the scale prefetch alone gave all of it. Form: a chunk-local body with SSA accumulators and per-step scale
  prefetch, touching register memory only at chunk boundaries (`_wgrad_ssa_chunk`, MX8:166).
- Scale loads must be few and wide. *(field)* A whole-loop MXFP8 wgrad that fetched scales with 16 `buffer_load_ubyte`
  per phase into 16 scattered VGPRs ran 811 µs against 479 µs for the chunked kernel. Removing those loads alone
  gave 448 µs; cutting the scale registers from 16 to 2 gave 506 µs. Byte loads are issue-bound and prefetching
  does not hide them. Pack four E8M0 per dword, per group: `flydsl_grouped_lowprec.md` §3.
- The scaled MFMA instruction itself is free (`flydsl_grouped_lowprec.md` §4).

## 5. Dispatch order: wgrad tiles are not equivalent
In NT every tile of a launch costs the same and skew changes how many tiles a group has. In wgrad every group has
the same tiles and skew changes what each costs. The mechanics of XCD orders are in
`flydsl_grouped_scheduling.md`; what follows is specific to wgrad.
- **Never slice the tile list contiguously across XCDs.** That hands whole groups to one XCD
  (`flydsl_grouped_scheduling.md` §1).
- **Keep the hardware round-robin (`bid % 8`, `num_xcd = 1`)**, which spreads every group evenly over the XCDs,
  and take L2 reuse from a `group_m`/`group_n` band inside a group. Turbo's tensorwise candidates are all
  `num_xcd = 1` ("xcd=1 keeps group-major LPT order", FP8:4654-4656); its XCD-affine variant keeps the hardware
  split and sizes the super-block to one group "so every XCD holds the same fraction of every group's K"
  (FP8:3897-3902). *(field)* On a tensorwise wgrad at G=32, `num_xcd=2` was 6.94% faster on a balanced load and
  22.07% slower under skew; `num_xcd=8` +3.72% and −42.84%.
- **Group-major order is longest-first only if the token histogram decreases with the group index.** Choose the
  walk direction or rank the groups on the device from the group table (`flydsl_grouped_scheduling.md` §5).
- **Group-interleaved band order** (band b of every group, then band b+1) is a different mechanism from the XCD
  band-cyclic remap. It helped under skew at the hardware XCD order (`num_xcd = 1`) and caused the skew collapse
  on top of a contiguous XCD remap (`flydsl_grouped_scheduling.md` §3).
- **Do not copy NT dispatch parameters or defaults to wgrad** (`flydsl_grouped_scheduling.md` §1). Each path gets
  its own candidates and constants.

## 6. Short contractions turn store-bound
- The output `[G, OUT_M, OUT_N]` has a size set by G alone: an 8-token group writes as many bytes as an
  8192-token one. *(field)* MXFP4 wgrad, G=32, constant total tokens: from balanced to a 159:1 skew, the ratio to a
  large-M reference fell from 0.94 to 0.22 (~3800 → ~870 TF/s); forward and dgrad were immune.
- Switch blocking for short contractions and store C non-temporally (`_GMXFP4_WGRAD_CFG_SHORT`, MX4:1124).
  *(field)* gate_up at 159:1 skew 916 → 3969 TF/s, at 36:1 1288 → 4002, at 10:1 2133 → 3971; the reference cell
  rose too.
- **The crossover scales with tiles per group.** *(field)* Sweeping 2048–32768 tokens per group at G=32, the
  short/regular crossover was 24576–32768 for a 276-tile-per-group shape and 12288–16384 for a 144-tile one: about
  85–120 tokens per output tile in both (the sweep brackets the crossover). Gate on
  `M_total / G > c · tiles_per_group`, not on a token count (MX4 uses a fitted `_GMXFP4_WGRAD_SHORT_MG = 8192`,
  MX4:1152, selector MX4:2083-2091).
- **The skew tax that remains.** *(field)* Tensorwise wgrad, G=32, 131072 tokens (4096 per expert on average),
  one expert with 97–99.8% of them, split-K active: the main kernel still paid +159 µs (gate_up) and +114 µs
  (down) over balanced; skewed time stayed 1.13–1.20× the balanced time. It comes from 31 eight-token groups
  that each compute a full output, plus one group that eight pieces cannot spread over 256 CUs.

## 7. Two K-loop bodies: choose by per-group M, not by M_total
- *(field)* Tensorwise wgrad with a masked chunked body and an exact-trip body (2-stage prefetch, runs exactly
  `M_g`): at `M_g = 512` the exact body was 60% faster (856 → 1369 TF/s); it won by 3–20% at `M_g ≤ 1024`, the
  two were within ±4% between 1024 and 2048, and masked won by 5–15% at `M_g ≥ 2048`.
- Gate on `M_total / G`. *(field)* A gate on `M_total` sent G=8, `M_g`=512 (`M_total` 4096) to the masked body,
  where the exact body was 13% faster.
- A single whole-loop body with a run-time count and a fused tail (§2) removes the choice; FP8 uses one 4-wave body
  for every length and keeps the masked kernel as the fallback for 64-bit traversal and for a
  candidate that fails to build.

## 8. Four waves beat eight on per-tensor fp8 wgrad
- *(field)* Tensorwise fp8 grouped, 28 points (4 token counts × 7 shapes): the 4-wave whole loop (256 threads, one
  wave per SIMD, 256×256 tile, AGPR accumulators) won 28/28 on wgrad (+3–17%), 6/28 on dgrad and 1/28 on the
  forward NT. At 1024 tokens per group it still beat the exact-trip 8-wave body on 5 of 7 shapes and the masked
  one on 7 of 7: keep short contractions in the 4-wave race.
- Why: both operands are transpose-read, and a 128×128 block per wave issues fewer reads per MFMA. *(field)* 2647
  against 2204 TF/s on one shape, the 8-wave kernel at two waves per SIMD; bank conflicts were 2.4% and memory
  stalls 0.05%: the limit is transpose-read feed per MFMA, which more waves on one LDS read port make worse.
- Flat or negative on that kernel *(field)*: more waves per SIMD, smaller tiles, store-defer (−18.6% once it
  cost LDS), plain `ds_read_b64` instead of transpose reads (0), wider epilogue stores (−2 to −7.7%), cross-tile
  overlap (−5%), epilogue double buffering (−1 to −3.5%), swizzle × XCD × `vmcnt` sweeps (<2%), persistent vs
  one tile per workgroup on balanced loads (±2%; under heavy skew one tile per workgroup won,
  `flydsl_grouped_scheduling.md` §6).
- What paid *(field)*: both B pools three buffers deep instead of only one of them, +2.1–3.8%. The partial drain
  must count the in-flight fills of every deep pool (`../gemm/flydsl_gemm_correctness.md` §4). Also the wave
  bank swizzle (`../gemm/flydsl_gemm_datapath.md` §2) and the lgkm-side partial drain
  (`../gemm/flydsl_gemm_schedule_and_locality.md` §4).

## 9. Deep groups: deterministic split-K
Under skew one group's contraction outlasts the rest and the launch ends with one window running while the other
CUs idle. The device is full, so the dense wave-quantization gate (`../gemm/flydsl_gemm_grid_pricing.md` §2) never
fires; the trigger is the deepest group. Determinism by fixed-order reduce is the same idea as
`../attention/flydsl_attn_backward.md` §1.
- **Decide on the device, from `group_offs`.** FP8 fixes the piece count from the static shape (tiles per group
  against CUs; `_wgrad_split_geom`, FP8:3688-3704) and cuts a group only if its own contraction clears a bar
  (`_wgrad_is_cut`, FP8:3747). A variant chose one window and its slice count S at run time. *(field)* That
  variant gained 7.57% on a six-cell score (two projections × balanced, moderate and heavy skew) of a tensorwise
  wgrad; the dispatch layer it added had to win its host cost back (`flydsl_grouped_host_autotune.md` §2).
- **Layout.** The piece that starts at K=0 writes C; every later piece writes a scratch band at C's pitch, so the
  store only swaps the descriptor base (FP8:3425-3433). A reduce kernel launched from the same `@flyc.jit` stub, on
  the same stream, folds the bands into C in ascending chunk order, in fp32, with no atomics: bit-reproducible
  (`kernel_grouped_tn_wgrad_reduce`, FP8:4450; launch FP8:4617). Its grid is the compile-time tile space, and a
  launch that cut nothing exits at once. *(field)* ~4 µs on a balanced load, 46–75 µs when it folds.
- **Scratch for CUDA graphs.** Keep it persistent per (shape, device) and size it for the deepest geometry any
  candidate can pick, so replay sees fixed addresses (`_wgrad_split_ws`, FP8:3758-3775).
- **What it buys.** *(field)* One expert with 99.8% of the tokens: gate_up 3024 → 2389 µs (21% less time), down
  1381 → 1298 µs (6%); balanced unchanged (down) or 2.7% faster (gate_up). BGrad SNR 52.8 dB with the reduce live.
- **Every output format keeps the split.** Tight C and an fp32 accumulator need bands at C's pitch and dtype and a
  reduce vector the store can issue: 8 fp32 lanes are 256 bits, beyond `buffer_store_dwordx4`, so FP8 halves the
  vector for fp32 (FP8:3989).
- **Granularity.** *(field)* Eight pieces were best; 16 were slower (down 1289 → 1428 µs); piece counts that are not
  powers of two break the power-of-two chunk geometry. Eight reduce workgroups per tile instead of four: ~0.3%.
- **A run-time divisor needs full coverage.** When the slice count is chosen at run time and the division uses a
  precomputed reciprocal table, the table must cover every S the policy can reach, and its exactness check must use
  the true dividend range. Assert the coverage at compile time; the gate that catches a gap is in
  `flydsl_grouped_correctness.md` §9.

## 10. beta = 1: accumulating into the gradient buffer
- **The read-back is a physical cost and scales with the accumulator dtype.** *(field)* Tensorwise 4-wave wgrad,
  one square tight-C shape, time beta=0 → beta=1: fp32 0.842 → 1.020 ms (21% more time), bf16 0.854 → 0.933 ms
  (9% more), fp16 0.847 → 0.907 ms (7% more). An fp32 accumulator is read at twice the bytes of a 16-bit one.
- **Do not move the read-back into the main loop at one wave per SIMD.** One quadrant's read-back is 64 fp32
  VGPRs held across the loop. *(field)* Hoisting one and then two quadrants raised the fp32 read-back cost from
  21% to 115% and 177% more time, bf16 from 9% to 37% and 142%: spills. Pipeline it in the epilogue
  (`../gemm/flydsl_gemm_epilogue.md` §7).
- **Fused still beats beta=0 plus a separate add.** *(field)* The add alone took 0.79 ms (fp32) and 0.27 ms (bf16),
  against 0.18 ms and 0.08 ms of read-back inside the fused kernel.
- **Ranking: two measured facts.** *(field)* Timing every candidate at both betas on a tensorwise wgrad (two
  projections × G = 24 and G = 32), the beta=0 winner differed from the beta=1 winner in two of four cases, both
  at G = 24 (1.1% and 0.5% lost at beta=1). A race at beta=0 picked the worse config in 4 of 6 runs on one case;
  ranking at the caller's beta picked the better one 6 of 6, at one extra compile per surviving candidate (cold
  autotune about doubled). The other side: ranking at the caller's beta lets a beta=0 caller and a beta=1 caller
  run different configurations of one shape. Which a race uses is the owner's choice. FP8 ranks at the caller's
  beta and keeps its numeric gate at beta=0 (FP8:4934-4945); race mechanics are in
  `flydsl_grouped_host_autotune.md` §6.
- beta can change code paths beyond the epilogue: the empty-group zero pass exists only at beta=0 (§11).

## 11. Empty groups
- Store zeros and skip the fetch pipeline. BF16 runs its prologue only when `group_tokens > 0` and then stores the
  zero accumulators (BF16:256-275); in MX8 a zero step count leaves the loop empty and the epilogue stores zeros.
- With the split active at beta=0, FP8 stops dispatching empty groups (FP8:4408) and zero-fills them flat in the
  reduce pass (`_zero_fill`, FP8:4558; `_ZERO = _SPLIT and not beta_is_one`, FP8:4001). As a GEMM tile body the
  zero store has no MFMA to hide under; Turbo's note puts that burst at ~0.27 µs/MB against a 0.17 µs/MB write
  roofline (FP8:3996-4000).
- At beta=1 an empty group's output is left as it was, so FP8 keeps its tiles in the GEMM. Descriptor clamps for
  empty groups: `flydsl_grouped_group_mapping.md` §8; gates: `flydsl_grouped_correctness.md`.

## 12. Output layout
- **Decouple C's pitch from the operands'.** Operands stay padded for the fill addressing; C gets the real
  extents (FP8's `m_real` / `n_real` / `c_tight`, FP8:3851-3853). Every C addressing site moves together: the
  store's row pitch, its row bound `(g + 1) · C_M`, the split band row, the reduce's element offset and column
  predicate, the C and scratch descriptors, the host allocation of output and scratch, and the cache key. A
  fallback that can only write the padded pitch must refuse a tight C (FP8:4826-4828). *(field)* Missing one
  site, the split band row, gave a relative error of 1.42 and non-reproducible output.
- **Pad free dims in the producer.** A free dim that is 64 mod 128 elements leaves every other 1-byte operand row
  straddling two 128 B lines (`../gemm/flydsl_gemm_datapath.md` §10) and the last output tile ragged. Padding
  that dim in the quantizer and slicing the result is byte-correct for any pad value, since the pad lands in
  sliced-off output columns, and lets the compile prove the per-column store bound dead (`_COL_SAFE`,
  FP8:3949). *(field)* +3%.
- **`trans_c` is a different store path.** *(field)* bf16 grouped wgrad writing `[G, N, K]` through the
  transposed store (BF16:350-369) instead of `[G, K, N]`: 1293 → 1010 TF/s (22% less throughput) on gate_up; on
  down the transposed store matched a vendor library that ran both layouts at one speed. Benchmark the layout
  each caller requests.

## Verify
- The gate includes skewed, one-hot and empty-group distributions and a large-M shape that forces the deepest
  split branch, compared with an fp32 reference, not only bitwise run-to-run.
- Dispatch order and `num_xcd` were judged on skewed cells as well as balanced ones.
- The runtime loop body carries only DSL values; the MX last chunk is guarded.
- beta=1 was timed at the accumulator dtype the caller uses, with the read-back in the epilogue.
- A tight-C or fp32 output reaches the same split path as the default output.

## Related
`flydsl_grouped_scheduling.md` · `flydsl_grouped_tile_pricing.md` · `flydsl_grouped_lowprec.md` ·
`flydsl_grouped_host_autotune.md` · `flydsl_grouped_correctness.md` · `flydsl_grouped_measurement.md` ·
`../gemm/flydsl_gemm_mainloop.md` · `../gemm/flydsl_gemm_grid_pricing.md` · `../gemm/flydsl_gemm_epilogue.md` ·
`../attention/flydsl_attn_backward.md` · `../bottleneck/debug-flydsl-kernel.md`
