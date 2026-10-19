---
title: FlyDSL GEMM — per-shape autotune that cannot regress, and a launch path that costs nothing
kind: language
lever: flydsl_gemm_autotune
gens: [gfx950]
dtypes: [fp8, mxfp4, mxfp8, bf16]
regimes: [prefill, training, cuda-graph]
updated: 2026-10-19
sources:
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_fp8_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp4_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp8_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/utils/gemm_helper.py
---

# Per-shape autotune and the launch path

> No single FlyDSL GEMM configuration is best across shapes. Two loop designs, tile shapes, L2 bands, store
> policy and split-K all change sign between shapes (`flydsl_gemm_mainloop.md` §1,
> `flydsl_gemm_epilogue.md` §5). Turbo's dense GEMMs therefore race a short candidate list on the first call
> for each shape and cache the winner. "FP8", "FP4", "MX8" and "GH" are the files under
> `primus_turbo/flydsl/{gemm,utils}` at Primus-Turbo main `7615a5a2`. Numbers marked *(field)* are measured
> values. Use them for direction and order of magnitude, not as targets.

## Route here when
- You are adding a configuration axis, or a second kernel design, behind one entry point.
- A benchmark gives a bimodal result for one shape across runs.
- Host launch time shows up next to a short kernel, or a kernel behaves differently under CUDA-graph capture.

## 1. The first-call race (FP8 `_dense_race`, :3040)

For each candidate `(cfg, factory)`:
1. Build it. Run it once on scratch output. Drop it if it raises, or if the first 1024 outputs are not
   finite. A geometry that does not hold for the shape simply leaves the race.
2. Run 200 throwaway launches of the leading candidate, so the clock is up before anything is timed
   (`_PICK_RAMP_ITERS`, FP8:80).
3. Time every candidate over forward and reversed passes, three round trips, and keep each candidate's
   minimum (`_pick_dense_candidate`, FP8:3067).
4. Cache the winner per static shape key. Call `torch.cuda.empty_cache()` to release the race buffers.

**Keep the race; do not replace it with a model.** *(field)* Replacing the TN race with "sort by an
analytic model, take the first" left the scored shapes unaffected, because the model ranked them right.
Off the scored set, wgrad shapes lost 13–14%. Restoring the race recovered every cell. A static
analysis is a cheap way to find suspect shapes; only a measurement decides. Any change that replaces a race
must be validated on shapes outside the scoring set.

**Keep the candidate list short.** Every candidate is compiled on the first call: "a race compiles every
entry, so the list is what first-call latency costs" (FP8:3188-3191). Turbo's NT 8-wave tables are four
wide, or two wide when whole-loop geometries join the race (`_NT_CANDIDATES`, FP8:3193-3198). Add a
candidate only after it has been shown to win somewhere. *(field)* A candidate that never wins on the
deployment shapes only costs compile time.

## 2. Never regress: the incumbent first, challengers win by a margin

- List the heuristic or incumbent configuration first. A challenger must beat it by a margin larger than
  the race's own noise. Turbo's margins:
  - whole-loop depth: 2% (`_WL_MARGIN = 1.02`, FP4:3274);
  - speculative swizzles: 0.5% (`_SWZ_MARGIN = 1.005`, FP4:3278);
  - split-K arms: 1% (`_race_mxfp4_ksplit(margin=0.99)`, FP4:3557);
  - MXFP8 C-store policy and split: 1% (MX8:973-976).
  These are measured values for those kernels. Derive your own margin from your race's spread.
- **Pure tile-order changes are safe to add.** A bijection on tile ids cannot change the result, so a
  swizzle candidate needs no correctness re-check (FP4:3182-3186).
- **Every extra candidate is another draw from the race's noise.** Add the margin before widening the
  list. *(field)* One candidate added without a margin cost 0.19%.
- **The first list entry is a decision.** With a margin, a challenger that wins by less never lands.
  *(field)* When no point on an axis is best everywhere, starting the list at one end of the axis let the margin
  chain step along it. Starting in the middle stranded it, because neither neighbour could clear the
  margin. Do not extend the ladder past the measured optimum.
- **Race noise above the margin makes the winner flip between runs.** *(field)* One shape alternated between two
  configurations about 3% apart. Clearing the cache and re-running the same shape showed a 2:1 split. The
  cause was the measurement order, not thermal noise.
  - Time base and candidate inside one window, as a palindrome `(base, cand, cand, base)` per rep, and take
    the median of `(c1+c2)/(b1+b2)` (`_robust_ab_ratio`, GH:2091). The DVFS ramp is odd in the slot index,
    so it cancels within a rep.
  - *(field)* With the palindrome, 48 of 48 re-runs picked the same configuration.
  - Check stability itself repeatedly. One stable check is not evidence; use at least 6 cleared-cache
    re-runs and report every adopted configuration.
- **Score at the deployment M.** *(field)* An arm that was +2.5–2.9% at the deployed M scored −0.94% and +0.54% at
  the race's two canonical M points, so the race never picked it. When one shape family is known, give it a
  static lead: reorder its candidate list under a predicate that matches only that family. Keep the race
  running, so the lead can still be beaten. Verify the lead is non-negative on every M the predicate
  covers.
- **Hard-coding beats a race that scores the wrong point.** *(field)* With a small candidate list and shape families
  that can be measured offline once, fixing the choice per family gained more than fixing the race did.
  Do not let two dispatch paths share one default constant.

## 3. Staged searches

Search one axis family at a time, holding the earlier winners, instead of a full cross product:
- MXFP8 (`_autotune_mxfp8`, MX8:865-981): `(BLOCK_M, GROUP_M, num_xcd)` at the seed band width; then the 2D
  N-band width, adopted only above 1.5%; then the C-store cache policy; then the split plan.
- MXFP4 (FP4:3170-3377 and the dispatch around it): swizzle × whole-loop depth; then the epilogue
  variants; then the launch mode (plain, uniform split, tail split over M or N). Launch modes are timed end
  to end, including the host reduce.

When a search stalls, change the family of the dimension you search (split-K, MFMA shape, cache modifier,
`waves_per_eu`, scale prefetch). Do not re-sweep the same tile grid.

## 4. Representative probe data

- **Time on production-magnitude data.** fp8 wall time depends on the data's magnitude through power and
  clock. "An all-zero probe would misrank candidates" (Turbo comment). Zero operands are a false ruler:
  *(field)* 155 µs against 205 µs for the same build and shape.
- **Rank at the caller's beta.** Race beta=1 configurations on scratch and rebuild the winner with the
  accumulate epilogue (`flydsl_gemm_epilogue.md` §7).
- **Check numerics, not only finiteness, for candidates that change arithmetic order** (split-K, different
  atoms). Compare each against the base configuration's output.

## 5. Cache keys

- **The key holds every input that changes the compiled code or the candidate list, and nothing else.**
  Never key on `id(tensor)`. A freed tensor's id is reused, so the cache returns another tensor's result.
  Turbo's compiled-launch key is `(shape, stride, dtype)` per tensor plus integer arguments; strides are in
  because the compile pins the operand layout (FP8:2988-3005).
- **A missing discriminator gives a wrong answer, not a slow one.** Turbo's MXFP4 artifact key carries
  `row_bytes` (one logical shape, two allocations, two kernels) and `scales_prepacked` (the two builds
  differ in whether the launch runs the scale preshuffle) beyond the shape (FP4:4518-4519, FP4:4650-4651).
  Leave either out and one call is handed the other's module.
- **A value used as a compile-time constant is part of the specialization.** Pass truly dynamic values as
  `fx.Int32` arguments. Closure-captured Python dicts are *not* in FlyDSL's JIT cache key
  (`../bottleneck/debug-flydsl-kernel.md` § What the cache key sees).
- Build knobs that select a code path (tile order, store policy) must be in the key. Otherwise two policies
  share one cache slot and silently run as one binary.

## 6. The launch path

- **Call the `flyc.compile` object, not the `@flyc.jit` function.** The jit call re-binds the signature,
  rebuilds the cache key and re-checks globals on every launch: 180 µs against 37 µs for a ~24-argument
  launch (`run_compiled`, GH:2049-2069). Compile once per key with `_static_layout`
  (`flyc.from_torch_tensor` per tensor, FP8:2973) so the object does not re-read shapes and strides.
- **Skip no-op tensor ops on the hot path.** A no-op `.to()` or `.reshape()` still costs a dispatch
  (`_scalar_scale`, FP8:3098-3105). Turbo reads the stream through `torch._C._cuda_getCurrentRawStream`
  (FP8:2970).
- **Under CUDA-graph capture: no timing, and call the raw closure.** A race cannot time inside capture, so
  fall back to the static heuristic configuration. Launch the raw `@flyc.jit` closure, because "a
  flyc.compile'd object regresses under CUDA-graph capture" (`_run_dense`, FP8:3081-3091). Warm up and
  synchronize the kernel with the same constexpr key before capture starts.
- **An adaptive dispatch layer adds host cost.** *(field)* Adding a runtime policy layer brought back the per-launch
  JIT overhead, and moving the entry to the compiled object recovered +6.5%. Re-measure the host path after
  adding any dispatch logic.

## 7. Failures the race must not hide

- **`except Exception: continue` turns a build bug into "no working config".** A `NameError`, a signature
  mismatch or a key-tuple mismatch looks exactly like an incompatible configuration. *(field)* This has removed a
  whole optimization axis three times while every benchmark stayed green. In one case, a launch mode never
  built at all.
  - Record `(candidate, repr(exception))` in a module-level list that a verbose path prints.
  - Add a smoke test that builds every launch mode once without timing.
  - Assert the derived knobs (tiles per workgroup, split factor, block_n) on one host-known shape.
- **A backend that raised once is disqualified for that shape.** Do not retry it on later calls.
- **A timing defect in the race can be load-bearing.** It may compensate for a cache-state difference
  between the race and the scored step. *(field)* Fixing the race's warm-up and ordering lost 0.68%. Change race
  timing only when the race and the scorer measure the same way.

## Related
`flydsl_gemm_mainloop.md` · `flydsl_gemm_epilogue.md` · `flydsl_gemm_measurement.md` (rulers, null controls) ·
`flydsl_gemm_correctness.md` · `../bottleneck/debug-flydsl-kernel.md`
