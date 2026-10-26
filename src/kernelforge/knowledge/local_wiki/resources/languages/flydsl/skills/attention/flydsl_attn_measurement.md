---
title: FlyDSL attention — measuring a change: steady state, rulers, zero points, FLOP accounting, host-bound shapes
kind: language
lever: flydsl_attn_measurement
gens: [gfx950]
dtypes: [bf16, fp8]
regimes: [prefill, backward, decode, kernel-A/B]
updated: 2026-10-26
sources:
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1633
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1651
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1662
---

# Measuring an attention kernel change

> The shared protocol — ruler choice, arm order, warm-up and first-arm bias, noise floors, statistics, closing a
> series of changes — is in `../gemm/flydsl_gemm_measurement.md` and
> `../../../../common_methodology/profiling/measure_protocol.md`. This card is what attention adds: small
> workloads, multi-cell scores with guards, workspace-heavy backward cells, FLOP conventions, and launch-bound
> decode kernels. Numbers marked *(field)* are measured values; use them for direction and order of magnitude.

## Route here when
- Setting up a benchmark or scorer for an attention kernel, or reading one you inherited.
- A delta is under ~1%, a cell is noisy, or two rulers disagree.
- Reporting TF/s for a causal, windowed, fp8 or sparse kernel.

## 1. Hot steady state
- **Never sleep between repetitions on a small workload.** *(field)* A B=1 backward read 1024 TF/s cold,
  727–969 with `sleep(0.3)` between calls, and a different value again under continuous load: a 40% swing that
  buried a 5–14% gap.
- Warm with back-to-back calls and no sleep until the clock settles (60–90 s from an idle card,
  `../gemm/flydsl_gemm_measurement.md` §3), then take the median of ≥7 back-to-back repetitions.
- B=1 cells also drift with sweep order: *(field)* one configuration read 482 TF/s inside a sweep and 759 when
  run alone. Decide on a cell large enough to hold the clock steady (B=4 on that kernel) and isolate sweeps (§4).

## 2. The bench mirrors the deployment
- **Each arm at its own best configuration** (`../gemm/flydsl_gemm_measurement.md` §7). The attention form of the
  trap: a bench that calls the kernel builder with its defaults while the deployed launcher passes other values
  (`block_kv`, `waves_per_eu`, LSE folding). *(field)* The two rulers differed by 1.3–1.4% on the same code,
  and the bench was optimising a configuration that did not ship.
- Make the builder's defaults equal the deployed values instead of relying on every bench to pass them. When a
  builder signature changes, update every bench that calls the builder directly.
- A ruler fix moves the score; report it separately from kernel speed-ups so history stays comparable.

## 3. Zero points and guards
A multi-cell score usually carries guard cells: cells the candidate does not change, scored as `min(ratio, 1)`.
- **The guard geometric mean is the session's zero point; use it.** *(field)* Seven runs of one scorer on
  byte-identical guard code read +0.52 / +0.35 / +0.87 / −0.26 / −0.94 / −0.20 / +0.15%: a 1.8% swing, same sign
  across three independent shapes, so a per-session slot bias that also hits the target cells. Decide on
  `raw / guard_geomean`; it reproduced an independent re-measurement within 0.26% where the raw score ranked two
  configurations backwards.
- The corrected value carries about ±0.5% *(field)* of its own noise: a difference under ~0.8% needs two independent
  sessions agreeing in sign before you keep it.
- **A hard cap is a one-sided tax.** *(field)* One attention guard cell spread 3.2% on identical code; capped at
  1.0 it could only subtract, and caught no real regression in 17 rounds. Set the cap tolerance at or above the
  measured spread of the guard cell.
- **Check for two modes before choosing an estimator.** *(field)* A 0.57 ms sliding-window cell, 24 processes on one
  binary: 0.567–0.574 or 0.634–0.642 ms, nothing between (the high mode 11.4% slower). For a bimodal cell use the
  minimum over ≥12 processes; a mean or median moves with the fraction of runs that land high, and a guard in the high
  mode costs the score ~1.5% by itself.

## 4. Isolation and arm order
- **One process per backward cell.** Each cell allocates gigabytes of dQ workspace, and allocator fragmentation
  sets the next cell's time. *(field)* The same cell read 11.23 and 6.64 ms depending on what ran before it, and
  `empty_cache()` did not help; forking a process per cell cut the cell swing from 69% to 0.29–2.50% and the
  score floor from 2.6% to 0.18%.
- Arm order and when a palindrome applies: `../gemm/flydsl_gemm_measurement.md` §2 (palindromes only for two
  module instances interleaved in one process; one process per arm runs A B).
- **Ratio rulers between two arms that share a body do not credit body changes.** For `bf16_ms / fp8_ms` with one
  shared backward body, an optimisation of that body speeds up both arms and the score does not move. Only
  changes that act on one arm count; fork the body per arm (separate cache-key entries) when that is intended.
- **Expect an in-process kernel ruler to read larger than the end-to-end score.** *(field)* +1.3% in-process
  against +0.8% on the full bench for one forward change; the in-process ruler omits fixed per-call costs.

## 5. FLOP and byte accounting
- **The numerator is the mathematically required work, not what the hardware issues.** Forward:
  `4·B·Hq·Sq·Skv·D × attended fraction`. Backward: 2.5× the forward (5 GEMMs). For square causal,
  `10·B·Hq·S²·D·(1 − (S−1)/(2S))`; for a `(W, 0)` window the attended fraction is about `(W+1)/S`. Rectangular
  and windowed shapes need the real attended fraction, not the square-causal factor.
- Work an implementation issues beyond the convention is not credited: a deterministic two-kernel backward
  issues 7 GEMM-equivalents for the convention's 5 (~29% of MACs) *(field)*. Use the gap as the redundancy
  ratio of `flydsl_attn_workflow.md` Step 3, not as throughput.
- **Lower precision shows up as a smaller denominator with the same numerator.** Changing the numerator for an fp8
  arm is cheating. Produce every quantised byte inside the timed region: *(field)* with Q/K quantisation moved
  into the clock, an fp8 backward that had "won" three times read 0.58× of bf16.
- **A gather-bound kernel is priced in bytes, not FLOPs.** *(field)* A sparse-MLA prefill at 385 TF/s was 12% of
  the fp8 peak and looked 8× short, while it moved 98.5% of the measured read-only ceiling (arithmetic intensity
  60.6 FLOP/B against a ridge of 511). Count logical bytes (rows × row width), not HBM traffic after cache hits,
  and compare against measured bandwidth for the same access pattern, not against FLOP peak or a stream copy.

## 6. Small shapes, the host and the launcher
- **FlyDSL v0.3.2: call the object returned by `flyc.compile(...)` in hot loops and benchmarks.** A direct call of
  a `@flyc.jit` launcher re-binds the signature, re-checks captured globals and rebuilds the cache key on every
  call. ~31 µs per call for an 8-argument launcher against 2.4–2.7 µs for the compiled object
  (`torch.add`: 1.8–2.3 µs); launchers with more arguments were not measured.
- The compiled object takes positional arguments only and ignores new values of `Constexpr` arguments: they are
  baked in at compile time (`jit_function.py:1651`). Recompile when one changes.
- **`flyc.compile(fn, *args)` runs `fn` once with those arguments** while compiling (`jit_function.py:1662`).
  Compile-then-call therefore executes twice, and a kernel that accumulates or atomically updates its outputs is
  applied one extra time. Compile on scratch buffers or re-initialise outputs and counters afterwards.
- **Every placeholder tensor built in the launcher is a kernel launch.** *(field)* A `torch.zeros(1)` placeholder
  that the dense path never read cost a 4.6–5.6 µs fill kernel per backward call, ~0.4% of a 1.22 ms call.
  Look for `fill_`/`zero_` entries that match no kernel of yours in the profiler table.
- **For kernels of 20–200 µs, a `synchronize()`-bracketed wall time is not a kernel ruler.** *(field)* ~27 µs
  of host time per call on one side and more on the other (a slower Python launcher) turned a 16.8% kernel gap into
  38% on wall, and the
  wall ruler produced three false scaling conclusions. Rulers for launch-bound kernels, worst to best:
  1. `synchronize()`-bracketed wall time;
  2. event timing without a graph — measures the Python launch rate *(field: a 21 µs kernel read 33.9 µs and
     two arms read 0.06% apart whatever their GPU work)*;
  3. the same configuration repeated across processes *(field: 12.9% range)*;
  4. ~20 calls captured in one HIP graph, timed with events, arms interleaved ABBA in one process, after
     45–75 s of warm-up *(field: 0.08–0.28% range across repetitions)*; or kernel time from
     `rocprofv3 --kernel-trace`.

## 7. Cache hygiene
- Clear or isolate the FlyDSL cache when comparing builds: *(field)* one commit read 55–83% apart before and after
  clearing a cache shared across code versions. What the cache key does and does not see is in
  `../bottleneck/debug-flydsl-kernel.md` § What the cache key sees.
- The first pass in a fresh cache directory includes compilation and any autotuning; discard it. Do not time with
  `FLYDSL_DUMP_IR=1` (`../gemm/flydsl_gemm_measurement.md` §6).

## 8. Coverage traps
- **A single scored shape gets traded against the others, silently.** *(field)* With only full-causal S=16384
  scored, one round bought +1.8% there and −12.2% on the same shape with a sliding window; at a 3:1 window to
  full-causal deployment mix that round lost. Close every series of changes on the deployment mix: all shapes × all mask
  modes, weighted by the mix. Before gating a regime-specific change, align the other levers for that regime
  (`flydsl_attn_workflow.md` § Rules that save rounds); gate on the trait only if it still loses there.
- **Weight cells by their share of step time.** *(field)* Full causal was 92% of a backward step and the window
  cells 8%; an equal-weight geometric mean let a 13.6% window gain lift the score while step time did not move.
- **A shared file is not a shared gain.** *(field)* 40 rounds on the D=128 path gained 19.3%; the D=64 path in the
  same file went from 997.2 to 990.0 TF/s, because four `D == 128` gates kept the work out. Measure every unscored
  variant before and after; a zero delta means look for a gate (opening these was worth 6.36% in one round).
- **Audit the dispatch per cell before claiming "no regression".** Wrap the dispatcher and log the path each cell
  takes. A window covering the whole sequence is normalised to full causal; a framework's default backend may not
  be FlyDSL, so the probe never reaches the change; a command that pins the backend and one that resolves it
  automatically are different rulers *(field: a false collapse from 936 to 123 TF/s)*.

## 9. Helper kernels in concurrent traces
- **Do not divide a helper kernel's trace duration by its bytes when it runs beside the main body.** *(field)* A
  reduction recorded as 670 MB in 317 µs (2.1 TB/s, "4× from peak") ran at 5.4–6.2 TB/s when launched alone, against
  a 5.95–6.2 TB/s stream floor on that capped card; the 317 µs was its time competing for CUs.
- Read per-call min and max first. *(field)* The same delta kernel ran 28.1 µs serially (7.2 TB/s, already above the
  read stream floor, so not slow) and 59.8–1527 µs beside an occupancy-1 body: the inflation was CU starvation, not
  extra work.
- **Kernel-time share is not wall share.** *(field)* A helper was 14.95% of the summed kernel time of a backward
  but 1.9–3.9% of exposed wall on the timeline. Rank helper-kernel work by exposed time; fusion pricing is in
  `flydsl_attn_backward.md` §7.

## Related
`../gemm/flydsl_gemm_measurement.md` · `../../../../common_methodology/profiling/measure_protocol.md` ·
`flydsl_attn_attribution.md` · `flydsl_attn_correctness.md` · `flydsl_attn_decode_split_kv.md` §8 ·
`../bottleneck/debug-flydsl-kernel.md` · `../../../../hardware/mi350_clocks.md`
