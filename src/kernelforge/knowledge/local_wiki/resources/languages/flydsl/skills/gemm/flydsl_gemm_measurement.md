---
title: FlyDSL GEMM — measuring a change so the number means something
kind: language
lever: flydsl_gemm_measurement
gens: [gfx950]
dtypes: [fp8, bf16, mxfp4, mxfp8]
regimes: [kernel-A/B, autotune, campaign]
updated: 2026-10-19
sources:
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1414
---

# Measuring a FlyDSL GEMM change

> The effects worth chasing on a tuned GEMM are 0.3–3%; the apparatus moves by that much on its own.
> Every rule here comes from a FlyDSL GEMM measurement on MI355X *(field)* that reported a confident
> number and was wrong. The general protocol is `../../../../common_methodology/profiling/measure_protocol.md`;
> this card is what a GEMM A/B adds to it.

## Route here when
- Comparing two builds, configs or kernels; deciding whether a delta is real.
- Writing a probe, a micro-ruler or an autotune race.
- Closing a campaign: which of the changes actually paid.

## 1. Pick the ruler for the effect size

| Ruler | What it cancels | Use for |
|---|---|---|
| **One process, two module instances, arms interleaved per rep** (load the frozen base file and the candidate file as two modules; median of 6–8 groups) | drift, clocks, allocator state — both arms see them | effects of 0.1–2% that live in the loaded file. *(field)* A-vs-A geomean −0.009% against 0.61% for the per-process scorer |
| **One process per arm, full score, A B (A B A B)** | nothing within the run; relies on per-arm warmup | changes outside the loaded file (toolchain, other modules, env), final confirmation |
| Isolated single-op loop | nothing about the surrounding workload | ranking arms of equal occupancy and equal cache behaviour only |

- The in-process ruler only works when every change lives in the file you load twice: both instances
  share the rest of the package. Prove the arms are two binaries (different ISA hash or kernel name)
  before reading the result.
- Cross-process comparisons of effects under ~3% are unreliable: *(field)* the same comparison read
  +1.66% in one pair of processes and −2.2% in another; a bit-identical control cell drifted 2.6% in
  an hour. Bracket each candidate with a fresh baseline (A B A) instead of reusing one from earlier.

## 2. Arm order

- **Per-process kernel A/B: `A B`, or `A B A B` for more confidence. No palindromes** (`A B B A`):
  each arm already carries its own warmup and median, so a palindrome only doubles machine time.
- **In-process multi-arm probes have a position term.** On a power-capped card the slot an arm runs in
  changes its clock: *(field)* the last slot read +1.3% on three independent processes; reversing the
  order moved the "gain" onto the unmodified arm. Fixed palindromes pin the term to arm order instead
  of cancelling it, and cyclic rotation does not help either — it keeps who-follows-whom. **Shuffle
  the arm order randomly every rep** *(field: an A-vs-A floor fell from 2.2% to 0.94%)*, never park the
  candidate last by habit, and discard the first pass.
- **Add an inert arm** — the candidate switch compiled to the baseline's exact configuration — and report
  every gain as "candidate minus inert". *(field)* An identical pair read +1.11% in the same batch where
  a candidate read +0.78%: the real effect was −0.33%.

## 3. Warmup, clocks and load

- **An idle MI355X needs 60–90 s of sustained real-shape load to reach its working clock.** *(field)* The
  same config climbed from 3990 to 4910 TF/s while each reading looked stable (0.26% spread). Short
  warmups make every absolute number ~20% low.
- **On a power-capped kernel the first arm of a batch runs slow** while the clock policy settles
  (*field:* ~1.5% slower, ~4% lower sclk). Burn a throwaway arm first, or use the end-of-batch baseline.
- **Record sclk per arm**, derived as `GRBM_GUI_ACTIVE / 8 XCDs / duration` rather than sampled with
  `rocm-smi` between batches (which catches idle windows). Two arms at different clocks are two
  machines, not two kernels.
- **Measure CU-fill levers under the real load mix.** *(field)* An isolated loop let the under-filled
  arm boost to 1699 MHz and made a tail split look negative; under the 12-shape mix both arms ran at
  ~1605 MHz and the split won by 21.5% on the scored table.
- **Cache, residency, swizzle and placement levers do not transfer from isolated back-to-back loops.**
  *(field)* A write-placement change read +1.1…6.5% isolated and −0.45% on the scored bench; an `nt`
  hint read −2.1% isolated and was worth +2.4% inside the step. Levers that delete bytes or requests
  do transfer (the isolated reading is then a lower bound).

## 4. Noise floors and null controls

- The floor belongs to a ruler and a cell, and it moves between sessions. Measure it every session
  with a same-binary arm; never estimate it from two reads (*field:* two reads suggested 0.13%, the
  third showed 0.5%).
- **Use cells the change cannot touch as a free drift ruler.** Prove they are untouched (identical ISA
  hash), then subtract their mean movement. *(field)* Untouched cells moved +0.21…0.37% in one
  direction and erased three "wins"; with the inert cells as reference a +0.29% gm effect came out of
  a run whose raw gm spread was larger than the effect.
- Report per cell, not only the geometric mean: an 18-cell gm can sit inside 0.02 pp while one cell
  moved 3%.

## 5. Statistics

- Compare means with their standard error (Welch t, non-overlap of the arm distributions), not with
  the range or the per-read spread; the range grows as you add reads.
- Small-n permutation tests have a floor (2 vs 2: p ≥ 0.167; 3 vs 3: 0.05; 4 vs 4: 0.014). Hitting the
  floor is the best that design can say, not a refutation.
- A large t does not fix a systematic bias in the apparatus (§6).
- `min` of N defends against in-process outliers; a paired win count defends against shifts between
  processes; a cell with two modes needs the minimum over many processes.
- Trial counts that held up: |Δ| > 2% — 9 trials; 0.5–2% — at least 11; < 0.5% — at least 11 in each
  of two sessions.

## 6. Apparatus traps

- **Swapping arms by copying files biases ~0.4%.** *(field)* An arm whose file was never imported read
  +0.39%: a different file size meant a different idle time before the run, so a different
  temperature. Equalize the gap (sleep ~20 s after each swap) or let a compile separate the arms.
- **Operand values change GEMM time.** *(field)* All-zero operands ran 151–155 µs against 205 µs on real
  data, same build and shape. Time on tensors from the real path; SNR-gate every probe.
- **Output-buffer placement moves the same binary 10–15%.** Share buffers across arms and pre-warm all
  of them; per-arm allocation in a tuning sweep biases the winner.
- **Do not time with `FLYDSL_DUMP_IR=1`** (it bypasses the cache and recompiles, `jit_function.py:1414`;
  *field:* +15% on a ±0.2% ruler).
- **Keep the host out of the GPU window.** Time R consecutive calls inside one event pair; a
  `synchronize` per call puts Python into the measurement (*field:* 650 TF/s read for ~1150 real). If
  time is flat across sizes, you are measuring the host.
- **One call per CUDA/HIP graph measures the launch floor**, and `torch.cuda.current_stream()` must be
  read inside the captured closure.
- Give every arm an identity fingerprint — config, grid, kernel name, error signature, opcode counts —
  plus one arm that must lose (e.g. an injected `s_sleep`). Arms silently collapsing into one binary
  is the most common way a probe lies.

## 7. The operating point and the denominator

- **Benchmark the deployment's per-call shape:** M per call (not per step) and the contraction
  padding. Check the launched grid against a trace.
- **Each arm at its own best configuration.** *(field)* Pinning both arms to `block_m=64` turned a 2.4%
  loss into a reported 1.8% win, because 64 was the new kernel's best and the old kernel's worst.
- **Re-measure the competitor column yourself**, with its own calling convention, and confirm it took
  its fast path; *(field)* the competitor reached only 0.966–0.992 of its own frozen column on the
  same box, which turned a 10/12 result into 12/12.
- **Aggregate with weights of the right dimension.** Time-share weights take an arithmetic mean of
  times; harmonic means are for FLOP-weighted rates. *(field)* A harmonic mean over time-share weights
  reported +8.0% for a true +6.9%.
- Use measured ceilings with the same read/write ratio as the kernel, never the datasheet or your own
  fastest kernel.

## 8. Autotune races

A per-shape race is a measurement too, with its own noise and order bias. How to build one that cannot
regress is in `flydsl_gemm_autotune.md`.

## 9. Closing a campaign

- **Ablate every component on the full table, including an all-off arm**: BASE → each one off → all
  off → BASE again. Turning off one at a time makes two compensating components look like two
  winners. *(field)* Two tile pickers, removed one at a time, priced at 1.6% together; removing both
  while keeping the machinery they enabled lost 3.0% — more than reverting the whole change (2.1%).
  The machinery was net negative without its lever.
- Keep a change only if its price exceeds the BASE-to-BASE drift and its sign agrees across cells.
  Re-verify any causal claim made from a single shape on the whole table (*field:* a narrow-tile rule
  credited with +1.9% on one shape was +0.09% better removed).
- The best banked score of a campaign is the maximum of a noise distribution; re-measure it.
- Equivalence by ISA hash only after confirming the same tree gives a stable hash (in-process
  autotuning can change the ISA); never drop outliers or let sample counts differ between arms.

## Related
`flydsl_gemm_attribution.md` (pricing a change before and after) · `flydsl_gemm_correctness.md` ·
`../../../../common_methodology/profiling/measure_protocol.md` ·
`../../../../hardware/mi350_clocks.md`
