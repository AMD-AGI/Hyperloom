---
title: FlyDSL attention — correctness gates: spike inputs, paths the gate must reach, shared LSE, seeds and mask conventions
kind: language
lever: flydsl_attn_correctness
gens: [gfx950]
dtypes: [bf16]
regimes: [prefill, backward, causal, sliding-window, rectangular]
updated: 2026-10-26
---

# Correctness for a FlyDSL attention kernel

> Most cases marked *(field)* below passed a gate that reported clean. Use them to design the gate before
> trusting it. Numbers marked *(field)* are measured values; use them for direction and order of magnitude.
> The general GEMM rules — gates that reach the changed path, calling twice, races found by bitwise repetition,
> waits and barriers, inline-asm hazards, a third reference after a reduction-order change — are in
> `../gemm/flydsl_gemm_correctness.md`. Debugging a kernel already known to be wrong starts in
> `../bottleneck/debug-flydsl-kernel.md` (its §1 covers `-inf − -inf` and divide-by-zero NaNs).

## Route here when
- You change the softmax shift, the LSE, a scale, an accumulator seed, a mask, a barrier or an anchor.
- A gate passes and you have not shown it runs the path you changed, at the dynamic range that breaks it.
- One output is wrong and the others are right, or a metric reads exactly 1.000.

## 1. What a gate measures
For each output — O, LSE, dQ, dK, dV — against an fp32 reference:
- **SNR**, to catch structural errors. *(field)* Correct bf16 kernels read ~50–53 dB on O and the gradients; a
  mask with one missing edge key reads 22–38 dB.
- **Norm ratio `|got| / |ref|`**, to catch "finite but biased", which SNR and finiteness checks miss. *(field)* dV
  fell to 0.94 at high spike with no non-finite value anywhere.
- **Non-finite count** on every output, LSE included.
- **Bitwise determinism**: two runs, `torch.equal`. In synchronisation changes it usually fails before SNR does.
- Score accuracy by the **worst** cell, not the mean: an average lets one row that is 5% off hide behind twenty
  correct rows.
- Check dQ, dK and dV separately; a change to a dQ scaling constant is invisible in dV.

## 2. Inputs that reach the numerics
- **Independent Gaussian inputs cannot exercise the softmax range.** The row maximum then grows slowly with the
  logit scale and stays near the first tile; trained attention is concentrated, and the maximum is often far
  from it. Build spike inputs: a few keys with large logits, placed away from the first KV tile, at several
  spike levels.
- **Push the spike to failure, not to "enough".** *(field)* A gate that stopped at spike 24 reported clean; the same
  kernel produced its first non-finite values at spike 36, growing with the spike. A gate that never fires is
  not a gate.
- **Prove the gate on known-bad code.** Run it on a build known to fail and see `ok=false`; passing on good code
  proves nothing about the gate. *(field)* Run on a build known to overflow, the extended gate failed it at
  spike 36.
- **Prove the probe on known-good code.** *(field)* A gradient probe passed the LSE as `[B·Sq, Hq]` to an entry
  that expected `[B, Hq, Sq]`, gave every row another row's normaliser and reported gradients 7.5× too large; the
  same reading on the unmodified baseline exposed the probe.

## 3. The gate must reach the path under test
- **Defaults choose paths per head_dim, causality and window.** A kernel whose softmax path differs by head_dim
  can pass every D=128 check while the D=64 path never runs. List, per gated cell, which path ran
  (`flydsl_attn_measurement.md` §8).
- **Every configuration sweep carries the SNR gate.** *(field)* Several `block_kv` values computed dQ silently
  wrong (8–14 dB) and one read 854 TF/s, exactly on target, because it skipped work.
- **Synchronisation, anchor and fence changes need a grid of configurations, with determinism.** *(field)* Two
  faster-but-wrong variants (+2.53% and +0.31%) failed only at `q_split=4, BLOCK_KV=128` and at
  `BLOCK_KV=64`; the main shape passed. Cover at least the split and tile values the dispatcher can pick.
- **Each path that applies a window needs its own test.** *(field)* A window bound that admitted W+2 keys sat in a
  fold path that the D=64 tests never took, while the mask-style path they did take was right.
- Call at least twice and compare from the second call (`../gemm/flydsl_gemm_correctness.md` §2).

## 4. Poison workspaces
A `torch.empty` workspace that is read before it is written returns whatever the allocator left there; unit
tests pass, full runs fail. Fill workspaces with NaN in the gate. *(field)* With a window narrower than the
q-block (W=32 against `BLOCK_Q=64`), a reduction read half-written bands; poisoning reproduced it exactly
(NaN on rows with `row % 64 ≥ 32`, only at W=32).

## 5. Forward and backward share state
- **Gate end to end.** The backward consumes the forward's O and LSE, so an edit to the LSE's layout or meaning
  keeps the forward's own SNR perfect and breaks the backward silently. Drive the backward with the optimised
  forward's own outputs, not the reference's:
  ```python
  o, lse = forward(q, k, v)                   # the side being optimised
  dq, dk, dv = backward(do, q, k, v, o, lse)  # its own o and lse
  ```
  *(field)* Forward 51.7 dB, end-to-end gradients 48.8–49.3 dB; a 45 dB gate leaves normal variation alone.
- **Recompute P in the backward from the same rounded operand as the forward.** If the forward rounds
  `q·(sm·log2e)` to bf16 and the backward rounds `k·(sm·log2e)` instead, the two scores differ by δ ∝ |s|, P is
  multiplied by `2^δ`, and since `E[2^δ] > 1` the bias is systematic and grows with concentration. *(field)* dV
  +6% at high spike. Computing the backward scale in fp32 ("more exact") was worse in a CPU replay that reproduced
  the GPU readings (1.041 at D=64); mirroring the forward's rounding gave 1.0000 / 0.9999 there.
- **Keep one owner for each convention.** A pre-scaled tensor may be read elsewhere (the same LDS K feeding two
  GEMMs); a second path may reduce dQ. Publish the scale from the builder and have every consumer read it. A
  gradient off by a constant factor names the wrong constant: log2e (1.443), ln 2 (0.693), 1/√D, or
  1/(sm·log2e) (7.84 at D=128).
- The LSE pre-scaling format must match the exp mode on both sides (`flydsl_attn_softmax.md`).

## 6. Accumulator seeds
When the softmax scale c is applied to the score accumulator rather than folded into Q, everything seeded into
that accumulator must already be divided by c, or the scale applies twice.
- Enumerate every source of the accumulator's initial value (host LSE seeds, the reference written by a rebase
  branch) and audit the set closed. *(field)* A missed second source broke only D=64 at high spike (dq ratio
  0.036), because the branch that produced it was enabled only for D ≤ 64 and fired only when a tile overran the
  reference. Find such a bug by intersecting the trigger conditions, not by reading line by line.
- **Dividing a masked-row floor by c overflows fp32.** A floor of −3.0e38 divided by c = 0.1275 (D=128) is
  −2.35e39. It fires only where a whole tile is masked for a row (a left window) and no finite bound replaced the
  floor first. **LSE breaks and O stays correct**, because O is normalised: *(field)* 15360–29696 of 32768 LSE
  values non-finite with O clean. Seed zero and fold the reference into the scaling FMA (`fma(acc, c, −ref)`);
  gate LSE, not only O.

## 7. Locating layout errors
- **Use position-encoded inputs, not permutation-invariant checks.** Row sums of P equal to 1, a constant C, or
  matching value multisets all survive a wrong element-to-row map. *(field)* A multiset comparison of two byte
  operand streams overlapped 89.1% and decided nothing.
- **Ramp-in-GEMM:** make one operand encode position (`K[kv, 0] = kv`, `Q[:, 0] = 1`, so S = kv) and read the map
  back element by element; a q-ramp checks the other axis. To locate a read, inject values constant along the
  contraction axis and varying along M: every lane must then return one constant, which separates a wrong value,
  a wrong M map and an unwritten byte.

## 8. Reading a difference
- **Compute `max|d|` with `torch.where(valid, d, 0)`, never `d * mask`.** Unwritten causal tails hold `-inf`;
  `-inf * 0` is NaN, and Python's `max(0.0, nan)` returns 0.0, so a whole table reads 0.0e+00 and looks
  bit-identical.
- **A metric at exactly 1.000 can be two errors cancelling.** *(field)* A clamp turned a +6% dV bias into −6%; a dQ
  loss stayed hidden under a positive P bias until that bias was fixed. Confirm against an independent
  reference, or switch each known error source off and see whether the metric moves.
- **A "no-op" whose removal changes the result is doing something.** A clamp that shifts the result by 13% when
  removed is firing often; one that never fires on real data but drops SNR when removed is carrying a hazard.
  *(field)* A `min(s, 0)` clamp never fired (largest argument −5.7), yet removing it took SNR from 50.5 to 10.5 dB:
  it was the only compiler-visible VALU read of an MFMA result feeding an inline-asm exp
  (`../gemm/flydsl_gemm_correctness.md` §7).

## 9. Mask conventions
- **`window_size = (W, 0)` keeps W+1 keys**, `kv ∈ [q + off − W, q + off]`, diagonal included. A reference or a
  kernel site that keeps W leaves out one edge key: *(field)* ~22 dB at W=128; ~37 dB at W=1024, for D=64 and D=128
  alike. Check the convention of the reference before blaming the kernel.
- **SDPA's `is_causal` is top-left aligned.** It equals the kernel's bottom-right causal mask only when Sq == Skv
  *(field: a rectangular shape read SNR 0.25)* and cannot express a left window. Give rectangular and windowed
  references an explicit bottom-right mask.
- A masked SDPA reference falls back to the math backend, materialises `[B, H, Sq, Skv]` and keeps it for its own
  backward: put the reference forward and backward inside the same out-of-memory guard.

## Related
`../gemm/flydsl_gemm_correctness.md` · `../bottleneck/debug-flydsl-kernel.md` · `flydsl_attn_softmax.md` ·
`flydsl_attn_backward.md` · `flydsl_attn_measurement.md` · `flydsl_attn_schedule_sync.md`
