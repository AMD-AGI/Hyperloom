---
title: FlyDSL attention — exp2, scale and shift, row reductions and the online-softmax rescale
kind: language
lever: flydsl_attn_softmax
gens: [gfx950]
dtypes: [bf16, fp8]
regimes: [prefill, decode, backward, causal]
updated: 2026-10-26
sources:
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/math.py:192
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/utils/arith.py:526
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/kernel_function.py:250
  - ROCm/FlyDSL@dd8373349:lib/Conversion/FlyToROCDL/FastMathOps.cpp:35
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:432
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3317
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3710
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3766
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3840
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_gfx950.py:1018
---

# Softmax and exp2 in a FlyDSL attention kernel

> Everything between the QK MFMAs and the P·V MFMAs: the exp, the scale and shift, the row max and row sum,
> and the online rescale. Guards for `-inf − -inf` and empty partitions are in
> `../bottleneck/debug-flydsl-kernel.md` §1; carrying the running max and sum through a FlyDSL loop is in
> `../optimize/prefetch-data-load.md`; how many exps fit under the MFMAs is in `flydsl_attn_mfma_layout.md` §1
> and `flydsl_attn_registers_occupancy.md` §8.
>
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as targets.

## Route here when
- `v_exp_f32` is the largest VALU class, or `v_ldexp_f32` / an `__ocml_exp2_f32` call sits next to it.
- You are deciding where the softmax scale, the max shift or a clamp is applied.
- A fixed reference max, a skipped or lazy rescale, or a cheaper exp is proposed.
- A backward output is all NaN, or an accuracy gate fails for one exp mode only.

## 1. What one exp costs on gfx950
- `v_exp_f32` (also `v_log_f32`, `v_rcp_f32`, `v_sqrt_f32`) takes 8 cycles per wave64 instruction;
  `v_mul_f32`, `v_add_f32` and `v_fmac_f32` take 4 from one wave and reach 2 per SIMD once two or more waves
  issue them. The exp stays at 8, so at SIMD level one exp costs as much VALU time as four multiplies.
- A transcendental never runs beside other VALU work: exp + mul in one wave take 12 cycles, and while one wave
  streams exps no other wave on that SIMD issues VALU. It draws on the same MFMA shadow as every other VALU op,
  twice as fast as a multiply.
- It is a trans op (CDNA4 ISA Table 12): a non-trans VALU consumer of its result needs one wait state. The
  compiler pads that for native instructions only (`flydsl_attn_schedule_sync.md`).
- To place exps between MFMAs, use the transcendental class `0x400` in `sched_group_barrier`; the VALU class
  `0x002` does not match `v_exp_f32` (`flydsl_attn_schedule_sync.md`).
- The tile fixes the count: one `v_exp_f32` per 64 scores. Only fewer scores (causal tile skipping, tile
  geometry) lowers it.
- *(field)* Deleting every softmax exp of a D=64 forward loop (wrong results, diagnostic only) bought ~8%,
  deleting half of them about half of that. In backward bodies hardware exp cost 1–7% against a no-exp body
  (4–7% on most) and a Schraudolph exp 2.5–3%. Price your own with a subtractive arm that keeps the consumers alive
  (`flydsl_attn_attribution.md` §5).

## 2. Writing exp2 in FlyDSL v0.3.2
| Spelling | Lowering | ISA |
|---|---|---|
| `fx.math.exp2(x)`, `ArithValue.exp2()` / `fx.Float32(x).exp2()` without `afn` | `__ocml_exp2_f32` | ~8 VALU: compare with −126, +64 bias, `v_exp_f32`, `v_ldexp_f32` |
| the same with `fastmath="afn"` or `"fast"`, or under the `fast_fp_math` compile hint | `rocdl.exp2` (pass `convert-rocdl-fastmath-ops`) | one `v_exp_f32` |
| `rocdl.exp2(T.f32, x)`, or `llvm.amdgcn.exp2.f32` | intrinsic | one `v_exp_f32` |

- The afn rewrite covers f32 scalars and 1-D f32 vectors only. A denormal-mode function attribute is not a
  switch for it: a bare `"passthrough"` key is dropped (`../gemm/INDEX.md` § FlyDSL version).
- The range reduction only exists to return denormal results; `v_exp_f32` flushes denormals to zero (ISA).
  After the max shift every argument is ≤ 0 and every result lies in (0, 1], where a denormal is worthless
  next to bf16 P, so the single instruction is enough. *(field)* Replacing the range-reduced exp with one
  `v_exp_f32` bought +20% and +25% in two backward kernels.
- `v_exp_f32(-inf) = 0` exactly (ISA): write `-inf` into masked scores and no select is needed after the exp.
- Do not emit `v_exp_f32` as inline asm: `rocdl.exp2` gives the same instruction and stays visible to the
  scheduler and to the hazard recognizer.
- **Reference pattern** — FlyDSL's gfx950 forward: Q is multiplied once by `rsqrt(D)·log2e` in f32 and truncated
  back to bf16 (`scale_all`), so scores and the running max are in log2 units; every softmax exp is
  `rocdl.exp2` (the D=128 causal ISA has 197 `v_exp_f32`, no `v_ldexp_f32`, no OCML call); the LSE is converted
  back to natural log at the store. Its compile hints include `fast_fp_math`.

## 3. Scale, shift, clamp and pack at zero instruction cost
- **Shift through the MFMA C operand.** MFMA computes A·B + C, so seeding C with the negative shift (`-ref` in
  the forward, `-log2e·lse` in the backward) makes the subtraction free. Build the seed inside the QK step from
  a scalar; a seed vector materialized outside the loop stays resident (16 VGPRs for a 32x32 C fragment). The
  measured price of carrying it is in `flydsl_attn_registers_occupancy.md` §3; fold it in the masked loop too
  (`flydsl_attn_backward.md` §8).
- **Choose where the scale goes.** Pre-scaling Q costs nothing but rounds `q·sm·log2e` to bf16; the backward must
  then reproduce that rounded operand (`flydsl_attn_correctness.md` §5). Scaling the accumulator leaves no residual
  *(field: it removed a dQ correction pass that cost up to 8.5× when triggered, for an 8–13% slower forward)*, but
  every value seeded into it must be pre-divided by the scale; one missed seed site applies the scale
  twice, and a `-3e38` floor divided by a scale < 1 overflows fp32. If you must scale the accumulator, do the
  shift inside the scaling FMA (`fma(acc, c, -ref)`) and seed zero. Gates: `flydsl_attn_correctness.md`.
- **A seed must not cost the accumulator its precision.** The Schraudolph-prescaled LSE (~1e9) cannot be seeded: f32
  spacing at 1e9 is 64, the trick needs the low mantissa bits exact *(field: output SNR fell to 1.5 dB)*. Keep that
  FMA as the last per-element step.
- **Clamp in the exp's modifier.** `min(max(exp2(x), 0), 1)` (max first) folds into the VOP3 clamp bit:
  one `v_exp_f32_e64 … clamp`, no extra VALU. `max(min(e, 1), 0)` does not fold (`min(NaN, 1) = 1`, the clamp
  gives 0) and costs two VALU. A clamp turns an overflow into silent mass loss: use it as a backstop on a path
  whose arguments are already provably ≤ 0, never as the fix (§6.1).
- **Pack natively.** An f32→bf16 `fptrunc` of a 2-element vector (`Vec(...).to(fx.BFloat16)`) lowers to one
  `v_cvt_pk_bf16_f32`. `rocdl.cvt_pk_bf16_f32` emits the same instruction as inline asm, invisible to the
  hazard recognizer; an MFMA reading that P directly needs explicit padding (`flydsl_attn_schedule_sync.md`).

## 4. Precision tiers and the LSE contract
| exp | per element | accuracy *(field)* |
|---|---|---|
| Schraudolph bit trick (FMA into the exponent field, convert to int, reinterpret) | one convert; the FMA exists anyway | ~35 dB: a linear mantissa, ~1.5% RMS shape error |
| `v_exp_f32` | one 8-cycle trans op | ~51–52.6 dB, the ceiling set by packing P/dS to bf16 |
| polynomial `2^n·P(f)` | round, subtract, convert, shift-add, 2–3 FMAs | degree 3 bit-identical to hardware, degree 2 ~1 dB lower; never higher |

- An accurate software exp2 needs six or more VALU instructions against one `v_exp_f32` worth two plain VALU
  ops from one wave, four with ≥ 2 waves. It cannot win on gfx950. *(field)* Degree 2: −11 to −18%; degree 3:
  worse. `v_exp_f16` is a trans op too and the ISA has no packed transcendental; the f32↔f16 round trip only
  adds VALU *(field: −8 to −39%)*.
- Pick the exp by the accuracy gate. The bit trick is ~3–9% faster than hardware exp *(field)* and passes only
  gates below ~35 dB.
- **The LSE prescale belongs to the exp mode.** Hardware exp takes `-log2e·lse`; the bit trick takes
  `-log2e·lse·2^23 + (127·2^23 − 486411)`. Fed the other mode's LSE, exp2 overflows and the output is all NaN.
  Derive both from one function keyed by the mode. When one output of a kernel is all NaN and the others are
  fine, check that output's input preprocessing first.
- **R ≡ 1.** With the forward's true LSE, every row of `exp2(s − lse₂)` sums to 1. A backward that carries a row
  sum and renormalizes by it does dead work. *(field)* Removing it from a dq kernel: about +4% with either exp
  mode, SNR unchanged within 0.01 dB (the Schraudolph errors cancel along the row to bf16 precision).
- **The backward's exp arguments are ≤ 0 only because lse ≥ the row max.** A forward shortcut that loses mass
  (a clamp, a dropped tile) puts lse below the max, and the backward's `exp2(s − lse)` overflows. *(field)* A
  forward with clean outputs produced ~8·10⁵ non-finite gradients. Gate forward and backward together, and make
  the backward reproduce the forward's scale rounding (`flydsl_attn_correctness.md`).

## 5. Row max and row sum
- Orientation sets the cross-lane cost. With S computed as Sᵀ a lane holds the scores of one q row, and one
  `permlane32_swap` finishes each reduction (`flydsl_attn_mfma_layout.md` §4).
- Max: LLVM folds a linear max chain over a lane's scores into `v_max3_f32` (16 for 32 values). A balanced binary
  tree breaks the fold *(field: 107 → 132 instructions)*; a ternary tree of `max(max(a, b), c)` groups keeps the
  count nearly unchanged (107 → 112) and cuts the depth from 16 to 4 *(field: +0.3–0.4%)*. gfx950 has no
  `v_pk_max_f32` or `v_pk_min_f32`; packed f32 exists for add, mul, fma and mov only.
- Sum: dependency depth did not matter *(field: four accumulators, depth 32 → 8, 0.00%)*; packing it into
  `v_pk_add_f32` removed 40 instructions at equal VGPRs and lost 1.45% *(field)*. The matrix-pipe row sum
  (`flydsl_attn_mfma_layout.md` §6) removes the adds and the cross-lane step.
- Sum the same bf16-rounded P that feeds P·V, so numerator and denominator share one rounding
  *(field: 51.26 → 51.61 dB)*.

## 6. Online softmax
### 6.1 A fixed reference max is not a general lever
- With a constant reference r, P = 2^(s − r). Scores more than 128 log2 above r overflow; a clamp P ≤ 1 stops
  that but silently drops the mass above r; scores ~126 log2 below r flush to 0, and a row entirely below gives
  l = 0 and 0/0. A margin only splits those ~126 binades between headroom and tail: any fixed r is exact
  inside a data-dependent window and silently wrong outside it.
- *(field)* Reference = first-block row max + 96 log2; a row whose max sat 158.7 log2 above the first block:
  O 28.6 dB, dq 6.5 dB, against 44.8 dB for true online softmax on the same input. Training diverged after some
  steps. iid Gaussian inputs only reached the window's edge (gap ≈ 97 log2 at 4× logit scale): trained
  attention is concentrated, random data is not.
- Gate every softmax shortcut with spike inputs (a few keys with large logits far from the first block),
  raise the spike until the gate fails, and check norm ratios of O, LSE, dq, dk, dv, not `isfinite` counts.
- A max over a fully masked block (sliding-window start, first causal pair) is `-inf`, and
  `exp2(s − (−inf)) = +inf` → NaN once a real score meets it. Floor it before use: a row max applied only to its
  own masked scores can sit at `-3e38` (FlyDSL's forward, `floor_masked_max`); a bound reused as the reference
  for later tiles needs a floor those scores cannot overflow against.
- With P packed to fp8 e4m3 (max 448) for the P·V MFMA, a fixed reference cannot replace the rescale: without a
  clamp P overflows the format, with one it loses mass.
- The exact path can be made nearly as fast. *(field)* A tuned online forward reached ~99–100% of the
  fixed-reference speed (D=128 and D=64) with the rules below.

### 6.2 What the rescale must touch
- Per tile: `m_new = max(m_old, m_tile)` before the tile's exps, `corr = exp2(m_old − m_new)`, then O and l
  are multiplied by `corr`. The running max must be loop-carried. *(field)* Fingerprint of a reset max: correct
  at short S, the row-sum ratio falling with S (0.80 at 1024, 0.53 at 2048).
- Rescale O and l, not P that is already packed. When tile j's packed P is still in flight as tile j+1 moves the
  max, let the P·V steps consume it on the old reference, then scale O and l once: every term and the sum share
  that reference. *(field)* Dropping the unpack-multiply-repack of the carried P removed 5 pp of a 21% online
  overhead, results bit-identical.
- The cost centre is the per-tile multiply of the whole O accumulator, not the max. *(field)* D=128 forward,
  15.5% online overhead: O scaling 8.7 pp, the shift 3.4 pp, the max reduction ~2.8 pp.
- Do not seed the shift with the previous iteration's max to save the subtractions: P loses its ≤ 1 bound
  (it is bounded only by `2^(m_new − m_old)`) and a narrower window comes back *(field: peak 6·10¹⁵ at spike 8,
  inf at spike 24)*.

### 6.3 Skipping rescales
- A skip branch must be wave-uniform, and one wave holds 32 rows. *(field)* Per row only 6.9% / 19.1% of tiles
  moved the max (S = 8192 / 2048), yet only 39% / 9.2% of tiles were skippable per wave.
- An `scf.if` in a hand-pipelined loop costs by itself *(field: ~1.3 pp per branch point)*. Count branch points;
  the instructions moved into a rarely taken branch are nearly free, and static counts misprice both.
- Lazy rescale — keep the old reference while the max moved less than a threshold — has no general positive
  gain. FlyDSL's forward does it per wave: skip when every lane has `m_tile − m_row ≤ 8.0` (log2), tested as
  `ballot == exec`, so P may reach 2⁸. *(field)* Ported into another forward, the same branch was 31% slower on one
  path (it broke a store pipeline) and was neutral on another.

### 6.4 Rare rebase without a window
- Keep the reference loop-carried and seeded (§3). Per tile, decide whether any score can exceed it; only then
  compute the exact tile max, raise the reference, multiply O and l by `2^−Δ` and recompute the tile.
- Decide before exp2. A test on the row sum is one step late: the sum is built from exp2 outputs, so it sees an
  overflow after it happened, at any check frequency *(field: non-finite from spike 36 whether checked once per
  8 tiles or 4× as often)*. A clamp under such a test only makes the error silent.
- A test that cannot miss: `‖q_i‖·max_{j∈tile}‖k_j‖` (q already scaled by `sm_scale·log2e`) bounds every score
  of row i in the tile (Cauchy–Schwarz). Tiles whose bound stays under the reference skip the max; the rest
  reduce exactly; the reference still updates online. *(field)* NaN-free to spike 96 and bit-identical to the
  kernel without the skip.

### 6.5 fp8 P in decode: fold the rescale into the P·V MFMA
- If each tile's P·V goes into a fresh accumulator merged as `O·alpha + tile·beta`, exponentiate the tile
  against `max(m_running, m_tile)` (beta then sits inside P and its sum) and pass `O·alpha` as the P·V MFMA's C.
  *(field)* 16 `v_pk_mul_f32` + 16 `v_pk_add_f32` fewer per tile and wave (−10% instructions), producer
  1.2–3.8% faster, and lower error (relL2 0.0262 → 0.0249): beta had been applied after fp8 quantization,
  scaling the quantization error up.

## Verify
- ISA: one `v_exp_f32` per score per lane, no `v_ldexp_f32` or `__ocml` in the hot loop, clamps as `clamp`
  modifiers rather than `v_min`/`v_max`.
- A spike-input gate raised until it fails, with norm ratios on every output; forward and backward gated
  together.
- The LSE prescale and the exp mode come from one place.
- Every value seeded into a score accumulator is accounted for: all seed sites, the scaling, the floor.

## Related
`flydsl_attn_mfma_layout.md` · `flydsl_attn_registers_occupancy.md` · `flydsl_attn_schedule_sync.md` ·
`flydsl_attn_correctness.md` · `flydsl_attn_backward.md` · `flydsl_attn_decode_split_kv.md` ·
`flydsl_attn_attribution.md` · `../bottleneck/debug-flydsl-kernel.md` · `../optimize/prefetch-data-load.md`
