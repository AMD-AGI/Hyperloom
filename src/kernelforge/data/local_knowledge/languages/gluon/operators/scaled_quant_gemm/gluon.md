---
title: scaled_quant_gemm on Gluon (FP8/BF8, CDNA) — authoring card
kind: sota_card
operator: scaled_quant_gemm
backend: gluon
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3, fp8_e5m2, fp8_e4m3_fnuz, fp8_e5m2_fnuz]
regimes: [prefill, training]
status: experimental
updated: 2026-08-23
sources:
  - https://github.com/ROCm/gfx950-gluon-tutorials
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
---

# scaled_quant_gemm × Gluon

## TL;DR (one-line decision)
> The FP8/BF8 GEMM is the **same Gluon design as the FP16 one** with a larger `BLOCK_K` — AMD's `a8w8`
> kernel reaches roughly **~99.7% MFMA efficiency** on gfx950, the highest of the three dtypes, because
> unlike MXFP4 it needs no separate scale pipeline. If you already have a working Gluon FP16 GEMM,
> this is a dtype change plus a `BLOCK_K` change, not a redesign.

## Reference design
`ROCm/gfx950-gluon-tutorials:kernels/gemm/a8w8/`. Deltas from the FP16 card
([`../dense_gemm/gluon.md`](../dense_gemm/gluon.md)), which you should read first:

| Element | a16w16 (FP16) | a8w8 (BF8) |
|---|---|---|
| Tile (M×N×K) | 256×256×64 | **256×256×128** |
| Everything else | M+N slicing, 3-stage pipeline, unroll 2, llirSched + amdgcnas | identical |

The larger `BLOCK_K` follows from the operand being half the width: the same LDS budget and the same
MFMA cadence want twice as many K elements per stage.

## Measured ceiling
| dtype | shape | TFLOPS | MFMA eff |
|---|---|---|---|
| BF8 | 4096×4096×16384 | ~3257 | ~99.7% |

AMD-measured, MI355X gfx950, ROCm 7.0. See the caveat in
[`../../skills/optimize/gluon_levers/overview.md`](../../skills/optimize/gluon_levers/overview.md) —
these are large-K compute-bound shapes and AMD's own numbers vary between sources.

## The trap that dominates this operator: fp8 dialect

**fp8 is FNUZ on CDNA3 (gfx942) and OCP on CDNA4 (gfx950).** A mismatched dialect corrupts the descale
and produces **wrong numbers, not an error**. This is inherited from the Triton substrate and it is the
single most common way an fp8 kernel silently fails.

Consequences for a Gluon kernel that must run on both:
- Select the pointer/operand dtype from the detected arch, not from a constant.
- OCP `float8_e4m3fn` fed into a CDNA3 matrix op does not lower; the failure mode there is at least
  loud. The dangerous direction is the quiet one — using the wrong *scale interpretation* and getting
  plausible output.
- Your correctness check must run on the arch you will ship on. A parity pass on gfx950 says nothing
  about gfx942.

See `../../../../hardware/mi350_dtypes.md` for the FNUZ-vs-OCP dialect details (language-independent),
and the aiter source's `torch.finfo(quant_dtype).max` usage for how the dialect is actually selected.

## Scaling model
This card is per-tensor / per-channel fp8 scaling, where the scale is applied in the epilogue or folded
into the accumulator — **not** microscaling. Block-scaled MXFP4/MXFP8 with an E8M0 per-32-element scale
goes through the native scaled MFMA and is a different kernel shape entirely; see
[`../quant_fp4_mxfp/gluon.md`](../quant_fp4_mxfp/gluon.md).

## Knobs worth sweeping
As for FP16, plus: `BLOCK_K` matters more here (it is the thing that changed), and the epilogue descale
placement — folding the scale into the accumulator loop vs applying it once after — is a real fork with
different register cost. Both are `constexpr`-able and therefore cheap sweeps.

## Cross-links
`../dense_gemm/gluon.md` — the base design this specializes.
Math contract, parity bands and backend landscape are not documented in this repo — read the kernel
source. For the library fp8 GEMM path and its tuned tables, see
`../../../../framework/aiter/overall/dispatch_and_rebind.md` + `../../../../framework/aiter/overall/tuning_db.md`;
hipBLASLt fp8 is a strong bar and should be measured before authoring.
