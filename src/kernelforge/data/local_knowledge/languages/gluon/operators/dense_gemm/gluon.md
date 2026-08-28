---
title: dense_gemm on Gluon (CDNA) — authoring card
kind: sota_card
operator: dense_gemm
backend: gluon
gens: [gfx942, gfx950]
dtypes: [fp16, bf16]
regimes: [prefill, training]
status: experimental
updated: 2026-08-23
sources:
  - https://github.com/ROCm/gfx950-gluon-tutorials
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://arxiv.org/abs/2511.08083
---

# dense_gemm × Gluon

## TL;DR (one-line decision)
> Dense GEMM is the workload the whole Gluon-on-CDNA ladder was developed on, so it is the best-mapped
> operator in this folder — AMD's public a16w16 kernel reaches **~99% MFMA efficiency** on gfx950 at
> 4096×4096×8192, roughly 3× a naive Gluon baseline. But for a *plain* GEMM in a production path,
> tuned **hipBLASLt / aiter** is still the default answer; Gluon is for when you need a fused or
> otherwise non-library-expressible GEMM at near-peak, or when the library has no entry for your shape
> or dtype.

## When this card applies
- Triton autotune has converged on this GEMM and PMC still shows the matrix core far from peak.
- The GEMM has an epilogue or a fusion the library cannot express, so hipBLASLt is not an option, and
  Triton's schedule is the limit.
- Large-K, compute-bound shapes. **Skinny / decode-shaped GEMM is a different regime** — the ceilings
  here do not transfer, and split-K / stream-K in Triton is usually the better lever there — see
  `../../../triton/skills/optimize/triton_levers/triton_templates.md`.

## Reference design
AMD's `a16w16` FP16 kernel (`ROCm/gfx950-gluon-tutorials:kernels/gemm/a16w16/`, MIT) is the reference,
and its final shape is the thing to copy:

| Element | Value |
|---|---|
| Tile (M×N×K) | 256×256×64 |
| Pipeline | **3-stage** software pipeline, hand-authored |
| Slicing | **M+N slicing** (this is what resolves the register-pressure cliff) |
| Unroll | loop unrolling by 2 |
| Global→LDS | async copy direct to LDS, no register staging, no `ds_write` in the inner loop |
| Global loads | AMD buffer ops (scalar base + offset tensor) |
| LDS layout | chosen by measurement between raw / swizzled / padded |
| Scheduling | `TRITON_ENABLE_LLIR_SCHED=1` + `TRITON_ENABLE_AMDGCN_AS=1` |
| Workgroups | XCD-aware remapping |

The rung-by-rung path to that shape, including the 73% regression at v6 and why it matters, is in
[`../../skills/optimize/gluon_levers/overview.md`](../../skills/optimize/gluon_levers/overview.md) —
read it rather than jumping to the final shape, because the intermediate diagnoses are what let you
adapt it to a kernel that is not this one.

## Measured ceilings (AMD-measured, MI355X gfx950, ROCm 7.0)
| version | dtype | shape | TFLOPS | MFMA eff |
|---|---|---|---|---|
| v0 naive | FP16 | — | ~520 | ~25% |
| v9 | FP16 | 4096×4096×8192 | ~1489 | ~99% |

⚠️ AMD's own two READMEs disagree (~541 → ~1421 vs ~520 → ~1489) and pin different Triton tags. Treat
these as "roughly 3× is available from a naive Gluon start", not as a target. For scale on the same
hardware, HipKittens reports BF16 at ~1610 TFLOPS — i.e. hand-written still leads, but not by much
anymore.

## Cross-gen
gfx942 runs this design: buffer ops, async copy to LDS, manual pipelining, and both wave patterns all
apply, and HipKittens reports the same 8-wave schedule reaching >95% of peak on **both** CDNA3 and
CDNA4 with only shared-memory-size adjustments. What does not transfer is anything scaled-MFMA — see
[`../quant_fp4_mxfp/gluon.md`](../quant_fp4_mxfp/gluon.md). Note also LDS is 64 KB/CU on CDNA3 vs
160 KB/CU on CDNA4, so a padded LDS layout tuned on gfx950 may not fit on gfx942.

## Knobs worth sweeping
These are `constexpr` in your own source, so they are cheap sweeps (`FORGE_SWEEP_*`), not edits:
`BLOCK_M` / `BLOCK_N` / `BLOCK_K`, pipeline depth (2 vs 3), unroll factor, and the LDS layout selector
if you parameterize it. Sweep coupled ones jointly — tile shape and pipeline depth both spend LDS and
registers, so they are not independent.

## Numerics
FP16/BF16 operands, **FP32 accumulate**. Nothing operator-specific beyond the usual: the accumulation
order changes when you change the pipeline or the slicing, so a tolerance that passed at v3 can fail at
v7. The task's own `correctness_command` decides — see
[`../../skills/optimize/gluon_levers/forge_integration.md`](../../skills/optimize/gluon_levers/forge_integration.md).

## Cross-links
Math contract, shape regimes and backend landscape are not documented in this repo — read the kernel
source for *what* to build. For the library path you would be competing with:
`../../../../framework/aiter/overall/dispatch_and_rebind.md` (how a dense GEMM call resolves) and
`../../../../framework/aiter/overall/tuning_db.md` (its per-shape tuned tables) — read those before
deciding to author at all.
