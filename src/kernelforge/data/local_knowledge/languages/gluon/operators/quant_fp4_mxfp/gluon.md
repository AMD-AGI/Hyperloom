---
title: quant_fp4_mxfp on Gluon (MXFP4 via CDNA4 scaled MFMA) — authoring card
kind: sota_card
operator: quant_fp4_mxfp
backend: gluon
gens: [gfx950]
dtypes: [fp4_e2m1, mxfp4, mxfp8]
regimes: [prefill, training]
status: experimental
updated: 2026-08-23
sources:
  - https://triton-lang.org/main/gluon/api/amd.cdna4.html
  - https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
  - https://triton-lang.org/main/dialects/TritonAMDGPUOps.html
  - https://github.com/ROCm/gfx950-gluon-tutorials
---

# quant_fp4_mxfp × Gluon

## TL;DR (one-line decision)
> **gfx950 only.** This is the operator Gluon exists for on CDNA4: the native scaled MFMA
> (`v_mfma_scale_f32_16x16x128_f8f6f4`) consumes packed FP4 operands *and* an E8M0 block scale in one
> instruction, and AMD's `a4w4` kernel reaches **~5255 TFLOPS at ~92.4% MFMA efficiency**. The ~92%
> rather than ~99% is **structural, not a tuning failure** — the kernel runs a second, separate scale
> pipeline alongside the data pipeline and the two contend for LDS ports.

## Hard prerequisite
CDNA4. There is **no native `v_mfma_scale_*` on gfx942**, so this route simply does not exist on
CDNA3 — an MXFP4 GEMM there means manual upcast plus an ordinary MFMA, which is a different (and much
slower) kernel. Gate on a detected-arch check, the `is_hip_cdna4()` idiom (backend is `'hip'` **and**
the arch matches), never on a SKU name or an env var.

## Reference design
`ROCm/gfx950-gluon-tutorials:kernels/gemm/a4w4/`. Same skeleton as the FP16 card
([`../dense_gemm/gluon.md`](../dense_gemm/gluon.md)) — M+N slicing, 3-stage pipeline, unroll 2,
llirSched + amdgcnas — with two changes:

| Element | value |
|---|---|
| Tile (M×N×K) | **256×256×256** |
| Extra structure | a **separate scale pipeline: GR → LW → LR** (global-read → LDS-write → LDS-read) running alongside the data pipeline |

That second pipeline is the whole difference. It is why the tile K is 256, and it is why the ceiling is
lower: two pipelines reading LDS contend for ports.

## Measured ceiling
| dtype | shape | TFLOPS | MFMA eff |
|---|---|---|---|
| MXFP4 | 4096×4096×32768 | ~5255 | ~92.4% |

AMD-measured, MI355X gfx950, ROCm 7.0. **~92% is the realistic target here** — chasing the ~99% that
BF8 reaches is chasing a ceiling this operator does not have.

## Data format — get this exactly right

- **FP4 (e2m1) is packed two elements per `uint8`**, normally along the reduction (K) dimension. The
  **low 4 bits are the first element, the high 4 bits the second.** Getting the nibble order backwards
  transposes every pair and produces plausible garbage.
- **MX scales are E8M0**: 8 exponent bits, 0 mantissa bits, representing powers of two from `2**-127`
  to `2**127`, with **255 reserved as NaN**. One scale per group of **32 elements**.
- The dialect-level upcast path takes fp4-as-i8 plus an E8M0 scale **encoded as BF16** and lowers to
  `v_cvt_scalef32_*`.

## ⚠️ The scale packing order differs between MFMA variants

| Variant | Scale packing order |
|---|---|
| `mfma_scaled_16x16x128` | `op_0, op_2, op_1, op_3` |
| `mfma_scaled_32x32x64`  | `op_0, op_1, op_2, op_3` |

**This is the highest-risk item on this card.** The order is not symmetric, so changing the MFMA shape
without re-deriving the scale packing compiles cleanly, runs at full speed, and returns wrong numbers.
If you switch variants — which is a natural thing to try while tuning — **re-run the task's
`correctness_command` before you read the timing at all.** SNR will not reliably catch it.

## Knobs worth sweeping
Tile dims and pipeline depth as for FP16, plus the scale pipeline's own depth and whether the scales
are staged through LDS at all for your shape (for small K the LDS round-trip may not pay). The two
pipelines couple through LDS capacity and ports, so sweep their depths **jointly** — independently
tuned they will both look neutral.

## Cross-links
[`../../API_docs/amd_targets.md`](../../API_docs/amd_targets.md) § 4 — the scaled-MFMA API surface and
the data-format details. `../dense_gemm/gluon.md` — the base design.
Math contract, quantization semantics and parity gating are not documented in this repo — read the
kernel source and `op_tests/`. For the library path, see
`../../../../framework/aiter/overall/operator_catalog.md` (the MXFP4/MXFP8 entry points) and
`../../../../framework/aiter/overall/dispatch_and_rebind.md`.
