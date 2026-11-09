---
title: Datatype numerics — FP8 FNUZ/OCP, FP6/FP4, MXFP, TF32 (CDNA cross-gen)
kind: hardware
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp64, fp32, tf32, bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1, mxfp8, mxfp6, mxfp4, int8, int4]
regimes: [both]
updated: 2026-11-07
---

# Datatype numerics across CDNA

> Numbers in brackets point to the Sources at the end.

## TL;DR
> Two correctness traps dominate: **(1) FP8 is FNUZ on CDNA3 but OCP on CDNA4** [3] — different bias and
> saturation [1][2], so a checkpoint must be **re-cast**, never bit-copied across gens; **(2) MXFP4/6/8**
> pack a block of **32 elements** sharing one **E8M0** (8-bit, exponent-only) scale [7][8] — the scale
> layout must match the `mfma_scale_*` expectation. Always **accumulate in FP32/INT32**.

## Concepts

### Format zoo
| Format | Bits | Exp/Mant | Where | Notes |
|---|---|---|---|---|
| FP64 | 64 | 11/52 (IEEE 754 double [4]) | all gens (matrix on CDNA2+ [10]) | HPC; CDNA4 halves the matrix FP64 rate [6] |
| FP32 | 32 | 8/23 [9] | all | IEEE |
| TF32 | 19 | 8/10 [5] | **CDNA3 only** (MFMA `xf32` [1]) | **moved to software emulation via BF16 in CDNA4** [6] |
| BF16 | 16 | 8/7 [9] | all | RD summation in CDNA3 MFMA [11] |
| FP16 | 16 | 5/10 [9] | all | RD summation in CDNA3 MFMA [11] |
| FP8 E4M3 (fp8) | 8 | 4/3 [3] | CDNA3 FNUZ / CDNA4 OCP [3] | activations/weights |
| FP8 E5M2 (bf8) | 8 | 5/2 [3] | CDNA3 FNUZ / CDNA4 OCP [3] | larger range, less mantissa |
| FP6 E2M3 (fp6) | 6 | 2/3 [3] | **CDNA4** [3] | runs at the FP4 rate [6] |
| FP6 E3M2 (bf6) | 6 | 3/2 [3] | **CDNA4** [3] | |
| FP4 E2M1 (fp4) | 4 | 2/1 [3] | **CDNA4** [3] | 2 per byte (`__hip_fp4x2_e2m1`, `__amd_fp4x2_storage_t`) [3] |
| MXFP8/6/4 | block | + E8M0 | **CDNA4** [6] | 32-element block shares one E8M0 scale [7][8] |
| INT8 | 8 | — | all [9] | INT32 accumulate [1][9] |

### FP8 FNUZ vs OCP — the critical gen split
- **CDNA3 (gfx942) = FNUZ** (finite, no inf; NaN is the sign bit with all other bits zero) [3][4]:
  - E4M3FNUZ: **bias 8**, max **±240**, **no inf**, single zero (+0 only), NaN = `0x80` [1].
  - E5M2FNUZ: bias 16, max **±57344**, no inf [1][3].
- **CDNA4 (gfx950) = OCP** (the OCP 8-bit floating point format [6]):
  - E4M3FN: **bias 7**, max **±448**, ±0, NaN `0x7F`/`0xFF`, no inf [2].
  - E5M2: bias 15, max ±57344, **with ±inf** [2].
- **Implication:** an FP8 model quantized for OCP must be **re-cast** for CDNA3 (different bias and
  saturation point). In HIP, `hip_fp8.h` has both: the FNUZ types (`__hip_fp8_e4m3_fnuz`,
  `__hip_fp8_e5m2_fnuz`) and the OCP types (`__hip_fp8_e4m3`, `__hip_fp8_e5m2`); on gfx94x the type
  defaults to FNUZ [3][4]. gfx950 also has the `__amd_fp8_*` extension types [3]. **Never bit-copy**
  OCP FP8 into a CDNA3 MFMA, or vice-versa.

### MXFP microscaling (CDNA4)
- A **block of 32 elements** shares one **E8M0** scale [7][8]: 8-bit, exponent-only, bias 127,
  exponents −127 … 127, no infinity and no zero, NaN = `0xFF` [7]. So a scale E means `2^(E−127)`, and
  E = 127 means no scaling (derived; the CDNA4 MFMA treats exponent `0x7f` as running without scale [2]).
  If the scale is NaN, the whole block is NaN [7][8].
- The block-scaled MFMA takes A, B, C, the A/B formats (in `CBSZ`/`BLGP`) and an E8M0 scale for A and
  for B [2]: `v_mfma_scale_f32_32x32x64_f8f6f4`. Format codes: `0=E4M3, 1=E5M2, 2=E2M3(fp6),
  3=E3M2(bf6), 4=E2M1(fp4)` [2].
- For the 32×32×64 shape: A=32×64, B=64×32, C=32×32, so (derived) the scales are Ax=32×2 and Bx=2×32
  (one per 32-element block along K), and per lane 32 A, 1 Ax, 32 B, 1 Bx, 16 C. It takes 32 cycles
  with FP4/FP6 inputs and 64 if A or B is FP8 [2].

### Rounding & subnormals (accuracy caveats)
- A bit-accurate study (MMA-Sim) found that **CDNA3**'s mixed-precision MFMAs — TF32, BF16, FP16 and
  FP8 inputs — use **round-down (RD)** in their internal summation, an asymmetric mode that gives a
  small negative bias, which matters for long-K reductions and training-like accumulation [11].
- **CDNA3 and CDNA4** MFMAs ignore the MODE denorm control and keep input/output denorms [1][2]; MMA-Sim
  found **CDNA2**'s FP16 MFMA flushing input subnormals to zero, which it links to training instability
  [11] — a workaround needed only on CDNA2.
- Always accumulate in **FP32/INT32**; never down-convert inside the K-loop.

## The levers
1. **Pick the lowest precision the task tolerates** — FP8 for most inference GEMM/attention; MXFP4/6
   on CDNA4 for weight-heavy layers with a task-accuracy gate.
2. **Match the FP8 variant to the arch** (FNUZ/OCP) in both quantization and kernel [3].
3. **Block-scale (MXFP)** instead of a single per-tensor scale when dynamic range varies within a
   tensor — the 32-element E8M0 block gives each block its own exponent [7][8].
4. **Gate on task accuracy, not byte parity**, for any quant path: the formats and the summation
   rounding above change the bits.

## Pitfalls
- **FNUZ↔OCP bit-copy** — silent garbage; always convert.
- **Assuming TF32 on CDNA4** — it moved to software emulation via BF16 [6]; code paths must fall back
  to BF16 or FP32.
- **Per-tensor scale on a wide-range tensor** — overflow/underflow; use MXFP block scales.
- **Subnormal flush assumptions** carried from CDNA2 to CDNA3/4 — unnecessary there [1][2].

## Verify
- Round-trip a tensor through the target FP8/FP4 cast and check max/relative error against an FP32
  reference; confirm bias/saturation match the arch (MMA-Sim publishes bit-accurate models [11]).
- On CDNA1–CDNA3, use `amd_matrix_instruction_calculator --detail-instruction` to confirm an MFMA's
  register usage and modifiers; the calculator has no CDNA4 target [12], so check scaled-MFMA operand
  layouts against the CDNA4 ISA guide [2].

## Sources
1. AMD, *"AMD Instinct MI300" Instruction Set Architecture* (CDNA3 ISA; §7.1 MFMA incl. denorm control and XF32, §7.2 BF8/FP8 formats) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
2. AMD, *AMD Instinct CDNA4 Instruction Set Architecture* (§7.1 MFMA incl. scaled MFMA formats, scale and cycles; FP8/BF8 format table) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
3. HIP documentation, *Low precision floating point types* — https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html
4. ROCm documentation, *Data types and precision support* — https://rocm.docs.amd.com/en/latest/reference/precision-support.html
5. AMD, *AMD CDNA 3 Architecture* white paper (TF32 format) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
6. AMD, *AMD CDNA 4 Architecture* white paper (TF32, MX formats, Table 1) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
7. Open Compute Project, *OCP Microscaling Formats (MX) Specification v1.0* (E8M0, Table 7) — https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
8. B. D. Rouhani et al., *Microscaling Data Formats for Deep Learning*, arXiv:2310.10537 — https://arxiv.org/abs/2310.10537
9. AMD, *AMD CDNA Architecture* white paper (MI100; numerical formats) — https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-white-paper.pdf
10. AMD, *AMD CDNA 2 Architecture* white paper (matrix FP64) — https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf
11. P. Xie et al., *Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units* (MMA-Sim), arXiv:2511.10909 (§VI-B, Table VI) — https://arxiv.org/abs/2511.10909
12. ROCm, *AMD Matrix Instruction Calculator* README — https://github.com/ROCm/amd_matrix_instruction_calculator
