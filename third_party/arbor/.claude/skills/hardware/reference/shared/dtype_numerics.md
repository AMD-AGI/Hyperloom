---
title: Datatype numerics — FP8 FNUZ/OCP, FP6/FP4, MXFP, TF32 (CDNA cross-gen)
kind: hardware
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp64, fp32, tf32, bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1, mxfp8, mxfp6, mxfp4, int8, int4]
regimes: [both]
updated: 2026-06-08
---

# Datatype numerics across CDNA

## TL;DR
> Two correctness traps dominate: **(1) FP8 is FNUZ on CDNA3 but OCP on CDNA4** — different bias and
> saturation, so a checkpoint must be **re-cast**, never bit-copied across gens; **(2) MXFP4/6/8** pack
> a block of **32 elements** sharing one **E8M0** (8-bit, exponent-only) scale — the scale layout must
> match the `mfma_scale_*` expectation. Always **accumulate in FP32/INT32**.

## Concepts

### Format zoo
| Format | Bits | Exp/Mant | Where | Notes |
|---|---|---|---|---|
| FP64 | 64 | 11/52 | all gens (matrix on CDNA2+) | HPC; CDNA4 halves matrix FP64 rate |
| FP32 | 32 | 8/23 | all | IEEE |
| TF32 | 19 used | 8/10 | **CDNA3 only** | emulated; **removed in CDNA4** (BF16/FP32 emulation) |
| BF16 | 16 | 8/7 | all | wide range; RD-accumulate on CDNA3 |
| FP16 | 16 | 5/10 | all | more mantissa; RD-accumulate on CDNA3 |
| FP8 E4M3 (fp8) | 8 | 4/3 | CDNA3 FNUZ / CDNA4 OCP | activations/weights |
| FP8 E5M2 (bf8) | 8 | 5/2 | CDNA3 FNUZ / CDNA4 OCP | larger range, less mantissa |
| FP6 E2M3 (fp6) | 6 | 2/3 | **CDNA4** | runs at FP4 rate |
| FP6 E3M2 (bf6) | 6 | 3/2 | **CDNA4** | |
| FP4 E2M1 (fp4) | 4 | 2/1 | **CDNA4** | 2 per byte (`__amd_fp4x2_storage_t`) |
| MXFP8/6/4 | block | + E8M0 | **CDNA4** | 32-elem block shares one E8M0 scale |
| INT8 / INT4 | 8/4 | — | all (INT8) / packed | INT32 accumulate |

### FP8 FNUZ vs OCP — the critical gen split
- **CDNA3 (gfx942) = FNUZ** (`F`inite, `U`nsigned `Z`ero):
  - E4M3FNUZ: **bias 8**, max **±240**, **no inf**, single zero (+0 only), NaN = `0x80`.
  - E5M2FNUZ: bias 16, max **±57344**, no inf.
- **CDNA4 (gfx950) = OCP** (matches NVIDIA/most vendors):
  - E4M3FN: **bias 7**, max **±448**, ±0, NaN (no inf).
  - E5M2: bias 15, max ±57344, **with ±inf**.
- **Implication:** an FP8 model quantized for OCP must be **re-cast** for CDNA3 (different bias and
  saturation point). Use the arch-matching `__hip_fp8_*` (CDNA3, `hip_fp8.h`) vs `__amd_fp8_*`
  (gfx950, `hip_ext_ocp.h`) helpers. **Never bit-copy** OCP FP8 into a CDNA3 MFMA, or vice-versa.

### MXFP microscaling (CDNA4)
- A **block of 32 consecutive elements** shares one **E8M0** scale (8-bit, exponent-only, value
  `2^(scale-127)`; `scale=127` ⇒ no scaling; `E=255` reserved for NaN; range `2^-127 … 2^127`).
- The block-scaled MFMA applies the scale **after the dot product, before accumulation**:
  `v_mfma_scale_f32_32x32x64_f8f6f4(A, B, C, Atype, Btype, opsel_a, scale_a, opsel_b, scale_b)`.
  Type codes: `0=E4M3, 1=E5M2, 2=E2M3(fp6), 3=E3M2(bf6), 4=E2M1(fp4)`.
- For the 32×32×64 shape: A=32×64, Ax(scales)=32×2, B=64×32, Bx=2×32, C=32×32; per-thread 32 A, 1 Ax,
  32 B, 1 Bx, 16 C. FP6/FP4-only runs at the **lower** cycle count; involving FP8 raises it.

### Rounding & subnormals (accuracy caveats)
- **CDNA3** FP16/BF16/TF32 MFMA conversion/accumulation uses an asymmetric **round-down (RD)** mode —
  a small systematic bias that matters for long-K reductions and training-like accumulation; the FP8
  path was specifically adjusted to mitigate it (MMA-Sim, arXiv 2511.10909).
- **CDNA3/CDNA4 fully support subnormals**; **CDNA2 flushed some** subnormals (hurt training
  stability) — gen-specific workaround needed only on CDNA2.
- Always accumulate in **FP32/INT32**; never down-convert inside the K-loop.

## The levers
1. **Pick the lowest precision the task tolerates** — FP8 for most inference GEMM/attention; MXFP4/6
   on CDNA4 for weight-heavy layers with a task-accuracy gate.
2. **Match the FP8 variant to the arch** (FNUZ/OCP) in both quantization and kernel.
3. **Block-scale (MXFP)** instead of a single per-tensor scale when dynamic range varies within a
   tensor — the 32-element E8M0 block recovers accuracy at low bit-width.
4. **Gate on task accuracy, not byte parity** for any quant path; bf16↔bf16 solution swaps can use
   byte/err-ratio parity.

## Pitfalls
- **FNUZ↔OCP bit-copy** — silent garbage; always convert.
- **Assuming TF32 on CDNA4** — it is gone; code paths must fall back to BF16 or FP32.
- **Per-tensor scale on a wide-range tensor** — overflow/underflow; use MXFP block scales.
- **Subnormal flush assumptions** carried from CDNA2 to CDNA3/4 — unnecessary and lossy there.

## Verify
- Round-trip a tensor through the target FP8/FP4 cast and check max/relative error against an FP32
  reference; confirm bias/saturation match the arch (MMA-Sim is a bit-accurate reference).
- Use `amd_matrix_instruction_calculator --detail-instruction` to confirm the in/out dtypes and the
  scale layout of a scaled MFMA before wiring up scales.
