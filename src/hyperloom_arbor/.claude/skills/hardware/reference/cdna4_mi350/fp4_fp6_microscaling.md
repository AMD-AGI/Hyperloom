---
title: CDNA4 / MI350 (gfx950) — FP4 / FP6 / MXFP microscaling
kind: hardware
gens: [gfx950]
dtypes: [fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1, mxfp8, mxfp6, mxfp4]
regimes: [both]
updated: 2026-06-08
---

# CDNA4 / MI350 (gfx950) — FP4 / FP6 / MXFP microscaling

> The new low-bit numerics on CDNA4. Matrix-core instruction detail in
> [matrix_core_blockscale.md](matrix_core_blockscale.md); cross-gen format facts in
> [../shared/dtype_numerics.md](../shared/dtype_numerics.md).

## TL;DR
> CDNA4 adds **FP6 (E2M3/E3M2)** and **FP4 (E2M1)**, and **MXFP** block scaling where a **32-element
> block** shares one **E8M0** (8-bit, exponent-only) scale. FP6 and FP4 both run at **10 PF** (the FP4
> rate). MXFP recovers accuracy at low bit-width by giving each 32-element block its own dynamic range.

## Concepts

### The low-bit formats
| Format | Bits | Layout | Max (per element) | Use |
|---|---|---|---|---|
| FP8 E4M3 | 8 | 4-exp/3-mant | ±448 (OCP) | activations/weights |
| FP8 E5M2 | 8 | 5/2 | ±57344 | wide range |
| **FP6 E2M3** | 6 | 2/3 | small range, more mantissa | weights |
| **FP6 E3M2** | 6 | 3/2 | wider range | weights/grad |
| **FP4 E2M1** | 4 | 2/1 | ±6 (E2M1) | aggressive weight quant |

FP4 packs **2 values per byte** (`__amd_fp4x2_storage_t`, alias `uint8_t`); helpers
`__amd_extract_fp4`, `__amd_create_fp4x2` from `hip_ext_ocp.h`. Addressing granularity is 8 bits.

### MXFP = microscaling (OCP MX spec)
- A **block of 32 consecutive elements** along K shares **one E8M0 scale**.
- **E8M0**: 8 bits, exponent only, value `2^(scale-127)`; `scale=127` ⇒ ×1 (no scaling); `E=255`
  reserved for NaN; range `2^-127 … 2^127`.
- `MXFP8` = 32× FP8 + 1 E8M0; `MXFP6` = 32× FP6 + 1 E8M0; `MXFP4` = 32× FP4 + 1 E8M0.
- Effective bits/element ≈ element_bits + 8/32 = element_bits + 0.25 (the scale amortizes over 32).

### Why block scaling matters
A single per-tensor scale wastes dynamic range when values vary across a tensor; outliers force a coarse
scale that underflows small values. A **per-32-element** E8M0 lets each block self-normalize, so FP4/FP6
keep usable precision where per-tensor FP4 would collapse. This is what makes **MXFP4 weight-only
quant** viable for inference with a small task-accuracy hit.

### How the hardware applies the scale
The block-scaled MFMA (`v_mfma_scale_f32_*_f8f6f4`) takes the A/B operands **plus** their E8M0 scale
operands (Ax/Bx) and applies the scale **after the dot product, before accumulation**. Type codes:
`0=E4M3, 1=E5M2, 2=E2M3, 3=E3M2, 4=E2M1`. A and B types/scales are independent → mix FP4 weights with
FP6/FP8 activations. Layout (32×32×64): A 32×64, **Ax 32×2**, B 64×32, **Bx 2×32**, C 32×32; per-thread
32 A / 1 Ax / 32 B / 1 Bx / 16 C.

### Throughput
| Precision | CDNA4 peak | vs FP32 |
|---|---|---|
| FP16/BF16 | 2.5 PF | 16× |
| FP8 (OCP) | 5 PF | 32× |
| FP6 | **10 PF** | 64× |
| FP4 | **10 PF** | 64× |
| MXFP8/6/4 | matches the underlying element rate | — |

FP6 runs at the FP4 rate (not between FP8 and FP4) — choosing FP6 over FP4 costs **accuracy headroom,
not throughput**, so prefer FP6 when FP4 is too lossy.

## The levers
1. **MXFP4 weight-only** for the largest weight tensors; **MXFP6** when FP4 is too lossy (same speed).
2. **Mix A/B types** — FP4 weights × FP6/FP8 activations as the accuracy gate dictates.
3. **Block-scale (32-elem E8M0)** rather than per-tensor scale on wide-range tensors.
4. **Always gate on task accuracy**, not byte parity, for any FP4/FP6 path.

## Pitfalls
- **Per-tensor FP4** — underflows; use MXFP block scaling.
- **Wrong Ax/Bx scale layout** → silent corruption; verify with the calculator.
- **Choosing FP4 over FP6 "for speed"** — same 10 PF rate; FP6 just gives more mantissa.
- **Assuming MXFP on CDNA3** — CDNA4-only.

## Verify
- Round-trip weights through the MXFP4/6 cast; measure per-block error and end-task accuracy vs FP16.
- `amd_matrix_instruction_calculator --architecture cdna4 --instruction
  v_mfma_scale_f32_32x32x64_f8f6f4 --get-register --Ax`/`--Bx` for the exact scale placement.
