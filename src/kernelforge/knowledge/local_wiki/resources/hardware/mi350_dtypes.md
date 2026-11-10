---
title: MI350X — number formats, OCP FP8, FP6/FP4, and MX block scaling
kind: hardware
topic: dtypes
gens: [gfx950]
updated: 2026-08-28
---

# Number formats on gfx950

## The three things that decide whether your kernel is correct
1. **FP8 here is OCP, not FNUZ.** A checkpoint quantized against FNUZ has to be converted, never
   reinterpreted. Bit-copying produces wrong numbers with no diagnostic.
2. **MX formats put 32 elements under one E8M0 scale.** The scale operand's layout has to match what
   `mfma_scale_*` reads, or you get silent corruption.
3. **Accumulate in FP32 or INT32.** Always. There is no format below that where narrowing the
   accumulator is a good trade.

## What exists
| Format | Bits | Exp/Mant | Notes |
|---|---:|---|---|
| FP64 | 64 | 11/52 | 78.6 TF vector; **matrix rate is half what CDNA3 gave you** |
| FP32 | 32 | 8/23 | IEEE; 157 TF on the matrix core |
| BF16 | 16 | 8/7 | wide exponent; instruction shapes are now 16×16×32 and 32×32×16 |
| FP16 | 16 | 5/10 | more mantissa, same new shapes |
| **FP8 E4M3** (`fp8`) | 8 | 4/3 | **OCP E4M3FN**: bias 7, max ±448, has ±0 and NaN, **no infinities** |
| **FP8 E5M2** (`bf8`) | 8 | 5/2 | **OCP**: bias 15, max ±57344, **does** have ±inf |
| **FP6 E2M3** (`fp6`) | 6 | 2/3 | mantissa-favoured, narrow range — weights |
| **FP6 E3M2** (`bf6`) | 6 | 3/2 | range-favoured — weights, gradients |
| **FP4 E2M1** (`fp4`) | 4 | 2/1 | max ±6; two values per byte |
| MXFP8 / MXFP6 / MXFP4 | block | + E8M0 | 32 elements share one scale |
| INT8 / INT4 | 8 / 4 | — | accumulate in INT32 |
| **TF32** | — | — | **does not exist on this part** — use BF16, or stay in FP32 |

## The FP8 encoding trap
gfx950 implements OCP. Earlier CDNA parts implemented **FNUZ** — bias 8, maximum ±240, no infinities, a
single zero, and NaN encoded as `0x80`. Those are not cosmetic differences: the **bias and the
saturation point both move**, so the same byte means a different number on each part.

What follows from that:

- Use the OCP helpers — `__amd_fp8_*` from `hip_ext_ocp.h`. The older `__hip_fp8_*` entry points are the
  FNUZ path.
- **Never hand FNUZ bytes to a gfx950 MFMA.** Nothing raises, nothing produces NaN, and the output looks
  like plausible numbers. This is the failure mode that survives a code review.
- Before trusting a downloaded quantized model, find out which flavour its quantizer emitted.

## FP4 storage
Two FP4 values occupy one byte, in `__amd_fp4x2_storage_t` (an alias for `uint8_t`), with
`__amd_extract_fp4` and `__amd_create_fp4x2` in `hip_ext_ocp.h` for packing and unpacking. Addressing
granularity is therefore 8 bits — you cannot address a single FP4 element.

## MX microscaling
The OCP MX spec, as implemented here:

| Property | Value |
|---|---|
| Block size | 32 consecutive elements along K |
| Scale format | E8M0 — 8 bits, exponent only |
| Scale value | `2^(scale − 127)`; `scale = 127` means ×1 |
| Scale range | `2^-127` … `2^127`; encoding 255 is reserved for NaN |
| MXFP8 / MXFP6 / MXFP4 | 32 elements of that width, plus one E8M0 |
| Effective width | `element_bits + 8/32` = element bits + 0.25 |

That last row is the point of the design: one 8-bit scale amortized across 32 elements costs a quarter
of a bit each.

### Why per-block beats per-tensor
With one scale for the whole tensor, the outliers set it. Everything else then has to fit underneath
that scale, and in a 4-bit format the small values simply underflow to zero — per-tensor FP4 collapses
on any heavy-tailed distribution.

Giving each group of 32 its own exponent lets every block normalize itself. Outliers stop poisoning
their neighbours. **That is what makes MXFP4 weight-only quantization usable in production** rather
than a benchmark curiosity: the accuracy cost becomes small enough to trade for the throughput.

### How the instruction applies the scale
The scaled MFMA takes A and B along with their E8M0 scale operands, and applies the scale **after the
dot product but before accumulation**.

| Type code | Format |
|---|---|
| 0 | E4M3 |
| 1 | E5M2 |
| 2 | E2M3 |
| 3 | E3M2 |
| 4 | E2M1 |

A's and B's types and scales are chosen **independently**. That is what makes mixed configurations
legal — FP4 weights against FP6 or FP8 activations, for instance, which is often the right accuracy
trade.

Operand shapes at 32×32×64:

| Operand | Shape | Per thread |
|---|---|---|
| A | 32×64 | 32 values |
| Ax (A's scales) | 32×2 | 1 |
| B | 64×32 | 32 values |
| Bx (B's scales) | 2×32 | 1 |
| C | 32×32 | 16 values |

Full instruction detail lives in `mi350_matrix_core.md`.

## Rates
| Precision | Peak | Relative to FP32 |
|---|---|---|
| FP16 / BF16 | 2.5 PF | 16× |
| FP8 (OCP) | 5 PF | 32× |
| **FP6** | **10 PF** | 64× |
| **FP4** | **10 PF** | 64× |
| MXFP8 / 6 / 4 | same as the underlying element rate | — |

**Read the FP6 row again.** It runs at the FP4 rate, not somewhere between FP8 and FP4. So choosing FP6
over FP4 costs you nothing in throughput — only in memory footprint. Whenever FP4 is too lossy for a
tensor, FP6 is free speed-wise, and picking FP4 "because it is faster" is based on an assumption that
does not hold on this part.

## Rounding and subnormals
- Subnormals are **fully supported**. The flush-to-zero workarounds you may be carrying from older
  parts are unnecessary.
- Keep the accumulator at FP32 or INT32 and never narrow it inside the K-loop. Accumulator precision is
  invisible on short reductions and decisive on long ones.
- If you need to predict MFMA conversion and accumulation behaviour exactly, MMA-Sim
  (arXiv 2511.10909) is a bit-accurate reference model.

## Turning this into kernel decisions
1. Use the **lowest precision the task tolerates** — FP8 covers most inference GEMM and attention;
   MXFP4/6 for weight-dominated layers, behind an accuracy gate.
2. **MXFP4 weight-only** on the largest weight tensors. **MXFP6** where FP4 loses too much — same speed.
3. **Mix A and B types** wherever the accuracy gate allows it; the hardware does not require symmetry.
4. **Block-scale, do not per-tensor-scale**, on anything with a wide dynamic range.
5. **Confirm OCP on both sides** — the quantizer that produced the weights, and the kernel consuming
   them.
6. **Gate quantization changes on task accuracy, never on byte parity.** Byte or err-ratio parity is
   the right gate only for a BF16↔BF16 solution swap, where the math is unchanged.

## Failure modes
| Symptom | Cause | Fix |
|---|---|---|
| Output is plausible but wrong, FP8 path | FNUZ bytes reinterpreted as OCP | convert properly; never reinterpret |
| Code referencing TF32 does not compile or behaves oddly | TF32 is gone on gfx950 | use BF16 or stay in FP32 |
| Small values vanish after quantization | one scale for a heavy-tailed tensor | move to MX block scales |
| MXFP result is corrupted, no error raised | Ax/Bx laid out differently than the instruction reads them | verify with the instruction calculator before wiring it up |
| FP4 chosen over FP6 for throughput | they run at the same 10 PF rate | use FP6 and keep the accuracy |
| FP64 matrix code slower than the CDNA3 estimate | the matrix rate is halved on this part | re-derive the budget |

## Verify
| Check | How |
|---|---|
| The cast is the right flavour | round-trip a tensor through the target FP8/FP6/FP4 cast; compare max and relative error to an FP32 reference, and confirm the bias and saturation point are **OCP** |
| Input and output dtypes of an instruction | `amd_matrix_instruction_calculator --architecture cdna4 --detail-instruction` |
| Scale operand placement, before writing MXFP code | the same tool with `--get-register --Ax` / `--Bx` |
| MXFP weights are acceptable | per-block error **and** end-task accuracy against FP16 — the first alone will mislead you |

## Related
`mi350_matrix_core.md` (the scaled intrinsics and the shape table) ·
`mi350_overview.md` (peak rates in context) ·
`../common_methodology/optimization/lever_numerics.md` (how to run the accuracy gate)
