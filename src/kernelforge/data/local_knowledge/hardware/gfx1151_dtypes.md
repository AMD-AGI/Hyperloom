---
title: gfx1151 — numeric formats, WMMA dtypes and software-route boundaries
kind: hardware
topic: dtypes
gens: [gfx1151]
updated: 2026-09-02
---

# Number formats on gfx1151

The RDNA3.5 ISA and the serving software stack answer different questions. The ISA defines
arithmetic/WMMA forms; a framework defines storage layouts, quantization metadata, loaders and
routing. Do not turn support at one layer into a claim about another.

## Architecturally relevant forms

| Format/domain | ISA-relevant use | Robust accumulation |
|---|---|---|
| F32 | scalar/vector arithmetic; WMMA accumulator for F16/BF16 inputs | F32 |
| F16 | scalar/vector/packed arithmetic; WMMA input and optional narrow output form | F32 for long reductions |
| BF16 | scalar/vector/packed arithmetic; WMMA input and optional narrow output form | F32 for long reductions |
| I32/U32 | scalar/vector integer arithmetic; integer matrix accumulator | I32/U32 as algebra requires |
| I8/U8 | packed/dot arithmetic; IU8 WMMA input | I32 |
| I4/U4 codes | packed/dot and IU4 WMMA code domain | I32 |

The `IU` name means per-operand signed/unsigned selection. It does not describe a complete tensor
format, zero point, group scale, packing order or quantization method.

## Packed arithmetic

- 16-bit packed math places two independent halves in one 32-bit VGPR.
- 8-/4-bit dot and WMMA forms consume packed codes according to instruction-specific lane layouts.
- Sub-byte elements are not independently byte-addressable; software must define nibble order and
  aligned storage.
- Packed storage savings do not guarantee compute savings if unpack/reorder/scale traffic dominates.

## WMMA dtype boundary

The official RDNA3.5 WMMA list includes:

- F16→F32 and BF16→F32;
- F16→F16 and BF16→BF16 forms;
- IU8→I32;
- IU4→I32.

It does **not** justify importing gfx950's OCP FP8, FP6, FP4, MXFP or scaled-MFMA tables. If a
framework serves an FP8/MX/low-bit model on gfx1151, identify whether it uses:

- conversion plus F16/BF16 WMMA;
- integer dot/WMMA;
- VALU dequantization plus GEMM;
- Triton software kernels;
- an external library;
- a fallback path.

The route—not the model-card label—defines the hardware claim.

## Floating-point behavior

- WMMA floating forms use round-to-nearest-even and generate no ALU exceptions.
- General VALU rounding/denormal behavior is controlled by MODE fields where the instruction permits.
- SALU floating support has narrower semantics than general VALU: do not substitute it when denormal,
  rounding or exception behavior matters.
- Memory/LDS operations do not magically convert tensor formats unless the selected instruction does.
- Long reductions should retain F32 accumulation through K and narrow at the output boundary.

## Integer quantization algebra

For an integer matrix path, write the exact recurrence before code. A typical asymmetric dot has
terms for:

```text
sum((w_code - w_zero) * (a_code - a_zero))
```

Expanding that expression introduces row/column sums and zero-point corrections. Whether those sums
are precomputed, packed or fused is part of the ABI. Dropping a term can yield plausible but wrong
model output.

For IU4/IU8 WMMA specifically:

- signedness controls must match operand code domains;
- accumulate in I32;
- test extreme codes and nonzero zero-points;
- prove nibble/byte order and any matched-K permutation;
- apply per-output scale metadata to the correct output coordinate, not merely the current lane.

## Software formats retained on Strix

The retained gfx1151 bridge/framework work contains software routes for formats such as FP8-FNUZ,
MXFP4, W8A8, W4A8, W4A4 and W4A16. Their existence proves only what their exact tests and serving
receipts prove. For each route record:

- source artifact format and metadata;
- loader/quantization scheme name;
- selected kernel implementation;
- physical code route;
- accumulator/conversion semantics;
- quality and correctness limits;
- supported framework/image/version.

Do not describe all these formats as native RDNA3.5 matrix dtypes.

## Choosing a route

1. Preserve source/model quality constraints first.
2. Determine whether the serving framework can load the exact artifact.
3. Inspect the selected physical kernel and its accumulator.
4. Include packing/reordering/scaling costs in the metric.
5. Gate numerical correctness independently of performance.
6. Gate model quality independently of format validity.
7. Keep fallback behavior explicit.

## Failure modes

| Symptom | Likely class | Required check |
|---|---|---|
| Plausible but shifted output | signedness/zero-point/correction error | exact integer reference |
| Alternating columns/rows wrong | fragment or nibble permutation | basis matrices/output-coordinate map |
| Microkernel wins, model loses | packing/scale/dispatch overhead | end-to-end route timing |
| Model loads but wrong kernel runs | framework fallback | request-owned route proof |
| FP8/MX label assumed native | cross-architecture claim leak | inspect generated ISA and loader |
| Small tests pass, long reduction drifts | narrow accumulator | long-K F32/I32 reference |

## Verify

- Round-trip storage packing independently of matrix arithmetic.
- Test scalar/reference dequantization for extreme and random blocks.
- Run basis/nonuniform matrix cases through the physical target kernel.
- Inspect the exact code object and target instruction family.
- Verify finite outputs and model-level quality against the immutable source model.
- Treat any unproven fallback as a failure of route qualification.

## Sources

- AMD RDNA3.5 ISA guide, packed VALU, dot and WMMA sections.
- Retained gfx1151 route evidence is software-specific and must be cited separately per framework.

## Related

`gfx1151_matrix_core.md` · `gfx1151_isa.md` · `gfx1151_memory.md` ·
`common_methodology/optimization/lever_numerics.md`
