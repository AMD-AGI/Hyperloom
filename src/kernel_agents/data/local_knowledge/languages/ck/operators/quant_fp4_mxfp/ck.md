---
title: quant_fp4_mxfp on ck — SOTA card
kind: sota_card
operator: quant_fp4_mxfp
backend: ck
gens: [gfx950]
dtypes: [mxfp4, mxfp6, mxfp8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/README.md
  - https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
---

# quant_fp4_mxfp × ck

## TL;DR
CK / CK-Tile is the C++ template path for MXFP on CDNA4 (gfx950). It owns the **block-scaled GEMM** that
consumes MXFP4/6 + E8M0 scales via the `v_mfma_scale_f32_*_f8f6f4` scaled WarpGemm
(`mfma_scale_f32_32x32x64` / `16x16x128`), and the cast itself is done in the GEMM's input pipeline or a
small CK cast kernel. Reach for CK when you want the fused **quant-in-the-GEMM-prologue + block-scaled
MFMA + epilogue** as one template — the production form for MXFP4 weight GEMM on MI350/MI355. The quant
alone is memory-bound; CK's value is the integrated low-bit GEMM and its **scale pipeline** that reshuffles
E8M0 scales through LDS into the layout the scaled-MFMA expects. → languages/composable_kernel.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| CK-Tile block-scaled GEMM (mxfp4/6) with E8M0 scaled WarpGemm | CK-Tile `ops/gemm` + `mfma_scale_f32_32x32x64_f8f6f4`; CDNA4 GEMM blog | gfx950, mxfp4/6/8 | up to **2×** vs naive (AMD-reported); MXFP4 ~5255 TFLOPS / 92% MFMA util (Gluon analog) | weight MXFP4 GEMM, integrated cast |
| CK cast / prologue (bf16→MXFP4 + E8M0) | CK-Tile prologue / aiter `gemm_a4w4` CK path | gfx950 | fused into the GEMM input pipeline | quant fused with the matrix core |
| Flatmm MX FP8/FP4 mixed pipeline (A4W6 / A6W4) | CK CHANGELOG (MX mixed in Flatmm) | gfx950 | mixed FP4×FP6 operands | mixed-precision MoE/GEMM |

### SOTA detail — the scale pipeline (GR → LW → LR)
CK does not just stream tiles; it runs a **separate scale pipeline** alongside the tile pipeline because
the E8M0 scale layout in global memory is **not** the layout the scaled-MFMA consumes, and no instruction
reads scales from registers directly into MFMA layout:
```
Global Read (scales → regs) → LDS Write (relayout) → LDS Read (feed scaled MFMA)
mfma_scale_f32_32x32x64_f8f6f4  // A,B ∈ {fp4,fp6,fp8} independently; E8M0 scale per 32-block
```
This is the C++ analog of aiter's `e8m0_shuffle` — let the CK **Policy** generate the
`tile_distribution_encoding` rather than hand-writing it.

## Config space / knobs
| knob | values | effect |
|---|---|---|
| WarpGemm | `mfma_scale_f32_32x32x64_f8f6f4` / `16x16x128` | scaled-MFMA; A/B element types independent (FP4×FP6) |
| group size | 32 | E8M0 block (OCP MX); fixed |
| scale layout | via Policy `tile_distribution_encoding` | let the Policy generate the E8M0 distribution |
| pipeline | `GemmPipelineAgBgCr*` | tile + scale double-buffering |
| epilogue | CShuffle | store relayout |
| tile `kM×kN×kK` | shape-dependent | fill the 256 CUs on MI350 → languages/composable_kernel |
| `preshuffleB` / `eightwarps` | on/off | abquant blockscale optimizations (CK CHANGELOG) |

## Measured performance
| config | metric | value @ hw / ver / date | source |
|---|---|---|---|
| MXFP4 block-scaled GEMM vs naive | speedup | up to **2×** (AMD-reported) | cdna4-gemm-kernels blog |
| MXFP4 GEMM (Gluon analog) | MFMA util | ~5255 TFLOPS, 92.4% of peak | gluon-gemm tutorial |
| MXFP6 vs MXFP4 | rate | same throughput on CDNA4 | matrix-cores-cdna blog |
| gfx942 | support | none — no FP4 MFMA (CDNA3) | HW matrix |

> The 2× and TFLOPS figures are AMD-reported / tutorial numbers, **not** measured on this box. Build the
> CDNA4 GEMM example and bench your shapes.

## Numerics / parity
- fp32 accumulate inside the scaled MFMA; the **E8M0 block scale is applied by the hardware** after the
  dot (per-32-block).
- A and B may be different MX element types (FP4×FP6) — the `f8f6f4` opcode family takes independent
  operand types.
- FP4/FP6/E8M0 semantics per numerics.md and ../../../quantization/block_scaling_mxfp.
- Task-accuracy gate; confirm the scaled MFMA + scale layout with the matrix-instruction calculator.

## Integration (rebind seam)
- Build the CK-Tile example / instance and bench at your shapes; aiter dispatches `gemm_a4w4` to a CK path
  on gfx950.
- **Not a Python-only edit** — CK is C++ templates (rebuild). → ../../../kernel_workflow/integrating_a_new_kernel.

## Pitfalls & anti-patterns
- **gfx950-only** (no FP4 HW on gfx942) — ../../../quantization/hardware_support_matrix.
- **Hand-writing the E8M0 `tile_distribution_encoding`** — let the Policy generate it; the GR→LW→LR scale
  pipeline is the part that goes wrong.
- CK repo **moved** to `ROCm/rocm-libraries:projects/composablekernel` (the `ROCm/composable_kernel`
  develop branch is now a read-only mirror) — pin the right source (languages/composable_kernel).
- The vendor 2× is AMD-reported, not measured here — bench your shapes.
- Slow CK-Tile GEMM vs old universal_gemm on some shapes (known issue) — validate before adopting.

## How to verify
- Build the CDNA4 GEMM example; bench vs the FP8 GEMM at the same shapes.
- `amd_matrix_instruction_calculator --get-register` to confirm the scaled-MFMA register/scale layout.
- e2e gsm8k on gfx950.

## Alternatives / cross-links
aiter · triton · hip · [overview.md](overview.md) ·
[numerics.md](numerics.md) · languages/composable_kernel · backends/composable_kernel_lib ·
operators/scaled_quant_gemm · ../../../quantization/block_scaling_mxfp.

## Worked example
Wire a CK MXFP4 weight GEMM through aiter's `gemm_a4w4` dispatch on gfx950:
```python
# Activations + weights quantized to MXFP4 (group 32, E8M0) by aiter per_1x32_f4_quant(shuffle=True);
# aiter.gemm_a4w4 then routes to a CK-Tile block-scaled instance that runs
#   mfma_scale_f32_32x32x64_f8f6f4 with the E8M0 scales fed via the CK scale pipeline (GR→LW→LR).
# To author a standalone CK instance, build the CDNA4 block-scaled GEMM example and pass
#   A_fp4, B_fp4, A_scale_e8m0, B_scale_e8m0 ; let the Policy own the scale distribution.
```
The scale **must** be in the Policy-generated layout — passing aiter's `e8m0_shuffle` output (or an
unshuffled scale) without matching the CK Policy corrupts results silently.

## Sources
- CK-Tile components + Policy/distribution: https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/README.md
- Block-scale / MX FP8-FP4 mixed in Flatmm, preshuffleB, eightwarps: https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md
- FP8/MXFP GEMM on AMD CDNA4 (scaled MFMA, 2× vs naive): https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
- Block-scaled MFMA, MXFP6=MXFP4 rate, type codes: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- Scaled-MFMA 16x16x128 / 32x32x64 packing + E8M0 repeat_interleave(32): https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
