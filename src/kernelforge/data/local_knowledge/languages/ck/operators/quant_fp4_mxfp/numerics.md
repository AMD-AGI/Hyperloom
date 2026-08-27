---
title: quant_fp4_mxfp — numerics
kind: operator_overview
operator: quant_fp4_mxfp
gens: [gfx950]
dtypes: [fp4_e2m1, fp6_e2m3, fp6_e3m2, mxfp4, mxfp6]
regimes: [both]
updated: 2026-06-08
sources:
  - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  - https://arxiv.org/pdf/2310.10537
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/quant.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# quant_fp4_mxfp — numerics

> The star numerics page. MXFP is **block floating point**: 32 elements share an exponent (the E8M0
> scale), so each block self-normalizes. This is what makes 4-bit usable. CDNA4-only HW; CDNA3 simulates.
> Cross-refs: hardware/cdna4_mi350, hardware/shared/dtype_numerics.

## The low-bit element formats
| format | bits | exp/mant | per-element max | role |
|---|---|---|---|---|
| **FP4 E2M1** | 4 | 2/1 | **±6** | aggressive weight quant (2/byte, `__amd_fp4x2_storage_t`) |
| **FP6 E2M3** | 6 | 2/3 | small range, more mantissa | weights when FP4 too lossy |
| **FP6 E3M2** | 6 | 3/2 | wider range | weights/grad |
FP4 E2M1 representable values: `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` — only **15 levels**. Without a
per-block scale this is hopeless for real tensors; the E8M0 block scale is what rescues it.

## E8M0 block scale (the core mechanism)
- **8 bits, exponent-only, no sign, no mantissa**: value `2^(scale-127)`. `scale=127` ⇒ ×1; `E=255`
  reserved NaN; range `2^-127 … 2^127` (it is literally an FP32 biased exponent).
- **One scale per 32 consecutive elements** (group size **32**, fixed by the OCP MX spec — do not tune).
- Effective bits/element ≈ element_bits + 8/32 = element_bits + **0.25** (the scale amortizes over 32):
  MXFP4 ≈ 4.25 b/elt, MXFP6 ≈ 6.25, MXFP8 ≈ 8.25.

## Scale computation (amax → power-of-2)
For each 32-block (aiter `per_1x32_f4_quant`):
1. `block_amax = max(|x_block|)`.
2. `FP4_MAX = 2^floor(log2(6)) = 4.0` (kernel uses the **power-of-2** max, not 6, because E8M0 scales are
   powers of two and the largest element must land at the top representable power).
3. `s_e8m0 = f32_to_e8m0(block_amax / FP4_MAX)` — rounds the ratio to the nearest power-of-2 exponent.
4. `x_fp4 = f32_to_mxfp4(x_block / e8m0_to_f32(s_e8m0))`.
The OCP spec: pick the scale so the **max element of the block lands at the max element-type exponent**,
maximizing range without overflow; a rounding mode converts the max to the nearest power-of-2.

## Scale layout & shuffle (hardware correctness)
The block-scaled MFMA (`v_mfma_scale_f32_32x32x64_f8f6f4`) expects the E8M0 scales in specific operand
slots (Ax 32×2, Bx 2×32 for the 32×32×64 shape). aiter `shuffle=True` rearranges the scale tensor
(`e8m0_shuffle`) and pads it (`((m+255)//256*256, ((n+31)//32+7)//8*8)`) to match. **Wrong shuffle = silent
corruption** — verify with `amd_matrix_instruction_calculator --get-register --Ax/--Bx`.

## FP4 vs FP6 — accuracy at the same throughput
Both run at **10 PF** (FP4 rate) on CDNA4. So FP6 is "free" extra accuracy:
- **MXFP4** (4.25 b): near-lossless on very large models (AMD: DeepSeek-R1-0528); noticeable degradation
  on small/mid models.
- **MXFP6** (6.25 b): more mantissa (E2M3) or range (E3M2); consistently better than MXFP4 at the *same
  FLOPs* — AMD recommends MXFP6 or mixed MXFP4/MXFP6 when MXFP4 alone degrades.
- **Mixed precision**: independent A/B types in the scaled MFMA → FP4 weights × FP6/FP8 activations, set
  by the accuracy gate (hardware/cdna4_mi350).

## Accuracy techniques (beyond plain cast)
- **Rotations / Hadamard (QuaRot)** + **SmoothQuant** before MXFP4 to spread outliers — AMD's
  "near-lossless" MXFP4 recipe combines fine-tuned rotations with SmoothQuant.
- **even_round** scaling mode (`scaling_mode="even"` in `dynamic_mxfp4_quant`) — the Quark
  `even_round` block-scale rounding.

## CDNA3 vs CDNA4 (HW support — critical)
| | CDNA3 (gfx942) | CDNA4 (gfx950) |
|---|---|---|
| FP4/FP6 MFMA | **✗ (no HW)** | ✓ (10 PF) |
| MXFP block-scaled MFMA | **✗** | ✓ (`v_mfma_scale_*`, E8M0) |
| MXFP4 inference | **software simulation** (dequant-on-the-fly to fp16 in a fused kernel) | native |
- **vLLM `FP4BMM` crashes gfx942** (no FP4 HW; set `VLLM_ROCM_USE_AITER_FP4BMM=0`).
- On MI300/MI325/MI250 you can *simulate* MXFP4 GEMM (Quark on-the-fly dequant) for accuracy validation,
  but there is **no throughput win** — the value is footprint + forward-compat to MI350.

## Accuracy gates
- **Never byte parity.** Gate on per-block round-trip error AND end-task accuracy (gsm8k/mmlu/perplexity).
- MXFP4 is the most aggressive path here — budget a larger task-accuracy band and prefer MXFP6/mixed if it
  fails. Same `err_ratio<0.05` isolated convention, but the e2e task gate dominates.

## Pitfalls
- **Per-tensor FP4** — collapses; MXFP block scaling is mandatory.
- **Wrong E8M0 scale shuffle/layout** → silent corruption (calculator check).
- **Group size ≠ 32** — fixed by spec; `dynamic_mxfp4_quant` comments "Do not tune this."
- **Assuming MXFP on CDNA3** — simulation only, no speedup; FP4BMM crashes gfx942.
- **FP4 over FP6 "for speed"** — same 10 PF; FP6 just adds accuracy.

## Verify
- Round-trip weights through MXFP4/6 cast; measure per-block error + end-task accuracy vs FP16.
- `amd_matrix_instruction_calculator --architecture cdna4 --instruction v_mfma_scale_f32_32x32x64_f8f6f4
  --get-register --Ax/--Bx` before wiring scales.

## Sources
- OCP MX spec (group 32, E8M0, scale selection = max→max-exponent): https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- Microscaling Data Formats for Deep Learning (the MX paper): https://arxiv.org/pdf/2310.10537
- aiter scale computation (`f32_to_e8m0`, power-of-2 max, shuffle, group 32): `ROCm/aiter@a6bb49937:aiter/ops/quant.py`, `aiter/ops/triton/quant/quant.py`.
- FP6@FP4 rate, scaled MFMA, type codes, CDNA3 vs CDNA4: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- MXFP4 near-lossless (rotations+SmoothQuant), MXFP6/mixed recommendation: https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-online-rotation/README.html ; https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html
