---
title: quant_dequant_fp8 — numerics
kind: operator_overview
operator: quant_dequant_fp8
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://arxiv.org/html/2511.10909v1
  - vllm-project/vllm@HEAD:csrc/quantization/fp8/common.cu
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/quant.py
---

# quant_dequant_fp8 — numerics

> The accuracy heart of FP8. Cross-gen format facts: hardware/shared/dtype_numerics;
> CDNA4 block-scaled MFMA: hardware/cdna4_mi350.

## The two FP8 formats
| format | exp/mant | bias (OCP) | max (OCP) | inf | use |
|---|---|---|---|---|---|
| **E4M3** | 4/3 | 7 | **±448** | no (NaN only) | activations + weights — **the default** (more mantissa) |
| **E5M2** | 5/2 | 15 | **±57344** | yes (±inf) | gradients / very wide range; rarely for inference activations |

E4M3 is preferred for inference: 3 mantissa bits give ~2× the relative precision of E5M2, and ±448 is
plenty once a per-token/per-tensor scale brings the tensor into range. E5M2 only wins when the dynamic
range exceeds ±448 *after* scaling (uncommon for well-scaled activations; common for raw gradients).

## FNUZ (CDNA3) vs OCP (CDNA4) — the gen split that bites
- **CDNA3 / gfx942 = FNUZ** ("Finite, Unsigned Zero"):
  - **E4M3FNUZ: bias 8, max ±240, no inf, single (+0) zero, NaN = `0x80`.**
  - E5M2FNUZ: bias 16, max ±57344, no inf.
- **CDNA4 / gfx950 = OCP**: E4M3FN bias 7, ±448, ±0, NaN (no inf); E5M2 bias 15, with ±inf.
- **Consequence:** a checkpoint quantized for OCP and bit-copied into a CDNA3 FNUZ MFMA is **off by an
  exponent** → values ~2× wrong, silently. Re-cast across gens; never bit-copy. Use the arch-matching
  helpers: `__hip_fp8_*` (`hip_fp8.h`, CDNA3) vs `__amd_fp8_*` (`hip_ext_ocp.h`, gfx950).
- vLLM encodes this directly: on ROCm `FP8_TYPE = c10::Float8_e4m3fnuz`
  (`csrc/quantization/fp8/common.cu`).

## The 224.0 ROCm cap (a real, sourced gotcha)
vLLM dynamic FP8 quant on ROCm clamps to **224.0**, not the FNUZ max 240:
```cpp
// vllm csrc/quantization/fp8/common.cu
// "Using the default max value from pytorch (240.0) will cause accuracy
//  issue when running dynamic quantization. Here use 224.0f for rocm."
constexpr auto FP8_E4M3_MAX = 224.0f;
```
Reason: leaving a little headroom below the saturation point avoids round-to-saturation artifacts on the
largest-magnitude elements during dynamic (amax-derived) quant. Static/calibrated paths can use the full
range; dynamic paths benefit from the margin.

## Scale computation (amax)
- **scale = amax / dtype_max**, then `x_fp8 = x / scale` (or `x * (1/scale)` with a reciprocal scale).
- aiter `pertoken_quant`: `per_token_amax = max(|x|, dim=-1)`, `scale = amax / dtypeMax`, and crucially
  **`scale[scale==0] = 1`** — an all-zero row must not divide by zero (`aiter/ops/quant.py`).
- **per-tensor dynamic** needs a global amax → a two-pass or atomic-max reduction
  (`segmented_max_reduction` in vLLM, `atomicMaxFloat`); **per-token** is a single row-reduction (cheaper,
  better accuracy because each row gets its own range).
- **Reciprocal-scale trick:** store `1/s` so the per-element hot path is a multiply, not a divide
  (`is_scale_inverted` in vLLM `scaled_fp8_conversion`).

## Scale granularity vs accuracy (the central tradeoff)
| granularity | accuracy | cost | notes |
|---|---|---|---|
| per-tensor static | lowest | cheapest | needs calibration; outliers force a coarse scale → small values underflow |
| per-tensor dynamic | low–mid | +amax reduce | no calibration; still one scale for the whole tensor |
| **per-token dynamic** | **good** | one row-reduce | each row self-normalizes; the SmoothQuant-style default for activations |
| per-block (1×128 / group) | best (FP8) | per-group reduce | DeepSeek block-fp8; each 128-wide block scaled; → fine-grained GEMM |

Per-tensor wastes dynamic range when a few outlier tokens dominate amax — the coarse scale underflows the
quiet tokens. Per-token / per-block fixes this by giving each row/block its own scale. This is the FP8
analog of the MXFP 32-element block argument (operators/quant_fp4_mxfp, hardware/cdna4_mi350).

## Stochastic rounding
Round-to-nearest-even (RNE) is the default. **Stochastic rounding (SR)** — round up with probability
proportional to the residual — removes the systematic bias of RNE in long accumulations and is used
mainly in **FP8 training** (weight updates) and some KV paths, not standard inference activation quant.
CDNA3's MFMA conversion uses an asymmetric **round-down** mode for FP16/BF16 that introduces a small
systematic bias on long-K reductions (the FP8 path was specifically adjusted to mitigate it — MMA-Sim,
arXiv 2511.10909). For inference quant, RNE + per-token scaling is the practical default; reach for SR
only when measured accumulation bias matters.

## Accuracy gates (do NOT gate on byte parity)
- **bf16↔bf16** library swaps → byte/err-ratio parity OK (`err_ratio < 0.05`, aiter gradlib gate,
  `gradlib/gradlib/GemmTuner.py`).
- **Any FP8 path is lossy by construction** → gate on **task accuracy** (gsm8k / mmlu / perplexity), not
  bit parity. A reasonable isolated gate: round-trip max-rel-error and the err-ratio (fraction of
  elements beyond rtol/atol) below a threshold; aiter's `checkAllclose(..., tol_err_ratio=0.05)`
  (`aiter/test_common.py:400`) is the convention used across aiter tests.
- The serving gate is e2e: tok/s up AND task metric within band (e.g. gsm8k drop < ~0.5–1 pt).

## Pitfalls
- **FNUZ↔OCP bit-copy** — silent ~2× error. Match the dialect to the arch.
- **Per-tensor on outlier-heavy activations** — quiet tokens underflow; use per-token.
- **Forgetting the 224 cap** on ROCm dynamic quant — saturation artifacts on max elements.
- **Dividing by a zero scale** on all-zero rows — clamp scale to 1.
- **E5M2 for activations** — wastes mantissa; E4M3 is the inference default.

## Verify
- Round-trip a representative activation through the FP8 cast; check max-rel error and err-ratio vs an
  fp32 reference; confirm the saturation point matches the arch (MMA-Sim is bit-accurate).
- e2e: gsm8k/mmlu before/after FP8 enable, same seed/temp=0.

## Sources
- FNUZ vs OCP bias/max, E4M3/E5M2 ranges: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- CDNA3 round-down rounding, FP8 adjustment, SR context (MMA-Sim): https://arxiv.org/html/2511.10909v1
- 224.0 ROCm cap, scale-inverted, per-token amax: `vllm-project/vllm@HEAD:csrc/quantization/fp8/common.cu`.
- amax/scale, zero-scale clamp, err-ratio gate: `ROCm/aiter@a6bb49937:aiter/ops/quant.py`, `aiter/test_common.py`, `gradlib/gradlib/GemmTuner.py`.
