---
title: numerical stability (fp32 accumulate, online softmax, Welford, fp8 FNUZ/OCP)
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp4_e2m1, fp32]
regimes: [prefill, decode, training, both]
updated: 2026-06-05
---

# numerical stability

## TL;DR
Speed must not break accuracy. The non-negotiables on MI300X/MI350X: **accumulate matmul/reductions in
fp32** (MFMA already accumulates in fp32 — keep it), use **online (streaming) softmax** with a running
max for attention, use **Welford** for single-pass mean/variance in norms, and watch the **fp8 format
trap**: CDNA3 fp8 is **FNUZ** (`*_fnuz`, different bias/no-inf) while CDNA4 adds **OCP E4M3/E5M2** — a
silent ~2× scale/exponent mismatch if you reuse FNUZ scales on OCP (or vice-versa). Gate every fast path
against an fp32 reference (`err_ratio < 0.05` for GEMM tuning,
`[[optimization/autotuning_methodology.md]]`). See `[[hardware/shared/dtype_numerics.md]]`,
`[[quantization/]]`, `[[hardware/cdna4_mi350/fp4_fp6_microscaling.md]]`.

## The invariants
### fp32 accumulation
- MFMA computes `D = A·B + C` with **fp32 accumulators in AGPRs** even for bf16/fp16/fp8 inputs — do not
  down-cast the accumulator inside the K-loop (`[[optimization/mfma_scheduling.md]]`). Cast to the
  output dtype only in the epilogue.
- Long reductions (softmax denom, norm sum, logsumexp) accumulate in fp32 regardless of I/O dtype;
  bf16 accumulation of a long sum loses bits fast (`[[operators/reduction/overview.md]]`).

### Online softmax (attention)
- Stream K-blocks keeping running `m` (max) and `l` (denom); rescale the partial output by
  `exp(m_old − m_new)` per block. Avoids overflow from `exp(large)` and needs no second pass — the basis
  of flash-style attention (`[[operators/attention_prefill_fmha.md]]`,
  `[[operators/attention_decode_paged.md]]`, `[[operators/softmax/overview.md]]`).

### Welford (norms)
- Single-pass mean/variance with a numerically stable running update (vs the catastrophic-cancellation
  `E[x²]−E[x]²`). Use for RMSNorm/LayerNorm, fp32 accumulators
  (`[[operators/rmsnorm/overview.md]]`, `[[operators/layernorm/overview.md]]`).

## The fp8 format trap (FNUZ vs OCP)
| | CDNA3 (MI300X) | CDNA4 (MI350X) |
|---|---|---|
| fp8 E4M3 | **FNUZ** (`fp8_e4m3_fnuz`) — bias 8, no inf, distinct NaN | **OCP** E4M3 (and FNUZ) |
| fp8 E5M2 | **FNUZ** (`fp8_e5m2_fnuz`) | **OCP** E5M2 |
| risk | — | reusing a scale/codepoint mapping across FNUZ↔OCP shifts values ~2× (different exponent bias / special-value handling) |

- Pick the dtype id explicitly in frontmatter and code; never assume "fp8 is fp8" across gens.
- CDNA4 adds **microscaled MX (fp8/fp6/fp4)** with per-block scales — the scale granularity is part of
  numeric correctness (`[[hardware/cdna4_mi350/fp4_fp6_microscaling.md]]`,
  `[[operators/quant_fp4_mxfp/overview.md]]`).

## Scaling (quantization hygiene)
- **Per-tensor / per-channel / per-block** scales: choose the finest the kernel can afford; compute
  `amax` in fp32, derive scale, clamp to the format max before cast
  (`[[operators/scaled_quant_gemm/overview.md]]`, `[[operators/quant_dequant_fp8/overview.md]]`).
- Fuse `amax`+quant into the producing pass to avoid an extra read
  (`[[optimization/kernel_fusion_strategy.md]]`), but keep the `amax` reduction in fp32.
- For dynamic activations, recompute scale per tensor per step; for static, calibrate offline.

## Pitfalls
- Down-casting the MFMA accumulator inside the K-loop ⇒ accuracy loss for "free", no speed gain.
- bf16/fp16 accumulation of softmax denom or norm sum ⇒ drift, sometimes NaN at long context.
- Reusing FNUZ scales on OCP fp8 (or porting an H100 OCP-fp8 recipe to MI300X FNUZ) ⇒ ~2× error.
- Skipping the fp32 oracle gate when autotuning ⇒ a "faster" kernel that is wrong
  (`err_ratio<0.05`, `[[optimization/autotuning_methodology.md]]`).
- Per-tensor scale on a heavy-tailed activation ⇒ clipping; use per-channel/block.

## Verify
- Oracle: compare fast path vs fp32 reference; max relative error / `err_ratio` within tolerance.
- Attention: check logsumexp stability at long context (no NaN/inf).
- fp8: confirm the exact dtype id (`*_fnuz` vs OCP) matches the target gen and the scale source.
