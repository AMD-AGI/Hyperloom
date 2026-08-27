---
title: rmsnorm — numerics & parity
kind: technique
operator: rmsnorm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [both]
updated: 2026-07-14
sources:
  - https://github.com/vllm-project/vllm/issues/42325
  - ROCm/aiter@b467ce342:aiter/ops/triton/normalization/rmsnorm.py
  - https://github.com/ROCm/aiter/issues/1972
---

# rmsnorm — numerics & parity

RMSNorm is numerically simple but has **three sharp edges** that cause real serving bugs on MI300X.

## 1. fp32 accumulate is mandatory
`Σx²` over N=4096..8192 in bf16 loses ~3 mantissa bits → wrong `rms` → drift that compounds over layers.
Every correct impl loads bf16/fp16 and accumulates the sum-of-squares in **fp32** (`x.to(tl.float32)`
then `tl.sum(x*x)`; HIP: `float acc`). The output convert back to bf16 happens *after* the `·γ` multiply.
Tolerance vs fp64 reference: rel-err ~1e-2 for bf16 IO, ~1e-3 for fp16 — gate on that, not byte parity.

## 2. γ (weight) must be fp32-promoted before the multiply
**This is a live regression class.** vLLM #42325: the HIP `rms_norm_kernel` multiplied by `γ` in the
input dtype instead of fp32, silently changing results (introduced v0.20.0 via the FP8-quant PR #40860,
where fp32-multiply was *correct* for the fused-quant kernel but wrongly mirrored into the plain kernel).
The contract: `y = fp32(x)·rsqrt(...)·fp32(γ)`, convert to out dtype last. When porting/authoring, promote
`γ` explicitly.

## 3. ε placement and reduction order
- ε is added to the **mean** (`mean(x²)+ε`), inside the rsqrt — not to the sum, not to the norm. Llama
  uses `1e-5`/`1e-6`; mismatched ε → small but real divergence. Carry ε from the checkpoint config.
- Reduction order differs across **HIP / opus / Triton / vLLM-HIP** (tree vs sequential, block size). bf16
  RMSNorm is *not* associative → outputs differ in the last bit. After any backend swap re-run
  **greedy/temp=0 e2e parity** + a small eval; aiter #1972 is a concrete RMSNorm-kernel divergence in
  SGLang traced to a kernel version bump.

## 4. fp8 quant variant (→ fused_norm_quant)
When RMSNorm output is fed to a per-token/per-tensor fp8 quant (`rmsnorm2d_fwd_with_dynamicquant`,
`gated_rmsnorm_fp8_group_quant`), accuracy is gated at the **task** level, not byte parity:
- **gfx942 = FNUZ fp8** (e4m3fnuz, exponent bias off-by-one vs OCP). A wrong-dialect read is off by
  **exactly 2×** — the #1 silent MI300X FP8 bug. Confirm the checkpoint is normalized to fnuz.
- dynamic-quant scale = `max(|y|)/fp8_max` computed in fp32 inside the fused kernel; the scale dtype and
  rounding (RNE) must match the consumer GEMM's dequant. Gate with gsm8k/MMLU delta, not allclose.

## 5. The parity gate (use this)
1. isolated: `rms_norm(x)` vs fp64 reference, rel-err within band (bf16 ~1e-2).
2. e2e: greedy decode, same prompts, before/after backend swap → token-identical or eval-delta < noise.
3. fp8 variant: task eval (gsm8k) delta within run-to-run noise, AND confirm fnuz dialect on gfx942.

## Sources
- γ-dtype regression (fp32 promote contract): https://github.com/vllm-project/vllm/issues/42325 (PR #40860).
- fp32 accumulate / `x.to(tl.float32)` / quant variants: `aiter/ops/triton/normalization/rmsnorm.py`.
- RMSNorm kernel divergence across versions (parity re-gate): https://github.com/ROCm/aiter/issues/1972.
- FNUZ fp8 off-by-2× on gfx942: perf_knowledge backends/sglang_kernels/overview, quant_dequant_fp8.
