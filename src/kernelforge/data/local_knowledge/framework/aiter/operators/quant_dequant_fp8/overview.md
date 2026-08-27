---
title: quant_dequant_fp8 — overview
kind: operator_overview
operator: quant_dequant_fp8
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2, bf16, fp16]
regimes: [prefill, decode, both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/quant.py
  - ROCm/aiter@b467ce342:csrc/kernels/quant_kernels.cu
  - vllm-project/vllm@HEAD:csrc/quantization/fp8/common.cu
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# quant_dequant_fp8  (`x_fp8 = round(x / s)`, `x ≈ x_fp8 · s`)

## TL;DR
Cast bf16/fp16 activations (and weights) to **FP8** with a scale `s` so the matrix core can run the
GEMM at 2× (CDNA3) / FP8-rate (CDNA4) — and dequant on the way out. The single most important fact:
**FP8 is FNUZ on CDNA3 (gfx942) and OCP on CDNA4 (gfx950)** — different bias and saturation, so a
checkpoint and its quant kernel must agree on the dialect or you read garbage off by ~2×. The standalone
quant op is **memory-bound** (one pass over the tensor); the real lever is **fusing** it into the
producer (RMSNorm/act) or the consumer (GEMM epilogue) so the FP8 tensor never round-trips HBM →
fusion.md, operators/fused_norm_quant, operators/scaled_quant_gemm.

## Math contract
For input `x[M,N]` (bf16/fp16) and target FP8 `e4m3`/`e5m2`:
- **Quant**: `s = amax(x_block) / FP8_MAX`; `x_fp8 = clamp(x / s, -FP8_MAX, FP8_MAX)` cast to FP8.
- **Dequant**: `x ≈ x_fp8.to(f32) * s`.
- Some kernels store the **reciprocal** scale (`is_scale_inverted` → multiply by `1/s`) to turn the hot
  per-element divide into a multiply (vLLM `scaled_fp8_conversion<true>`).
- `FP8_MAX` depends on dialect: OCP E4M3 **448**, FNUZ E4M3FNUZ **240** — but vLLM uses **224.0** on
  ROCm for *dynamic* quant (240 hurts accuracy; see numerics.md).

**Scale granularity** (the design axis that dominates accuracy/perf):
| granularity | scale shape for `x[M,N]` | when |
|---|---|---|
| **per-tensor static** | scalar (precomputed, calibration) | fastest; weights, well-behaved activations |
| **per-tensor dynamic** | scalar (amax this call) | activations without calibration |
| **per-token dynamic** | `[M,1]` (amax per row) | activations (per-token is the SmoothQuant-friendly default) |
| **per-block / per-group** | `[M, N/128]` or 1×128 / 1×32 | fine-grained (DeepSeek-style block fp8, group=128); 1×32 is the MX path → operators/quant_fp4_mxfp |

## Shape regimes
- **prefill**: quantize the full activation `[M=tokens, hidden]` before each linear; `M` 1k–16k. Bandwidth
  ∝ tensor size; per-token amax is a row-reduction over `hidden`.
- **decode**: skinny `M=batch` (1..256); the quant is tiny but launch overhead matters → fuse.
- **weights**: quantized once offline (Quark) → per-tensor or per-channel/per-block, static.

## Where it matters (Amdahl)
A standalone quant pass is ~1–3% of GPU time but it **gates** the FP8 GEMM (the ~80% Amdahl head). The
win is not the quant kernel itself — it is enabling the FP8 matrix core and *removing* the pass by
fusion. Unfused, every linear pays an extra full-tensor read+write of the activation.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (live path: dispatches HIP/CK/triton quant) | [aiter.md](aiter.md) |
| vllm_kernels | 🟢 sota (own `scaled_fp8_quant` HIP, ROCm 224 cap) | vllm_kernels |
| hip | 🟢 sota (the editable HIP source under aiter/vLLM) | hip |
| triton | 🟡 competitive (portable, fusion-friendly) | triton |
| asm | 🟡 (hand-tuned cast inside fused asm kernels) | asm |

## Fusion neighbors
`rmsnorm + quant` (fused_rms_fp8_*_quant), `act_and_mul + quant`, GEMM **dequant epilogue**
(`gemm_a8w8` fuses the `*s` back), KV-store quant. → fusion.md,
operators/fused_norm_quant, operators/scaled_quant_gemm, operators/kv_cache_quant.

## Numerics
e4m3 vs e5m2 ranges, FNUZ↔OCP, amax/scale, 224-cap, stochastic rounding, accuracy gates →
numerics.md, hardware/shared/dtype_numerics, hardware/cdna4_mi350.

## How to bench
Isolated: time `dynamic_per_token_scaled_quant(out, x, scale)` over `[M, hidden]` at prefill/decode M;
oracle = round-trip max/rel error vs an fp32 reference cast (gate per numerics.md). e2e: measure the
FP8 *linear* (quant+GEMM+dequant) vs bf16, gate on tok/s delta AND task accuracy (gsm8k/mmlu), not byte
parity.

## Sources
- aiter quant entrypoints (`pertoken_quant`, `static/dynamic_per_tensor`, `dynamic_per_token_scaled_quant`):
  `ROCm/aiter@b467ce342:aiter/ops/quant.py`.
- HIP quant kernels (`scaled_quant_impl`, `dynamic_per_token_scaled_quant`): `ROCm/aiter@b467ce342:csrc/kernels/quant_kernels.cu`.
- vLLM FP8 quant (224.0 ROCm cap, scale-inverted, per-token): `vllm-project/vllm@HEAD:csrc/quantization/fp8/common.cu`.
- FNUZ vs OCP, FP8 ranges: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
