---
title: act_and_mul_silu_gelu — overview
kind: operator_overview
operator: act_and_mul_silu_gelu
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1, mxfp4]
regimes: [prefill, decode, both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/activation.py
  - ROCm/aiter@b467ce342:aiter/ops/triton/activation.py
  - https://github.com/vllm-project/vllm/blob/main/csrc/activation.cu
---

# act_and_mul_silu_gelu  (`y = act(x[:d]) · x[d:]`, the gated MLP/MoE activation)

## TL;DR
This is the **gated activation** at the heart of every SwiGLU/GeGLU MLP and MoE expert: split the
up/gate projection `[M, 2d]` into two halves, apply SiLU/GeLU to the gate half, and **elementwise-multiply**
by the up half → `[M, d]`. It is **memory-bound** (reads 2d, writes d) and runs **once per layer** (×n_experts
in MoE). The dominant optimization is **fusion**: fold it into the up/gate GEMM epilogue, and/or fuse the
**fp8/fp4 quant** of its output into the same kernel so the down-proj GEMM reads quantized input.

## Math contract
Input `x[M, 2d]` (concatenated gate||up from the fused up/gate Linear). `d = x.shape[-1]//2`.
- **SiLU/SwiGLU**: `y = silu(x[:, :d]) · x[:, d:]`, `silu(z)=z·sigmoid(z)`. `silu_and_mul` takes an optional
  `limit` (default 0 = off) for the clamped-SwiGLU used by gpt-oss.
- **GeLU/GeGLU**: `y = gelu(x[:, :d]) · x[:, d:]` (exact erf or tanh approx `gelu_tanh`).
- dtype: bf16/fp16 in, **fp32 act compute**, bf16/fp16
  out (or fp8/fp4 + scale in the quant-fused variant). Layout: contiguous last dim → split is a stride.

## Shape regimes (hidden 4096/5120/8192, inter 14336/17408/...)
- **prefill**: `M = tokens` (1k–64k), `2d = intermediate·2` (28k–35k). Many rows → fills the chip.
- **decode**: `M = batch` (1..256) → few rows, latency-bound; persistent grid.
- **MoE**: per-expert `M = tokens routed to expert`; the activation is inside the fused-MoE stage-1
  epilogue (gate/up GEMM → act_and_mul → stage-2 down GEMM). See fused_moe_grouped_gemm.

## Where it matters (Amdahl)
Standalone act_and_mul is **1–3%** GPU time, but as a separate pass it forces a `[M, 2d]` read + `[M, d]`
write between two GEMMs. Fusing it into the GEMM epilogue removes that whole HBM round-trip; fusing the
output quant halves the down-proj input traffic. In MoE it's part of the kernel that AMD's FlyDSL rewrite
drove to **+162% throughput** on Kimi-K2.5 (the `silu_and_mul_fq` fused activation+quant).

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (HIP `silu_and_mul` + flydsl fused quant) | [aiter.md](aiter.md) |
| triton | 🟢 sota (aiter act+quant Triton kernels) | triton |
| hip | 🟢 sota (vLLM `silu_and_mul` HIP) | hip |
| vllm_kernels | 🟢 sota (HIP act.cu + AITER wiring) | vllm_kernels |
| flydsl | 🟢 sota (SiLU·mul + fused MXFP4/MXFP8 quant `silu_and_mul_fq`) | flydsl |

## Fusion neighbors
`+up/gate GEMM` epilogue (gemm_epilogue_fused, dense_gemm); `+fp8/fp4 dynamic quant` of the output
→ fused_norm_quant / quant_dequant_fp8 / quant_fp4_mxfp (aiter `scaled_silu_and_mul`,
`act_mul_and_fp8_group_quant`, `act_mul_and_mxfp4_quant`; flydsl `silu_and_mul_fq`); inside fused-MoE
stage-1 (fused_moe_grouped_gemm). See [fusion.md](fusion.md).

## Numerics
fp32 act compute; exact-erf vs tanh GeLU; fp8/fp4 quant accuracy gate. See [numerics.md](numerics.md).

## How to bench
Isolated `silu_and_mul(out, x)` at `(M, 2d)`; fp64 oracle; median ≥3 reps. e2e: toggle the GEMM-epilogue
fusion / output-quant fusion. See [tuning.md](tuning.md).

## Sources
- aiter activation ops (`silu_and_mul`, `gelu_and_mul`, `gelu_tanh_and_mul`, `scaled_silu_and_mul`): `aiter/ops/activation.py`.
- aiter Triton act+quant (`act_mul_and_fp8_group_quant`, `act_mul_and_mxfp4_quant`): `aiter/ops/triton/activation.py`.
- flydsl fused activation+quant `silu_and_mul_fq`: `aiter/ops/flydsl/kernels/silu_and_mul_fq.py`.
- vLLM HIP `silu_and_mul`/`gelu_and_mul`: https://github.com/vllm-project/vllm/blob/main/csrc/activation.cu.
