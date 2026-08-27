---
title: skinny_gemv_decode — overview
kind: operator_overview
operator: skinny_gemv_decode
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
updated: 2026-06-05
sources:
  - https://github.com/ROCm/aiter
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
---

# skinny_gemv_decode

## TL;DR
> The tall-skinny GEMM / GEMV of LLM decode — `M=1..8` tokens × large weight matrix — which is **memory
> bandwidth bound**, so the key fact is: you optimize for **weight read bandwidth and CU utilization**
> (split-K across CUs), not MFMA throughput.

## Math contract
- `Y[M,N] = X[M,K] @ W[K,N] (+ bias)` with `M` tiny (1..8). For `M=1` it is a matrix-vector product (GEMV).
- Output dtype bf16/fp16; weights bf16/fp16 or fp8 (→ scaled_quant_gemm/overview.md);
  accumulate fp32.
- Dominant decode GEMMs: QKV proj, O proj, MLP up/gate/down at batch≈1.

## Shape regimes
- **Decode only**: M = batch (1..8 typical, up to ~16 with small batching). N,K = model hidden dims (large).
  Arithmetic intensity is low → bandwidth bound. Large-batch/prefill uses dense GEMM instead
  (dense_gemm/overview.md).

## Where it matters (Amdahl)
- At batch≈1 decode, these projections are a large fraction of per-token latency and are bandwidth-limited;
  the GPU's MFMA peak is irrelevant — only how fast you stream W and how well you fill the 304 CUs
  (MI300X) / 256 CUs (MI350X) matters. This is why aiter ships a dedicated skinny / wvSplitK-style path.

## Backend landscape (link table → SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota | aiter |
| triton | 🟡 competitive | triton |
| asm | 🟡 competitive | [asm.md](asm.md) |
| hip | 🟡 competitive | hip |
| flydsl | 🟡 competitive (gfx950 small-M family) | flydsl |

## Fusion neighbors
- Bias/activation epilogue, fp8 dequant; for decode MoE the per-expert M is also tiny →
  grouped_gemm_moe/overview.md. Split-K is the core technique →
  splitk_streamk_gemm/overview.md.

## Numerics
- fp32 accumulate; split-K reduction order; fp8 weight scales → [numerics.md](numerics.md).

## How to bench
- Bench at M=1 and M=2/4/8 separately (regimes differ); median ≥3 warm reps; oracle = dense fp32
  reference. Report achieved GB/s vs HBM peak (bandwidth bound), not TFLOPS.

## Sources
- AITER (skinny GEMM path): https://github.com/ROCm/aiter
- vLLM decode/GEMV custom kernels on ROCm: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
