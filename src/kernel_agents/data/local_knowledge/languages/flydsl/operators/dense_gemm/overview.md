---
title: dense_gemm — overview
kind: operator_overview
operator: dense_gemm
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, mxfp4]
regimes: [prefill, decode]
updated: 2026-06-08
---

# dense_gemm  (`C = A · Bᵀ [+ bias] [+ act]`)

## TL;DR
Dense GEMM is the Amdahl head of LLM prefill (~78–81% of GPU time). On MI300X the **executed** kernels
are hipBLASLt Tensile `Cijk_*`, but on sglang/vllm they are **dispatched by aiter** — so the live tuning
lever is aiter's per-shape DB, *not* PyTorch TunableOp. Use `mfma_16x16`, ≥1024 workgroups, 8-multiple
tiles; expect ~45–55% of peak, not peak.

## Math contract
`D[M,N] = A[M,K] · B[N,K]ᵀ + bias[N]` (sglang `nn.Linear` → `transpose_b=true`). dtype: bf16/fp16 in,
fp32 accumulate, bf16 out; quant variants take fp8/fp4 inputs + scales. **The `bias` flag is part of the
lookup key** for aiter's tuned DB (most live GEMMs are `bias=false`).

## Shape regimes (typical LLM, hidden 5120 / inter 17408)
- **prefill (large M ~ chunk×batch, 1k–16k)**: up/gate `K=5120, N∈{14336,16384,34816}`; down/qkv
  `N=5120, K∈{17408,6144}`. This is where the 78–81% lives.
- **decode (M = running batch ≤ conc, 1..256)**: same N/K, skinny M → split-K / skinny kernels matter.

## Where it matters (Amdahl)
A 1.15× on the ~80% GEMM mass ≈ +10% e2e ceiling. Even a blended 1.03× clears a 0.5% gate. This is the
single biggest lever — but the library is already near-optimal, so real wins are modest (measured
**+2.23% e2e** from a full bias-correct aiter tune on Qwen3.5-27B; see backends/aiter.md).

## Backend landscape
Competitive positioning for this operator; this folder's card is [flydsl.md](flydsl.md). Other backends
live in their own folders (`framework/aiter/`, `languages/{triton,ck,asm,hip}/`).

| backend | status |
|---|---|
| aiter | 🟢 sota (live path) |
| hipblaslt | 🟢 sota (executed kernels) |
| flydsl | 🟢 sota (authorable, mixed-precision/MoE) |
| triton | 🟡 competitive (loses to tuned hipBLASLt on plain GEMM) |
| tilelang | 🟡 (~0.94–1.05× of Triton GEMM on MI300X) |
| ck / asm | 🟢 (asm = peak; CK templates) |
| rocblas | 🟡 (small/odd M; raced by TunableOp) |

## Fusion neighbors
`+bias`, `+silu/gelu act_and_mul` (up/gate epilogue), `+residual add`, `+fp8 quant` epilogue → see
[fusion.md](fusion.md). Fusing the up/gate GEMM with `act_and_mul` removes a separate [M,34816] pass.

## Numerics
hipBLASLt solution swaps are same-math bf16 (parity-safe, tuner err_ratio<0.05). Quant variants need a
task-accuracy gate, not byte parity. See [numerics.md](numerics.md).

## How to bench
Isolated: `op_bench.py`/gradlib on the exact (M,N,K,bias,dtype). e2e: same-session 2-launch A/B at the
target ISL/OSL/conc, gate on delta>0.5% AND non-overlapping AND engagement proven.
