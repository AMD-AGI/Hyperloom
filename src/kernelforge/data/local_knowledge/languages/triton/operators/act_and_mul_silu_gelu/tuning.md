---
title: act_and_mul_silu_gelu — tuning
kind: technique
operator: act_and_mul_silu_gelu
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, mxfp4]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/activation.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# act_and_mul_silu_gelu — tuning

Memory-bound elementwise op: reads `[M, 2d]`, writes `[M, d]` (½ if quant-fused). The ceiling is
`(2d + d·out_bytes/in_bytes)/BW`. Tuning = saturate bandwidth + fuse.

## 1. The two halves: stride, not gather
`gate = x[:, :d]`, `up = x[:, d:]` are contiguous halves → load both with the **same vectorized 128-bit
access pattern** (offset by `d`). No gather; consecutive lanes hit consecutive addresses in each half.

## 2. Grid + block (the aiter Triton act+quant heuristics, verified)
From `act_mul_and_mxfp4_quant` / `act_mul_and_fp8_group_quant`:
```python
BLOCK_SIZE_N = min(256, next_power_of_2(N_half)); BLOCK_SIZE_N = max(32, BLOCK_SIZE_N)
BLOCK_SIZE_M = min(8, next_power_of_2(M))
NUM_WARPS    = 1 if BLOCK_SIZE_M < 4 else 4
grid = (cdiv(M, BLOCK_SIZE_M), cdiv(N_half, BLOCK_SIZE_N * NUM_ITER))
# blocks rounded to multiples of 32 (MXFP4 quant block size)
```
- 2D tiling `(BLOCK_M, BLOCK_N)` over the output `[M, d]`; small `BLOCK_M` (≤8) since each row is wide.
- `NUM_WARPS=1` for tiny M (decode), 4 otherwise — memory-bound, don't over-warp.
- MXFP4 quant forces `BLOCK_N` multiple of 32 (the e8m0 scale block).

## 3. Knob table
| knob | setting | note |
|---|---|---|
| `BLOCK_SIZE_N` | `min(256, next_pow2(d))`, ≥32, ×32 if mxfp4 | 128-bit loads |
| `BLOCK_SIZE_M` | `min(8, next_pow2(M))` | wide rows → small M tile |
| `num_warps` | 1 (tiny M) / 4 | memory-bound |
| grid | 2D over `[M, d]` | fill 304 CUs (≥1024 WGs) |
| act compute | fp32 | sigmoid/erf stability |

## 4. The real lever: fuse
- **Into the up/gate GEMM epilogue**: the GEMM already has `[M, 2d]` in registers/LDS — apply act_and_mul
  in the epilogue, write `[M, d]` directly. Removes a full standalone pass. See gemm_epilogue_fused.
- **Output quant fusion**: while `y` is in fp32 registers, quantize to fp8/fp4 + scale → down-proj reads
  ½ (fp8) or ¼ (fp4) bytes. aiter `act_mul_and_fp8_group_quant` / `act_mul_and_mxfp4_quant`; flydsl
  `silu_and_mul_fq`. This is the MoE win.
- **In fused-MoE stage-1**: the activation is inside the grouped GEMM; no separate tuning. See
  fused_moe_grouped_gemm.

## 5. SiLU vs GeLU cost
SiLU = `z·sigmoid(z)` (1 exp). GeLU-erf is pricier (erf); `gelu_tanh` approx is cheaper (1 tanh). All
memory-bound so the compute difference rarely shows — but on tiny-M decode where it's latency-bound,
prefer `gelu_tanh` over exact erf if the model tolerates it.

## Sources
- aiter Triton act+quant block heuristics: `/sgl-workspace/aiter/aiter/ops/triton/activation.py`.
- 128-bit loads / ≥1024 grid / memory-bound num_warps: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
- fused activation+quant MoE win: perf_knowledge languages/flydsl/kernel_families (silu_and_mul_fq, Kimi-K2.5).
