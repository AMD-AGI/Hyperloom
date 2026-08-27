---
title: elementwise — fusion (the prime fusion donor)
kind: operator_overview
operator: elementwise
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [both]
updated: 2026-06-08
sources:
  - https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://github.com/ROCm/aiter
---

# elementwise — fusion

Elementwise ops are the **#1 fusion donor** in the whole knowledge base: each one is a full HBM
round-trip with ~0 compute, so the best optimization is to make it ride along on a neighbor that's
*already* touching the data. The rule: **never let a pointwise op be its own kernel if a producer or
consumer can absorb it.**

## Why fusion dominates (the byte math)
A standalone `out = silu(x) * gate` on `[8192, 17408]` bf16 reads 2 tensors + writes 1 = ~855 MB of HBM
traffic ≈ 0.2 ms at 4.3 TB/s — pure overhead if `x`/`gate` were *just* produced by the up/gate GEMM.
Fusing it into the GEMM epilogue removes the read-back **and** the intermediate write. N chained
pointwise ops fused into one kernel: **N round-trips → 1**.

## The fusion patterns (donor → host)

| elementwise op | fuse into | result | where |
|---|---|---|---|
| `+bias`, `*scale` | **GEMM epilogue** | done in fp32 accumulator, free | [`../dense_gemm/fusion.md`] · `OPTIMIZE_EPILOGUE=1` |
| `silu/gelu × gate` | **up/gate GEMM epilogue** | removes a `[M,inter]` pass | `../act_and_mul_silu_gelu/overview.md` |
| `+residual` | **RMSNorm/LayerNorm** input | fused-add-norm, one pass | `../fused_add_rmsnorm/overview.md` |
| `*scale` + cast→fp8 | **norm or GEMM** epilogue (quant) | fused-norm-quant | `../fused_norm_quant/overview.md` · `../quant_dequant_fp8/overview.md` |
| any chain of pointwise | **each other** | one Triton kernel | Inductor (automatic) |
| `clamp`/`abs`/`where` | adjacent reduction | fused into reduce input | `../reduction/fusion.md` |

## How it actually gets fused
- **PyTorch Inductor (automatic, the default path)**: Inductor's core job is fusing pointwise/reduction
  chains into single Triton kernels — the main free win on any `torch.compile` graph. It walks the FX
  graph and merges adjacent pointwise nodes (and a trailing reduction) into one kernel. See
  pytorch_inductor. This is why a standalone elementwise card
  for Inductor is mostly "let it do its thing."
- **Triton (manual)**: write the chain in one `@triton.jit` body — load once, compute the whole
  expression, store once. `OPTIMIZE_EPILOGUE=1` drops the epilogue convert when the chain ends a GEMM.
- **aiter (library-fused)**: ships pre-fused ops — `fused_add_rmsnorm`, act+mul, norm+quant — so the
  pointwise never appears as a separate kernel on the serving path. See `../../backends/aiter/overview.md`.
- **HIP (epilogue in your own kernel)**: when hand-writing the producer (GEMM/norm), apply the pointwise
  in the fp32 accumulator before the store — zero extra traffic.

## Anti-patterns
- Materializing an intermediate just to run one more pointwise kernel on it (the thing fusion exists to
  kill). Check rocprof `WRITE_SIZE`/`FETCH_SIZE` — an unfused chain shows N× the necessary traffic.
- Fusing across a **non-elementwise reshape/transpose** that breaks contiguity — the fused kernel then
  loses 128-bit coalescing on one operand; sometimes a separate contiguous copy + fuse is faster (measure;
  see `../cast_fill_copy/overview.md`).
- Over-fusing into one giant kernel that **spills VGPRs** (too many live intermediates) — fusion helps
  until register pressure cuts occupancy; then split.

## Verify
rocprof the graph before/after: total HBM bytes (`FETCH_SIZE+WRITE_SIZE`) should drop by ≈
`(unfused_kernels − 1) × tensor_bytes`; kernel count should fall. `TORCH_COMPILE_DEBUG=1` →
`output_code.py` shows the fused Triton kernel for the Inductor path.

## Sources
- Inductor pointwise/reduction fusion is the core free win: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
- `OPTIMIZE_EPILOGUE=1` epilogue-convert fusion: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- aiter pre-fused norm/act/quant ops: https://github.com/ROCm/aiter
