---
title: elementwise — overview
kind: operator_overview
operator: elementwise
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [prefill, decode, training, both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
---

# elementwise  (`out[i] = f(a[i], b[i], …)` — add/mul/clamp/cast/where)

## TL;DR
Elementwise ops (add, mul, sub, div, clamp/min/max, abs, where/select, dtype cast, scale) are pure
**HBM-bandwidth** kernels: ~0 arithmetic intensity, so the *only* lever is moving bytes at peak. On
MI300X that means **128-bit vectorized access** (`global_load_dwordx4` / `global_store_dwordx4` = 16 B/lane
= 512 B/wave), a **grid-stride loop** over a grid sized to saturate 304 CUs, and `block=256` (4 wave64s)
to keep all 4 SIMDs busy. Peak HBM is 5.3 TB/s; a well-written elementwise kernel sustains
**~3.5–4.3 TB/s (~66–81%)** (BabelStream-class). The single biggest *real* win, though, is **not running
the kernel at all** — elementwise ops are the #1 fusion donor: fold them into the producer/consumer GEMM
epilogue, norm, or each other so the data is touched once.

## Math contract
- **Unary**: `out[i] = f(a[i])` — `abs`, `neg`, `clamp(a, lo, hi)`, `scale = a * s`, cast `a.to(dtype)`.
- **Binary**: `out[i] = a[i] ⊕ b[i]` — `add`, `mul`, `sub`, `div`, `min`, `max`, with NumPy/PyTorch
  **broadcasting** (a `[M,N]` ⊕ `[N]` row-vector is the common LLM case: bias-like add/scale).
- **Ternary**: `where(cond[i], a[i], b[i])` (select / masked fill).
- dtype: any in/out; arithmetic upcasts to fp32 for bf16/fp16 then rounds back (matches torch). Cast
  ops are their own contract → see `../cast_fill_copy/overview.md`.
- Layout: contiguous is the fast path; **strided/broadcast** breaks 128-bit coalescing (see numerics +
  [fusion.md](fusion.md)).

## Shape regimes
- **Activation tensors** `[tokens, hidden]` — prefill `tokens = chunk×batch` (1k–16k), decode
  `tokens = batch` (1..256); `hidden ∈ {4096, 5120, 8192}`. These dominate elementwise byte traffic.
- **MLP intermediate** `[tokens, inter]`, `inter ∈ {14336, 17408, 34816}` — biggest single tensors; the
  `act_and_mul` (SiLU·gate) on this is the highest-value elementwise-adjacent op
  (`../act_and_mul_silu_gelu/overview.md`).
- **Residual / bias** broadcast adds `[tokens, hidden] + [hidden]`.

## Where it matters (Amdahl)
Individually each elementwise op is **<1% of GPU time**, but they are *numerous* and each is a full
HBM round-trip (read inputs + write output). On a fusion-naive graph the *aggregate* unfused pointwise
traffic can be 10–20% of memory time. The win is almost never "make one add faster" — it is **fuse N of
them into one pass** (Inductor's main job) or into an adjacent GEMM/norm epilogue. A standalone
elementwise kernel is only worth hand-writing when it's already memory-bound *and* can't be fused.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| pytorch_inductor | 🟢 sota (fuses pointwise chains automatically) | pytorch_inductor |
| triton | 🟢 sota (authoring + Inductor codegen target) | [triton.md](triton.md) |
| hip | 🟢 sota (peak bandwidth, full vectorization control) | hip |
| aiter | 🟡 competitive (fused add+rmsnorm, quant epilogues) | `../../backends/aiter/overview.md` |
| ck | 🟡 (CK elementwise device ops / fused epilogues) | n/a here (see [`../../languages/composable_kernel/ck_tile.md`]) |

## Fusion neighbors
The prime fusion donor. Folds into: GEMM epilogue (`+bias`, `*scale`, `+residual`, act) ·
RMSNorm/LayerNorm (residual-add fused into norm) · quant (`scale` + cast to fp8) · other elementwise
(any chain → one kernel). See [fusion.md](fusion.md). Cross-links:
`../act_and_mul_silu_gelu/overview.md`,
`../fused_add_rmsnorm/overview.md`,
`../quant_dequant_fp8/overview.md`.

## Numerics
bf16/fp16 arithmetic upcasts to fp32 then rounds; `clamp` order vs NaN, `div` by zero, fp8 saturation
all matter for parity. See [numerics.md](numerics.md).

## How to bench
Isolated: time `out = a + b` (or the fused chain) at the exact `[tokens, hidden]`, dtype; compute
achieved GB/s = `(read_bytes + write_bytes) / time`; gate against measured BabelStream peak (~4.3 TB/s),
not theoretical 5.3. e2e: the only honest elementwise win is fewer kernels / less HBM traffic — measure
total bytes moved (rocprof `FETCH_SIZE`/`WRITE_SIZE`) before vs after fusion.

## Sources
- 16 B / `global_load_dwordx4`, subgroup-contiguous 512 B, block=256, ≥1024 grid (MI300X workload opt):
  https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- `global_load_dwordx4` in loops + `_b128` LDS (Triton ISA guidance):
  https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- Vectorized memory access (float4/int4, grid-stride, register-pressure tradeoff):
  https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
- ~4.3 TB/s measured (81% of 5.3 peak), saturation behavior: https://arxiv.org/pdf/2510.27583
