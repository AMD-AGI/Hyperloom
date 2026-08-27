---
title: elementwise — tuning (vectorization, occupancy, grid-stride)
kind: operator_overview
operator: elementwise
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
  - https://github.com/ROCm/ROCm/issues/341
---

# elementwise — tuning

The whole game for a bandwidth-bound op is **bytes/instruction × instructions-in-flight**. Three knobs,
in priority order: vectorization width, grid/occupancy, and (only if it can't be fused) staying on the
contiguous fast path.

## 1. Vectorization width — the #1 lever
Use the **widest aligned access**: 128-bit (`global_load_dwordx4`/`global_store_dwordx4`) = 16 B/lane.
AMD names this the optimal access size; a full wave then moves **64 lanes × 16 B = 1024 B** per
instruction (the workload guide's "subgroup-contiguous 512 B" is the per-phase figure).

| dtype | elems per 128-bit load | HIP vector type | Triton |
|---|---|---|---|
| fp32 | 4 | `float4` / `int4` | load a `[BLOCK]` with `BLOCK%4==0`, contiguous |
| fp16/bf16 | 8 | `__half2`×4 via `int4` | same |
| fp8/int8 | 16 | `int4` | same |

- Vectorization **cuts instruction count ~4×** and is the difference between ~50% and ~80% of LDS/HBM
  peak.
- **Cost: VGPR pressure.** A `float4` holds 4 regs; a wide grid-stride loop with several live vectors can
  cross a 16-VGPR occupancy boundary (512 VGPR/EU, 16-granule) and drop waves/EU. If a kernel is already
  register-heavy or low-parallelism, **scalar can win** — verify with `-Rpass-analysis=kernel-resource-usage`.
- **Emit verification matters**: historically `uint4`/`ulonglong2` didn't always lower to `dwordx4`
  (ROCm#341); modern HIP handles `float4`/`int4` well but **always grep the ISA for `_dwordx4`** (Triton:
  `AMDGCN_ENABLE_DUMP=1`; HIP: `--save-temps`).

## 2. Alignment / the contiguous fast path
128-bit loads require **16-byte alignment** and the leading dim divisible by 4 (fp32) / 8 (fp16) / 16
(fp8). Strategy: **runtime branch** — if `(ptr % 16 == 0) && (n % vec == 0)` use the vectorized kernel,
else a scalar tail/fallback. A broadcast/strided input (e.g. `[M,N] + [N]`) cannot be 128-bit-coalesced on
the broadcast operand → either materialize, or keep that operand scalar/`tl.load`-broadcast and vectorize
only the contiguous side.

## 3. Grid & occupancy (saturate 304 CUs)
- **block = 256** (4 wave64s) → all 4 SIMDs busy. 128 only to save power.
- **grid ≥ 1024 workgroups** so the scheduler hides HBM latency; a common saturating choice is
  `304 × k` (e.g. `304*8`). With a **grid-stride loop** the grid is fixed and each thread loops over the
  array — this is the canonical idiom (keep the `dwordx4` inside the loop).
- Memory-bound kernels want **high occupancy to hide latency**, so trim VGPRs to lift waves/EU: HIP
  `__launch_bounds__(256, minWavesPerEU)`, Triton `waves_per_eu=3/4`, `num_warps=2/4`, `num_stages=1`.
  Don't over-trim into spills.

## 4. Grid-stride loop template (the canonical shape)
```cpp
__global__ void axpy_vec(int n, float a, const float4* __restrict__ x, float4* __restrict__ y) {
    int nv = n / 4;                                   // vectorized element count
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < nv; i += blockDim.x*gridDim.x) {
        float4 xv = x[i], yv = y[i];                  // -> global_load_dwordx4 ×2
        yv.x = a*xv.x + yv.x; yv.y = a*xv.y + yv.y;
        yv.z = a*xv.z + yv.z; yv.w = a*xv.w + yv.w;
        y[i] = yv;                                    // -> global_store_dwordx4
    }
    // scalar tail for n % 4 != 0
}
// grid = 304*8 blocks, block = 256
```
See hip and `../../languages/hip_cpp/patterns.md` §2.

## 5. Don't tune — fuse
Before tuning a standalone elementwise kernel, ask if it can be **fused** (it almost always can). A fused
chain reads inputs once and writes once; tuning a separate kernel just optimizes traffic you shouldn't be
paying. See [fusion.md](fusion.md); on `torch.compile` this is automatic via
pytorch_inductor.

## Sources
- 16 B optimal access, 512 B subgroup-contiguous, block=256, ≥1024 grid: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- float4/int4 vectorization, ~4× instruction reduction, register-pressure caveat, alignment: https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
- dwordx4 emission caveat for some HIP vector types: https://github.com/ROCm/ROCm/issues/341
- 512 VGPR/EU, 16-granule occupancy math: `../../triton_levers/knobs.md`
