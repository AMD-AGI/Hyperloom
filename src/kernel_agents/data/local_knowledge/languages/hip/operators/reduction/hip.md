---
title: reduction on HIP / C++ — SOTA card
kind: sota_card
operator: reduction
backend: hip
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int32]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# reduction × hip

## TL;DR
HIP gives full control of the **three-level reduce** (wave64 shuffle → LDS → grid) and the **split**
decision (atomic single-pass vs deterministic two-call), reaching the bandwidth ceiling and serving as the
Tier-C seam. The two AMD must-knows: the wave reduce is **6 shuffle steps over 64 lanes** (not 5 over 32),
and fp32 grid-atomics need `-munsafe-fp-atomics`.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| wave-shuffle + LDS block reduce + `atomicAdd` | `../../../hip_levers/patterns.md` §1 | all gfx9, fp32 acc | bandwidth-bound, **~4.3 TB/s** input read @ MI300X gfx942 | sum/mean, parity-tolerant |
| two-call (partial array + reduce kernel) | this card | all gfx9 | deterministic; +1 launch | max/min, parity-critical, no fp atomic |

```cpp
__device__ float wave_reduce_sum(float v) {            // 64-lane: 6 steps
    for (int o = warpSize/2; o > 0; o >>= 1) v += __shfl_down(v, o);  // 32,16,8,4,2,1
    return v;
}
__global__ void reduce_sum(const float* __restrict__ x, float* out, int n) {
    __shared__ float part[64];                          // <= nwaves
    float v = 0.f;
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        v += x[i];                                       // grid-stride, vectorize w/ float4
    v = wave_reduce_sum(v);
    int lane = threadIdx.x % warpSize, w = threadIdx.x / warpSize;
    if (lane == 0) part[w] = v;
    __syncthreads();
    if (w == 0) { int nw = blockDim.x/warpSize;
        v = (lane < nw) ? part[lane] : 0.f; v = wave_reduce_sum(v);
        if (lane == 0) atomicAdd(out, v); }              // -munsafe-fp-atomics
}
```

## Config space / knobs
- **block 256** (4 wave64s). **grid `304×k`** + grid-stride (split the axis across blocks → fills CUs at
  low output count).
- **vectorize the thread-pass load** (`float4`/`int4`, 128-bit) — the bandwidth-bound part.
- `__launch_bounds__(256, 4)` to lift occupancy (hide latency); back off if it spills.
- **combine choice**: `atomicAdd` (single pass, `-munsafe-fp-atomics`, nondeterministic) vs **two-call**
  (write `partial[block]`, 2nd kernel reduces it — deterministic). `max`/`min` have no fp `atomicAdd`: use
  `atomicMax` on int-bits or the two-call path.
- Cooperative groups `tiled_partition<64>` is the portable wave reduce (tile ≤ 64 on CDNA).

## Numerics / parity
fp32 accumulate; atomic split is nondeterministic-order (bf16 LSB run-to-run); two-call is deterministic.
`v_max_f32` returns non-NaN operand (differs from torch NaN-propagation). See [numerics.md](numerics.md).

## Integration (rebind seam)
`hipcc --offload-arch=gfx942 -munsafe-fp-atomics` → `.so`, bound as a torch custom op
(`TORCH_LIBRARY`). HIP source is the edit surface (rebuild after editing).

## Pitfalls & anti-patterns
- **5-step (32-lane) reduce** carried from CUDA → drops half the wave → wrong result / half throughput.
- `atomicAdd(float)` without `-munsafe-fp-atomics` → CAS loop (slow) or compile fallback.
- Single block per output for low output count → 8 busy CUs, 296 idle (split it).
- No `__syncthreads()` between LDS write and read → race.
- Non-vectorized thread pass → ~2–4× fewer bytes/instr.

## How to verify
`--save-temps` ISA: `global_load_dwordx4`, `ds_*` for LDS, `global_atomic_add_f32` (if atomic), no
`scratch_`; GB/s vs ~4.3 TB/s; rocprof CU utilization for split; fp32 atol vs torch.

## Alternatives / cross-links
triton (faster to author) · [composable_kernel.md](ck.md) (ready instances)
· [tuning.md](tuning.md) · `../../../hip_levers/patterns.md` §1.

## Sources
- wave64 `__shfl_down` 6-step reduce, 64-bit masks, v_max NaN, atomics: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- cooperative_groups tile ≤ warpSize: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html
- block=256, ≥1024 grid, CU saturation: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
