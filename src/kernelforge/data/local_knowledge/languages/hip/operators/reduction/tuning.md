---
title: reduction — tuning (wave reduce, LDS, split reduction)
kind: operator_overview
operator: reduction
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
  - https://github.com/ROCmSoftwarePlatform/composable_kernel/pull/82
---

# reduction — tuning

The combine adds two AMD-specific concerns on top of elementwise bandwidth: the **cross-lane reduce
shape** (wave64 → LDS → grid) and **whether to split** the reduced axis to fill 304 CUs.

## 1. The three-level reduction (the canonical shape)
```
per-thread (registers/VGPR)  →  per-wave (__shfl_down, 64 lanes)  →  per-block (LDS)  →  per-grid (atomic / 2nd kernel)
```
1. **Thread**: each lane sums its slice of the axis in an fp32 register (grid-stride, 128-bit loads).
2. **Wave64**: `for (off=32; off>0; off>>=1) v += __shfl_down(v, off);` — **6 steps** (32,16,8,4,2,1)
   because the wave is **64 lanes, not 32**. (Carrying a 5-step CUDA reduce loses half the wave.)
3. **Block**: lane 0 of each wave writes to `__shared__ partial[64]`; wave 0 reduces those. Pad/avoid
   bank conflicts. 256 threads = 4 waves → `partial[4]` used.
4. **Grid**: either `atomicAdd(out, block_result)` (single pass, needs `-munsafe-fp-atomics` for fp32) or
   write block partials and launch a tiny second kernel (deterministic; see split reduction).

Template in `../../hip_levers/patterns.md` §1.

## 2. When to split the axis (the key decision)
- **Row reduce, many rows** (`[16k, 5120] → [16k]`): one block per row already gives ≥1024 blocks → fills
  CUs. **No split needed.** Tune: block 256, vectorized 128-bit loads, fp32 acc.
- **Few output rows / huge axis** (e.g. `[8, 1e6] → [8]`, or full reduce → scalar): one-block-per-output
  = ≤8 busy CUs, 296 idle. **Split**: tile the reduced axis across many blocks, each does a partial, then
  combine. This is the single biggest standalone-reduction lever — turns 8 CUs into 304.

## 3. Two ways to combine the split partials
| method | how | cost | when |
|---|---|---|---|
| **AtomicAdd** (single pass) | each block `atomicAdd`s its partial into the one global output | needs fp atomic (`-munsafe-fp-atomics`); **nondeterministic order** (bf16/fp32 LSB) | sum/mean, parity-tolerant; lowest latency |
| **Two-call** (partial + reduce) | block partials → array; 2nd kernel reduces the array | extra kernel launch + array write; **deterministic** | max/min, parity-critical, or no fp atomic |

CK exposes both as `DeviceReduceMultiBlockAtomicAdd` and `MultiBlockTwoCall` (= `multiblock_partial_reduce`
then a `BlockWise` second call); see ck.

## 4. LDS & occupancy knobs
- **block size 256** (4 wave64s, all SIMDs). The LDS combine buffer is tiny (`partial[nwaves]`) so LDS is
  not the limiter — occupancy is set by VGPRs (acc + addresses). Trim with `__launch_bounds__(256, 4)` /
  Triton `waves_per_eu=3/4` to hide load latency.
- **Vectorize the input load** (128-bit) — the thread-level pass is the bandwidth-bound part; the
  shuffle/LDS combine is cheap.
- **Contiguous, hole-free shuffle masks** reduce faster on AMD (`0xFF...` beats scattered) — reduce over
  lanes `0..N-1`.

## 5. `max`/`min` & L2 specifics
- `max`/`min` reduce: same shuffle pattern with `v = fmaxf(v, __shfl_down(v, off))`. **No fp atomicAdd**
  for max — use `atomicMax` on the int-bits trick or the two-call path (CK has `AtomicMax` trait, gated by
  dtype).
- **L2 / Σx²**: square in the thread pass (fp32), sum-reduce, `sqrt` in the finalize. Welford for
  mean+var in one pass (norms).

## Triton equivalents
`tl.sum(x, axis)` / `tl.max(x, axis)` lower to the wave reduce + LDS automatically. Knobs: `num_warps`
controls how many waves combine via LDS; `BLOCK` = `next_pow2(axis)` so the reduced dim is wave-full (a
reduced dim < 64 wastes lanes). For split, use a 2-D grid + `tl.atomic_add` (the reduction analogue of
GEMM SPLIT_K). See triton.

## Sources
- wave64 `__shfl_down` (6 steps), 64-bit masks, hole-free mask perf: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- LDS banks / occupancy / block sizing: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- CK MultiBlockAtomicAdd vs MultiBlockTwoCall (partial+blockwise second call): https://github.com/ROCmSoftwarePlatform/composable_kernel/pull/82
