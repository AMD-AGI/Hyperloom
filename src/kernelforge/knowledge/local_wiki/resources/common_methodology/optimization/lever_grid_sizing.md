---
title: grid and wave sizing — wave64, workgroup shape, launch bounds, persistent kernels
kind: lever
lever: grid_sizing
gens: [gfx950]
bottleneck: latency / occupancy-bound
updated: 2026-08-28
---

# Grid and wave sizing

## Route here when
- CUs are idle: the launch produces fewer workgroups than the device can hold.
- Latency-bound with occupancy already reasonable — you need **more in-flight work**, not more
  registers.
- Decode-shape kernel (M = 1..8) that cannot fill the device from its natural grid.
- You are porting from CUDA and have not re-derived any lane math.

## gfx950 constants

| Fact | Value |
|---|---|
| Wavefront | **64 lanes** — never 32 |
| SIMDs per CU | 4 |
| Wave slots | 8/SIMD → **32 waves/CU** |
| Active CUs | **256** (8 XCD × 32) |
| Fill target | **≥1024 workgroups** (≈4/CU, gives tail slack) |
| Tile-count rule | **multiple of 8** for even XCD spread |
| Block size | multiple of **64** threads |

**Do not hardcode the CU count.** Query `hipGetDeviceProperties → multiProcessorCount`. 304 is MI300X;
gfx950 is 256.

## Wave64 is the thing CUDA ports get wrong

Every lane-width constant is 64: `__shfl`/`__ballot` masks are `unsigned long long` with `__popcll`,
reductions are mod 64, coalescing windows are 64 lanes wide, divergence uses the 64-bit `EXEC` mask.
32-lane code **runs correctly and uses half the machine** — it will not error, it will just be slow.
In Triton, `num_warps=N` means N × 64 threads.

## What to change, in order

### 1. Count your workgroups
```
workgroups = ceil(M/BLOCK_M) * ceil(N/BLOCK_N) * SPLIT_K
```
| Count | Verdict |
|---|---|
| < 256 | CUs literally idle — fix this first, nothing else matters |
| 256–1024 | device covered but no tail slack |
| ≥ 1024 **and** `% 8 == 0` | target |

### 2. Set `num_warps` / block size
4–8 wavefronts (256–512 threads) is the usual GEMM range. Larger blocks share LDS better but raise
per-block register and LDS footprint, which lowers blocks/CU. Tune jointly with tile size and
`num_stages` — these three are not independent.

### 3. `__launch_bounds__(maxThreadsPerBlock, minWavesPerEU)`
Caps the register allocation so the requested occupancy is achievable, and tells the compiler the real
block size so it does not over-allocate. **Set it below the actual block size and you force spills** —
verify in the ISA (`lever_occupancy.md`).

### 4. Decode shapes: manufacture parallelism with split-K
A skinny GEMM (M = 1..8) has naturally few tiles and starves 256 CUs. Use small `BLOCK_M` (16/32) plus
**split-K** to create enough workgroups, then reduce the partials. Without this the kernel is
latency-bound on a nearly empty device.

### 5. Persistent kernels when you want control
Launch exactly `256 × blocks_per_CU` workgroups that loop over output tiles:
```
for tile in my_tiles: compute(tile)
```
Buys: amortized launch overhead, resident weights/state, **explicit tile→XCD mapping** for L2 locality
(`lever_xcd_locality.md`), and natural Stream-K reduction. Costs: you own load balancing — a naive
static partition reintroduces the tail imbalance you were trying to remove. Use an atomic work queue
or Stream-K.

## Prefill vs decode

| | Prefill (large M) | Decode (M = 1..8) |
|---|---|---|
| Class | compute-bound | memory / latency-bound |
| Tile | large | small `BLOCK_M` (16/32) |
| Grid | ≥1024 WGs naturally | needs split-K to reach the CU count |
| Occupancy | 1–2 wg/CU + deep prefetch is fine | want ≥4 waves/CU |
| Next lever | `lever_mfma_sched.md` | `lever_coalescing.md`, `lever_prefetch.md` |

## Verify

| Check | How | Pass |
|---|---|---|
| Device size | `hipGetDeviceProperties` → `multiProcessorCount` | 256 on gfx950; use the queried value |
| Grid | arithmetic | ≥1024 workgroups, tile count `% 8 == 0` |
| Idle CUs | `rocprof-compute` per-CU occupancy / wavefront launch count | no idle dies or CUs |
| No spills from launch bounds | ISA VGPR count vs your target, scratch traffic | none |
| A/B | `num_warps ∈ {4,8}`; persistent vs non-persistent for decode | keep fastest |

## Expected magnitude
Going from a grid that covers half the device to a full one: **near-linear** in the coverage ratio.
Split-K on a decode GEMM that was under-filling: **2–4×** is common. `num_warps` tuning on an already
well-filled kernel: usually **<10%**.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Half the CUs idle | grid smaller than 256 workgroups | raise the grid; split-K for skinny shapes |
| Reduction math wrong / mask asserts | wave32 assumption from CUDA | all lane math is **64-wide** |
| Spills appeared after adding `__launch_bounds__` | bound set below actual block size | match it to the real block size |
| Persistent kernel has a long tail | naive static tile partition | atomic work queue or Stream-K |
| Grid ≥1024 but one XCD lags | tile count not a multiple of 8 | `lever_xcd_locality.md` |
| Tuned for 304 CUs | MI300X value hardcoded | query the device |

## Deeper
`hardware/mi350_execution.md` (execution model) ·
`hardware/mi350_overview.md` (topology, CU counts) ·
`lever_occupancy.md` · `lever_xcd_locality.md` · `lever_coalescing.md`
