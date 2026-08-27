---
title: HIP / C++ authoring — programming model, hardware constants, toolchain
kind: language
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [both]
status: sota
updated: 2026-07-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# HIP / C++ on AMD Instinct — authoring overview

## TL;DR
HIP/C++ is the **lowest-level portable** way to author CDNA kernels: full control of LDS, registers,
wave/cross-lane ops, MFMA intrinsics, and instruction scheduling, compiled by `hipcc`/`amdclang++`. The
two facts that break every CUDA habit: **wavefront = 64 lanes** (not 32) and **LDS = 64 KB/CU** (CDNA3;
160 KB on CDNA4). Reach for HIP when Triton/CK/FlyDSL can't express a fusion, or to read/own the exact
ISA. For the matrix-core and async-LDS layer see [intrinsics.md](intrinsics.md) and
[lds_async.md](lds_async.md); for the tile-abstraction alternative that avoids raw asm see
[hipkittens.md](hipkittens.md).

## MI300X (gfx942) hardware constants — memorize
| Resource | Value |
|---|---|
| Compute Units | **304** (8 XCDs × 38) |
| SIMDs / CU | 4 |
| Wavefront (warp) | **64 lanes** |
| VGPRs | **512 / SIMD lane-slot**, granularity 16 |
| SGPRs | ~102 usable / wave |
| AGPRs (MFMA accum) | up to 256 / lane (shared budget with VGPR on CDNA3) |
| LDS (`__shared__`) | **64 KB / CU**, 32 banks × 4 B (128 B/clk) |
| L1 vector cache | 32 KB / CU |
| L2 | 4 MB / XCD |
| Infinity Cache (LLC) | 256 MB |
| HBM3 | 192 GB, ~5.3 TB/s |

CDNA4 (gfx950, MI350X/MI355X) deltas: **LDS 160 KB/CU**, 256 B/clk; OCP fp8 + MXFP block-scaled MFMA;
`tf32` removed. Full tables live in the shared tree — `local_knowledge/hardware/cdna3_mi300/` and
`local_knowledge/hardware/cdna4_mi350/`.

## Toolchain
```bash
hipcc --offload-arch=gfx942 -O3 kernel.hip -o kernel               # CDNA3
hipcc --offload-arch=gfx942 --offload-arch=gfx950 -O3 ... -o fat   # fat binary
amdclang++ -x hip --offload-arch=gfx942 -O3 -munsafe-fp-atomics kernel.hip -o kernel
```
| Flag | Purpose |
|---|---|
| `--offload-arch=gfx942/gfx950` | target arch (required) |
| `-munsafe-fp-atomics` | HW fp atomics (`global_atomic_add_f32`) — big for split-K/reductions |
| `--save-temps` | keep `.s` AMDGCN ISA |
| `-Rpass-analysis=kernel-resource-usage` | print VGPR/SGPR/LDS/scratch per kernel |
| `-mllvm -amdgpu-waves-per-eu=N` | global occupancy hint |
| `-ffast-math` / `-fgpu-flush-denormals-to-zero` | relax FP (check accuracy) |

Inspect: `rocminfo | grep -E "Compute Unit|SIMD|Wavefront"`;
`llvm-objdump -d --arch=amdgcn kernel | less`.

## The wave64 programming model
A workgroup is partitioned into **64-lane** wavefronts on one SIMD. `warpSize == 64`.
```cpp
int lane = threadIdx.x % warpSize;   // 0..63 — NOT 0..31
int wave = threadIdx.x / warpSize;
```
- **Block size = multiple of 64** (64/128/256). 256 threads = 4 waves is a common sweet spot.
- **Grid ≥ 1024 workgroups** so 304 CUs stay fed across 8 XCDs.
- `__launch_bounds__(maxTPB, minWavesPerEU)` caps registers: `minWavesPerEU=2` forces VGPR ≤ 256;
  `=4` forces ≤ 128. Too aggressive → scratch spills (HBM) → 3–5× slower. It's the C++ analogue of
  Triton `waves_per_eu`.
- `__restrict__` on pointers enables wider `global_load_dwordx4` and reordering.

## What HIP gives you that higher levels don't
| Capability | HIP | Triton | FlyDSL |
|---|---|---|---|
| Explicit LDS layout / padding / swizzle | full | indirect (`kpack`) | explicit (`SmemAllocator`, `swizzle_xor16`) |
| MFMA intrinsic choice / fragment layout | full | `tl.dot` picks | explicit (`rocdl.mfma_*`) |
| Hand-built sched pipeline | `sched_group_barrier` builtins | `schedule_hint` | `rocdl.sched_*` |
| Direct-to-LDS / async copy | `global_load_lds` builtin | `knobs.amd.use_async_copy` | `rocdl.raw_ptr_buffer_load_lds` |
| 64-bit wave masks | full | hidden | via `gpu`/`rocdl` |

## Deep-dive map (this levers folder)
- [intrinsics.md](intrinsics.md) — MFMA builtins, buffer descriptors, cross-lane, sched builtins.
- [lds_async.md](lds_async.md) — LDS banks/padding/swizzle, direct-to-LDS, barriers, wait counters.
- [patterns.md](patterns.md) — wave reductions, grid-stride, streams/graphs, tiled LDS GEMM, MFMA µkernel.
- [pitfalls.md](pitfalls.md) — the CUDA→HIP porting traps + ISA verification checklist.
- [hipkittens.md](hipkittens.md) — HK tile abstractions, scheduling patterns, register pinning.

## Sources
- HIP kernel language (warpSize, __launch_bounds__, 64-bit masks): https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- HIP programming model (wave64, SIMD, block sizing): https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- HIP hardware implementation (LDS banks, 64 KB/CU, occupancy): https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- MI300X workload optimization (304 CUs, VGPR/LDS, ≥1024 grid): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- CDNA4 LDS 160 KB / MXFP: CDNA4 whitepaper, https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
