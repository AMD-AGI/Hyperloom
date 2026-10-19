---
title: HIP — the authoring model, toolchain, and when to use it
kind: language
lever: hip_authoring_model
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# HIP authoring model

**Read this first.** HIP/C++ is the lowest-level *portable* way to author CDNA kernels — full control
of LDS, registers, cross-lane ops, MFMA intrinsics and scheduling. This card decides whether you should
be here, and gives the constants and toolchain everything else assumes.

## Route here when
- Triton / CK / rocWMMA / HipKittens / FlyDSL **cannot express** the fusion you need.
- You must **own the exact ISA** for a hot path.
- You are porting CUDA and need the delta list before you start.

**Raw HIP is the escape hatch, not the default.** The higher levels already encode the
tied-accumulator, software-pipeline and double-buffer patterns correctly and avoid the known AGPR-spill
trap (LLVM #131954). If you find yourself hand-writing `__builtin_amdgcn_mfma_*` +
`sched_group_barrier` + double-buffering, first check whether **rocWMMA**, **ck_tile**, **HipKittens**
or **FlyDSL** already expresses it.

## gfx950 constants — memorize these

| Resource | Value |
|---|---|
| Compute Units | **256** (8 XCD × 32) |
| SIMDs / CU | 4 |
| Wavefront | **64 lanes** |
| Registers | **512 / SIMD**, granularity **16** |
| AGPRs (MFMA accumulators) | ≤ 256 / lane, **unified pool** with VGPR |
| SGPRs | ~102 usable / wave |
| LDS (`__shared__`) | **160 KiB / CU**, **64 banks** × 4 B, **256 B/clk** |
| L1 vector cache | 32 KiB / CU |
| L2 | **per-XCD** |
| Infinity Cache | 256 MiB |
| HBM3E | 288 GB, **8.0 TB/s** |
| FP8 | **OCP** (E4M3FN / E5M2) — not FNUZ |
| TF32 | **removed** |

**Do not hardcode the CU count** — query `hipGetDeviceProperties → multiProcessorCount`. 304 is MI300X.
Full tables: `local_knowledge/hardware/`.

## The two facts behind almost every ported bug

1. **Wavefront = 64 lanes, not 32.** `warpSize == 64`. Every `__shfl` / `__ballot` / manual reduction,
   every mask (`unsigned long long` + `__popcll`), every grid and occupancy calculation, and the
   block-size rule (multiple of 64) traces back here. **32-lane CUDA code runs correctly and uses half
   the machine** — it will not error.
2. **LDS is 160 KiB/CU** on gfx950. H100 habits (228 KB) still overflow; MI300X habits (64 KB) leave
   2.5× on the table. Re-derive, do not inherit either.

## The wave64 programming model

```cpp
int lane = threadIdx.x % warpSize;   // 0..63 — NOT 0..31
int wave = threadIdx.x / warpSize;
```

- **Block size a multiple of 64** (64 / 128 / 256). 256 threads = 4 waves is the common sweet spot.
- **Grid ≥ 1024 workgroups** so 256 CUs stay fed across 8 XCDs.
- **`__launch_bounds__(maxTPB, minWavesPerEU)`** caps registers — the C++ analogue of Triton's
  `waves_per_eu`. `minWavesPerEU=2` forces VGPR ≤ 256; `=4` forces ≤ 128. **Too aggressive → scratch
  spills to HBM → 3–5× slower.**
- **`__restrict__`** on pointers enables wider `global_load_dwordx4` and reordering.

## Toolchain

```bash
hipcc --offload-arch=gfx950 -O3 kernel.hip -o kernel
hipcc --offload-arch=gfx942 --offload-arch=gfx950 -O3 ... -o fat     # fat binary
amdclang++ -x hip --offload-arch=gfx950 -O3 -munsafe-fp-atomics kernel.hip -o kernel
```

| Flag | Purpose |
|---|---|
| `--offload-arch=gfx950` | target arch (required) |
| `-munsafe-fp-atomics` | HW fp atomics (`global_atomic_add_f32`) — **big for split-K and reductions** |
| `--save-temps` | keep the `.s` AMDGCN ISA |
| `-Rpass-analysis=kernel-resource-usage` | print VGPR/SGPR/LDS/scratch per kernel |
| `-mllvm -amdgpu-waves-per-eu=N` | global occupancy hint |
| `-ffast-math` / `-fgpu-flush-denormals-to-zero` | relax FP — **check accuracy** |

Inspect: `rocminfo | grep -E "Compute Unit|SIMD|Wavefront"` ·
`llvm-objdump -d --arch=amdgcn --mcpu=gfx950 kernel | less`

## What HIP gives you that higher levels don't

| Capability | HIP | Triton | FlyDSL |
|---|---|---|---|
| Explicit LDS layout / padding / swizzle | **full** | indirect | explicit |
| MFMA intrinsic choice / fragment layout | **full** | `tl.dot` picks | explicit |
| Hand-built scheduling pipeline | `sched_group_barrier` builtins | `schedule_hint` | `rocdl.sched_*` |
| Direct-to-LDS / async copy | `global_load_lds` builtin | `knobs.amd.use_async_copy` | `rocdl.raw_ptr_buffer_load_lds` |
| 64-bit wave masks | **full** | hidden | via `gpu`/`rocdl` |

That column of "full" is the whole argument for being here — and the reason it is more work.

## Predict occupancy before you measure

```
occ_vgpr    = floor(512 / round_up_16(vgpr_used))            # waves/SIMD
occ_lds     = floor(163840 / lds_bytes_per_workgroup)        # workgroups/CU  (gfx950: 160 KiB)
occ (wg/CU) = min(floor(occ_vgpr * 4 / num_waves), occ_lds)  # 4 SIMD/CU
```

On gfx950 the **LDS term rarely binds** — register pressure is usually the limiter. Full model and
worked examples: `hardware/mi350_execution.md`.

## Where next

| Question | Card |
|---|---|
| MFMA builtins, buffer descriptors, cross-lane, scheduling | `hip_builtins.md` |
| LDS banks, swizzle, direct-to-LDS, barriers, double-buffer | `hip_lds_staging.md` |
| Give me a working kernel body | `hip_templates.md` |
| It compiles but is wrong or slow | `hip_traps.md` |
| Tile abstractions instead of raw asm | `hipkittens.md` |

## Sources
- HIP kernel language (`warpSize`, `__launch_bounds__`, 64-bit masks): https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- HIP programming model (wave64, SIMD, block sizing): https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- HIP hardware implementation (LDS banks, occupancy): https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- MI300X workload optimization (VGPR/LDS, grid sizing): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- CDNA4 whitepaper (LDS 160 KiB, MXFP): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
