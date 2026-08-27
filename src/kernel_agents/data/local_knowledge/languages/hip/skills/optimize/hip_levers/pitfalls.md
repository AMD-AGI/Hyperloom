---
title: HIP / C++ — CUDA→HIP pitfalls & ISA verification
kind: language
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: sota
updated: 2026-07-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://github.com/ROCm/HIP/issues/3667
  - https://github.com/llvm/llvm-project/issues/131954
---

# HIP — pitfalls & ISA verification

## CUDA → HIP porting checklist
| Pitfall | Symptom | Fix |
|---|---|---|
| `warpSize`/mask assumed 32 | wrong reductions, static-assert on mask | use **64**; `unsigned long long` masks |
| Block size not multiple of 64 | wasted lanes (32-thread block = half a wave) | use 64/128/256 |
| `__ballot` stored in `unsigned` | truncated mask | `unsigned long long` + `__popcll` |
| LDS > 64 KB (ported from H100) | launch fail / occupancy → 1 | shrink tile; budget to 64 KB (160 KB CDNA4) |
| No LDS padding/swizzle | bank conflicts, serialized LDS | pad inner dim `+1` or XOR-swizzle |
| `__launch_bounds__` too tight | scratch spill (HBM), 3–5× slower | check `.private_segment_fixed_size == 0` |
| Scalar global loads | low BW | `float4`/`__restrict__` → `dwordx4` |
| fp atomics slow/unused | reduction bottleneck | `-munsafe-fp-atomics` |
| Mask with holes | slower cross-lane | use contiguous prefix masks |
| Half-float `__shfl` | unsupported | shuffle as int/float, repack |
| `v_accvgpr_*` in MFMA loop | perf back to small-tile levels | keep acc in a stable vector var (tied accum); LLVM #131954 |
| OCP fp8 MFMA on gfx942 | wrong result (FNUZ vs OCP) / no lowering | use FNUZ on gfx942; OCP only gfx950 |

## The two facts behind almost every bug
1. **Wavefront = 64**, not 32. Every `__shfl`/`__ballot`/manual reduction, every grid/occupancy
   calc, and the static-assert on mask width trace here. "Wave-aware" 32-lane CUDA code *runs* but
   uses **half** the machine.
2. **LDS = 64 KB/CU** (CDNA3). H100 habits (228 KB) overflow LDS → launch failure or occupancy 1.

## Occupancy: predict it before you measure
```
occ_vgpr = floor(512 / round_up_16(vgpr_used))   # waves/SIMD from VGPR
occ_lds  = floor(LDS_CAP / lds_bytes_used)        # blocks/CU from LDS (65536 / 163840)
occ (wg/CU) = min(floor(occ_vgpr * 4 / num_warps), occ_lds)   # 4 SIMD/CU
```
`__launch_bounds__(maxTPB, minWavesPerEU)` is your lever: `minWavesPerEU=2` → VGPR ≤ 256, `=4` → ≤ 128.
Going past what the kernel needs forces scratch spills — verify, don't guess.

## ISA verification checklist
Build with `--save-temps` (or `AMDGCN_ENABLE_DUMP=1`) and confirm in the inner loop:
| Look for | Good | Bad → retune |
|---|---|---|
| Global loads | `global_load_dwordx4` / `buffer_load_dwordx4` | `global_load_dword` (scalar) |
| LDS access | `ds_read_b128` / `ds_write_b128` | `ds_read_b32` |
| MFMA | dense `v_mfma_f32_16x16x16` | sparse, gaps = starved core |
| Accumulator | accumulators in `a[0:n]` (AGPR) | `v_accvgpr_read/write` in loop |
| Scratch | `.private_segment_fixed_size: 0` | nonzero → spilling to HBM |
| Waitcnt | minimal, overlapped | `s_waitcnt vmcnt(0)` after every load = no overlap |

Resource usage at a glance: `-Rpass-analysis=kernel-resource-usage` prints `.vgpr_count`,
`.sgpr_count`, `.group_segment_fixed_size` (LDS), `.private_segment_fixed_size` (scratch).

## When NOT to write raw HIP
If you're reaching for `__builtin_amdgcn_mfma_*` + `sched_group_barrier` + double-buffering by hand,
first check whether **rocWMMA**, **ck_tile / Composable Kernel**, **HipKittens**
([hipkittens.md](hipkittens.md)), or **FlyDSL** already express it — they encode the tied-accumulator
and pipeline patterns correctly and avoid the LLVM #131954 trap. Raw HIP is for fusions those can't
express, or when you must own the exact ISA.

## Sources
- warpSize/masks/__shfl/half-float: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- `__ballot` 64-bit return & mask requirements: https://github.com/ROCm/HIP/issues/3667
- MFMA + pipelining AGPR spill / tied accumulator: https://github.com/llvm/llvm-project/issues/131954
- VGPR/LDS limits, ≥1024 grid, occupancy: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
