---
title: HIP / C++ — MFMA, buffer descriptors, cross-lane & sched intrinsics
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8, mxfp4]
regimes: [both]
status: sota
updated: 2026-07-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://github.com/ROCm/amd_matrix_instruction_calculator
  - https://reviews.llvm.org/D128158
  - https://github.com/llvm/llvm-project/issues/131954
---

# HIP — low-level AMDGCN intrinsics

The hand-written-kernel layer: `__builtin_amdgcn_mfma_*`, buffer resource descriptors, LDS builtins,
cross-lane permutes, and the scheduling builtins that build a software pipeline.

## 0. Register classes you must reason about
| Class | What | CDNA3 budget | Role in MFMA |
|---|---|---|---|
| **VGPR** | per-lane vector | 512/lane-slot, gran 16 | A/B operands, addresses |
| **AGPR** | accumulation | up to 256/lane (shared w/ VGPR) | MFMA C/D accumulators |
| **SGPR** | scalar (wave-uniform) | ~102 usable | buffer descriptors, loop counters |

On CDNA3, MFMA accumulators live in **AGPRs**; moving to/from VGPR costs `v_accvgpr_read/write`. A
classic pipelining bug (large tiles + SW pipelining) is the compiler inserting `v_accvgpr_*` in the
inner loop → kills perf back to small-tile levels (LLVM #131954). Fix: keep accumulators in a stable
`__attribute__((vector_size))` variable across iterations (so they stay in AGPRs); CK relies on the
"tied accumulator" flag (input accum tied to output) which inline asm alone doesn't give.

## 1. MFMA intrinsics
```
d = __builtin_amdgcn_mfma_<CDFmt>_MxNxK<ABFmt>(a, b, c, cbsz, abid, blgp);
```
`a/b/c` are **per-lane vector slices** (each lane holds `M·K/64`, `K·N/64`, `M·N/64` elems).
`cbsz/abid/blgp` = broadcast controls; **set 0** for standard GEMM.

| Intrinsic | M×N×K | A/B | C/D | A/B/C elems/lane |
|---|---|---|---|---|
| `mfma_f32_16x16x16f16` | 16×16×16 | fp16 | fp32 | 4/4/4 |
| `mfma_f32_16x16x16bf16_1k` | 16×16×16 | bf16 | fp32 | 4/4/4 |
| `mfma_f32_32x32x8f16` | 32×32×8 | fp16 | fp32 | 4/4/16 |
| `mfma_f32_16x16x32_fp8_fp8` | 16×16×32 | fp8 fnuz | fp32 | 8/8/4 |
| `mfma_f32_16x16x32_fp8_bf8` | 16×16×32 | fp8/bf8 fnuz | fp32 | 8/8/4 |
| `mfma_i32_16x16x32_i8` | 16×16×32 | int8 | int32 | 8/8/4 |
| `mfma_scale_f32_16x16x128_f8f6f4` | 16×16×128 | MXFP8/6/4 (CDNA4) | fp32 | block-scaled (E8M0) |

- **fp8 on gfx942 is FNUZ** (e4m3 fnuz / e5m2 "bf8"); the `_fp8_fp8`/`_fp8_bf8` suffixes pick A/B
  formats independently. **CDNA4/gfx950** adds OCP fp8 and the **block-scaled** `mfma_scale_*_f8f6f4`
  (MXFP8/6/4, 32-elem E8M0 scale) — not on gfx942.
- fp8 operands pack 8 fp8 / lane (cast to `long`). Use the **AMD Matrix Instruction Calculator** for
  the exact lane→element map rather than reverse-engineering it.
```cpp
using fp16x4 = __attribute__((vector_size(4*sizeof(_Float16)))) _Float16;
using fp32x4 = __attribute__((vector_size(4*sizeof(float)))) float;
fp32x4 acc = {0,0,0,0};                                  // -> AGPRs (keep stable across loop)
acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, acc, 0, 0, 0);
```

## 2. Buffer resource descriptors & buffer load/store
`buffer_*` use a **128-bit V#** (in SGPRs): base, stride, num-records (bounds), flags. Benefits over
plain `global_load`: **HW bounds checking** (OOB lanes return 0 / drop writes — no predication branch)
and sometimes better address gen. Triton's `knobs.amd.use_buffer_ops` emits these.
```cpp
float4 v = __builtin_amdgcn_raw_buffer_load_b128(rsrc, voffset, /*soffset=*/0, /*aux=*/0);
__builtin_amdgcn_raw_buffer_store_b128(value, rsrc, voffset, 0, 0);
```
Prefer **b128** (`global_load_dwordx4` equivalent) in inner loops. OOB (`voffset ≥ num_records`)
safely returns 0 — replaces predication masks in GEMM tail handling. Use AMD's
`__amdgcn_make_buffer_rsrc` builtin where available rather than hardcoding the flags word.

## 3. LDS builtins & cross-lane
```cpp
*reinterpret_cast<float4*>(&lds[off]) = v;                 // -> ds_write_b128
float4 r = *reinterpret_cast<float4*>(&lds[off]);          // -> ds_read_b128
int x = __builtin_amdgcn_ds_bpermute(srcLane << 2, val);  // gather via LDS crossbar (byte addr)
int y = __builtin_amdgcn_ds_permute (dstLane << 2, val);  // scatter
int z = __builtin_amdgcn_ds_swizzle(val, 0x1F);            // fixed swizzle within 32-lane group
```
| Builtin | Use |
|---|---|
| `ds_bpermute`/`ds_permute` | arbitrary lane gather/scatter via LDS crossbar (no LDS storage used) |
| `ds_swizzle` | fixed permutation within 32-lane group |
| `mov_dpp`/`update_dpp` | cheap neighbor shifts (row/bcast) — fast wave reductions |
| `permlane16`/`permlanex16` | 16-lane / cross-16 permute (CDNA3) |
| `readlane`/`readfirstlane` | broadcast a lane's value to scalar/all |
DPP and `permlane` beat `ds_*permute` for fixed neighbor patterns; `ds_bpermute` is the general gather.

## 4. Scheduling builtins (build a software pipeline)
```cpp
__builtin_amdgcn_sched_barrier(mask);                  // hard barrier; mask = categories allowed to cross (0 = block all)
__builtin_amdgcn_sched_group_barrier(mask, size, sync_id);  // group of `size` instrs of category `mask`, ordered by sync_id
__builtin_amdgcn_iglp_opt(variant);                    // predefined IGLP pipeline (0/1)
```
**`SchedGroupMask` category bits** (as used in CK GEMM pipelines):
| Mask | Category |
|---|---|
| `0x002` | VALU |
| `0x008` | **MFMA** |
| `0x020` | **VMEM read** |
| `0x040` | VMEM write |
| `0x100` | DS read |
| `0x200` | **DS write** |
```cpp
#pragma unroll
for (int i = 0; i < UNROLL; ++i) {
    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);   // 1 VMEM read  (prefetch next)
    __builtin_amdgcn_sched_group_barrier(0x008, 4, 0);   // 4 MFMA       (compute current)
    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);   // 1 DS write   (stage prefetched)
    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);   // 1 DS read    (feed next MFMA)
}
```
Use only after the default scheduler proves inadequate (verify via ISA) — **wrong ratios hurt**.
These are exactly the primitives FlyDSL exposes as `rocdl.sched_mfma/sched_vmem/sched_dsrd/sched_dswr/
sched_group_barrier` and Triton hides behind `schedule_hint`.

## 5. Production note
Hand-rolled MFMA microkernels are rarely worth it vs **rocWMMA** (C++ WMMA-style wrapper over MFMA),
**Composable Kernel / ck_tile** (templated, fully pipelined CDNA GEMM/attention), **HipKittens** (tile
primitives — [hipkittens.md](hipkittens.md)), or **FlyDSL** — these already encode the tied-accumulator
+ sched-group-barrier + double-buffer patterns correctly. Reach for raw intrinsics only when those
can't express your fusion or you must own the ISA.

## Sources
- Matrix Core programming CDNA3/CDNA4 (MFMA format, per-lane layouts, cbsz/abid/blgp, f8f6f4): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- AMD Matrix Instruction Calculator (exact lane→element maps): https://github.com/ROCm/amd_matrix_instruction_calculator
- sched_group_barrier semantics: https://reviews.llvm.org/D128158
- AGPR spill / tied accumulator: https://github.com/llvm/llvm-project/issues/131954
- buffer descriptors / ds builtins / sched builtins: https://llvm.org/docs/AMDGPUUsage.html
