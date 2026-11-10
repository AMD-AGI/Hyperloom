---
title: HIP — MFMA builtins, buffer descriptors, cross-lane, scheduling
kind: language
lever: hip_builtins
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://github.com/ROCm/amd_matrix_instruction_calculator
  - https://reviews.llvm.org/D128158
  - https://github.com/llvm/llvm-project/issues/131954
  - https://llvm.org/docs/AMDGPUUsage.html
---

# AMDGCN builtins from HIP

The hand-written-kernel layer: `__builtin_amdgcn_mfma_*`, buffer resource descriptors, LDS builtins,
cross-lane permutes, and the scheduling builtins that let you build a software pipeline.

## Route here when
- Writing a matrix inner loop at the intrinsic level.
- You need hardware bounds checking instead of predicated masks.
- The default scheduler's interleave is provably bad (you checked the ISA) and you want to pin it.

## Register classes you must reason about

| Class | What | gfx950 budget | Role in MFMA |
|---|---|---|---|
| **VGPR** | per-lane vector | 512/SIMD, granule 16 | A/B operands, addresses |
| **AGPR** | accumulation | ≤ 256/lane, **unified pool with VGPR** | MFMA C/D accumulators |
| **SGPR** | scalar (wave-uniform) | ~102 usable | buffer descriptors, loop counters |

MFMA accumulators live in **AGPRs**; moving to/from VGPR costs `v_accvgpr_read/write`.

> **The classic pipelining bug** (LLVM #131954): at large tiles the compiler inserts `v_accvgpr_*` in
> the inner loop and performance falls back to small-tile levels. **The signature is TFLOP/s that
> plateaus or regresses as the tile grows.**
> **Fix:** keep the accumulator in a stable `__attribute__((vector_size))` variable across iterations so
> it stays in AGPRs. CK relies on the "tied accumulator" flag (input accum tied to output), which
> inline asm alone does not give you.

## 1. MFMA builtins

```
d = __builtin_amdgcn_mfma_<CDFmt>_MxNxK<ABFmt>(a, b, c, cbsz, abid, blgp);
```

`a`/`b`/`c` are **per-lane vector slices** — each lane holds `M·K/64`, `K·N/64`, `M·N/64` elements.
`cbsz`/`abid`/`blgp` are broadcast controls; **set 0** for standard GEMM.

| Builtin (gfx950) | M×N×K | A/B → C | A/B/**C** per lane |
|---|---|---|---|
| `mfma_f32_16x16x32_f16` / `_bf16` | 16×16×32 | fp16/bf16 → fp32 | 8/8/**4** |
| `mfma_f32_32x32x16_f16` / `_bf16` | 32×32×16 | fp16/bf16 → fp32 | 8/8/**16** |
| `mfma_f32_16x16x128_f8f6f4` | 16×16×128 | fp8/fp6/fp4 → fp32 | 32/32/**4** |
| `mfma_f32_32x32x64_f8f6f4` | 32×32×64 | fp8/fp6/fp4 → fp32 | 32/32/**16** |
| `mfma_scale_f32_16x16x128_f8f6f4` | 16×16×128 | MXFP8/6/4, E8M0 | block-scaled |
| `mfma_i32_16x16x64_i8` | 16×16×64 | int8 → int32 | 16/16/**4** |
| `mfma_f64_16x16x4f64` | 16×16×4 | fp64 | 1/1/4 |

**Prefer the 16×16 shapes**: 4 C-registers/lane vs 32×32's 16, *and* the 32×32 op clocks lower under
power. Both reasons point the same way.

**FP8 on gfx950 is OCP**, and the `f8f6f4` family lets A and B pick formats **independently**. The FNUZ
`_fp8_fp8` / `_fp8_bf8` suffixes are the **gfx942** dialect — feeding those bits here is silently wrong.

```cpp
using bf16x8 = __attribute__((vector_size(8*sizeof(__bf16)))) __bf16;
using fp32x4 = __attribute__((vector_size(4*sizeof(float)))) float;
fp32x4 acc = {0,0,0,0};                                   // -> AGPRs; keep stable across the loop
acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a_reg, b_reg, acc, 0, 0, 0);
```

**Use the matrix calculator for the lane→element map** rather than reverse-engineering it:
`matrix_calculator.py --architecture cdna4 --instruction <op> --register-layout --A-matrix`.

## 2. Buffer resource descriptors

`buffer_*` operations use a **128-bit V#** held in SGPRs: base, stride, num-records (bounds), flags.

Two wins over plain `global_load`:
- **Hardware bounds checking** — OOB lanes return 0 and writes are dropped, so **no predication
  branch** in your tail handling.
- Sometimes better address generation.

```cpp
float4 v = __builtin_amdgcn_raw_buffer_load_b128(rsrc, voffset, /*soffset=*/0, /*aux=*/0);
__builtin_amdgcn_raw_buffer_store_b128(value, rsrc, voffset, 0, 0);
```

Prefer **b128** (the `global_load_dwordx4` equivalent) in inner loops. `voffset ≥ num_records` returns 0
safely — this is what replaces predication masks in GEMM tails. Build the descriptor with
`__amdgcn_make_buffer_rsrc` where available rather than hardcoding the flags word.

This is exactly what Triton emits behind `knobs.amd.use_buffer_ops`.

## 3. LDS and cross-lane builtins

```cpp
*reinterpret_cast<float4*>(&lds[off]) = v;                // -> ds_write_b128
float4 r = *reinterpret_cast<float4*>(&lds[off]);         // -> ds_read_b128
int x = __builtin_amdgcn_ds_bpermute(srcLane << 2, val);  // gather via LDS crossbar (byte addr)
int y = __builtin_amdgcn_ds_permute (dstLane << 2, val);  // scatter
int z = __builtin_amdgcn_ds_swizzle(val, 0x1F);           // fixed swizzle within a 32-lane group
```

| Builtin | Use |
|---|---|
| `ds_bpermute` / `ds_permute` | arbitrary lane gather/scatter through the LDS crossbar (**uses no LDS storage**) |
| `ds_swizzle` | fixed permutation within a 32-lane group |
| `mov_dpp` / `update_dpp` | cheap neighbour shifts (row/broadcast) — **fastest wave reductions** |
| `permlane16` / `permlanex16` | 16-lane / cross-16 permute |
| `readlane` / `readfirstlane` | broadcast a lane's value to scalar / all lanes |

**DPP and `permlane` beat `ds_*permute` for fixed neighbour patterns**; `ds_bpermute` is the general
gather. Reach for the cheapest one that expresses your pattern.

## 4. Scheduling builtins

```cpp
__builtin_amdgcn_sched_barrier(mask);                        // hard barrier; mask = categories allowed to cross (0 = block all)
__builtin_amdgcn_sched_group_barrier(mask, size, sync_id);   // a group of `size` instrs of category `mask`, ordered by sync_id
__builtin_amdgcn_iglp_opt(variant);                          // predefined IGLP pipeline (0/1)
```

`SchedGroupMask` category bits (as used in CK's GEMM pipelines):

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

**Use these only after the default scheduler has provably failed** (check the ISA first) — **wrong
ratios hurt.** These are the same primitives FlyDSL exposes as `rocdl.sched_*` and Triton hides behind
`schedule_hint`.

## Verify

| Check | Pass |
|---|---|
| MFMA shape emitted | the 16×16 form you asked for |
| `v_accvgpr_*` in the K-loop | **none** (epilogue-only is expected) |
| `scratch_` | **none** |
| Load width | `buffer_load_dwordx4` / `global_load_dwordx4` |
| Fragment layout | matches `--register-layout` — never guess |

```bash
amdclang++ -x hip --offload-arch=gfx950 -O3 -S kern.cpp -o kern.s
grep -cE 'v_accvgpr|scratch_' kern.s        # both ~0 in a clean hot loop
```

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| **TFLOP/s regresses as the tile grows** | LLVM #131954 — spurious `v_accvgpr` / spills | stable accumulator variable; shrink tile |
| Silent wrong answer | guessed the fragment lane order | `--register-layout` |
| FP8 results garbage | fed FNUZ bits to an OCP MFMA | convert, never bit-copy |
| Predication branches in the tail | plain `global_load` + mask | use `buffer_*` (HW bounds) |
| Scheduling builtins made it slower | wrong instruction ratios | remove them; let the scheduler work |

## When not to be here

Hand-rolled MFMA microkernels are rarely worth it against **rocWMMA**, **ck_tile**, **HipKittens**
(`hipkittens.md`), or **FlyDSL** — they already encode the tied-accumulator + sched-group-barrier +
double-buffer patterns correctly. Reach for raw builtins only when those cannot express your fusion.

## Sources
- Matrix Core programming CDNA3/CDNA4 (MFMA format, per-lane layouts, `cbsz`/`abid`/`blgp`, f8f6f4): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- AMD Matrix Instruction Calculator (exact lane→element maps): https://github.com/ROCm/amd_matrix_instruction_calculator
- `sched_group_barrier` semantics: https://reviews.llvm.org/D128158
- AGPR spill / tied accumulator: https://github.com/llvm/llvm-project/issues/131954
- Buffer descriptors, `ds` builtins, sched builtins: https://llvm.org/docs/AMDGPUUsage.html
