---
title: asm — MFMA builtins, shapes, and the block-scaled MXFP family
kind: language
lever: asm_mfma_builtins
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://github.com/ROCm/amd_matrix_instruction_calculator
  - https://github.com/llvm/llvm-project/pull/116723
  - https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html
---

# MFMA builtins

**This is the default sub-level.** `__builtin_amdgcn_mfma_*` gives you the matrix core with the
compiler still able to schedule around it. Only drop lower when disassembly proves it necessary
(`asm_decision.md`).

## Route here when
- Writing or hand-tuning a matrix inner loop at the intrinsic level.
- Choosing an MFMA shape, or wiring up block-scaled MXFP.
- The pipeliner is not overlapping your loads with compute (check: is your MFMA an *intrinsic*?).

## The call

```
d_reg = __builtin_amdgcn_mfma_<ODType>_<M>x<N>x<K><InDType>(a_reg, b_reg, c_reg, cbsz, abid, blgp);
```

- `ODType` — accumulate type: `f32` / `i32` / `f64`. **Always 32-bit.** Never accumulate narrower.
- `InDType` — A/B element type.
- `cbsz, abid, blgp` — broadcast / block-select flags. **`0,0,0` for plain GEMM.**
  For SMFMAC these are re-purposed as the compression-index selector — see `asm_inline_and_raw.md`.

MFMA is a **wave-level** op: all 64 lanes cooperate on one M×N×K block, with A/B/C/D fragments scattered
across lane registers in a fixed packed pattern.

## gfx950 shape table

| Intrinsic | M×N×K | in→out | cycles | A/lane | B/lane | **C/lane** |
|---|---|---|---:|---:|---:|---:|
| `..._f32_16x16x32_f16` / `_bf16` | 16×16×32 | f16/bf16→f32 | 16 | 8 | 8 | **4** ← prefer |
| `..._f32_32x32x16_f16` / `_bf16` | 32×32×16 | f16/bf16→f32 | 32 | 8 | 8 | **16** |
| `..._f32_16x16x128_f8f6f4` | 16×16×128 | fp8/fp6/fp4→f32 | 16 or 32 | 32 | 32 | **4** |
| `..._f32_32x32x64_f8f6f4` | 32×32×64 | fp8/fp6/fp4→f32 | 32 or 64 | 32 | 32 | **16** |
| `..._i32_16x16x64_i8` | 16×16×64 | i8→i32 | 16 | 16 | 16 | **4** |
| `..._i32_32x32x32_i8` | 32×32×32 | i8→i32 | 32 | 16 | 16 | **16** |
| `..._f64_16x16x4f64` | 16×16×4 | f64→f64 | 64 | 1 | 1 | 4 |

> **f8f6f4 cycle rule:** the **lower** count applies when neither A nor B is FP8 (FP6/FP4 only); the
> **higher** when either is FP8. That is why FP6/FP4 reach 10 PF while FP8 tops at 5 PF.

**FP8 on gfx950 is OCP** (E4M3FN / E5M2), not FNUZ. A and B formats can be **mixed** in the f8f6f4
family. TF32 was **removed**.

## Why 16×16 wins — two independent reasons

Both point the same way, so the default is not close:

1. **Register footprint.** 16×16 carries **4** C-registers/lane; 32×32 carries **16**. That 4× comes out
   of the 512-register budget and directly costs occupancy (`asm_register_budget.md`).
2. **Power and clock.** The 32×32 op has better *software* efficiency (bigger payload, fewer
   instructions) but draws more power → the part clocks lower → **lower max-achievable FLOPs**
   (ROCm Max-Achievable-FLOPs Part 2).

**Default to 16×16.** Only test 32×32 for a specific large square shape, and check the register count
did not cross a tier boundary when you do.

HipKittens defaults register tiles to the smallest MFMA for maximum scheduling control, mixing shapes
only where the algorithm demands it.

## Block-scaled MXFP (gfx950, ROCm ≥ 7.0)

Different signature from classic MFMA:

```cpp
d = __builtin_amdgcn_mfma_scale_f32_MxNxK_f8f6f4(
        a, b, c, Atype, Btype, OPSEL_A, scale_a, OPSEL_B, scale_b);
// Atype/Btype: 0=E4M3(fp8) 1=E5M2(bf8) 2=E2M3(fp6) 3=E3M2(bf6) 4=E2M1(fp4)
// scale_*: E8M0, one 8-bit exponent per 32-element block -> factor 2^(scale-127)
```

- Shapes: **`16x16x128`** and **`32x32x64`**.
- A and B formats are chosen **independently** — mix FP4 weights with FP6/FP8 activations.
- The scale is applied **after the dot product, before accumulation**.
- At LLVM level: `llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4`, VOP3PX encoding bundling the pre-scale
  `v_mfma_ld_scale_b32`. The compiler shrinks the operand vector when the format needs <8 bits
  (e.g. fp6 → `v6i32`) — LLVM PR #117047.

### The scale-layout trap

**The scale layout that global memory delivers is not the layout the scaled MFMA consumes**, and no
instruction converts register→MFMA-layout directly. You need a three-step pipeline:

```
global read scales  →  LDS write (converts layout)  →  LDS read  →  feed scaled MFMA
```

Budget for it. Skipping the LDS round-trip is not an option, and getting the layout wrong is silent
corruption, not an error.

## What to change

- **Pick the smallest MFMA (16×16)** for both scheduling flexibility and power-limited FLOPs.
- **Push K-density with low precision** — fp8 doubles, fp4 quadruples K per instruction, so fewer
  instructions per K. Watch the operand register count (32 regs/lane at f8f6f4).
- **Prefer FP6 over FP4 when FP4 is too lossy** — identical 10 PF rate, more mantissa.
- **`blgp` / `cbsz` / `abid`** for A-broadcast tricks (e.g. broadcasting one A block across N). Rare;
  check the calculator first, and note they conflict with SMFMAC index selection.

## Verify

```bash
./matrix_calculator.py --architecture cdna4 --instruction v_mfma_f32_16x16x32_bf16 --detail-instruction
./matrix_calculator.py --architecture cdna4 --instruction v_mfma_f32_32x32x16_f16 --register-layout --A-matrix
./matrix_calculator.py --architecture cdna4 --instruction v_mfma_scale_f32_16x16x128_f8f6f4 --detail-instruction
# output Vx{y}.z : x = register offset, y = lane, .z = sub-register
```

Then confirm in the disassembly that the shape you asked for is what got emitted, and that the MFMA is
an intrinsic (not buried in an `asm volatile` block).

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| No software pipelining, loads not overlapped | MFMA written in inline asm — `SchedGroupMask` cannot see it | keep MFMA as the intrinsic |
| Silent wrong answer | guessed the fragment lane order | use the calculator, never guess |
| FP8 results garbage | fed FNUZ bits to a gfx950 OCP MFMA | convert, do not bit-copy |
| Scaled MFMA garbage | scale layout not converted through LDS | build the 3-step pipeline |
| `mfma_scale_*` won't compile | needs gfx950 **and ROCm ≥ 7.0** | check the toolchain |
| Chose 32×32 "because bigger" | it is not faster — 4× C-registers *and* lower clock | default 16×16 |
| Accuracy drift over long K | down-converted the accumulator | keep f32/i32 through the K-loop |

## Sources
- AMD CDNA4 ISA §7 (MFMA encodings, block-scaled MFMA, E8M0, f8f6f4): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- Matrix Core Programming CDNA3/CDNA4 (intrinsic forms, `mfma_scale` syntax, the scale pipeline): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- amd_matrix_instruction_calculator (per-instruction M/N/K, cycles, register/lane layouts): https://github.com/ROCm/amd_matrix_instruction_calculator
- LLVM PR #116723 (define `v_mfma_f32_{16x16x128,32x32x64}_f8f6f4`) and #117047 (shrink operand regs by format): https://github.com/llvm/llvm-project/pull/116723 · https://github.com/llvm/llvm-project/pull/117047
- ROCm Blog — Measuring Max-Achievable FLOPs Part 2 (16×16 vs 32×32 power/clock): https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html
