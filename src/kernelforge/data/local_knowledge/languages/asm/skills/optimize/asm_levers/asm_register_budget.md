---
title: asm — the register budget, AGPR accumulators, spills
kind: language
lever: asm_register_budget
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
  - https://arxiv.org/abs/2511.08083
  - https://github.com/llvm/llvm-project/issues/131954
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# The register budget

**Register pressure is the master occupancy knob on CDNA**, and at this level you own it explicitly.
Two failure modes account for most slow hand-written MFMA kernels: silent spills, and the compiler
inserting accumulator moves you did not ask for.

## Route here when
- The disassembly shows `scratch_` or unexplained `v_accvgpr_read/write` in the hot loop.
- TFLOP/s **plateaus or regresses** as you grow the tile — the classic signature.
- Occupancy is 1 wave/SIMD and you do not know which resource caused it.

## The budget

| Resource | gfx950 | Notes |
|---|---|---|
| Registers per lane per SIMD | **512 × 32-bit** | allocated in **16-granules** |
| One wave/SIMD split | **256 VGPR + 256 AGPR** | the budget a hand-built micro-kernel owns |
| Pool | **unified** | a wave flexes the VGPR/AGPR split |
| Occupancy | `max(VGPR, AGPR, LDS, wave-slot)`-limited | 8 slots/SIMD, 32 waves/CU |
| LDS | 160 KiB/CU | rarely the binding term on gfx950 |

256 VGPR/lane → ≤2 waves/SIMD → 8 waves/CU. Full occupancy arithmetic: `hardware/mi350_execution.md`.

- **VGPR** — per-lane vector registers, all VALU operands.
- **AGPR** — CDNA-specific accumulation registers. Classic MFMA codegen keeps the C accumulator here.
  Touching them from VALU code costs `v_accvgpr_read_b32` / `v_accvgpr_write_b32`.
- The compiler *can* keep MFMA accumulators in **VGPRs** when pressure allows, avoiding the AGPR move
  tax — but that competes with everything else for the same 512.

## The C-register arithmetic that decides your tile

```
A entries/lane = M·K / 64     B entries/lane = K·N / 64     C entries/lane = M·N / 64
```

| Shape | C/lane |
|---|---:|
| 16×16 | **4** |
| 32×32 | **16** |

**Prefer 16×16**: 4× fewer accumulator registers per lane means less AGPR pressure and more occupancy —
and it also clocks higher (`asm_mfma_builtins.md`). `MXdlPerWave × NXdlPerWave` multiplies this; over-size
it and you exhaust AGPRs down to 1 wave/SIMD.

## Fragment placement

Each MFMA scatters A/B/C across the 64 lanes in a **fixed packed pattern with no guaranteed element
order.** You must place data per the calculator's layout — guessing gives a silent wrong answer.

```cpp
using bf16x8_t = __attribute__((ext_vector_type(8))) __bf16;
using fp32x4_t = __attribute__((ext_vector_type(4))) float;
bf16x8_t a, b; fp32x4_t c{};
// ... place a/b per the calculator's --register-layout for this instruction ...
c = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);   // one wave-wide MFMA
```

Type your fragments with `ext_vector_type(N)` / `vector_size` so the compiler packs them into the
register pairs the instruction expects.

## The HIPCC AGPR-input constraint

AMD hardware allows AGPRs as matrix **inputs**; HIPCC does not expose that, forcing a redundant
`v_accvgpr_read` on paths that should not need one (HipKittens, arXiv 2511.08083).

HipKittens works around it with **pinned register tiles** — explicit register assignment so the
developer controls scheduling and lifetimes. Measured effect at seq 4096:
**HK 855 → HK-Pinned 1024 TFLOPS**, against aiter's hand-asm 1018 (Table 1). If you are hand-building a
micro-kernel and the compiler keeps inserting moves, this is the escape hatch.

## What to change

| Lever | Effect |
|---|---|
| **`__launch_bounds__(256)`** | caps VGPRs, makes occupancy explicit |
| **Shrink the tile / `MRepeat`×`NRepeat`** | first move when `v_accvgpr` or `scratch_` appears |
| **Prefer 16×16 MFMA** | 4 vs 16 C-registers/lane |
| **Let the accumulator stay in VGPR** | avoids the AGPR move tax, when pressure allows |
| **AGPR escape hatch** `-mllvm -amdgpu-mfma-vgpr-form=false -mllvm -amdgpu-agpr-alloc=256` | forces accumulators out of the architected budget |
| **Pinned register tiles** | exact lifetime control for hand-built micro-kernels |
| **Direct-to-LDS staging** | removes staging VGPRs entirely (`asm_inline_and_raw.md`) |

Remember the **16-granule rounding**: 170 used → 176 reserved. Shaving registers *within* a tier buys
nothing; crossing a boundary (64/80/96/128/168/256) jumps a whole occupancy tier.

> **2 waves/SIMD with zero spills beats 3 waves/SIMD that spill.** A spill turns a register access into
> scratch memory traffic inside the inner loop.

## Verify

```bash
amdclang++ -x hip --offload-arch=gfx950 -O3 -S kern.cpp -o kern.s
grep -cE 'v_accvgpr|scratch_' kern.s        # both ~0 in a clean hot loop
hipcc --offload-arch=gfx950 -Rpass-analysis=kernel-resource-usage ...   # VGPR/AGPR/LDS report
```

| Check | Pass |
|---|---|
| `scratch_` in the hot loop | **zero** |
| `v_accvgpr_*` in the K-loop | zero (epilogue-only is fine and expected) |
| `.vgpr_count` / `.agpr_count` | matches your budget, below the tier boundary you targeted |
| `rocprof-compute` occupancy panel | says **which** resource binds — do not guess |

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| **TFLOP/s plateaus or regresses as the tile grows** | LLVM #131954 — spurious `v_accvgpr_read/write` and spills at large tiles | grep `accvgpr`; shrink the tile or let the accumulator sit in VGPR |
| Silent slowdown, no obvious cause | `scratch_` spills | cut tile / `KPerBlock` / prefetch depth, or set `__launch_bounds__` |
| Occupancy stuck at 1 wave/SIMD | over-large `MXdlPerWave × NXdlPerWave` → AGPR exhaustion | shrink the accumulator tile |
| Cut registers, occupancy unchanged | shaved within a 16-granule | target the next boundary down |
| Silent wrong answer | guessed the fragment lane order | use `--register-layout` |
| Compiler keeps inserting accumulator moves | HIPCC will not feed AGPRs to MFMA as inputs | pinned register tiles |

## Sources
- AMD CDNA4 ISA (register file sizes, MFMA fragment layout): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- HipKittens (arXiv 2511.08083 — 256/256 VGPR/AGPR split, HIPCC AGPR-input limitation, pinned register tiles, Table 1: 855 → 1024 TFLOPS vs aiter 1018): https://arxiv.org/abs/2511.08083
- LLVM #131954 (large MFMA tiles → spurious `v_accvgpr` moves and spills): https://github.com/llvm/llvm-project/issues/131954
- Matrix Core Programming CDNA3/CDNA4 (fragment layout examples): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
