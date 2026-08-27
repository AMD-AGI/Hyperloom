---
title: Register allocation — VGPR/AGPR, MFMA fragments, occupancy & pinned tiles
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
  - https://arxiv.org/abs/2511.08083
  - https://github.com/llvm/llvm-project/issues/131954
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# Register allocation (VGPR / AGPR)

## TL;DR
Register pressure is the master occupancy knob on CDNA. The SIMD has **512 × 32-bit registers/lane**;
with one wave/SIMD they split **256 VGPR + 256 AGPR** (HipKittens, arXiv 2511.08083). MFMA accumulators
classically live in **AGPRs**; moving them to VALU costs `v_accvgpr_read/write`. The two failure modes:
(1) the compiler inserts *spurious* `v_accvgpr` moves at large tiles (LLVM #131954), and (2) spilling past
the budget collapses occupancy. HipKittens works around HIPCC's inability to feed AGPRs to MFMA via
**pinned register tiles**. Watch the disassembly, not just the config.

## Core concepts
- **VGPR** — per-lane vector regs, allocated in **16-granules**; 512/lane budget shared with AGPR. Drives
  how many waves fit per SIMD: 256 VGPR/lane → ≤2 waves/SIMD = 8 waves/CU.
- **AGPR** — CDNA-specific accumulation regs; classic MFMA codegen keeps the C accumulator here.
  Reading/writing from VALU code costs `v_accvgpr_read_b32` / `v_accvgpr_write_b32`.
- **Occupancy** = `max(VGPR, AGPR, LDS, wave-slot)`-limited resident waves. Spilling (`scratch_`) is the
  #1 cause of MFMA underperformance.
- On gfx942 the compiler *can* keep MFMA accumulators in **VGPRs** (no AGPR move tax) when pressure
  allows — but that competes with everything else for the 512 budget.
- **HIPCC constraint:** AMD hardware allows AGPRs as matrix *inputs*, but HIPCC does not, forcing
  redundant `v_accvgpr_read` (HipKittens). HK's **pinned register tiles** give explicit register
  assignment so the developer controls scheduling/lifetimes — recovering peak (seq 4096: HK 855 →
  HK-Pinned 1024 TFLOPS, vs AITER 1018; arXiv 2511.08083, Table 1).

## MFMA fragment layout
Each MFMA instruction scatters A/B/C/D across the 64 lanes in a **fixed packed pattern with no guaranteed
element order**. You must place data per the documented layout or the calculator, e.g. for
`v_mfma_f32_16x16x16f16` each lane holds 4 A + 4 B + 4 C elements:
```cpp
using fp16x4_t = __attribute__((ext_vector_type(4))) _Float16;
using fp32x4_t = __attribute__((ext_vector_type(4))) float;
fp16x4_t a, b; fp32x4_t c{};
a = *reinterpret_cast<const fp16x4_t*>(A + 4*(threadIdx.x/16) + 16*(threadIdx.x%16));
for (int i=0;i<4;++i) b[i] = *(B + i*16 + threadIdx.x%16 + (threadIdx.x/16)*64);
c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0,0,0);   // one wave-wide MFMA
```
fp8 packs 8 elems/lane (`fp8x8_t`); the C accumulator regs/lane scale with the MFMA N (4 for 16×16,
16 for 32×32).

## The levers
- **`__launch_bounds__(256)`** caps VGPRs and controls occupancy explicitly.
- **Shrink the tile / fewer MRepeat×NRepeat** if `v_accvgpr` or `scratch_` appear in the hot loop.
- **Prefer 16×16 MFMA** — fewer C accumulator regs/lane (4 vs 16) → less AGPR pressure, more occupancy.
- **Let the accumulator stay in VGPR** when pressure allows (avoids the AGPR move tax).
- **Pinned register tiles** (HipKittens style) for hand-built micro-kernels needing exact lifetime control.
- Type fragments with `ext_vector_type(N)` / `vector_size` so the compiler packs them correctly.

## Pitfalls
- **Spurious `v_accvgpr_read/write` at large tiles** (LLVM #131954): TFLOP/s plateaus or *regresses* as
  you grow the tile. Symptom of the compiler over-spilling the accumulator. Grep `.s` for `accvgpr`.
- **Silent spills** (`scratch_`): cut tile / `KPerBlock` / prefetch depth / `__launch_bounds__`.
- Over-large `MXdlPerWave×NXdlPerWave` → AGPR exhaustion → occupancy of 1 wave/SIMD.
- VGPR banking stalls from bad operand patterns — mostly a "give the scheduler room" concern, rarely
  hand-tunable.

## Verify
```bash
amdclang++ -x hip --offload-arch=gfx942 -O3 -S kern.cpp -o kern.s
grep -cE 'v_accvgpr|scratch_' kern.s          # both should be ~0 in a clean hot loop
# occupancy / reg counts:
hipcc --offload-arch=gfx942 -Rpass-analysis=kernel-resource-usage ...   # VGPR/AGPR/LDS report
```

## Sources
- AMD CDNA3 ISA (VGPR/AGPR file sizes, MFMA fragment layout): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- HipKittens (arXiv 2511.08083 — 256/256 VGPR/AGPR split, HIPCC AGPR-input limitation, pinned register tiles, Table 1 855→1024 TFLOPS): https://arxiv.org/abs/2511.08083
- LLVM #131954 (large MFMA tiles → spurious v_accvgpr moves & spills): https://github.com/llvm/llvm-project/issues/131954
- Matrix Core Programming CDNA3/CDNA4 (MFMA fragment layout examples): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
