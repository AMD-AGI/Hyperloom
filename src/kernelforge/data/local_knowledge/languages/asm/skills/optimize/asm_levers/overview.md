---
title: ASM / MFMA — the lowest level of the Instinct kernel stack
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1, int8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://arxiv.org/abs/2511.08083
---

# ASM / MFMA overview

## TL;DR
This is the foundation under CK / ck_tile / rocWMMA / Triton / TileLang. The **fastest** AMD AI kernels —
the AITER library's hot paths — are **hand-written assembly** by a handful of experts; everything else is
a tradeoff of editability vs that ceiling (HipKittens, arXiv 2511.08083). You reach this level for: the
last 10–20% over a library, a fused op no template expresses, or to diagnose why a higher-level kernel
underperforms. The three sub-levels: **MFMA intrinsics** (`__builtin_amdgcn_mfma_*`, scheduler-friendly,
the default), **inline `asm volatile`** (hand-scheduled micro-loops), and **raw `.s`** (peak
micro-kernels). See [mfma_intrinsics.md](mfma_intrinsics.md), [raw_asm.md](raw_asm.md),
[register_alloc.md](register_alloc.md), [pitfalls.md](pitfalls.md).

## Core concepts — the CDNA3 execution model (MI300X)
| Unit | count/size | notes |
|---|---|---|
| **XCD** | 8 per MI300X | ~38 CUs each; 304 CUs total; clock varies 3–10% across XCDs; private 4 MB L2/XCD, shared LLC before HBM |
| **CU** | 304 | 4 SIMDs each |
| **SIMD** | 4/CU | 64-lane; one wavefront issues per SIMD |
| **Wavefront** | 64 lanes | CDNA is **wave64 only** (RDNA can be wave32) |
| **VGPR** | 512 × 32-bit/lane/SIMD | drives occupancy; 16-granule alloc |
| **AGPR** | 256 × 32-bit/lane | accumulation GPRs for MFMA (CDNA-specific) |
| **LDS** | 64 KB/CU (gfx942); 160 KB (gfx950) | 32 banks × 4B |
| **Matrix cores** | per-SIMD MFMA units | the XDL engines |

With a single wave per SIMD, the 512 registers split into **256 VGPR + 256 AGPR** (HipKittens, arXiv
2511.08083). Occupancy = `max(VGPR, AGPR, LDS, wave-slot)`-limited; spilling past the budget collapses it
and is the #1 cause of MFMA kernels underperforming.

**Instruction classes & counters:** VALU (`v_fma_f32`, `v_pk_*`); MFMA/XDL (matrix core, in-order pipe);
VMEM (`buffer_load_*`, tracked by **vmcnt**); LDS/DS (`ds_read_b128`, tracked by **lgkmcnt**); SMEM
(`s_load_*`, lgkmcnt); scalar/control (`s_waitcnt`, `s_barrier`, `s_setprio`). CDNA4 adds **q_waitcnt**
for the async load queue (HipKittens). CDNA memory ops are **asynchronous**; overlap is achieved by
`s_waitcnt <counter>(N)` = "wait until ≤N outstanding" (NOT "wait N instructions").

## The levers (when to drop to each level)
| approach | use it for | avoid for |
|---|---|---|
| `__builtin_amdgcn_*` intrinsics | MFMA, `ds_*` perms, `buffer_load`, sched barriers, ballot/permute | — (the default; scheduler-friendly) |
| inline `asm volatile` | a tight hand-scheduled micro-loop, latency probes, forcing an encoding | **MFMA** (breaks SchedGroupMask pipelining); anything the compiler schedules well |
| full hand-written `.s` | a peak micro-kernel where you out-schedule LLVM (AITER's fastest paths) | maintainability; rarely worth it for production |

**Verdict:** hand-asm pays only when you can prove via disassembly the compiler's schedule is suboptimal
*and* the kernel is hot enough to amortize the maintenance. For MFMA loops, use intrinsics +
`sched_group_barrier` to *guide* the compiler rather than replace it. AITER ships raw asm because the
last few percent at scale is worth a hand-maintained kernel; HipKittens argues a tile DSL recovers most of
that with far less brittleness.

## Pitfalls (summary — full list in [pitfalls.md](pitfalls.md))
- Hand-written MFMA in inline asm is **not recognized by `SchedGroupMask`** → defeats the SW pipeliner.
- Inline asm needs **early-clobber `"=&v"`** + `"memory"` clobber + `volatile`, and **one block** for
  ordered sequences.
- 32×32 MFMA usually clocks lower than 16×16 (power) → 16×16×16 yields higher achievable FLOPs.

## Verify
```bash
/opt/rocm/bin/amdclang++ -x hip --offload-device-only --offload-arch=gfx942 -O3 -S kern.cpp -o kern.s
# grep -E 'v_mfma|s_waitcnt|accvgpr|ds_read|buffer_load|scratch_' kern.s
```
Use the [amd_matrix_instruction_calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
to confirm A/B/C/D and compression-index register layouts per instruction.

## Sources
- AMD CDNA3 ISA Reference Guide (Ch.7 Matrix Arithmetic; waitcnt; encodings): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- AMD CDNA4 ISA Reference Guide (Ch.7; block-scaled MFMA; gfx950): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- Matrix Core Programming on AMD CDNA3 and CDNA4 (ROCm Blog — intrinsics, register/lane layout): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- HipKittens: Fast and Furious AMD Kernels (arXiv 2511.08083 — peak kernels are raw asm; 256 VGPR/256 AGPR split; q_waitcnt): https://arxiv.org/abs/2511.08083
- amd_matrix_instruction_calculator: https://github.com/ROCm/amd_matrix_instruction_calculator
