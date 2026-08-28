---
title: asm — should you drop to this level, and to which sub-level
kind: language
lever: asm_decision
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://arxiv.org/abs/2511.08083
---

# Dropping to assembly — the decision

**Read this before writing a line of asm.** Most kernels that reach this folder should not be here, and
of the ones that should, most need only the *first* sub-level. This card decides both questions.

## Route here when

One of exactly three reasons:

1. **The last 10–20% over a tuned library**, and the kernel is hot enough to amortize hand maintenance.
2. **A fused op no template expresses** — CK/ck_tile/Triton/rocWMMA/HipKittens all failed to express it.
3. **Diagnosing** why a higher-level kernel underperforms — you are *reading* asm, not writing it.

**Skip this folder if** you have not yet proven from disassembly that the compiler's schedule is the
problem. "It feels slow" is not a reason. Go measure
(`common_methodology/profiling/measure_triage.md`) and try the language-level levers first.

## The honest tradeoff

The fastest AMD AI kernels — aiter's hot paths — **are hand-written assembly**, maintained by a handful
of experts. Everything above is a tradeoff of editability against that ceiling (HipKittens,
arXiv 2511.08083). Two facts to hold at once:

- Raw asm is where peak lives. aiter ships it because the last few percent at scale pays for a
  hand-maintained kernel.
- A tile DSL recovers **most** of that with far less brittleness. HipKittens' 4-wave interleave FP8 GEMM
  reaches **3327 TFLOPS** in 183 lines; AMD's own HIP 8-wave ping-pong hits **3204 TFLOPS** with **no
  assembly at all**, beating hipBLASLt's 3130 (MI355X, ROCm 7.1.0, M=N=K=8192).

**So: if a symmetric all-waves-compute schedule in HIP or a tile DSL gets you to ~95% of the library,
stop there.** Reach for asm for the residual, not the bulk.

## The three sub-levels — drop only as far as you must

| Level | Use it for | Do **not** use it for | Card |
|---|---|---|---|
| **1. MFMA intrinsics** `__builtin_amdgcn_mfma_*` | matrix ops, `ds_*` permutes, `buffer_load`, sched barriers, ballot/permute | — this is the **default** | `asm_mfma_builtins.md` |
| **2. inline `asm volatile`** | a tight hand-scheduled micro-loop, a latency probe, forcing a specific encoding | **MFMA itself** — see below | `asm_inline_and_raw.md` |
| **3. raw `.s`** | a peak micro-kernel where you provably out-schedule LLVM | production maintainability | `asm_inline_and_raw.md` |

> **The rule that catches most people:** hand-written MFMA in inline asm is **invisible to
> `SchedGroupMask`**, which defeats the software pipeliner entirely. Keep MFMA as the *intrinsic* and
> hand-schedule only the surrounding `buffer_load` / `ds_read`. Use `sched_group_barrier` to **guide**
> the compiler rather than replace it.

## gfx950 as it looks from the ISA level

| Unit | Value | Consequence at this level |
|---|---|---|
| XCD | **8** | private L2 per XCD; 3–10% clock spread across dies |
| CU | **256** (32/XCD) | grid math; `hipGetDeviceProperties`, never hardcode |
| SIMD | 4/CU, 64-lane | one wavefront issues per SIMD |
| Wavefront | **64 lanes only** | no wave32 on CDNA; all lane math mod 64 |
| Registers | **512 × 32-bit per lane per SIMD** | 16-granule allocation |
| One wave/SIMD split | **256 VGPR + 256 AGPR** | the budget a hand-built micro-kernel actually owns |
| LDS | **160 KiB/CU**, **64 banks** × 4 B | any 32-bank swizzle you inherited is wrong |
| Matrix cores | 4/CU (per-SIMD XDL engines) | 1024 device-wide |
| FP8 | **OCP** (E4M3FN / E5M2) | **not FNUZ** — re-cast, never bit-copy |
| TF32 | **removed** | no path will be emitted |

Occupancy is `max(VGPR, AGPR, LDS, wave-slot)`-limited. **Spilling past the budget collapses it and is
the #1 cause of slow MFMA kernels.** Full model: `hardware/mi350_execution.md`.

## Instruction classes and their counters

| Class | Examples | Counter |
|---|---|---|
| VALU | `v_fma_f32`, `v_pk_*` | — |
| MFMA / XDL | `v_mfma_*`, `v_smfmac_*` | in-order matrix pipe |
| VMEM | `buffer_load_*`, `global_load_*` | **`vmcnt`** |
| LDS / DS | `ds_read_b128`, `ds_write_b128` | **`lgkmcnt`** |
| SMEM | `s_load_*` | `lgkmcnt` |
| Scalar / control | `s_waitcnt`, `s_barrier`, `s_setprio` | — |
| Async load queue (gfx950) | direct-to-LDS staging | **`q_waitcnt`** |

**CDNA memory ops are asynchronous.** Overlap is expressed through
`s_waitcnt <counter>(N)` = **"wait until ≤ N outstanding"** — *not* "wait N instructions". That single
semantic is what the whole hand-scheduling game is built on (`asm_inline_and_raw.md`).

## Verify — the compile-and-look loop

```bash
/opt/rocm/bin/amdclang++ -x hip --offload-device-only --offload-arch=gfx950 -O3 -S kern.cpp -o kern.s
grep -E 'v_mfma|v_smfmac|s_waitcnt|accvgpr|ds_read|buffer_load|scratch_' kern.s
```

| Look for | Pass |
|---|---|
| `scratch_` in the hot loop | **none** — any spill is a bug |
| MFMA shape | the 16×16 form you asked for |
| `s_waitcnt` placement | between stages, not before every load |
| `ds_read_b128` / `buffer_load_dwordx4` | wide forms, not `b32` / `dword` |

Treat `amd_matrix_instruction_calculator --architecture cdna4` as authoritative for A/B/C/D and
compression-index register layouts — never guess lane order.

## Where to go next

| Question | Card |
|---|---|
| Which MFMA intrinsic, which shape, block-scaled MXFP? | `asm_mfma_builtins.md` |
| How do I hand-schedule the loads around it? SMFMAC? | `asm_inline_and_raw.md` |
| Why is my occupancy 1? Where did the spills come from? | `asm_register_budget.md` |
| It compiles but the answer/perf is wrong | `asm_traps.md` |
| Measured per-instruction cycles and hazards | `../intellikit/instructions/<op>.md` |
| Kernel family skeletons (GEMM, attention, GEMV) | `../intellikit/guides/kernel-architecture.md` |

## Sources
- AMD CDNA4 ISA Reference Guide (Ch.7 matrix arithmetic, block-scaled MFMA, waitcnt, encodings): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- Matrix Core Programming on AMD CDNA3 and CDNA4 (intrinsics, register/lane layout): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- HipKittens: Fast and Furious AMD Kernels (arXiv 2511.08083 — peak kernels are raw asm; 256/256 VGPR/AGPR split; `q_waitcnt`; 3327 TFLOPS 4-wave interleave): https://arxiv.org/abs/2511.08083
- amd_matrix_instruction_calculator: https://github.com/ROCm/amd_matrix_instruction_calculator
