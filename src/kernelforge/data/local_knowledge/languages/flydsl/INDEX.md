---
title: FlyDSL knowledge map — index, file roles & problem-routing
kind: index
scope: languages/flydsl
updated: 2026-07-14
---

# FlyDSL — knowledge map

This file is the entry index for everything under `languages/flydsl/`. It gives (1)
what FlyDSL is and the two ways you engage it, (2) the **reading order**, (3) for a given task/symptom
**which files to read and in what order**, and (4) the role of every file and folder.

> **Convention (KernelForge standard):** a knowledge folder that contains an `INDEX.md` is navigated
> **through this file** — load it whole. Folders without an `INDEX.md` fall back to a generated
> "filename — one-line description" listing.

## What FlyDSL is
FlyDSL (`flyc`) is AMD's **Python-embedded tile/layout DSL** for authoring CDNA kernels. You write device
code with `@flyc.kernel` + a `@flyc.jit` host launcher using a **CuTe-style layout algebra**
(Shape/Stride/Layout, tiled copy, tiled MMA); it compiles **Python AST → Fly MLIR dialect → ROCDL →
AMDGCN**. It sits between the compiler-DSL tier (Triton) and hand-asm: more layout control than Triton,
far less brittle than raw `.s`. Its kernels ship **inside aiter** (`aiter/ops/flydsl/`) as a backend
(split-K HGEMM, small-M decode HGEMM, fp8 GEMM, MoE, norm/softmax); the live dispatch/rebind seam is
documented in `framework/aiter/` (the `flydsl_path` lever). Backend-neutral hardware numbers live in
`hardware/`; this folder references them rather than duplicating.

**Two ways you engage FlyDSL — pick your path first, it changes which files you read:**
- **Path A — *use* the aiter-shipped FlyDSL kernel library.** Tune `flydsl_hgemm`-style knobs, pick a
  kernel family, handle preshuffle/dispatch. Entry: `skills/optimize/flydsl_levers/{kernel_families,knobs}.md`
  → `operators/<op>/flydsl.md`.
- **Path B — *author* a new `@flyc.kernel` from scratch.** Entry: `API_docs/` (language) +
  `API_docs/conventions.md` (style) → `skills/optimize/flydsl_levers/{authoring_optimization,authoring_gemm_levers}.md`.

## Reading order (three layers)
1. **`API_docs/`** — the language itself: `kernel_authoring_guide.md` (`@flyc.kernel`/`@flyc.jit`, launch,
   LDS, tiled copy/MMA) → `layout_system_guide.md` (layout algebra) → `architecture_guide.md` (compile
   stack). Read alongside **`API_docs/conventions.md`** (the authoring style guide) before writing any kernel.
2. **`skills/optimize/flydsl_levers/`** — the levers: `kernel_families.md` + `knobs.md` (Path A, using the
   library) or `authoring_optimization.md` + `authoring_gemm_levers.md` (Path B, structure-first authoring).
3. **`operators/<op>/`** — the operator you're shipping (overview → `flydsl.md` card → tuning/numerics/
   fusion), plus **`skills/`** *when you hit a problem*: `profile/` (trace, benchmark), `bottleneck/` (debug).

## Portable golden rules (FlyDSL-specific)
- **Layout-first.** Prefer the layout API (`fx.rocdl.make_buffer_tensor()` + logical layouts +
  `fx.copy_atom_call`) for new kernels; raw `buffer_ops.create_buffer_resource()` / manual byte offsets are **legacy**.
- **`range_constexpr(n)`** = compile-time unrolled loop; **`range(start, stop, step, init=[...])`** = `scf.for`
  with loop-carried values. Keep `scf.for` state explicit and compact.
- **Single explicit exit path** in traced functions — no early `return`, no branch-local `return`/`yield`;
  hoist values out of `if`/`else` so MLIR result types stay well-defined.
- **Shared memory via `SharedAllocator`** (over legacy `SmemPtr`); clear `SmemPtr._view_cache = None` after
  exiting `scf.for` when recreating shared-memory views (avoids MLIR dominance errors).
- **Stale cache is the #1 "my fix didn't work"** — `rm -rf ~/.flydsl /tmp/flydsl*` before re-testing.
- **Structure before parameters.** Fusion / pipelining / layout changes beat knob-tuning; tune last.
- **GEMM defaults**: fp32 accumulate, MFMA-16, XOR-swizzled LDS, 2-stage LDS pipeline, split-K via
  global-semaphore reduce; default tile 128×128×64, warps 1×4.

## Start here — problem → files → order
Substitute `<op>` with the operator (catalog below). Paths are relative to this folder.

| Task / symptom | Read in this order |
|---|---|
| "What is FlyDSL / where does it fit?" | `API_docs/architecture_guide.md` → `skills/optimize/flydsl_levers/kernel_families.md` |
| "Write my first FlyDSL kernel" | `API_docs/kernel_authoring_guide.md` → `API_docs/conventions.md` → `API_docs/examples/` (01→04) |
| "Understand the layout algebra (Shape/Stride/tiled copy/MMA)" | `API_docs/layout_system_guide.md` → `API_docs/cute_layout_algebra_guide.md` → `API_docs/flydsl-tile-programming.md` |
| "Use the aiter FlyDSL GEMM library — which knobs?" | `skills/optimize/flydsl_levers/kernel_families.md` → `.../knobs.md` → `operators/<op>/flydsl.md` |
| "Author & optimize a new kernel (structure-first)" | `skills/optimize/flydsl_levers/authoring_optimization.md` → (GEMM) `.../authoring_gemm_levers.md` |
| "Optimize a GEMM specifically" | `skills/optimize/gemm-optimization.md` → `skills/optimize/flydsl_levers/authoring_gemm_levers.md` |
| "LDS bank conflicts / double-buffer / swizzle" | `skills/optimize/lds-optimization.md` → `API_docs/conventions.md` (SharedAllocator) |
| "Overlap loads / prefetch / pipeline" | `skills/optimize/prefetch-data-load.md` |
| "Wrong output / NaN / won't compile / fix didn't take" | `skills/bottleneck/debug-flydsl-kernel.md` (clear cache first) → `API_docs/conventions.md` |
| "Profile / capture a kernel trace / find the hotspot" | `skills/profile/capture-kernel-trace.md` → `skills/profile/kernel-trace-analysis/SKILL.md` |
| "Benchmark / write tests / test tiering" | `skills/profile/testing_benchmarking_guide.md` → `skills/profile/tests_tiering_README.md` |
| "A perf regression appeared — find the culprit commit" | `skills/optimize/bisect-perf-regression.md` |
| "Which pre-built kernels exist + their dtype/config?" | `API_docs/prebuilt_kernels_guide.md` → `skills/optimize/flydsl_levers/kernel_families.md` |
| "Tune / numerics / fusion for operator X" | `operators/<op>/tuning.md` · `operators/<op>/numerics.md` · `operators/<op>/fusion.md` |

## Folder structure & file roles
```
languages/flydsl/
├── INDEX.md                              ← this map (load first)
├── API_docs/                             ← the FlyDSL language: how to write kernels
│   ├── kernel_authoring_guide.md         # @flyc.kernel/@flyc.jit, launch config, LDS, tiled copy/MMA, fx.* API
│   ├── conventions.md                    # kernel-authoring conventions & style guide (see note below)
│   ├── layout_system_guide.md            # layout algebra: Shape/Stride/Layout, products/divides, coord mapping
│   ├── architecture_guide.md             # compile stack: AST tracing → Fly MLIR passes → ROCDL/JIT/runtime
│   ├── prebuilt_kernels_guide.md         # pre-built kernel library (norm/softmax/GEMM/MoE/attention) + dtype/config
│   ├── cute_layout_algebra_guide.md      # CuTe layout algebra background (advanced reference)
│   ├── flydsl-kernel-authoring.md        # authoring reference + new-kernel step recipe
│   ├── flydsl-tile-programming.md        # tile-programming patterns
│   └── examples/                         # runnable skeletons: 01 vectorAdd · 02 tiledCopy · 03 tiledMma · 04 preshuffle_gemm
├── skills/                               ← techniques by phase
│   ├── optimize/
│   │   ├── flydsl_levers/
│   │   │   ├── kernel_families.md        # Path A: the shipped aiter FlyDSL families (HGEMM, small-M, fp8, MoE, norm)
│   │   │   ├── knobs.md                  # Path A: flydsl_hgemm knob set + constraints (verified against aiter source)
│   │   │   ├── authoring_optimization.md # Path B: structure-first optimization workflow for an authored kernel
│   │   │   └── authoring_gemm_levers.md  # Path B: GEMM authoring levers (tiling/LDS/MFMA-loop/epilogue)
│   │   ├── gemm-optimization.md          # GEMM optimization walkthrough
│   │   ├── lds-optimization.md           # LDS sizing / bank-conflict / buffering
│   │   ├── prefetch-data-load.md         # prefetch & load-overlap / pipelining
│   │   └── bisect-perf-regression.md     # locate the commit that regressed perf
│   ├── profile/
│   │   ├── capture-kernel-trace.md       # capture a kernel trace
│   │   ├── kernel-trace-analysis/SKILL.md# analyze the trace (+ scripts/ hotspot_analyzer.py, pmc_l2_analyzer.py)
│   │   ├── testing_benchmarking_guide.md # test infra, benchmark harness, perf measurement
│   │   └── tests_tiering_README.md       # test tiering (unit / lit / GPU kernel tiers)
│   └── bottleneck/debug-flydsl-kernel.md # NaN/wrong/crash: cache invalidation, tracing pitfalls, layout/MFMA/LDS debug
└── operators/<op>/                       ← per-operator knowledge (14 operators; catalog below)
    ├── overview.md   # what the operator is, shapes/regimes
    ├── flydsl.md     # implementation card: entry points, knob table + defaults, numerics/parity, pitfalls, verify
    ├── tuning.md     # tuning levers for this operator
    ├── numerics.md   # accuracy expectations & parity notes
    └── fusion.md     # fusion neighbors / opportunities
```

## `API_docs/conventions.md` — what it is
The **kernel-authoring conventions & style guide** for writing FlyDSL kernels (grounded in the `flydsl`
compiler repo, with PR references). It is *rules*, not a tutorial: prefer the layout API over legacy buffer
ops; `range_constexpr` vs `range(...init=[...])`; single explicit exit path in traced functions; hoist
values out of `if`/`else`; clear `SmemPtr._view_cache` after `scf.for`; use `SharedAllocator`; helper
placement (reuse `kernels/kernels_common.py` etc.); and the **`expr/` target-neutrality** rule (the
`python/flydsl/expr/` top-level modules must not import ROCDL/HIP bindings — backend code goes in
`expr/rocdl/`). Read it **before authoring** (Path B) and consult it when a kernel won't trace/compile.

## Operator catalog (→ `operators/<op>/`)
- **GEMM / linear**: `dense_gemm` · `splitk_streamk_gemm` · `scaled_quant_gemm` (fp8/fp4) · `gemm_epilogue_fused` · `skinny_gemv_decode` (M=1..16 decode)
- **MoE**: `fused_moe_grouped_gemm` · `grouped_gemm_moe`
- **Norm / activation**: `rmsnorm` · `layernorm` · `softmax` · `act_and_mul_silu_gelu`
- **Reduction**: `reduction`
- **Attention (linear)**: `linear_attention_gated_delta`
- **Layout / utility**: `layout_shuffle` (preshuffle / MFMA-native layout)

## Cross-links out of this folder
Backend-neutral hardware constants (CDNA3 MI300 / CDNA4 MI350 / shared) live in `local_knowledge/hardware/`.
Backend-agnostic optimization methodology (roofline, bottleneck classification, benchmarking) lives in
`local_knowledge/common_methodology/`. The library control plane that dispatches FlyDSL kernels into the
live sglang/vLLM path — and the decision of when to reach for the FlyDSL backend — is in
`framework/aiter/` (`skills/optimize/aiter_levers/flydsl_path.md`). Lower-level MFMA/ISA detail is in
`languages/asm/`.
