---
title: FlyDSL knowledge map — index, file roles & problem-routing
kind: index
scope: languages/flydsl
updated: 2026-10-22
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
documented in `framework/aiter/` (the `aiter_flydsl_libtype` lever). Backend-neutral hardware numbers live in
`hardware/`; this folder references them rather than duplicating.

**Where the techniques live.** This folder carries the FlyDSL *language* (`API_docs/`). The techniques for
making a FlyDSL kernel fast or correct live in the **FlyDSL optimization skill**, an external skill the knowledge
block points at when the campaign passes one with `forge-loop --skill-dir` (for SIKL kernels:
`<SIKL>/.claude/skills/flydsl-kernel-optimization`). It covers a dense-GEMM package, aiter's shipped FlyDSL kernels
and their knobs, the structure-first authoring method, LDS and prefetch, trace capture and analysis, debugging,
perf bisection and FlyDSL's own test suites. Below, a path starting `skill:` is relative to that skill's folder.
If the knowledge block lists no such skill, those techniques are not available in this run; work from
`API_docs/` and `common_methodology/`.

**Two ways you engage FlyDSL — pick your path first, it changes which files you read:**
- **Path A — *use* the aiter-shipped FlyDSL kernel library.** Pick a kernel family, its knobs and its dispatch.
  Entry: `skill: optimize/flydsl_levers/flydsl_kernel_library.md` → `skill: optimize/flydsl_levers/flydsl_knob_space.md`.
- **Path B — *author* a new `@flyc.kernel`.** Entry: `API_docs/` (language) + `API_docs/conventions.md` (style)
  → `skill: optimize/flydsl_levers/flydsl_authoring_method.md`; for a dense GEMM, `skill: gemm/INDEX.md`.

## Reading order
1. **`API_docs/`** — the language itself: `kernel_authoring_guide.md` (`@flyc.kernel`/`@flyc.jit`, launch,
   LDS, tiled copy/MMA) → `layout_system_guide.md` (layout algebra) → `architecture_guide.md` (compile
   stack). Read alongside **`API_docs/conventions.md`** (the authoring style guide) before writing any kernel.
2. **The optimization skill** — `skill: SKILL.md` routes a task or symptom to its card.

> **Per-operator cards are not in this folder — and largely not in this repo.** General operator theory
> (math contract, shape regimes, Amdahl weight, parity bands) is not maintained here; read the source.
> The FlyDSL *dispatch* decision does live in `framework/aiter/`
> (`skills/optimize/aiter_levers/aiter_flydsl_libtype.md`) — read that for whether FlyDSL is the right backend.

## Portable golden rules (FlyDSL-specific)
- **Layout-first.** Prefer the layout API (`fx.rocdl.make_buffer_tensor()` + logical layouts +
  `fx.copy_atom_call`) for new kernels; raw `buffer_ops.create_buffer_resource()` / manual byte offsets are **legacy**.
- **Loop forms.** `range_constexpr(n)` unrolls at compile time. A plain `range(...)` lowers to `scf.for` and
  carries the names its body reassigns; `range(..., init=[...])` carries exactly the values listed. Keep carried
  state compact.
- **Single explicit exit path** in traced functions — no early `return`, no branch-local `return`/`yield`;
  hoist values out of `if`/`else` so MLIR result types stay well-defined.
- **Shared memory via `SharedAllocator`** (over legacy `SmemPtr`); clear `SmemPtr._view_cache = None` after
  exiting `scf.for` when recreating shared-memory views (avoids MLIR dominance errors).
- **Stale cache is the #1 "my fix didn't work"** — `rm -rf ~/.flydsl /tmp/flydsl*` before re-testing.
- **Structure before parameters.** Fusion / pipelining / layout changes beat knob-tuning; tune last.

## Start here — problem → files → order
Paths are relative to this folder, except `skill:` paths (see above).

| Task / symptom | Read in this order |
|---|---|
| "What is FlyDSL / where does it fit?" | `API_docs/architecture_guide.md` → `skill: optimize/flydsl_levers/flydsl_kernel_library.md` |
| "Write my first FlyDSL kernel" | `API_docs/kernel_authoring_guide.md` → `API_docs/conventions.md` → `API_docs/examples/` (01→04) |
| "Understand the layout algebra (Shape/Stride/tiled copy/MMA)" | `API_docs/layout_system_guide.md` → `API_docs/cute_layout_algebra_guide.md` → `API_docs/flydsl-tile-programming.md` |
| "Which pre-built kernels exist + their dtype/config?" | `API_docs/prebuilt_kernels_guide.md` → `skill: optimize/flydsl_levers/flydsl_kernel_library.md` |
| Optimize, profile, debug, or write a GEMM | `skill: SKILL.md` (it routes by task and symptom) |
| "Tune / numerics / fusion for operator X" | not covered in this repo — read the source (`framework/aiter/overall/operator_catalog.md` gives the entry point) |

## Folder structure & file roles
```
languages/flydsl/
├── INDEX.md                              ← this map (load first)
└── API_docs/                             ← the FlyDSL language: how to write kernels
    ├── kernel_authoring_guide.md         # @flyc.kernel/@flyc.jit, launch config, LDS, tiled copy/MMA, fx.* API
    ├── conventions.md                    # kernel-authoring conventions & style guide (see note below)
    ├── layout_system_guide.md            # layout algebra: Shape/Stride/Layout, products/divides, coord mapping
    ├── architecture_guide.md             # compile stack: AST tracing → Fly MLIR passes → ROCDL/JIT/runtime
    ├── prebuilt_kernels_guide.md         # pre-built kernel library (norm/softmax/GEMM/MoE/attention) + dtype/config
    ├── cute_layout_algebra_guide.md      # CuTe layout algebra background (advanced reference)
    ├── flydsl-kernel-authoring.md        # authoring reference + new-kernel step recipe
    ├── flydsl-tile-programming.md        # the PROCEDURE: classify pattern → skeleton → compute → sync/LDS → verify
    └── examples/                         # runnable skeletons: 01 vectorAdd · 02 tiledCopy · 03 tiledMma · 04 preshuffle_gemm
```

## `API_docs/conventions.md` — what it is
The **kernel-authoring conventions & style guide** for writing FlyDSL kernels (grounded in the `flydsl`
compiler repo, with PR references). It is *rules*, not a tutorial: prefer the layout API over legacy buffer
ops; `range_constexpr` vs `range(...init=[...])`; single explicit exit path in traced functions; hoist
values out of `if`/`else`; clear `SmemPtr._view_cache` after `scf.for`; use `SharedAllocator`; helper
placement (reuse `kernels/kernels_common.py` etc.); and the **`expr/` target-neutrality** rule (the
`python/flydsl/expr/` top-level modules must not import ROCDL/HIP bindings — backend code goes in
`expr/rocdl/`). Read it **before authoring** (Path B) and consult it when a kernel won't trace/compile.

## Where operator knowledge lives
There is **no `operators/` folder here**. The per-operator FlyDSL cards were removed: `overview`/
`fusion`/`numerics`/`tuning` are operator-level facts that do not change with the authoring language, and
keeping a per-language copy meant the same card existed 3–5 times across `triton/`, `ck/`, `hip/`, `asm/`
and `flydsl/`.

Operator-level knowledge is **not maintained in this repo at all** — not per language, and no longer per
framework either. It rots faster than it can be kept true: which backend wins, what the knobs are, which
env var gates which path all turn over every release, and a stale card is worse than none — it sends you
to an entry point that no longer exists, confidently. Where to get those facts instead:
- **"Which API do I call for operator X?"** — `framework/aiter/overall/operator_catalog.md` (entry point
  + signature, pinned to a commit).
- **"Which backend will it dispatch to, and what can I tune?"** —
  `framework/aiter/overall/dispatch_and_rebind.md` + `tuning_db.md`.
- **"What are its shape constraints / numerics?"** — the `assert`s in the kernel source and `op_tests/`.
  Nothing else is authoritative.
- **`framework/mori/operators/`** — the one surviving operator folder: EP dispatch/combine, which is a
  cross-GPU protocol, not a per-release config.


For Path A (using the shipped library), `skill: optimize/flydsl_levers/flydsl_kernel_library.md` +
`flydsl_knob_space.md` and `API_docs/prebuilt_kernels_guide.md` carry the entry points and knob tables the
per-operator `flydsl.md` cards duplicated. For Path B (authoring), `skill: optimize/flydsl_levers/flydsl_authoring_method.md`
and, for a dense GEMM, `skill: gemm/INDEX.md` are the structure-first path.

**Coverage note:** none of the operators this folder used to cover (`gemm_epilogue_fused`,
`layout_shuffle`, `linear_attention_gated_delta`, `reduction`, `splitk_streamk_gemm`, the GEMM and MoE
families) has an operator card in `local_knowledge` any more. The skill's `flydsl_kernel_library.md` and
`API_docs/prebuilt_kernels_guide.md` still enumerate the FlyDSL kernels that implement them.

## Cross-links out of this folder
Backend-neutral hardware constants (gfx950 only) live in `local_knowledge/hardware/`.
Backend-agnostic optimization methodology (roofline, bottleneck classification, benchmarking) lives in
`local_knowledge/common_methodology/`. The library control plane that dispatches FlyDSL kernels into the
live sglang/vLLM path — and the decision of when to reach for the FlyDSL backend — is in
`framework/aiter/` (`skills/optimize/aiter_levers/aiter_flydsl_libtype.md`). Lower-level MFMA/ISA detail is in
`languages/hip/skills/optimize/hip_levers/`.
