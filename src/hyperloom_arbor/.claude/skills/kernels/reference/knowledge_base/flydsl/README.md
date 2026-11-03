# FlyDSL Knowledge Base

> Distilled from the upstream **ROCm/FlyDSL** GitHub repo at commit `2812676` (head of `main`, ingested 2026-05-18).
> Source pinned by `source_registry.json` entry `flydsl_repo_ingest_v1`.
> Working clone (for re-reading lines) lives at `_tmp_flydsl_src/FlyDSL/` (do not edit; delete if disk pressure).

## What FlyDSL Is

**FlyDSL = Flexible Layout Python DSL.** A Python frontend + MLIR compiler stack
for authoring high-performance AMD GPU kernels. Two halves:

1. **FlyDSL** — the Python DSL (`python/flydsl/`): trace-based JIT compilation
   via `@flyc.kernel` / `@flyc.jit` decorators, expression API in
   `flydsl.expr`, layout algebra primitives.
2. **Fly dialect** — the MLIR backbone (`include/flydsl/Dialect/Fly/`,
   `lib/Dialect/Fly/`): first-class layout IR (`!fly.layout`,
   `!fly.coord_tensor`, …) with explicit algebra ops (`fly.composition`,
   `fly.logical_divide`, `fly.crd2idx`), lowered through
   `fly-layout-lowering` and `convert-fly-to-rocdl` to ROCDL/LLVM/HSACO.

Inspired by NVIDIA CuTe (CUTLASS) and Composable Kernel; targets gfx942
(MI300X), gfx950 (MI350/MI355X), gfx1250 (MI450), gfx120x (RDNA4).

## Files in this KB

| File | Purpose |
|---|---|
| [overview.md](overview.md) | One-page mental model: stack, decorators, layout, atoms, pipeline, arch matrix |
| [dsl_api.md](dsl_api.md) | The `flydsl.expr.*` Python surface — every public symbol with file:line, including atom catalogs |
| [dialect.md](dialect.md) | C++ Fly + FlyROCDL dialects: types, ops, attrs, passes, conversion patterns, CAPI, runtime wrappers |
| [compilation_pipeline.md](compilation_pipeline.md) | MlirCompiler pass list, JIT cache key, autotune flow, IR dump workflow |
| [kernel_patterns.md](kernel_patterns.md) | The 5 canonical patterns (elementwise / copy / TiledCopy / TiledMma / buffer-load) with code skeletons |
| [gemm_optimization.md](gemm_optimization.md) | Preshuffle GEMM tiling, ping-pong LDS, XOR16 swizzle, prefetch pipeline, `hot_loop_scheduler`, epilogues, VGPR budget |
| [lds_optimization.md](lds_optimization.md) | LDS bank-conflict diagnosis (gfx942/gfx950), XOR swizzle math, padding, write-read distance |
| [env_vars.md](env_vars.md) | Every `FLYDSL_*` env var with default + effect (from `python/flydsl/utils/env.py`) |
| [kernels_inventory.md](kernels_inventory.md) | The 30+ kernels in `kernels/*.py` — what each does, tile shape, dtypes, arch gating |
| [skills_index.md](skills_index.md) | Index of `.claude/skills/*` inside the FlyDSL repo (kernel-authoring, GEMM-opt, debug, etc.) |
| [pitfalls.md](pitfalls.md) | Common LLM/author bugs: cache staleness, `range` vs `range_constexpr`, MFMA operand order, LSE domain mismatch, etc. |
| [dsl_patterns.md](dsl_patterns.md) | Legacy SLA/wpe-tuning notes from earlier ingest (kept for backwards reference) |

## When to Re-Ingest

The upstream repo moves quickly (active perf-tuning work on PA decode, flash-attn,
MoE on gfx1250). Re-run the ingest if you need:
- A new kernel that postdates `2812676`
- New gfx1250/CDNA4 MFMA atom or TDM async-copy semantics
- A pass-pipeline change (e.g. new `fly-*` pass)

Re-ingest by: `git clone --depth 1 https://github.com/ROCm/FlyDSL.git` + re-run
the 3 Explore agents and refresh this directory.
