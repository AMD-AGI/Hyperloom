---
title: Composable Kernel (CK) knowledge map — index, file roles, problem-routing & pinned sources
kind: index
scope: languages/ck
updated: 2026-08-28
---

# Composable Kernel (CK) — knowledge map

This file is the entry index for everything under `languages/ck/`. It gives (1) what
CK is and its one defining decision (classic vs ck_tile), (2) for a given task/symptom, **which files to
read and in what order**, (3) the role of every file and folder, and (4) the **pinned reference sources**
the cards cite.

> **Convention (KernelForge standard):** a knowledge folder that contains an `INDEX.md` is navigated
> **through this file** — load it whole. Folders without an `INDEX.md` fall back to a generated
> "filename — one-line description" listing.

## What CK is (and its two front-ends)
CK is AMD's **C++ template kernel-authoring framework** built on a compile-time
**coordinate-transform + tile** engine (index math folds into the load/store address — no runtime index
arithmetic in a well-written CK kernel). This folder documents CK *authoring* — the templates, knobs,
codegen, and pipelines — the same "language" shape as `hip/`, `triton/`, `flydsl/`. CK is also the layer
**many aiter kernels are written in** (`aiter/csrc/ck_*`), so aiter's `authoring_delegation` points here.

CK has **two front-ends, and choosing between them is the single load-bearing decision:**
- **Classic CK** (`include/ck`, the `DeviceGemm*` device-op family) — the mature path; for **dense square
  bf16/fp16 GEMM** the `DeviceGemmXdlUniversal` v3/Intrawave instance is often still the strongest
  baseline (~615 TFLOP/s @ 4096³ MI300X, **~1.7× faster than ck_tile at the same tile**, Issue #1727).
- **CK-Tile** (`include/ck_tile`, CUTLASS/CuTe-like tile programming) — the path AMD uses for **new** LLM
  kernels: FMHA (paged-KV prefill+decode), fused-MoE, fp8/mxfp4 GEMM. Production SOTA for attention;
  wins on **fusion + attention/MoE**, not raw square GEMM.

**Rule of thumb:** dense square GEMM → benchmark **classic v3** first; attention / MoE / fusion / low-bit
→ **ck_tile**. Never assume ck_tile is automatically faster (Issue #1727).

## Portable golden rules (internalize before authoring)
- **`IsSupportedArgument()` is a correctness gate** — forcing an instance past a `false` returns **silent
  garbage, not an error**. Always gate; use `GemmSpecialization::MNKPadding` for non-divisible shapes.
- **Repo moved:** standalone `ROCm/composable_kernel` is **DEPRECATED** → `ROCm/rocm-libraries`
  (`projects/composablekernel/`); the old `develop` branch is a read-only mirror. Pin the monorepo.
- **A pinned "winning instance" is build-specific** — tile/pipeline IDs drift across CK/ROCm versions;
  re-sweep after any bump, never ship a hand-copied instance table as portable.
- **`mfma_16x16` usually beats `32x32`** on MI300X (power/clock); **`AK1/BK1` ≥ 128-bit/load** (bf16=8,
  fp8=16); aim block count `ceil(M/MPerBlock)·ceil(N/NPerBlock) ≈ k·304`; **split-K (`KBatch≥2`)** fills
  CUs for skinny-M decode.
- **fp8 is FNUZ on CDNA3 (gfx942), OCP on CDNA4 (gfx950)** — match the dequant scale to the encoding.
- **CK is often consumed via aiter/hipBLASLt.** If the library already dispatches a strong CK instance for
  your shape, **tune via the aiter DB first**; author/modify CK templates only for a fusion the library
  can't express or a shape it doesn't cover.
- **MFMA/ISA facts and hardware constants are NOT duplicated here** — MFMA intrinsics live in
  `languages/hip/`; CU/VGPR/LDS/peak numbers in `local_knowledge/hardware/`.

## Start here — problem → files → order
| Task / symptom | Read in this order |
|---|---|
| "Which front-end — classic or ck_tile?" | `skills/optimize/ck_levers/ck_frontend_classic.md` + `ck_frontend_tile.md` (the decision) |
| "Write / tune a dense square GEMM" | `ck_levers/ck_frontend_classic.md` → `ck_gemm_stack.md` → `ck_tuning_knobs.md` |
| "Write / tune attention (FMHA prefill/decode/SWA/GQA/MLA)" | `ck_levers/ck_frontend_tile.md` → `ck_fmha_stack.md` |
| "Fused MoE / grouped GEMM" | `ck_levers/ck_frontend_tile.md` → `local_knowledge/framework/aiter/skills/optimize/aiter_levers/aiter_moe_pipeline.md` (the aiter-side dispatch + `tuned_fmoe` DB) |
| "Tune a CK GEMM — which knob first?" | `ck_levers/ck_tuning_knobs.md` → `ck_gemm_stack.md` |
| "Classic device-op API / call lifecycle" | `API_docs/device_op_api.md` → `ck_levers/ck_frontend_classic.md` |
| "CK-Tile API / tile verbs / kernel composition" | `API_docs/ck_tile_api.md` → `ck_levers/ck_frontend_tile.md` |
| "Build takes forever / instance selection / codegen" | `ck_levers/ck_instance_codegen.md` → `ck_levers/ck_traps.md` |
| "Kernel is wrong / garbage / won't build-select / slow" | `skills/bottleneck/debug-ck-kernel.md` (symptom table → §) |
| "What should I optimize next? (sweep + read profiler)" | `skills/profile/profiling-ck.md` → the knob it points to |
| "Common CK traps before integrating" | `ck_levers/ck_traps.md` |
| "fp8 gives wrong numbers" | `ck_levers/ck_traps.md` (fnuz/OCP) → `skills/bottleneck/debug-ck-kernel.md` (§6) → `local_knowledge/hardware/` |
| "Author / optimize operator X in CK" | the kernel source (`framework/aiter/overall/operator_catalog.md` for the aiter entry point) → back here: `ck_levers/ck_frontend_classic.md` or `ck_frontend_tile.md` → `ck_gemm_stack.md`/`ck_fmha_stack.md` → `ck_tuning_knobs.md` |
| "MFMA intrinsics / read the ISA" | `languages/hip/skills/optimize/hip_levers/hip_builtins.md` (CK does not re-doc) |
| "Hardware constants (CU / VGPR / LDS / peak)" | `local_knowledge/hardware/` (single source of truth) |

## Folder structure & file roles
```
languages/ck/
├── INDEX.md                              ← this map (load first; includes pinned sources)
├── API_docs/                             ← the CK interface standard ("what the calls are")
│   ├── device_op_api.md                  # classic DeviceGemm* family + the 5-call MakeArgument/IsSupportedArgument/Run lifecycle
│   └── ck_tile_api.md                    # ck_tile headers, the 5 tile abstractions, tile verbs, GemmKernel composition
├── skills/                               ← task playbooks (the entry points)
│   ├── profile/profiling-ck.md           # ckProfiler sweep + rocprofv3 PMC → classify → map to a CK knob; cross-check vs hipBLASLt/aiter
│   ├── bottleneck/debug-ck-kernel.md     # wrong/garbage/won't-select/slow: symptom→cause table, IsSupportedArgument, front-end, spills, fp8, ISA
│   └── optimize/ck_levers/               ← the "how to optimize" levers (NO overview.md — ck_classic + ck_tile are the entry points)
│       ├── ck_frontend_classic.md       # DeviceGemm* model: descriptors, CShuffle, pipelines v1-v5, Intra/Interwave, the sweep loop
│       ├── ck_frontend_tile.md          # tile-programming model: TensorView/TileWindow/TileDistribution, pipeline/policy/WarpGemm
│       ├── ck_tuning_knobs.md           # the knob space RANKED (block tile -> KPerBlock -> pipeline -> MFMA -> wave map -> load width)
│       ├── ck_gemm_stack.md             # the XDL parameter stack, the 3 inter-level constraints, the 128-bit-load rule
│       ├── ck_fmha_stack.md             # FA-2 -> CK-Tile mapping, pipeline variants, paged-KV, masking knobs
│       ├── ck_instance_codegen.md       # instance factory vs generate.py; trimming build time; how portable a pin is (it isn't)
│       └── ck_traps.md                  # the 11 CK traps, indexed BY SYMPTOM
(no operators/ — see "Where operator knowledge lives" below)
```

## Where operator knowledge lives
There is **no `operators/` folder here**. The per-operator CK cards were removed: `overview`/`fusion`/
`numerics`/`tuning` are operator-level facts that do not change with the authoring language, and keeping
a per-language copy meant the same card existed 3–5 times across `triton/`, `ck/`, `hip/`, `asm/` and
`flydsl/`.

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


For "write operator X in CK", get *what* you are building from the kernel source, then use this folder
for *how*: `ck_levers/ck_frontend_classic.md` or `ck_frontend_tile.md` for the front-end decision,
`ck_gemm_stack.md` / `ck_fmha_stack.md` for the parameter stack, `ck_tuning_knobs.md` for the tune order.
The two templates carry the CK-specific structure that the per-operator `ck.md` cards duplicated.

**Coverage note:** none of the operators this folder used to cover (`conv2d`,
`sliding_window_attention`, `gemm_epilogue_fused`, `reduction`, `splitk_streamk_gemm`, the attention and
MoE families) has an operator card in `local_knowledge` any more. The CK-side substance survives in the
two templates: `ck_fmha_stack.md` documents FA-2 mapping, paged-KV and SWA masking; `ck_gemm_stack.md`
documents the XDL parameter stack and the CShuffle epilogue.

## Reading-depth guide (how much to load)
- **Deciding the front-end / a single fact**: `ck_levers/ck_frontend_classic.md` or `ck_frontend_tile.md` TL;DR — don't
  load the whole levers folder.
- **Authoring/tuning a GEMM**: `ck_frontend_classic.md` (or `ck_frontend_tile.md`) → `ck_gemm_stack.md` → `ck_tuning_knobs.md`;
  add `ck_traps.md` before trusting a pinned config.
- **Authoring attention**: `ck_frontend_tile.md` → `ck_fmha_stack.md`.
- **Diagnosing a failure**: go straight to `skills/bottleneck/debug-ck-kernel.md` and follow its
  symptom→section table; `skills/profile/profiling-ck.md` when the question is "what next?".
- **Per-operator work**: start from the kernel source and `framework/aiter/overall/dispatch_and_rebind.md`
  (is CK even the backend this call resolves to?), then come back here for the .
- **Hardware / MFMA numbers**: defer to `local_knowledge/hardware/` and `languages/hip|asm/` — never
  duplicated here.

## Pinned reference sources
Single place for the `repo@commit` / canonical-URL pins the `ck/` cards cite (cards also cite inline).

**Primary framework**
- **ROCm/rocm-libraries** `projects/composablekernel/` — https://github.com/ROCm/rocm-libraries — the live CK source. **Standalone `ROCm/composable_kernel` is DEPRECATED** → monorepo; `develop` is a read-only mirror. Paths (`include/ck`, `include/ck_tile`, `example/ck_tile`) identical in both.
- ROCm/composable_kernel — https://github.com/ROCm/composable_kernel — deprecated mirror (pin only for read-only reference).
- `ckProfiler` — classic-CK instance sweeper; build on a dev node (`make -j ckProfiler`); absent in many deployment images.

**Where CK kernels are consumed**
- aiter — `aiter/csrc/ck_*`, `3rdparty/composable_kernel` — CK GEMM/MoE/attention behind `gemm_a8w8_ck`, `ck_moe_*`, `*_cktile`; tune/dispatch via `local_knowledge/framework/aiter/`.
- flash-attention ROCm / vLLM / sglang — CK-Tile FMHA (`example/ck_tile/01_fmha`) — the `--attention-backend ck` path.

**AMD primary docs (canonical)**
- Optimizing with Composable Kernel (instance selection, ckProfiler, IsSupportedArgument): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- A Block GEMM on MI300 (descriptor hierarchy, tile sizing, 256×256 / 304 CU, LDS): https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
- Hands-On with CK-Tile GEMM (WarpGemm, policy, AK1/BK1): https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
- FlashAttention-v2 with CK-Tile (FMHA pipeline mapping): https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- ck_tile component docs (tile_window / tensor_views / sweep_tile): https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/
- Issue #1727 (ck_tile vs classic v3 dense-GEMM perf gap: 359 vs 615 TFLOP/s): https://github.com/ROCm/composable_kernel/issues/1727
- Matrix Core programming CDNA3/CDNA4 (MFMA, 16×16 vs 32×32): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- MI300X workload optimization (128-bit load, split-K, occupancy): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html

## Cross-links out of this folder
The MFMA layer CK's `WarpGemm` wraps: `languages/hip/skills/optimize/hip_levers/hip_builtins.md`.
Backend-neutral hardware constants: `local_knowledge/hardware/`. Tuning/dispatch of
CK-via-aiter: `local_knowledge/framework/aiter/`. Alternative authoring paths and cross-backend SOTA
cards: `languages/{triton,gluon,flydsl,hip}/`. Benchmark discipline: `local_knowledge/common_methodology/`.
