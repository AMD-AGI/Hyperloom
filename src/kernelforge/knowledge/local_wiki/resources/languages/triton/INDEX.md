---
title: Triton-on-AMD knowledge map — index, file roles, problem-routing & pinned sources
kind: index
scope: languages/triton
updated: 2026-08-28
---

# Triton on AMD — knowledge map

This file is the entry index for everything under `languages/triton/`. It gives (1) what
this knowledge base covers, (2) the **reading order**, (3) for a given task/symptom **which files to read
and in what order**, (4) the role of every file and folder, and (5) the **pinned reference sources** the
cards cite.

> **Convention (KernelForge standard):** a knowledge folder that contains an `INDEX.md` is navigated
> **through this file** — load it whole. Folders without an `INDEX.md` fall back to a generated
> "filename — one-line description" listing.

## What this knowledge base is
How to **author, tune, and debug Triton kernels on AMD Instinct** (CDNA3 gfx942 / MI300X·MI325X, CDNA4
gfx950 / MI350X·MI355X). The Triton Python API is **identical to NVIDIA**; everything here is about what
changes underneath — the `TritonGPU → TritonAMDGPU → AMDGCN` lowering, the AMD-only knobs
(`matrix_instr_nonkdim`, `kpack`, `waves_per_eu`, `num_stages`), and the CDNA hardware facts that break
CUDA habits. This is a **kernel-source authoring** folder: it sits **below** `framework/aiter/` (the
library control plane that dispatches Triton kernels) and references `hardware/` for the raw numbers
rather than duplicating them.

**Honest positioning (internalize before choosing Triton):** on a *plain* dense GEMM, AMD Triton typically
**loses to tuned hipBLASLt/aiter/CK/asm** (corroborated by HipKittens, arXiv 2511.08083). The real Triton
win is **fusion** (epilogue/attention the library can't express), **skinny split-K decode**, rapid shape
exploration, or the `torch.compile`/Inductor `max-autotune` path.

## Reading order (three layers)
1. **`skills/optimize/triton_levers/triton_amd_delta.md`** — the authoring overview: NVIDIA→AMD cheat sheet, the
   compilation pipeline, the five AMD mistakes that kill perf. **Read this first.**
2. **`API_docs/`** — the Python surface: how a kernel is declared/launched/autotuned
   (`programming_model.md`) and the kernel-body op reference (`language_api.md`).
3. **`skills/`** *when you hit a problem*: `profile/` (measure & target), `bottleneck/` (diagnose), and
   the deeper `optimize/triton_levers/` cards (knobs, patterns, codegen, ISA).

> **Per-operator cards are not in this folder.** This folder is **language-level only** — how to author,
> tune and debug Triton on AMD, independent of which operator you are writing. Operator-level knowledge
> (math contract, shape regimes, Amdahl weight, parity bands) is **not maintained in this repo** — read
> the source. `framework/aiter/overall/operator_catalog.md` gives the aiter entry point and signature;
> `framework/aiter/overall/dispatch_and_rebind.md` tells you which backend a call actually resolves to.
> Then come back here for the Triton authoring levers.

## Portable golden rules (the CUDA habits that break on AMD)
- **wavefront = 64 lanes** (not 32). `num_warps=N` → N·64 threads; all occupancy/reduction math is mod 64.
- **`num_warps=4` to start** — carrying `num_warps=8` from NVIDIA spills VGPRs to scratch (HBM) → 3–5× slower.
- **`num_stages` = 1–2**, not 3–4 — the AMD stream pipeliner pipelines a single GEMM best at 2 (fused FA at 1).
- **`num_stages` only does anything on a loop the pipeliner can schedule** — a `for`/`tl.range` with a
  loop-invariant bound and addresses affine in the induction variable. A `while` bounded by a `tl.load`,
  or a `tl.load` addressed from another `tl.load`, silently forfeits pipelining **and**
  `knobs.amd.use_async_copy` (default-on on gfx950). Rewriting a data-dependent gather as
  "shape-static `tl.range` + mask" once ran **~2× faster while visiting 2–4× more blocks**
  (`local_knowledge/common_methodology/optimization/lever_loop_form.md`).
- **LDS = 64 KB/CU (CDNA3) / 160 KB (CDNA4)**, **512 VGPR/EU** (16-granule) — big tiles silently drop to
  1 wg/CU or fail to compile.
- **FP8 is FNUZ on CDNA3, OCP on CDNA4** — OCP `float8_e4m3fn` into `tl.dot` on gfx942 fails to lower; use `*_fnuz`.
- **AMD-only knobs take effect only inside `triton.Config({...})`** — setting them as Python variables does nothing.
- **`mfma_16x16` > `mfma_32x32`**, ≥1024 programs, 8-multiple tiles, `OPTIMIZE_EPILOGUE=1` (avoid the 512B Tagram hotspot).
- **A config is not trusted until you've read the AMDGCN** — autotune timing catches *what*, the ISA catches *why* (scalar loads, spills, FNUZ mismatch).

## Start here — problem → files → order
Paths are relative to this folder unless prefixed with `local_knowledge/`.

| Task / symptom | Read in this order |
|---|---|
| "Onboard / understand Triton on AMD" | `skills/optimize/triton_levers/triton_amd_delta.md` → `API_docs/programming_model.md` → `API_docs/language_api.md` |
| "Write a kernel body / which `tl.*` op?" | `API_docs/language_api.md` → `API_docs/programming_model.md` → `skills/optimize/triton_levers/triton_templates.md` |
| "Port a CUDA / NVIDIA-Triton kernel to Instinct" | `skills/optimize/triton_levers/triton_amd_delta.md` (cheat sheet) → `.../triton_traps.md` → `.../triton_knob_space.md` |
| "Author / tune operator X" | the kernel source (`framework/aiter/overall/operator_catalog.md` for the aiter entry point) → back here: `skills/optimize/triton_levers/triton_knob_space.md` → `.../triton_templates.md` |
| "Which knobs / how to autotune?" | `skills/optimize/triton_levers/triton_knob_space.md` → `API_docs/programming_model.md` (`@triton.autotune`) |
| "Give me a starting template (GEMM/attention/reduction)" | `skills/optimize/triton_levers/triton_templates.md` |
| "Kernel is slow — what do I tune next?" | `skills/profile/profiling-triton.md` → `skills/optimize/triton_levers/triton_knob_space.md` |
| "Sparse / top-k / paged-gather kernel — `while` over selected blocks, `num_stages` sweeps flat" | `local_knowledge/common_methodology/optimization/lever_loop_form.md` → `skills/optimize/triton_levers/triton_lowering.md` (§3 stream pipeliner, §4 async copy) |
| "Wrong output / won't compile / lowering error" | `skills/bottleneck/debug-triton-kernel.md` → `skills/optimize/triton_levers/triton_traps.md` |
| "Verify the compiled kernel / read the ISA" | `skills/optimize/triton_levers/triton_isa_check.md` → `.../triton_lowering.md` |
| "Understand `tl.dot`→MFMA / the compile pipeline" | `skills/optimize/triton_levers/triton_lowering.md` → `.../triton_isa_check.md` |
| "Block-scaled MXFP8 / MXFP4 GEMM on CDNA4 (gfx950)" | `API_docs/tl_dot_scaled_gfx950.md` → `local_knowledge/hardware/mi350_dtypes.md` |
| "Should I even use Triton for this?" | `skills/optimize/triton_levers/triton_amd_delta.md` ("where it fits" table) |
| "Numerics / parity gate, or fusion neighbours, for operator X" | not covered in this repo — read the source (`framework/aiter/overall/operator_catalog.md` gives the entry point) |
| **"Autotune converged but MFMA utilization is still low"** | **`../gluon/INDEX.md` → `../gluon/skills/optimize/gluon_levers/overview.md`** — that is a *scheduling* limit, not a hardware one, and the next lever is Gluon (see below) |

## Folder structure & file roles
```
languages/triton/
├── INDEX.md                              ← this map (load first)
├── API_docs/                             ← the Triton Python surface (identical to NVIDIA; AMD notes inline)
│   ├── programming_model.md              # @jit, launch grid, @autotune/@heuristics, constexpr, param→HIPOptions map
│   ├── language_api.md                   # tl.* kernel-body op reference (load/store/dot/reduce/math, masking)
│   └── tl_dot_scaled_gfx950.md           # tl.dot_scaled → native v_mfma_scale_* (block-scaled MXFP8/MXFP4, CDNA4 only)
├── skills/                               ← authoring levers + problem-triggered diagnosis
│   ├── optimize/triton_levers/
│   │   ├── triton_amd_delta.md          # WHAT CHANGES vs NVIDIA + where Triton fits (READ FIRST)
│   │   ├── triton_knob_space.md         # the HIPOptions knob set, ranges, autotune space, baking a winner
│   │   ├── triton_templates.md          # CDNA-tuned starting bodies: dense GEMM, split-K decode, fp8, FA, softmax
│   │   ├── triton_lowering.md           # tl.dot->MFMA, layouts/convert_layout, stream-pipeliner, buffer/async loads
│   │   ├── triton_isa_check.md          # AMDGCN dump workflow; what good ISA looks like; the occupancy boundary check
│   │   └── triton_traps.md              # the traps, indexed BY SYMPTOM
│   ├── profile/profiling-triton.md       # read TRITON_PRINT_AUTOTUNING + rocprofv3 PMC → memory/compute verdict → which knob
│   └── bottleneck/debug-triton-kernel.md # classify wrong/won't-lower/slow; FNUZ 2× trap, num_warps spill, ignored knobs
(no operators/ — see "Where operator knowledge lives" below)
```

## Where operator knowledge lives
There is **no `operators/` folder here**. Per-operator cards were removed because they were
operator-level facts (math contract, shape regimes, tuning space, parity bands, fusion neighbours) that
do not change with the authoring language — keeping a copy per language meant the same card existed 3–5
times over.

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


When the task is "write this operator in Triton", get *what* you are building and where it matters from
the kernel source, then use this folder for *how* to author it. `skills/optimize/triton_levers/triton_templates.md`
carries the CDNA-tuned starting templates (dense GEMM, attention, reductions) that the per-operator
`triton.md` cards used to duplicate.

**Coverage note:** none of the operators this folder used to cover (`sparse_attention_nsa`,
`elementwise`, `reduction`, `splitk_streamk_gemm`, the GEMM / attention / norm families) has an operator
card in `local_knowledge` any more. For the NSA / data-dependent-gather case the load-bearing knowledge
is the loop-form rule in
`local_knowledge/common_methodology/optimization/lever_loop_form.md` plus
`skills/optimize/triton_levers/triton_lowering.md` §3–4, both of which survive.

## Pinned reference sources
Cards cite inline; this consolidates the most-used pins. Grow as cards are added.

**Primary language / compiler**
- **triton-lang/triton** — https://github.com/triton-lang/triton — upstream; AMD backend in `third_party/amd/`, CDNA3/CDNA4 first-class. `backend/compiler.py::HIPOptions` is the authoritative knob set — `grep` it on your build.
- **ROCm/triton** — https://github.com/ROCm/triton — AMD staging fork; carries perf patches + tuning utils (`occ.sh`); ROCm PyTorch wheels build from here. Knob defaults drift vs upstream.
- AMD backend dir: `triton/third_party/amd/{backend,lib,include,language/hip}` (HIPOptions, MLIR passes, MFMA lowering).

**SOTA reference kernels (where tuned Triton kernels live)**
- **ROCm/aiter** — https://github.com/ROCm/aiter — production Triton kernels + per-shape tuned tables; the dense-GEMM live path.
- sgl-project/sglang — https://github.com/sgl-project/sglang — Triton attention/MoE kernels + per-shape JSON dispatch.
- vllm-project/vllm — https://github.com/vllm-project/vllm — V1 Triton attention/MoE/sampling backends, per-shape configs.
- pytorch/pytorch (Inductor) — https://github.com/pytorch/pytorch — `torch.compile`/`max-autotune` emits Triton (AMD GEMM knobs, PR #143286).

**AMD primary docs**
- Optimizing Triton kernels on AMD (knobs, OPTIMIZE_EPILOGUE, ISA verify): https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- MI300X workload optimization (num_warps, ≥1024 grid, Tagram, occupancy): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- AI Developer Hub — Triton kernel dev tutorial: https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/triton_kernel_dev.html
- Triton AMD backend HIPOptions / pass pipeline: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- Enabling vLLM V1 on AMD GPUs with Triton (num_warps spill, per-shape configs): https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/
- Matrix Core programming CDNA3/CDNA4 (MFMA shapes, AGPR): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- AMDGPU backend (ISA, s_waitcnt, buffer descriptors): https://llvm.org/docs/AMDGPUUsage.html
- Honest compiler-vs-asm limits (Triton loses plain GEMM to asm/CK): HipKittens, https://arxiv.org/abs/2511.08083

## Going lower: Gluon is the same toolchain, one level down
`languages/gluon/` is **not a different backend** — Gluon is Triton's low-level dialect and shares this
folder's entire substrate: the same `@…jit` frontend, the same
`Triton → TritonGPU → TritonAMDGPU → AMDGCN` lowering, the same JIT cache, the same `@triton.autotune`,
and the same AMDGCN ISA-verification workflow (`skills/optimize/triton_levers/triton_isa_check.md` applies
verbatim). What it changes is *who decides*: tile layouts, the software pipeline (there is no
`num_stages` — you author the stages), the register budget, and the MFMA instruction all become source
you write.

Reach for it when **autotune has converged and the matrix core is still far from peak** — that is the
compiler's schedule binding, and it is the one ceiling no knob in this folder can lift. Do NOT reach for
it while cheaper axes here remain untested: a naive Gluon kernel loses to a tuned Triton one.

Read `../gluon/skills/optimize/gluon_levers/overview.md` for whether the drop is justified and what the
measured rung ladder is, and `../gluon/skills/optimize/gluon_levers/forge_integration.md` before any
edit inside a campaign (change shape, and the version traps — Gluon is `triton.experimental` and has
shipped release-to-release breakage).

## Cross-links out of this folder
Backend-neutral hardware constants (gfx950 only) live in `local_knowledge/hardware/` —
Triton cards reference it rather than duplicating numbers. Backend-agnostic optimization methodology
(roofline, bottleneck classification, benchmarking) lives in `local_knowledge/common_methodology/`. The
library control plane that dispatches these kernels into the live sglang/vLLM path is `framework/aiter/`.
