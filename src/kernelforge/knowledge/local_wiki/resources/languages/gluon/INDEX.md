---
title: Gluon on AMD Instinct knowledge map — index, file roles, problem-routing & pinned sources
kind: index
scope: languages/gluon
updated: 2026-08-29
---

# Gluon on AMD — knowledge map

This file is the entry index for everything under `languages/gluon/`. It gives (1) what this knowledge
base covers, (2) the **reading order**, (3) for a given task/symptom **which files to read and in what
order**, (4) the role of every file, and (5) the **pinned reference sources** the cards cite.

> **Convention (KernelForge standard):** a knowledge folder that contains an `INDEX.md` is navigated
> **through this file** — load it whole. Folders without an `INDEX.md` fall back to a generated
> "filename — one-line description" listing.

## What this knowledge base is
How to **author, tune, and debug Gluon kernels on AMD Instinct** (CDNA3 gfx942 / MI300X·MI325X, CDNA4
gfx950 / MI350X·MI355X). Gluon is **Triton's lower-level dialect** — same Python frontend, same
`@…jit` JIT infrastructure, same `Triton → TritonGPU → TritonAMDGPU → AMDGCN` compilation pipeline,
same launch/`constexpr`/autotune surface, same `TRITON_CACHE_DIR`. What changes is *who decides*:
in Triton the compiler assigns tile layouts, pipeline depth, register allocation and MFMA selection;
in Gluon **you write all four explicitly**.

**This folder is deliberately thin on shared substrate.** Everything Gluon inherits from Triton —
the compile pipeline internals, the AMDGCN ISA-verification workflow, the CDNA hardware facts — lives
in `languages/triton/` and `hardware/` and is cross-linked below rather than duplicated. Read those
for the substrate; read this folder for what Gluon adds on top.

## Reading order (three layers)
1. **`skills/optimize/gluon_levers/overview.md`** — is Gluon even the right move here, and what the
   optimization ladder looks like. **Read this first**; it can send you back to Triton.
2. **`API_docs/`** — the surface: `programming_model.md` (declare/launch/autotune, the layout-typed
   value model), `layouts.md` (the layout objects and what they cost), `amd_targets.md` (the
   `gl.amd.cdna3` / `gl.amd.cdna4` namespaces — buffer ops, async copy to LDS, scaled MFMA).
3. **`skills/optimize/gluon_levers/forge_integration.md`** — **read before your first edit inside a
   forge campaign.** How a Gluon change is shaped so forge can actually keep it, and the version/ABI
   traps that otherwise burn a whole session.

## Portable golden rules (what Gluon changes relative to Triton)
- **Every tensor carries an explicit layout.** `gl.arange(0, XBLOCK, layout=…)` is the seed; layouts
  propagate forward through type inference from there, so you usually annotate only the index tensor.
- **`num_stages` does not exist.** You author the software pipeline yourself. Prefetch depth is code,
  not a knob — but `@triton.autotune` still stacks above `@gluon.jit` for `constexpr` hyperparameters.
- **There is no canonical layout.** Different layout objects can describe the same element mapping.
  Use `gl.convert_layout(x, layout, assert_trivial=True)` to *assert* a conversion is free; a
  conversion that is not free moves data across lanes and warps, and cross-warp movement goes through
  LDS.
- **Do not `convert_layout` before a reduction.** The compiler emits efficient reductions and scans
  from any input layout, so converting first usually costs more than it saves.
- **`gl.warp_specialize` is Hopper-and-newer NVIDIA only.** On CDNA the pipelining mechanism is the
  async-copy group (`async_copy.buffer_load_to_shared` + `commit_group` / `wait_group`) plus
  hand-authored wave scheduling. Do not port an NVIDIA warp-specialized kernel shape verbatim.
- **Wave specialization is the *wrong* pattern on CDNA anyway** — it reaches only ~80% of peak BF16
  GEMM on MI355X because static register allocation starves the producer waves. The two patterns that
  do reach peak are **8-wave ping-pong** and **4-wave interleave** (HipKittens, arXiv 2511.08083).
- **CDNA4-only:** native scaled MFMA (`v_mfma_scale_f32_16x16x128_f8f6f4`) and therefore the MXFP4
  route. CDNA3 runs Gluon fine — buffer loads, async copy to LDS, manual pipelining and the wave
  patterns all apply — but has no native scaled MFMA.
- **Gluon is `triton.experimental`.** The API is not stabilized and it has shipped release-to-release
  breakage. Probe before you build; see `forge_integration.md` § Version traps.

## Start here — problem → files → order
Paths are relative to this folder.

| Task / symptom | Read in this order |
|---|---|
| "Should I use Gluon at all / Triton has plateaued" | `skills/optimize/gluon_levers/overview.md` (decision table) |
| "First Gluon kernel — how do I declare and launch one?" | `API_docs/programming_model.md` → `API_docs/layouts.md` |
| "I'm inside a forge campaign, about to edit" | `skills/optimize/gluon_levers/forge_integration.md` (**first**) → the two above |
| "Which layout do I give this tensor?" | `API_docs/layouts.md` → `../triton/skills/optimize/triton_levers/triton_lowering.md` |
| "`convert_layout` is showing up in my ISA / it's slow" | `API_docs/layouts.md` (§ Conversions and what they cost) |
| "Loads are branchy / masked-load overhead" | `API_docs/amd_targets.md` (§ Buffer ops) |
| "I want global→LDS without staging in registers" | `API_docs/amd_targets.md` (§ Async copy to LDS) |
| "MXFP4 / block-scaled GEMM on gfx950" | `API_docs/amd_targets.md` (§ Scaled MFMA) → `operators/quant_fp4_mxfp/gluon.md` |
| "Bank conflicts on `ds_read`" | `API_docs/layouts.md` (§ Shared layouts) → `../../hardware/` LDS cards |
| "Register spills / occupancy collapsed after a change" | `skills/optimize/gluon_levers/overview.md` (§ The ladder, v6 lesson) → `../triton/skills/optimize/triton_levers/triton_isa_check.md` |
| "MFMA efficiency is low but autotune converged" | `skills/optimize/gluon_levers/overview.md` (§ When Gluon is the answer) |
| "Won't compile / `gluon.aggregate` missing / ABI error" | `skills/optimize/gluon_levers/forge_integration.md` (§ Version traps) |
| "Verify the compiled kernel / read the ISA" | `../triton/skills/optimize/triton_levers/triton_isa_check.md` (shared — Gluon lowers the same way) |
| "Bottleneck classification, roofline" | `../../common_methodology/` (backend-agnostic) |
| "Wavefront / LDS / MFMA / occupancy numbers" | `../../hardware/` (backend-neutral facts) |

## Folder structure & file roles
```
languages/gluon/
├── INDEX.md                                   ← this map (load first)
├── API_docs/                                  ← the Gluon surface
│   ├── programming_model.md                   # @gluon.jit, launch, autotune, the layout-typed value model
│   ├── layouts.md                             # BlockedLayout/SliceLayout/shared layouts, convert_layout costs
│   └── amd_targets.md                         # gl.amd.cdna3 / cdna4: buffer ops, async copy to LDS, scaled MFMA
├── skills/optimize/gluon_levers/
│   ├── overview.md                            # WHEN Gluon (decision table) + the measured optimization ladder
│   └── forge_integration.md                   # forge-campaign shape rules + version traps (READ BEFORE EDITING)
└── operators/<op>/gluon.md                    ← per-operator authoring card (catalog below)
```

## Operator catalog (→ `operators/<op>/gluon.md`)
- `dense_gemm` — the reference workload the whole Gluon ladder was developed on (FP16/BF16).
- `scaled_quant_gemm` — FP8/BF8 with the same skeleton and a larger `BLOCK_K`.
- `quant_fp4_mxfp` — MXFP4 via CDNA4 native scaled MFMA, plus its separate scale pipeline.

Nothing else has a card yet. For an operator with no card, read the source for the math contract and
shape regimes — this repo does not maintain general operator theory — and check
`local_knowledge/framework/aiter/overall/operator_catalog.md` for the aiter entry point plus
`.../dispatch_and_rebind.md` for which backend it resolves to. Then bring the Gluon levers from this
folder. (The sibling language folders no longer carry per-operator cards; `../triton/` is now
language-level only.)

## Pinned reference sources
Cards cite inline; this consolidates the most-used pins.

**Primary language / compiler**
- **Gluon overview** — https://triton-lang.org/main/gluon/index.html — what Gluon is and its scope.
- **Gluon tutorials** — https://triton-lang.org/main/getting-started/tutorials/gluon/index.html —
  `01-intro`, `02-layouts` are the load-bearing two for AMD; the `tcgen05` / TMA / multi-CTA /
  `warp_specialize` ones are NVIDIA-specific and do not transfer.
- **Gluon AMD API** — https://triton-lang.org/main/gluon/api/amd.html and
  https://triton-lang.org/main/gluon/api/amd.cdna4.html — the authoritative `gl.amd.*` listing.
  Read it on YOUR build; this is `triton.experimental` and signatures move.
- **`gluon` dialect** — https://triton-lang.org/main/dialects/GluonDialect.html ·
  **GluonOps** — https://triton-lang.org/main/dialects/GluonOps.html ·
  **TritonAMDGPUOps** — https://triton-lang.org/main/dialects/TritonAMDGPUOps.html (fp4 upcast →
  `v_cvt_scalef32_*`).
- **Linear layouts** (the escape hatch under every layout) — `include/triton/Tools/LinearLayout.h`;
  paper https://arxiv.org/abs/2505.23819.

**AMD reference kernels & measured ceilings**
- **ROCm/gfx950-gluon-tutorials** — https://github.com/ROCm/gfx950-gluon-tutorials — the v0→v9 GEMM
  ladder (a16w16 FP16, a8w8 BF8, a4w4 MXFP4), plus `docs/lds_throughput.md` and
  `docs/memory_bandwidth_model.md`. MIT. Reproduces against an annotated tag in triton-lang/triton.
- **From Naive to Near-Peak: GEMM with Gluon** —
  https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html — the
  narrated walkthrough of that ladder.
- **CDNA4 GEMM kernels (ping-pong / interleave origin)** —
  https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
- **HipKittens** — https://arxiv.org/abs/2511.08083 — why wave specialization loses on CDNA and what
  the two winning wave patterns are; also the honest compiler-vs-hand-tuned gap.
- **ROCm/aiter** — https://github.com/ROCm/aiter — production Gluon in a shipping library; see
  `aiter/ops/triton/attention/pa_mqa_logits.py` — one public entry with a Gluon kernel and a
  `@triton.jit` fallback selected at dispatch, the dual-backend shape forge should copy.

## Cross-links out of this folder
The Triton substrate Gluon shares is in `languages/triton/` — read
`skills/optimize/triton_levers/triton_lowering.md` for the lowering pipeline and MFMA layout selection,
and `.../triton_isa_check.md` for the `AMDGCN_ENABLE_DUMP` workflow (identical for Gluon: same backend, same
ISA). Backend-neutral hardware constants are in `local_knowledge/hardware/`; bottleneck
classification, roofline and benchmarking methodology are in `local_knowledge/common_methodology/`.
The library control plane that dispatches these kernels into a live sglang/vLLM path is
`framework/aiter/`.
