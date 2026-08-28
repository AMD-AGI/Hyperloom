---
title: AMDGCN ASM / ISA knowledge map — index, file roles, problem-routing & pinned sources
kind: index
scope: languages/asm
updated: 2026-08-28
---

# AMDGCN ASM / ISA — knowledge map

This file is the entry index for everything under `languages/asm/`. It gives (1) what
this knowledge base covers and how it relates to the rest of the stack, (2) the **reading order**, (3) for
a given task/symptom **which files to read and in what order**, (4) the role of every file and folder
(including the vendored IntelliKit sub-base and the raw ISA extracts), and (5) the **pinned reference
sources**.

> **Convention (KernelForge standard):** a knowledge folder that contains an `INDEX.md` is navigated
> **through this file** — load it whole. Folders without an `INDEX.md` fall back to a generated
> "filename — one-line description" listing.

## What this knowledge base is
AMDGCN assembly is the **lowest level of the Instinct kernel stack** — the foundation under CK / ck_tile /
rocWMMA / Triton / HIP. Unlike the other language folders, **there is no framework control plane here**:
this is a **shared authoring + ISA-reference layer** that the higher folders delegate to (aiter `*_asm` /
HSACO paths, CK `WarpGemm`, HIP inline `asm volatile` / `__builtin_amdgcn_*`). You reach this level for one
of three reasons: **the last 10–20% over a tuned library**, **a fused op no template expresses**, or **to
diagnose why a higher-level kernel underperforms**. The fastest AITER paths are hand-written asm; the
honest tradeoff is editability vs that ceiling (HipKittens, arXiv 2511.08083). Backend-neutral hardware
numbers live in `hardware/`; this folder references them rather than duplicating.

**The three authoring sub-levels** (drop only as far as you must):
1. **MFMA intrinsics** (`__builtin_amdgcn_mfma_*`) — scheduler-friendly, **the default**; the SW pipeliner
   / `SchedGroupMask` only recognizes these.
2. **inline `asm volatile`** — a tight hand-scheduled micro-loop, latency probe, or forced encoding.
3. **raw `.s`** — a peak micro-kernel where you out-schedule LLVM (AITER's fastest paths).

## Reading order (three layers)
1. **`skills/optimize/asm_levers/asm_decision.md`** — the decision itself (should you be here at all,
   and at which sub-level) + the gfx950 execution model as it looks from the ISA (wave64,
   256 VGPR / 256 AGPR, async memory + `s_waitcnt`, instruction classes and counters). **Read this first.**
2. **`API_docs/`** — the "standard doc": the ABI/launch contract (`abi_and_kernel_descriptor.md`) and the
   ISA spec itself (`cdna4_isa_reference.md` → raw `cdna4-isa/*.txt` extracts).
3. **`skills/`** *when you hit a problem*: `profile/`, `bottleneck/`, and the vendored **IntelliKit**
   reference (MI355X-measured per-instruction/hazard ground truth). `ik/guides/kernel-architecture.md`
   carries the GEMM/attention/GEMV/grouped-GEMM kernel families.

> **Per-operator cards are not in this folder — and largely not in this repo.** General operator theory
> (math contract, shape regimes, Amdahl weight, parity bands) is not maintained here at all; read the
> source; `framework/aiter/overall/operator_catalog.md` gives the aiter entry point and signature.
> This folder is the ISA-level *how*.

## Portable golden rules (the ISA-level non-negotiables)
- **wave64 only**; with one wave/SIMD the 512 regs/lane split **256 VGPR + 256 AGPR**. Occupancy =
  `max(VGPR, AGPR, LDS, wave-slot)`-limited; **spilling past the budget collapses it** — the #1 cause of slow MFMA kernels.
- **MFMA accumulates in 32-bit** (f32/i32/f64) — a wave-level op with A/B/C/D fragments scattered across lanes.
- **Prefer intrinsics over inline asm for MFMA** — hand-written MFMA in inline asm is **not seen by
  `SchedGroupMask`** and defeats the software pipeliner. Use intrinsics + `sched_group_barrier` to *guide* the compiler.
- **CDNA memory is asynchronous**: `s_waitcnt <counter>(N)` means "wait until **≤N outstanding**", NOT
  "wait N instructions". Counters: `vmcnt` (VMEM), `lgkmcnt` (LDS/SMEM); CDNA4 adds `q_waitcnt` (async load queue).
- **`mfma_16x16` > `mfma_32x32`** — two reasons: 4 vs 16 C-registers/lane, **and** 32×32 draws more power so the part clocks lower. Both favour 16×16.
- **inline asm hygiene**: early-clobber `"=&v"` + `"memory"` clobber + `volatile`, and **one block** for ordered sequences.
- **NOP/wait-state hazards cause silent corruption** — consult `intellikit/instructions/nop_hazard_summary.md`; never guess.
- **Never guess ISA behavior** — Read/Grep the spec extract or the IntelliKit instruction doc. IntelliKit
  workflow: **disassemble a working `.co` → round-trip validate bit-identical → one targeted change → profile.**

## Start here — problem → files → order
Paths are relative to this folder; `ik/` abbreviates `skills/optimize/asm_levers/intellikit/`.

| Task / symptom | Read in this order |
|---|---|
| "Should I even drop to asm? which sub-level?" | `skills/optimize/asm_levers/asm_decision.md` |
| "Write an MFMA GEMM/attention loop (recommended path)" | `skills/optimize/asm_levers/asm_mfma_builtins.md` → `ik/guides/kernel-architecture.md` |
| "Hand-schedule a raw `.s` micro-loop / memory overlap" | `skills/optimize/asm_levers/asm_inline_and_raw.md` → `ik/instructions/s_waitcnt.md` → `ik/guides/memory-coherence-formats.md` |
| "Reduce VGPR/AGPR pressure / raise occupancy" | `skills/optimize/asm_levers/asm_register_budget.md` → `ik/guides/register-allocation.md` (+ `ik/tools/scripts/vgpr_liveness.py`) |
| "Exact instruction semantics / encoding / wait-states" | `API_docs/cdna4_isa_reference.md` → `API_docs/cdna4-isa/<chapter>.txt` |
| "Measured cycle count / hazard for instruction X" | `ik/instructions/<X>.md` |
| "NOP hazards / silent data corruption" | `ik/instructions/nop_hazard_summary.md` → `skills/bottleneck/debug-asm-kernel.md` |
| "Kernel won't launch / AGPR aliasing / accum_offset" | `API_docs/abi_and_kernel_descriptor.md` → `ik/instructions/kernel_descriptor.md` |
| "Wrong output / broken kernel" | `skills/bottleneck/debug-asm-kernel.md` → `ik/guides/debugging-playbook.md` |
| "Kernel is slow — what do I change next?" | `skills/profile/profiling-asm.md` → `ik/guides/kernel-optimization-workflow.md` |
| "LDS bank conflicts / double-buffer / swizzle" | `ik/guides/lds-patterns.md` → `ik/instructions/ds_read_b128.md` / `ds_write_b128.md` |
| "Block-scaled MXFP8 / FP4 MFMA on CDNA4" | `skills/optimize/asm_levers/asm_mfma_builtins.md` → `API_docs/cdna4-isa/Ch7_block_scaled_p63-68.txt` → `ik/instructions/v_mfma_f32_16x16x128_f8f6f4_scales.md` |
| "Author / tune operator X in asm" | the kernel source (`framework/aiter/overall/operator_catalog.md` for the aiter entry point) → back here: `asm_levers/asm_mfma_builtins.md` → `ik/guides/kernel-architecture.md` → `asm_levers/asm_register_budget.md` |
| "Wrong numbers / no pipelining / TFLOPs regress as tile grows" | `skills/optimize/asm_levers/asm_traps.md` (indexed by symptom) |
| "Numerics / parity gate for operator X" | not covered in this repo — read the source (`framework/aiter/overall/operator_catalog.md` gives the entry point) |

## Folder structure & file roles
```
languages/asm/
├── INDEX.md                              ← this map (load first)
├── API_docs/                             ← the "standard doc": ABI/launch contract + the ISA spec itself
│   ├── abi_and_kernel_descriptor.md      # code-object ABI: kernel descriptor, HSA AQL dispatch, EXEC/SGPR/arg conventions
│   ├── cdna4_isa_reference.md            # pointer file → the raw ISA spec extracts below (Read/Grep on demand)
│   └── cdna4-isa/                        # 30 raw .txt chapter extracts of the AMD CDNA4 ISA spec (~588 KB)
│       │                                 #   Ch7_MFMA_complete / _layout / _sparse / _block_scaled / FP8_BF8_formats,
│       │                                 #   Ch4 wait-state hazards, Ch5/6 SALU/VALU, Ch8/11 SMEM/LDS,
│       └─                                #   Ch9/10 global·FLAT·MUBUF, Sec12/13 VOP3P/DPP/SDWA encodings
├── skills/                               ← authoring levers + problem-triggered diagnosis
│   ├── optimize/asm_levers/              ← 5 levers, all gfx950; each: route-in → constants → change → verify → traps
│   │   ├── asm_decision.md               # SHOULD you drop to asm, and to which of the 3 sub-levels (READ FIRST)
│   │   ├── asm_mfma_builtins.md          # __builtin_amdgcn_mfma_* + the block-scaled MXFP family (the default sub-level)
│   │   ├── asm_inline_and_raw.md         # sub-levels 2-3: s_waitcnt overlap, sched barriers, inline-asm hygiene, SMFMAC
│   │   ├── asm_register_budget.md        # 256 VGPR + 256 AGPR, C-register arithmetic, fragment layout, spills
│   │   ├── asm_traps.md                  # the 10 ISA-level traps, indexed BY SYMPTOM
│   │   └── intellikit/                   # VENDORED AMD RAD IntelliKit — MI355X-measured ground truth (see below)
│   ├── profile/profiling-asm.md          # read the ISA hot loop + rocprofv3 PMC → VALU/MFMA/VMEM/LDS-bound verdict
│   └── bottleneck/debug-asm-kernel.md    # NOP hazards, waitcnt FIFO, AGPR-alias launch fails, disassemble→round-trip workflow
(no operators/ — see "Where operator knowledge lives" below)
```

### The vendored IntelliKit reference (`skills/optimize/asm_levers/intellikit/`)
AMD RAD's IntelliKit ASM Skills, **measured on real MI355X (gfx950/CDNA4) silicon** — hazard rules, NOP
counts, and cycle counts that are **not in the public ISA docs**. It is a self-contained sub-knowledge-base:
- `README.md`, `METHODOLOGY.md`, `CONTRIBUTING.md` — the "start from reference asm → round-trip → one change → profile" methodology.
- `guides/` (6) — `kernel-optimization-workflow.md`, `debugging-playbook.md`, `kernel-architecture.md`
  (GEMM/attention/GEMV/grouped-GEMM/uber-kernel families), `register-allocation.md`, `lds-patterns.md`,
  `memory-coherence-formats.md`.
- `instructions/` (67) — per-instruction docs (syntax, measured cycles, counter tracked, hazards, patterns).
  Key entries: `nop_hazard_summary.md` (silent-corruption reference), `kernel_descriptor.md` (launch-failure
  bugs), `s_waitcnt.md`, `buffer_load_lds.md`, and the `v_mfma_*` family.
- `tools/scripts/vgpr_liveness.py` — VGPR liveness analyzer (dead-register windows + remap suggestions; `--json`).

## Where operator knowledge lives
There is **no `operators/` folder here**. The per-operator asm cards were removed: `overview`/`fusion`/
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


For "write operator X in asm", get *what* you are building from the kernel source, then use this folder
for *how*. The asm-specific kernel structure the per-operator `asm.md` cards carried is in
**`ik/guides/kernel-architecture.md`** (GEMM, attention, GEMV, grouped-GEMM, uber-kernel families) —
that guide is measured on MI355X and is the better source anyway.

**Coverage note:** no operator this folder used to cover has an operator card in `local_knowledge` any
more. The asm-side substance survives in `ik/guides/kernel-architecture.md` (GEMM / attention / GEMV /
grouped-GEMM / uber-kernel families, incl. split-K scheduling) and the `ik/instructions/` set.

## Pinned reference sources
**Primary ISA references**
- **AMD CDNA4 ISA Reference (MI350)** — the reference for this folder; Ch.7, block-scaled MFMA, gfx950: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- **IntelliKit ASM Skills** (vendored, `asm_levers/intellikit/`) — AMD RAD, **measured on MI355X**: 67 instruction docs + 6 guides + `vgpr_liveness.py`.
- **amd_matrix_instruction_calculator** — MFMA lane/element layout (authoritative over any table): https://github.com/ROCm/amd_matrix_instruction_calculator
- Matrix Core programming CDNA3/CDNA4 (ROCm Blog) — intrinsics + register/lane layout: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- AMDGPU backend user guide (LLVM) — buffer descriptors, s_waitcnt, sched builtins: https://llvm.org/docs/AMDGPUUsage.html
- HipKittens (arXiv 2511.08083) — "peak kernels are raw asm"; 256 VGPR / 256 AGPR split; q_waitcnt: https://arxiv.org/abs/2511.08083

**Where hand-asm is consumed** (this folder is the delegated authoring layer)
- **aiter** — `aiter/hsa/{gfx}/…` HSACO, `*_asm` ops, `hsa/codegen.py`; tune/dispatch via `framework/aiter/`.
- **CK** — `WarpGemm` wraps MFMA intrinsics; see `languages/ck/`.
- **HIP** — inline `asm volatile`, `__builtin_amdgcn_*`; `languages/hip/skills/optimize/hip_levers/hip_builtins.md`
  covers the **builtins** (recommended default). This `asm/` folder goes one level lower — raw `.s`, register
  allocation, and per-instruction cycle/hazard truth. The two are complementary; neither duplicates the other.

## Cross-links out of this folder
Backend-neutral hardware constants (gfx950 only) live in `local_knowledge/hardware/`.
Backend-agnostic optimization methodology (roofline, bottleneck classification, benchmarking) lives in
`local_knowledge/common_methodology/`. Higher-level backends that consume hand-asm and dispatch it into the
live path: `framework/aiter/`, `languages/ck/`, `languages/hip/`.
