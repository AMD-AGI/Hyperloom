---
title: aiter knowledge map — index, file roles & problem-routing
kind: index
scope: framework/aiter
updated: 2026-08-28
pinned_source: ROCm/aiter@b467ce3425cceeafe4f5587212d36df46feeb265 (v0.1.16-283)
---

# aiter — knowledge map

This file is the entry index for everything under `framework/aiter/`. It gives (1) what
aiter is + the source version these docs are grounded on, (2) the **reading order**, (3) for a given
task/problem **which files to read and in what order**, and (4) the role of every file and folder.

> **Convention (KernelForge standard):** a knowledge folder that contains an `INDEX.md` is navigated
> **through this file** — load it whole. Folders without an `INDEX.md` fall back to a generated
> "filename — one-line description" listing.

## Reading order (two layers)
1. **`overall/`** — universal basics that apply to *every* aiter operator (repo structure, dispatch,
   DB tuning, config system, build/JIT, API catalog, tune-vs-author decision). **Read this first.**
2. **`skills/`** — pick one *when you hit a problem*: `profile/` (measure & target), `bottleneck/`
   (diagnose), `optimize/aiter_levers/` (domain-specific optimize levers: MoE / attention·MLA / FlyDSL).

**There is no per-operator layer here — by design.** See "Where operator knowledge comes from" below.

## What aiter is
aiter (`ROCm/aiter`) is AMD's unified **operator library + per-shape dispatcher** for LLM inference. Its
kernels are **CK / ASM / HIP / Triton / FlyDSL / opus split-K / hipBLASLt** under the hood. This folder
documents the **library control plane** — which op to call, how it builds/dispatches, how to tune its
per-shape DB — and **delegates kernel-source authoring** to `languages/{hip,triton,gluon,flydsl,ck,asm}/`.

- **Pinned source**: `ROCm/aiter@b467ce342` (v0.1.16-283) — the commit every card is grounded on; re-pin per install.
- **Live-path integration (the rebind seam)**: SGLang `SGLANG_USE_AITER=1` (dense → `aiter.tuned_gemm:gemm_a16w16` / `tgemm.mm`); vLLM `vllm/_aiter_ops.py` registers aiter kernels as `torch.ops` custom ops gated by `VLLM_ROCM_USE_AITER*`.
- **Golden rule**: read `overall/` → find the Amdahl-dominant op (profile) → look up its entry point in `overall/operator_catalog.md` → **tune the per-shape DB** → author a kernel only if tuning plateaus. Always prove engagement (`AITER_LOG_TUNED_CONFIG=1` → `is tuned on cu_num`) before trusting any delta.

## Start here — problem → files → order
| Task / symptom | Read in this order |
|---|---|
| Onboarding / "understand aiter" | `overall/repo_layout.md` → `overall/dispatch_and_rebind.md` → `overall/tuning_db.md` |
| "Make operator X faster" (don't know where to start) | `overall/` (basics) → `skills/profile/profiling-aiter.md` (find the Amdahl op) → `overall/tuning_db.md` |
| "Which `aiter.ops.*` API do I call for X?" | `overall/operator_catalog.md` (covers every operator family) → the source |
| "Tune the per-shape DB (GEMM / MoE)" | `overall/tuning_db.md` → `overall/config_files_and_merge.md` → (MoE) `skills/optimize/aiter_levers/aiter_moe_pipeline.md` |
| "Deployed a tuned CSV but it does nothing" (0-engagement) | `skills/bottleneck/debug-aiter.md` (§2) → `overall/dispatch_and_rebind.md` → `overall/config_files_and_merge.md` |
| "Wrong results / crash / won't build / edit didn't take effect" | `skills/bottleneck/debug-aiter.md` → `overall/jit_and_build.md` |
| "Where does kernel Y live? repo layout?" | `overall/repo_layout.md` |
| "DB tuning plateaued — should I write a kernel?" | `overall/authoring_delegation.md` → the matching `languages/<lang>/` folder |
| Domain deep-dive | MoE → `skills/optimize/aiter_levers/aiter_moe_pipeline.md` · attention/MLA → `.../aiter_attention_entries.md` · FlyDSL → `.../aiter_flydsl_libtype.md` |
| Numerics / parity gate for operator X | not covered here — read the source and gate on a task metric, not `allclose`. `common_methodology/optimization/lever_numerics.md` has the general rules |
| Tuning knobs for operator X | `overall/tuning_db.md` — the per-shape DB *is* the aiter lever → `overall/config_files_and_merge.md` |

## Folder structure & file roles
```
framework/aiter/
├── INDEX.md                          ← this map (load first)
├── overall/                          ← LAYER 1: universal basics (read first; applies to every operator)
│   ├── repo_layout.md        # repo structure & source distribution + the dispatcher/build model
│   ├── dispatch_and_rebind.md# how a call resolves (solMap libtype routing) + engages sglang/vLLM
│   ├── tuning_db.md          # per-shape DB tuning (capture→tune→deploy) — the primary optimization lever
│   ├── config_files_and_merge.md # CSV schemas, AITER_CONFIG_* resolution, merge + lowest-us rules
│   ├── jit_and_build.md      # build/JIT system, @compile_ops cache, hsa/codegen.py, optCompilerConfig.json
│   ├── operator_catalog.md   # which aiter.ops.* entry point + signature per operator family
│   └── authoring_delegation.md # decision: tune the DB (default) vs author a kernel (→ languages/*)
├── skills/                           ← LAYER 2: pick one when you hit a problem
│   ├── profile/profiling-aiter.md    # profile a real workload; prove engagement; pick the Amdahl target
│   ├── bottleneck/debug-aiter.md     # diagnose: 0-engagement, build/JIT, ABI, variant/parity traps
│   └── optimize/aiter_levers/        # DOMAIN levers, one per area:
│       ├── aiter_moe_pipeline.md     #   fused MoE: what fuses, tuned_fmoe key, quant routing
│       ├── aiter_attention_entries.md#   which attention entry → which kernel, per generation
│       └── aiter_flydsl_libtype.md   #   the flydsl libtype's three gates + A4W4 → CK fallback
└── (kernel-source authoring is NOT here — see ../languages/{hip,triton,gluon,flydsl,ck,asm}/)
```

## Where operator knowledge comes from (there is no `operators/` folder)
This repo used to carry per-operator cards here. They were **removed**: operator-level knowledge
(which kernel is currently fastest, what the config knobs are this month, which env var gates which
path) is the fastest-rotting kind of knowledge in the stack, and a card that is one aiter release
behind is worse than no card — it sends the agent to an entry point that no longer exists.

**Get operator facts in this order instead:**

| Question | Where the answer actually is |
|---|---|
| "What API do I call for operator X?" | `overall/operator_catalog.md` — entry point + signature per operator family, regenerated against the pinned commit |
| "Which backend will my call dispatch to?" | `overall/dispatch_and_rebind.md`, then confirm at runtime with `AITER_LOG_MORE=1` / `AITER_LOG_TUNED_CONFIG=1` |
| "What can I tune on it?" | `overall/tuning_db.md` + `overall/config_files_and_merge.md`. The per-shape DB is the lever for **every** aiter operator; there is no per-operator knob list to memorize |
| "What are the numerics / shape constraints?" | The `assert`s in the aiter source and `op_tests/` are the ground truth. Read them; do not trust a doc |
| "MoE / attention·MLA / FlyDSL specifics" | `skills/optimize/aiter_levers/aiter_{moe_pipeline,attention_entries,flydsl_libtype}.md` — these are kept because they describe *aiter's own dispatch structure* for a domain, not a single operator's current best config |

> **Rule for adding anything back:** a doc belongs here only if it stays true across aiter releases —
> the dispatch model, the config-DB mechanics, the build system, the engagement-proof workflow. A
> "fastest kernel for operator X right now" card does not, and should be a benchmark run, not a file.

## `overall/` — universal basics (LAYER 1, read first)
- `repo_layout.md` — source-tree map, dispatcher model, build model (where everything lives).
- `dispatch_and_rebind.md` — `solMap` libtype routing (`hipblaslt/asm/skinny/triton/flydsl/opus/torch`) + SGLang/vLLM engagement gates.
- `tuning_db.md` — **primary lever**: capture→tune→deploy the per-shape DB; 10-tuple `gfx`-first key; multi-backend tuner `csrc/gemm_a16w16/gemm_a16w16_tune.py`.
- `config_files_and_merge.md` — the CSV schemas, how `AITER_CONFIG_*` resolves (and what setting it turns off), and the merge rules that decide which row wins.
- `jit_and_build.md` — build/JIT env (`GPU_ARCHS`/`ENABLE_CK`/`AITER_REBUILD`), `@compile_ops` cache, `hsa/codegen.py`, `optCompilerConfig.json`.
- `operator_catalog.md` — the exact `aiter.ops.*` entry point + signature per operator family.
- `authoring_delegation.md` — the decision: tune the DB (default) vs author a kernel; routes to the language folders.

## `skills/` — problem-triggered (LAYER 2)
- `profile/profiling-aiter.md` — profile a real workload; prove engagement before believing a delta; pick the Amdahl target.
- `bottleneck/debug-aiter.md` — diagnose 0-engagement, build/JIT failures, ABI mismatch, variant/parity traps.
- `optimize/aiter_levers/` — domain-specific optimize levers:
  - `aiter_moe_pipeline.md` — fused MoE end to end: what fuses, the `tuned_fmoe` key, quant routing, shared-expert fusion.
  - `aiter_attention_entries.md` — which attention entry maps to which kernel, and the per-generation differences (`flash_attn_func` / paged decode / `mla_decode_fwd` / v4 sparse).
  - `aiter_flydsl_libtype.md` — the one libtype that can be selected and still not run: the three gates, and the A4W4 → CK fallback.

## Kernel-source authoring (delegated) & shared facts
Editing kernel source ≠ tuning the DB. To author/replace a kernel, open the language folder by backend:
CK → `languages/ck/` · HIP/C++ → `languages/hip/` · Triton → `languages/triton/` ·
**Gluon → `languages/gluon/`** · FlyDSL → `languages/flydsl/`. opus split-K is aiter-internal — read
`aiter/ops/opus/*` and `csrc/opus_gemm/*` directly, and tune it via
`gemm_a16w16_tune.py --libtype opus` (see `overall/tuning_db.md`).

> **A path under `ops/triton/` does not mean the kernel is Triton.** aiter ships Gluon kernels there —
> `ops/triton/attention/pa_mqa_logits.py` holds a Gluon kernel and a `@triton.jit` fallback behind one
> public entry, selected at dispatch, and the Gluon path is the more capable one (it supports
> `Preshuffle` and `KVBlockSize > 1`, which the Triton path does not). Read the source before picking a
> language folder. Note also that a campaign inferred onto `aiter` gets these framework cards
> but **no language layer at all**, so pass `--kernel-backend gluon` (or `triton`) explicitly when
> the work is kernel authoring rather than DB tuning.

Backend-neutral hardware constants live in `local_knowledge/hardware/`.
