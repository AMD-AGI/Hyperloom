---
title: aiter knowledge map — index, file roles & problem-routing
kind: index
scope: framework/aiter
updated: 2026-07-14
pinned_source: ROCm/aiter@b467ce3425cceeafe4f5587212d36df46feeb265 (v0.1.16-283)
---

# aiter — knowledge map

This file is the entry index for everything under `framework/aiter/`. It gives (1) what
aiter is + the source version these docs are grounded on, (2) the **reading order**, (3) for a given
task/problem **which files to read and in what order**, and (4) the role of every file and folder.

> **Convention (KernelForge standard):** a knowledge folder that contains an `INDEX.md` is navigated
> **through this file** — load it whole. Folders without an `INDEX.md` fall back to a generated
> "filename — one-line description" listing.

## Reading order (three layers)
1. **`overall/`** — universal basics that apply to *every* aiter operator (repo structure, dispatch,
   DB tuning, config system, build/JIT, API catalog, tune-vs-author decision). **Read this first.**
2. **`operators/<op>/`** — the specific operator you're working on (overview → aiter SOTA card → numerics/…).
3. **`skills/`** — pick one *when you hit a problem*: `profile/` (measure & target), `bottleneck/`
   (diagnose), `optimize/aiter_levers/` (domain-specific optimize levers: MoE / attention·MLA / FlyDSL).

## What aiter is
aiter (`ROCm/aiter`) is AMD's unified **operator library + per-shape dispatcher** for LLM inference. Its
kernels are **CK / ASM / HIP / Triton / FlyDSL / opus split-K / hipBLASLt** under the hood. This folder
documents the **library control plane** — which op to call, how it builds/dispatches, how to tune its
per-shape DB — and **delegates kernel-source authoring** to `languages/{hip,triton,gluon,flydsl,ck,asm}/`.

- **Pinned source**: `ROCm/aiter@b467ce342` (v0.1.16-283) — the commit every card is grounded on; re-pin per install.
- **Live-path integration (the rebind seam)**: SGLang `SGLANG_USE_AITER=1` (dense → `aiter.tuned_gemm:gemm_a16w16` / `tgemm.mm`); vLLM `vllm/_aiter_ops.py` registers aiter kernels as `torch.ops` custom ops gated by `VLLM_ROCM_USE_AITER*`.
- **Golden rule**: read `overall/` → find the Amdahl-dominant op (profile) → read that operator card → **tune the per-shape DB** → author a kernel only if tuning plateaus. Always prove engagement (`AITER_LOG_TUNED_CONFIG=1` → `is tuned on cu_num`) before trusting any delta.

## Start here — problem → files → order
| Task / symptom | Read in this order |
|---|---|
| Onboarding / "understand aiter" | `overall/repo_layout.md` → `overall/dispatch_and_rebind.md` → `overall/tuning_db.md` |
| "Make operator X faster" (don't know where to start) | `overall/` (basics) → `skills/profile/profiling-aiter.md` (find the Amdahl op) → `operators/<X>/overview.md` + `aiter.md` → `overall/tuning_db.md` |
| "Which `aiter.ops.*` API do I call for X?" | `overall/operator_catalog.md` → `operators/<X>/aiter.md` |
| "Tune the per-shape DB (GEMM / MoE)" | `overall/tuning_db.md` → `overall/configs_db.md` → (MoE) `skills/optimize/aiter_levers/fmoe.md` |
| "Deployed a tuned CSV but it does nothing" (0-engagement) | `skills/bottleneck/debug-aiter.md` (§2) → `overall/dispatch_and_rebind.md` → `overall/configs_db.md` |
| "Wrong results / crash / won't build / edit didn't take effect" | `skills/bottleneck/debug-aiter.md` → `overall/jit_and_build.md` |
| "Where does kernel Y live? repo layout?" | `overall/repo_layout.md` |
| "DB tuning plateaued — should I write a kernel?" | `overall/authoring_delegation.md` → the matching `languages/<lang>/` folder |
| Domain deep-dive | MoE → `skills/optimize/aiter_levers/fmoe.md` · attention/MLA → `.../attn_mla.md` · FlyDSL → `.../flydsl_path.md` · opus → `operators/dense_gemm/backends/opus.md` |
| Numerics / parity gate for X | `operators/<X>/numerics.md` |

## Folder structure & file roles
```
framework/aiter/
├── INDEX.md                          ← this map (load first)
├── overall/                          ← LAYER 1: universal basics (read first; applies to every operator)
│   ├── repo_layout.md        # repo structure & source distribution + the dispatcher/build model
│   ├── dispatch_and_rebind.md# how a call resolves (solMap libtype routing) + engages sglang/vLLM
│   ├── tuning_db.md          # per-shape DB tuning (capture→tune→deploy) — the primary optimization lever
│   ├── configs_db.md         # tuned/untuned CSV schema, AITER_CONFIG_* env, AITER_CONFIGS merge semantics
│   ├── jit_and_build.md      # build/JIT system, @compile_ops cache, hsa/codegen.py, optCompilerConfig.json
│   ├── operator_catalog.md   # which aiter.ops.* entry point + signature per operator family
│   └── authoring_delegation.md # decision: tune the DB (default) vs author a kernel (→ languages/*)
├── operators/<op>/                   ← LAYER 2: per-operator knowledge (34 operators; catalog below)
│   ├── overview.md   # what/why, math contract, shape regimes, Amdahl weight, backend landscape, how-to-bench
│   ├── aiter.md      # SOTA card: live dispatch, config knobs, measured perf, integration seam, pitfalls
│   ├── fusion.md     # fusion neighbors (GEMM epilogues, fused attention-entry kernels, norm+quant)
│   ├── numerics.md   # dtype/accumulate contract, parity bands, accuracy gating
│   ├── tuning.md     # per-backend knob space + the tune recipe
│   └── backends/<backend>.md   # (where present) a specific backend, e.g. dense_gemm/backends/opus.md
├── skills/                           ← LAYER 3: pick one when you hit a problem
│   ├── profile/profiling-aiter.md    # profile a real workload; prove engagement; pick the Amdahl target
│   ├── bottleneck/debug-aiter.md     # diagnose: 0-engagement, build/JIT, ABI, variant/parity traps
│   └── optimize/aiter_levers/        # DOMAIN-specific optimize levers: fmoe.md · attn_mla.md · flydsl_path.md
└── (kernel-source authoring is NOT here — see ../languages/{hip,triton,gluon,flydsl,ck,asm}/)
```
The 24 established operators carry the full 5-file set (`overview`/`aiter`/`fusion`/`numerics`/`tuning`);
the 10 newer operators (marked ⁺ below) currently ship a single comprehensive `overview.md`.

## Operator catalog (→ `operators/<op>/`)
- **GEMM / linear**: `dense_gemm` (Amdahl head; 10-tuple `gfx` DB) · `batched_gemm` (dense-flatten + dedicated CK strided) · `skinny_gemv_decode` (decode M=1..8) · `scaled_quant_gemm` (fp8/fp4) · `gemm_a8w4_mxfp8`⁺
- **MoE**: `fused_moe_grouped_gemm` (the fused mega-kernel) · `grouped_gemm_moe` · `moe_routing_topk` (biased/grouped gate) · `shared_expert_fusion` · `moe_dispatch_combine` (EP all-to-all dispatch/combine — MoRI-EP seam) · `moe_sorting`⁺ · `moe_align_block_size`⁺ · `moe_stage2_a8w4_opus`⁺
- **Attention / MLA**: `attention_prefill_fmha` · `attention_decode_paged` · `gqa_mqa_attention` · `mla_attention` (+v4 / context-parallel) · `kv_cache_quant` · `fmha_sink_mxfp8`⁺ · `sparse_attention_mla`⁺ (DSV4) · `paged_mqa_logits`⁺ (DSV4 FP8 indexer)
- **Norm / activation**: `rmsnorm` (HIP fast path + opus) · `fused_add_rmsnorm` · `layernorm` · `softmax` · `act_and_mul_silu_gelu`
- **Position**: `rope` · `mrope`
- **Quantization**: `quant_dequant_fp8` · `quant_fp4_mxfp` (MXFP4 / MXFP8, e8m0 block scale)
- **Sampling**: `sampling_topk_topp` (dual-pivot rejection)
- **Collectives / TP**: `collectives_all_reduce`⁺ (RCCL-bypass AR + fused residual/RMSNorm/quant)
- **SSM / linear-attn**: `causal_conv1d`⁺ (Mamba conv + gated-delta neighbor)
- **Utility**: `weight_shuffle`⁺ (MFMA-native / bpreshuffle layout — prereq for tuned GEMM/MoE kernels)

(⁺ = newer operator, single `overview.md` so far.)

## `overall/` — universal basics (LAYER 1, read first)
- `repo_layout.md` — source-tree map, dispatcher model, build model (where everything lives).
- `dispatch_and_rebind.md` — `solMap` libtype routing (`hipblaslt/asm/skinny/triton/flydsl/opus/torch`) + SGLang/vLLM engagement gates.
- `tuning_db.md` — **primary lever**: capture→tune→deploy the per-shape DB; 10-tuple `gfx`-first key; multi-backend tuner `csrc/gemm_a16w16/gemm_a16w16_tune.py`.
- `configs_db.md` — tuned/untuned CSV schemas, `AITER_CONFIG_*` env, `AITER_CONFIGS` resolve + `:`-merge + `model_configs/` overlay.
- `jit_and_build.md` — build/JIT env (`GPU_ARCHS`/`ENABLE_CK`/`AITER_REBUILD`), `@compile_ops` cache, `hsa/codegen.py`, `optCompilerConfig.json`.
- `operator_catalog.md` — the exact `aiter.ops.*` entry point + signature per operator family.
- `authoring_delegation.md` — the decision: tune the DB (default) vs author a kernel; routes to the language folders.

## `skills/` — problem-triggered (LAYER 3)
- `profile/profiling-aiter.md` — profile a real workload; prove engagement before believing a delta; pick the Amdahl target.
- `bottleneck/debug-aiter.md` — diagnose 0-engagement, build/JIT failures, ABI mismatch, variant/parity traps.
- `optimize/aiter_levers/` — domain-specific optimize levers:
  - `fmoe.md` — fused-MoE (`tuned_fmoe` DB, quant routing, shared-expert fusion, grouped-fmoe/gfx1250).
  - `attn_mla.md` — attention & MLA (`flash_attn_func` / paged decode / `mla_decode_fwd` + v4).
  - `flydsl_path.md` — the FlyDSL backend path (split-K HGEMM / A4W4 MoE, with CK fallback).

## Kernel-source authoring (delegated) & shared facts
Editing kernel source ≠ tuning the DB. To author/replace a kernel, open the language folder by backend:
CK → `languages/ck/` · ASM → `languages/asm/` · HIP/C++ → `languages/hip/` · Triton → `languages/triton/` ·
**Gluon → `languages/gluon/`** · FlyDSL → `languages/flydsl/`. opus split-K is aiter-internal
(`operators/dense_gemm/backends/opus.md`).

> **A path under `ops/triton/` does not mean the kernel is Triton.** aiter ships Gluon kernels there —
> `ops/triton/attention/pa_mqa_logits.py` holds a Gluon kernel and a `@triton.jit` fallback behind one
> public entry, selected at dispatch, and the Gluon path is the more capable one (it supports
> `Preshuffle` and `KVBlockSize > 1`, which the Triton path does not). Read the source before picking a
> language folder. Note also that a campaign inferred onto `aiter-fellow` gets these framework cards
> but **no language layer at all**, so pass `--fellow gluon-fellow` (or `triton-fellow`) explicitly when
> the work is kernel authoring rather than DB tuning.
Backend-neutral hardware constants live in `local_knowledge/hardware/`. AMD primary refs (Matrix-Core
CDNA3/CDNA4 programming; MI300X workload optimization) are cited in the individual operator cards.
