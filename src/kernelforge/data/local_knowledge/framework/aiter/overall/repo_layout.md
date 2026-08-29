---
title: aiter repo layout — source-tree map, subsystem overview & how to navigate
kind: language
gens: [gfx942, gfx950, gfx1250]
dtypes: [both]
regimes: [both]
status: sota
updated: 2026-07-14
sources:
  - https://github.com/ROCm/aiter
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
  - ROCm/aiter@b467ce3425cceeafe4f5587212d36df46feeb265
---

# aiter repo layout & subsystem overview

## TL;DR
aiter (`ROCm/aiter`, "AI Tensor Engine for ROCm") is AMD's **default kernel backend for LLM inference** —
roughly *cuBLAS + cuDNN + FlashAttention + TransformerEngine combined*: one library owning GEMM, attention
(MHA/MLA), MoE, norm, RoPE, quant, sampling, and RCCL-bypass collectives. Crucially it is a **dispatcher,
not a monolith**: for each op it picks the fastest of hipBLASLt / hand-tuned asm / skinny HIP / Triton /
FlyDSL / CK from a per-shape config DB. This doc is the **map of where things live in the repo and how the
pieces connect**; for the tuning lever see [tuning_db.md](tuning_db.md), for dispatch/framework wiring see
[dispatch_and_rebind.md](dispatch_and_rebind.md), for the op API see [operator_catalog.md](operator_catalog.md).

## Source-tree map (what lives where)
```
aiter/                         (ROCm/aiter@b467ce342 = v0.1.16-283)
├── aiter/                     # Python package
│   ├── tuned_gemm.py          # dense GEMM DISPATCHER (gemm_a16w16 / tgemm.mm, solMap, 10-tuple gfx-first key)
│   ├── fused_moe.py           # fused-MoE dispatch (tuned_fmoe DB, sorting backends)
│   ├── fused_moe_dp_shared_expert.py   # DP shared-expert MoE
│   ├── mla.py                 # MLA decode / prefill / v4 (package ROOT, not under ops/)
│   ├── paged_attn.py, rotary_embedding.py           # paged attention / RoPE high-level wrappers
│   ├── ops/                   # Python op wrappers (the API surface you import)
│   │   ├── gemm_op_a8w8.py, gemm_op_a4w4.py, gemm_op_a8w4.py, gemm_op_a16w16.py, gemm_op_common.py  # GEMM entries + get_padded_m
│   │   ├── attention.py, mha.py, mhc.py             # MHA / paged / MLA-asm / multi-head-compression
│   │   ├── moe_op.py, moe_sorting.py, moe_sorting_opus.py, topk.py   # MoE gate/sort + routing (topk_softmax lives here)
│   │   ├── norm.py, rmsnorm.py, rope.py, activation.py, quant.py, cache.py, sample.py, sampling.py, enum.py
│   │   ├── communication.py, custom_all_reduce.py, quick_all_reduce.py   # collectives (RCCL-bypass)
│   │   ├── causal_conv1d_*.py, chunk_gated_delta_rule_fwd_h.py, deepgemm.py   # SSM / linear-attn / DeepGEMM
│   │   ├── fused_qk_norm_rope_cache_quant.py, fused_qk_norm_mrope_cache_quant.py, gated_rmsnorm_fp8_group_quant.py
│   │   ├── shuffle.py         # weight / scale preshuffle helpers
│   │   ├── opus/              # opus split-K GEMM + MoE-stage2 adapters (module_deepgemm_opus / module_moe_opus)
│   │   ├── triton/            # Triton kernels (aiter.ops.triton.*)  -> languages/triton/
│   │   └── flydsl/            # FlyDSL kernels (aiter.ops.flydsl.*)  -> languages/flydsl/
│   ├── configs/               # 125 tuned/untuned CSVs + model_configs/ per-model overlays (auto-merged)
│   ├── jit/                   # JIT engine: core.py (@compile_ops, AITER_CONFIGS, optCompilerConfig.json), utils/torch_guard.py
│   ├── utility/               # shared tuning infra (base_tuner.py, mp_tuner.py, pretune.py)
│   └── dist/                  # distributed comms (parallel_state, device_communicators)
├── csrc/                      # C++/HIP/CK/ASM kernel sources (compiled by JIT)
│   ├── gemm_a16w16/           # PRIMARY multi-backend bf16 GEMM tuner (gemm_a16w16_tune.py: asm/opus/flydsl/triton/skinny/torch)
│   ├── opus_gemm/, opus_moe/  # opus split-K GEMM / MoE kernels + tuners (codegen gfx942/gfx950/gfx1250)
│   ├── ck_gemm_a8w8*/, ck_gemm_a4w4_blockscale/, ck_batched_gemm_*/, ck_gemm_moe_2stages_codegen/  # per-format CK tuners  -> languages/ck/
│   ├── py_itfs_cu/            # pybind glue (gemm_common.cu = getPaddedM; asm_mla.cu / asm_mla_v4.cu)
│   ├── cpp_itfs/              # CK-tile codegen interfaces (sampling, pa, ...)
│   ├── kernels/               # hand-written HIP (rmsnorm_quant_kernels.cu, quant_kernels.cu, ...)  -> languages/hip/
│   └── pybind/, include/      # torch bindings + headers (incl. opus/)
├── hsa/                       # PRECOMPILED ASM code-objects (.co HSACO) + CSV metadata
│   ├── gfx942/, gfx950/, gfx1250/   # per-arch .co: pa/ mla/ mla_v4/ fmoe_*.co flatmm_uk_*.co gemm_a8w8_*.co ...
│   ├── {op}/{op}_asm.csv      # CSV metadata mapping kernel params -> .co filename + function ptr
│   └── codegen.py             # CSV -> C++ dispatch-table codegen (python hsa/codegen.py -m {pa,fmha,mla})
├── gradlib/                   # LEGACY hipBLASLt-only GEMM tuner (gradlib/gemm_tuner.py, GemmTuner.py)
├── op_tests/                  # per-operator tests + op_tests/tuning_tests/ (tuner regressions)
└── 3rdparty/composable_kernel # CK submodule (ENABLE_CK=1)
```

## Kernel families on disk (hsa/ asm HSACO)
| Family | Location | Purpose |
|---|---|---|
| **PA** (paged attention) | `hsa/gfx{942,950,1250}/pa/` + `pa_*.co` | decode + prefill attention, paged KV |
| **MLA** | `hsa/gfx{942,950,1250}/mla/`, `mla_v4/` (v4/sparse on gfx950+gfx1250) | DeepSeek MLA / DSV4 — see [aiter_attention_entries.md](../skills/optimize/aiter_levers/aiter_attention_entries.md) |
| **FMOE** | `hsa/gfx{942,950,1250}/fmoe_*.co` | fused MoE (GEMM-A + act + GEMM-B) — see [aiter_moe_pipeline.md](../skills/optimize/aiter_levers/aiter_moe_pipeline.md) |
| **GEMM** | `hsa/gfx942/{bf16,f4,i8,fp8gemm_blockscale}/`, `flatmm_uk_*.co`, `gemm_a8w8_*.co` | tuned per-shape GEMMs |
| **TopK-softmax** | `hsa/gfx942/topksoftmax/` | pre-FMOE expert selection |
| **AllReduce** | `all_reduce.co`, `allreduce_{layernorm,rmsnorm}_*.co` | XGMI ring + fused post-attn norm |
Note: aiter does **not** check in `.s` source — it ships `.co` binaries + CSV metadata + a round-trip ISA
toolchain. To read/edit an asm kernel, disassemble the `.co` with `llvm-objdump`. gfx1250 (CDNA-next) has
its own `hsa/gfx1250/` tree (FMHA / MLA / MLA-v4 / f4gemm).

## The dispatcher model (how a call resolves)
1. `aiter.ops.<family>` Python wrapper is called (e.g. `tuned_gemm.gemm_a16w16`).
2. It builds a lookup key and consults the per-shape **config DB** (`aiter/configs/*.csv`, schema in
   [config_files_and_merge.md](config_files_and_merge.md)); the winning row names a `libtype` + `solidx`.
3. `solMap` routes `libtype` → executor: `hipblaslt` / `asm` / `skinny` (HIP) / `triton` / `flydsl` /
   `opus` (split-K) / `torch`. No match → arch-dependent default (`hipblaslt`/`asm` for bpreshuffle,
   `skinny` for small-M, else `torch`). Details in [dispatch_and_rebind.md](dispatch_and_rebind.md).
4. The chosen kernel is JIT/AOT-compiled on first use and cached (`aiter/jit/`), then run.

## Build / JIT model (one line; full detail in jit_and_build.md)
Most C++/HIP/CK/asm kernels compile on **first use** via `@compile_ops` into `~/.aiter/jit/` (or the AOT
`aot/` blobs); later calls hit the cached `.so`. `optCompilerConfig.json` is the per-module recipe; env
knobs (`GPU_ARCHS`, `ENABLE_CK`, `PREBUILD_KERNELS`, `AITER_REBUILD`) control it. See
[jit_and_build.md](jit_and_build.md).

## torch.compile survival (why aiter kernels stay opaque)
aiter wraps dispatchers (e.g. `gemm_a16w16`) with `@torch_compile_guard` (`aiter/jit/utils/torch_guard.py`),
registering the op into a `torch.library.Library` with a **fake/meta impl** — so Inductor traces around it
without decomposing it into generated Triton (the AMD analog of vLLM's `direct_register_custom_op`). This
is what preserves the hand-tuned kernel through `torch.compile`.

## How to use aiter (typical flow)
1. **Import the op** from `aiter.ops.*` — pick the right entry via [operator_catalog.md](operator_catalog.md).
2. **Preprocess** if needed (quant via `aiter.ops.quant`, weight preshuffle via `aiter.ops.shuffle`) —
   see [operator_catalog.md](operator_catalog.md) / the quant integration patterns.
3. **Engage on the serving path**: `SGLANG_USE_AITER=1` / `VLLM_ROCM_USE_AITER=1` — see
   [dispatch_and_rebind.md](dispatch_and_rebind.md).
4. **Optimize** = tune the per-shape DB (not the kernel source): [tuning_db.md](tuning_db.md). Author a new
   kernel only when no libtype covers the shape/fusion → [authoring_delegation.md](authoring_delegation.md).

## Per-subsystem deep dives
[config_files_and_merge.md](config_files_and_merge.md) (CSV schema) · [tuning_db.md](tuning_db.md) (capture→tune→deploy) ·
[aiter_moe_pipeline.md](../skills/optimize/aiter_levers/aiter_moe_pipeline.md) (fused MoE) · [aiter_attention_entries.md](../skills/optimize/aiter_levers/aiter_attention_entries.md) (MLA decode) ·
[aiter_flydsl_libtype.md](../skills/optimize/aiter_levers/aiter_flydsl_libtype.md) (aiter→FlyDSL dispatch) · [dispatch_and_rebind.md](dispatch_and_rebind.md) ·
[jit_and_build.md](jit_and_build.md) · [operator_catalog.md](operator_catalog.md) ·
[authoring_delegation.md](authoring_delegation.md).

## Sources
- Repo structure / op catalog / dispatcher / custom-op: `ROCm/aiter@b467ce342` (`aiter/tuned_gemm.py`,
  `aiter/fused_moe.py`, `aiter/mla.py`, `aiter/jit/core.py`, `aiter/jit/utils/torch_guard.py`,
  `aiter/configs/`, `csrc/gemm_a16w16/`, `csrc/opus_gemm/`, `hsa/{gfx942,gfx950,gfx1250}/`, `gradlib/`).
- aiter as the central engine / default backend: https://github.com/ROCm/aiter ;
  https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- Kernel families on disk (hsa/ HSACO): synthesized from the on-box `hsa/` tree.
