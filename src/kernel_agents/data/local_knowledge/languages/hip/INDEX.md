---
title: HIP / C++ knowledge map — index, file roles, problem-routing & pinned sources
kind: index
scope: languages/hip
updated: 2026-07-14
---

# HIP / C++ — knowledge map

This file is the entry index for everything under `languages/hip/`. It gives (1) what
this knowledge base is and when to reach for HIP at all, (2) for a given task/symptom, **which files to
read and in what order**, (3) the role of every file and folder, and (4) the **pinned reference sources**
the cards cite.

> **Convention (KernelForge standard):** a knowledge folder that contains an `INDEX.md` is navigated
> **through this file** — load it whole. Folders without an `INDEX.md` fall back to a generated
> "filename — one-line description" listing.

## What HIP knowledge is (and when to use it)
HIP/C++ is the **lowest-level portable** way to author CDNA kernels: full control of LDS, registers,
wave/cross-lane ops, MFMA intrinsics, and instruction scheduling, compiled by `hipcc`/`amdclang++`. This
folder documents **how to author and debug hand-written HIP kernels** — the language surface, the
authoring levers (MFMA/LDS/scheduling), the profile/debug playbooks, and per-operator HIP cards.

- **When to reach for HIP:** only when Triton / Composable-Kernel (ck_tile) / rocWMMA / HipKittens /
  FlyDSL **cannot express** the fusion, or you must **own the exact ISA**. Those higher levels already
  encode the tied-accumulator + software-pipeline + double-buffer patterns correctly and avoid the
  known AGPR-spill trap (LLVM #131954). Raw HIP is the escape hatch, not the default.
- **Backend-neutral hardware constants are NOT here.** CU/VGPR/LDS/peak numbers are single-sourced in
  `local_knowledge/hardware/` (`cdna3_mi300/`, `cdna4_mi350/`, `shared/`); HIP cards reference them.

## The two facts behind almost every HIP bug (internalize first)
1. **Wavefront = 64 lanes, not 32.** `warpSize == 64`; every `__shfl`/`__ballot`/manual reduction, mask
   (`unsigned long long` + `__popcll`), grid/occupancy calc, and block-size (multiple of 64) traces here.
   32-lane CUDA code *runs* but uses **half** the machine.
2. **LDS = 64 KB/CU (CDNA3), 160 KB/CU (CDNA4).** H100 habits (228 KB) overflow → launch failure or
   occupancy 1.

Other load-bearing rules: **grid ≥ 1024 workgroups** to fill the device; **keep MFMA accumulators in a
stable `vector_size` variable** so they stay in AGPRs (no `v_accvgpr_*` in the K-loop); **fp8 is FNUZ on
gfx942, OCP on gfx950**; always **verify the inner loop in the ISA** (`--save-temps`) — a "win" that
doesn't change the ISA as expected is usually noise.

## Start here — problem → files → order
| Task / symptom | Read in this order |
|---|---|
| "Orient me on authoring HIP kernels" | `skills/optimize/hip_levers/overview.md` → `API_docs/kernel_language.md` |
| "Write / tune an MFMA (matrix-core) GEMM" | `skills/optimize/hip_levers/overview.md` → `intrinsics.md` → `lds_async.md` → `patterns.md` |
| "LDS bank conflicts / double-buffer / direct-to-LDS / async copy" | `skills/optimize/hip_levers/lds_async.md` → `intrinsics.md` (§2 buffer, §4 sched) |
| "Wave reductions / grid-stride / streams / graphs / tiled-GEMM skeleton" | `skills/optimize/hip_levers/patterns.md` |
| "CUDA→HIP port went wrong / slow / static-assert on mask" | `skills/optimize/hip_levers/pitfalls.md` → `skills/bottleneck/debug-hip-kernel.md` |
| "Kernel is wrong / crashes / hangs / underperforms" | `skills/bottleneck/debug-hip-kernel.md` (symptom table → §) |
| "What should I optimize next? (read the profiler)" | `skills/profile/profiling-hip.md` → the lever it points to |
| "Tile-abstraction alternative to raw asm / SOTA perf reference" | `skills/optimize/hip_levers/hipkittens.md` |
| "API: qualifiers, launch, `__shfl`/`__ballot`, atomics, cooperative groups" | `API_docs/kernel_language.md` |
| "Host API: memory, streams, events, HIP graphs, error handling" | `API_docs/runtime_api.md` |
| "Compile flags / CMake / PyTorch extension / arch guards / JIT cache" | `API_docs/compilation_and_build.md` |
| "Low-precision dtype headers (fp8/fp6/fp4/bf16) + MFMA vector types" | `API_docs/dtypes.md` |
| "fp8 gives wrong results (~2× off) on gfx942" | `API_docs/dtypes.md` → `skills/optimize/hip_levers/pitfalls.md` → `skills/bottleneck/debug-hip-kernel.md` (§6) |
| "Author / optimize operator X in HIP" | `operators/<X>/overview.md` → `operators/<X>/hip.md` → `fusion.md` / `numerics.md` / `tuning.md` |
| "Hardware constants (CU / VGPR / LDS / peak / roofline)" | `local_knowledge/hardware/` (single source of truth) |

## Folder structure & file roles
```
languages/hip/
├── INDEX.md                              ← this map (load first; includes pinned sources)
├── API_docs/                             ← the language/runtime API standard ("what the calls are")
│   ├── kernel_language.md                # device surface: qualifiers, launch, indexing, sync, wave64 cross-lane, atomics
│   ├── runtime_api.md                    # host surface: memory, streams, events, HIP graphs, occupancy query, errors
│   ├── compilation_and_build.md          # hipcc/amdclang++ flags, arch guards, CMake/PyTorch-ext, JIT cache gotchas
│   └── dtypes.md                         # fp8/fp6/fp4/bf16/fp16 headers, MXFP E8M0, MFMA operand vector types
├── skills/                               ← task playbooks (the entry points)
│   ├── profile/profiling-hip.md          # read rocprofv3 PMC → classify memory/compute/spill → point to a lever
│   ├── bottleneck/debug-hip-kernel.md    # wrong/crash/slow: symptom→cause table, wave64, LDS, AGPR, fp8, ISA check, hang recovery
│   └── optimize/hip_levers/              ← the "how to optimize" levers (indexed below)
│       ├── overview.md                   # programming model, MI300X constants, toolchain, HIP-vs-Triton/FlyDSL capability map
│       ├── intrinsics.md                 # __builtin_amdgcn_mfma_*, buffer descriptors, cross-lane, sched_group_barrier
│       ├── lds_async.md                  # LDS banks/padding/swizzle, direct-to-LDS, barriers/wait-counters, double-buffer
│       ├── patterns.md                   # wave reductions, grid-stride, cooperative groups, streams/graphs, tiled-GEMM + MFMA µkernel
│       ├── pitfalls.md                   # CUDA→HIP traps, occupancy prediction, ISA-verification checklist, when NOT to write HIP
│       └── hipkittens.md                 # HipKittens tile primitives / wave scheduling / register pinning (SOTA perf reference)
└── operators/<op>/                       ← per-operator HIP cards (13 operators; catalog below)
    ├── overview.md   # what/why, math contract, shape regimes, Amdahl weight, cross-backend landscape, how-to-bench
    ├── hip.md        # the HIP SOTA card: editable kernel source pointers, knobs, the HIP-specific implementation
    ├── fusion.md     # fusion neighbors (residual-add, norm+quant, GEMM epilogue, attention-entry mega-fusions)
    ├── numerics.md   # dtype/accumulate contract, parity bands, accuracy gating
    └── tuning.md     # tile/occupancy/scheduling knob space + the tune recipe
```
All 13 operators carry the full 5-file set (`overview`/`hip`/`fusion`/`numerics`/`tuning`).

## Operator catalog (→ `operators/<op>/`)
- **GEMM / linear**: `batched_gemm` (per-head/per-expert; fill all CUs across the batch) · `splitk_streamk_gemm` (K/tile decomposition — only helps when CU-underutilized) · `scaled_quant_gemm` (fp8/fp6/fp4 block-scaled MFMA; FP6 at FP4 rate) · `grouped_gemm_moe` (variable-M per-expert GEMMs in one fused launch)
- **Attention**: `attention_decode_paged` (paged-KV flash-decoding, splitKV, bandwidth-bound) · `mla_attention` (DeepSeek MLA, weight-absorbed MQA on the latent)
- **Norm**: `rmsnorm` (bandwidth-bound row reduce; fusion anchor) · `fused_add_rmsnorm` (the dominant serving form — residual-add fused) · `layernorm` (two-statistic μ/σ²; encoder/ViT)
- **Reduction / activation**: `softmax` (max-subtracted row reduce; usually fused into attention) · `reduction` (wave64 shuffle → LDS → split/atomic; the primitive inside norm/softmax)
- **Position**: `rope` (rotary embedding; must be fused, not naive)
- **Quantization**: `quant_dequant_fp8` (FNUZ vs OCP correctness gate; the win is fusing it away)

## Reading-depth guide (how much to load)
- **Just orienting / a single API fact**: `hip_levers/overview.md` or the relevant `API_docs/*` file —
  don't load the whole levers folder.
- **Authoring a matrix-core kernel**: the levers chain `overview → intrinsics → lds_async → patterns`
  is the core loop; add `pitfalls.md` before you trust a result.
- **Diagnosing a specific failure**: go straight to `skills/bottleneck/debug-hip-kernel.md` and follow
  its symptom→section table; use `skills/profile/profiling-hip.md` when the question is "what next?".
- **Per-operator work**: `operators/<op>/overview.md` first (cross-backend context + whether HIP is even
  the right backend), then `hip.md` for the HIP implementation, then `fusion`/`numerics`/`tuning`.
- **Hardware numbers**: always defer to `local_knowledge/hardware/` — this folder never duplicates them.

## Pinned reference sources
Single place for the `repo@commit` / canonical-URL pins the `hip/` cards cite (cards also cite inline).

**Primary language / runtime**
- **ROCm/HIP** — https://github.com/ROCm/HIP — HIP C++ runtime API + kernel-language definition
  (`__global__`, `warpSize`, `__launch_bounds__`, intrinsics). Docs: rocm.docs.amd.com/projects/HIP.
- ROCm/rocm-examples — https://github.com/ROCm/rocm-examples — official HIP examples/tutorials (educational, not SOTA perf).
- ROCm/hip-tests — https://github.com/ROCm/hip-tests — HIP conformance tests (API-behavior reference).

**SOTA reference kernels**
- **HazyResearch/HipKittens** — https://github.com/HazyResearch/HipKittens — C++ tile framework + SOTA hand-written HIP kernels; now an official AITER backend (ROCm/aiter PR #2039); CDNA3/CDNA4 branches differ. Research artifact — pin a commit, re-measure per shape. Paper: arXiv 2511.08083.
- **ROCm/aiter** `csrc/` — https://github.com/ROCm/aiter — production HIP/C++ kernels (the live optimization objects).
- ROCm/rocWMMA — https://github.com/ROCm/rocWMMA — C++ WMMA/MFMA wrappers over the matrix core.
- ROCm/composable_kernel (now ROCm/rocm-libraries `projects/composablekernel`) — https://github.com/ROCm/rocm-libraries — templated CDNA GEMM/attention (ck_tile); tied-accumulator + sched-group-barrier pipelines.

**AMD primary docs (canonical)**
- HIP kernel language (warpSize, __launch_bounds__, 64-bit masks): https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- HIP programming model (wave64, SIMD, block sizing): https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- HIP hardware implementation (LDS banks, 64 KB/CU, occupancy): https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- MI300X workload optimization (304 CUs, VGPR/LDS, ≥1024 grid): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Matrix Core programming CDNA3/CDNA4 (MFMA layouts, cbsz/abid/blgp, f8f6f4): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- AMD Matrix Instruction Calculator (exact lane→element maps): https://github.com/ROCm/amd_matrix_instruction_calculator
- AMDGPU backend (buffer descriptors, ds builtins, sched builtins, s_waitcnt): https://llvm.org/docs/AMDGPUUsage.html
- CDNA3 ISA — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- CDNA4 ISA — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- CDNA4 whitepaper (LDS 160 KB, MXFP, 256 B/clk) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf

## Cross-links out of this folder
Hardware constants: `local_knowledge/hardware/` (single source of truth). Higher-level / alternative
authoring paths and their operator cards live under `languages/{triton,gluon,flydsl,ck,asm}/` and
`framework/aiter/`; a HIP operator's `overview.md` links the cross-backend SOTA cards. Benchmark
discipline (warmup, median-of-≥3, in-context A/B) lives in `local_knowledge/common_methodology/`.
