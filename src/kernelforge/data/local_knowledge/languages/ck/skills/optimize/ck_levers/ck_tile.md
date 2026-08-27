---
title: CK-Tile — the tile-programming front-end of Composable Kernel
kind: language
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2, mxfp4]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
  - https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/README.md
  - https://github.com/ROCm/composable_kernel/issues/1727
---

# CK-Tile

## TL;DR
CK-Tile (`include/ck_tile`) is Composable Kernel's **CUTLASS/CuTe-like tile-programming** layer. It keeps
CK's compile-time coordinate-transform engine underneath but exposes *tiles / windows / distributions*
instead of raw nested `constexpr` descriptors. It is the path AMD now uses for **new** LLM kernels:
FMHA (paged-KV prefill+decode), fused-MoE, fp8/mxfp4 GEMM. For attention it is the production SOTA on
Instinct (the backend behind flash-attention ROCm and selectable in vLLM/sglang). For *dense square*
bf16 GEMM the older classic-CK `DeviceGemmXdlUniversal` v3 can still be ~1.7× faster (Issue #1727) — so
benchmark before shipping ck_tile as the dense path. See [ck_classic.md](ck_classic.md) for the classic
model and [knobs.md](knobs.md)/[pitfalls.md](pitfalls.md) for tuning.

> **Repo move (pin this):** standalone `ROCm/composable_kernel` is **DEPRECATED** — development moved into
> the monorepo `ROCm/rocm-libraries` under `projects/composablekernel/`. The old `develop` branch is a
> read-only mirror. Paths below are relative to the CK source root (`include/ck_tile/...`,
> `example/ck_tile/...`), identical in both layouts.

## Core concepts — the five abstractions
Each component is one header (`#include "ck_tile/core.hpp"`, `"ck_tile/ops/gemm.hpp"`,
`"ck_tile/ops/fmha.hpp"`).

| Abstraction | header | role |
|---|---|---|
| **TensorView** | `core/tensor/tensor_view.hpp` | strided, optionally padded N-D view over a raw pointer (global / LDS / VGPR) |
| **TileDistribution** | `core/tensor/tile_distribution.hpp` | the **thread↔element map**: which lane/wave owns which tile coordinate |
| **TileWindow** | `core/tensor/tile_window.hpp` | a *moving* sub-view + distribution → the load/store gateway (coalescing, vectorization, OOB guard) |
| **DistributedTensor** | `core/tensor/...` | the in-register result of `load_tile()`: per-lane storage + cooperation pattern |
| **Pipeline / Policy / Epilogue** | `ops/gemm/`, `ops/fmha/` | the K-loop mainloop schedule, its layout policy, and the writeback |

The golden rule (from the docs): the tile APIs (`make_naive_tensor_view`, `make_tile_window`) only
**declare** memory addresses; the real loading/writing happens inside the **pipeline** and **epilogue**.
A window is a cursor, not a copy.

**TileDistribution** is the single most important (and most cryptic) object: a
`tile_distribution_encoding` declaring, in compile-time `sequence`/`tuple` form, how the wavefront's 64
lanes and the block's waves tile a region and how many elements each lane holds. The `<Repeat, Warp,
Lane, Vector>` (e.g. `<4,2,8,4>`) pattern maps the tile onto MFMA lanes. You almost never write it by
hand — a **Policy** generates it from the tile sizes and the chosen WarpGemm.

Tile-level verbs on distributed tensors: `load_tile`, `store_tile`, `update_tile`, `async_load_tile`
(direct global→LDS via `buffer_load`), `shuffle_tile` (re-distribute across lanes, e.g. transpose),
`slice_tile`, `sweep_tile` (iterate the per-lane Y elements with a lambda), and `block_tile_reduce`
(the FMHA row-max/row-sum primitive).

## The levers
A GEMM kernel is assembled from **four** template pieces (this composition is the heart of CK-Tile):

```
GemmKernel< TilePartitioner, GemmPipeline, EpiloguePipeline >
              │                  │              │
              │                  │              └─ writeback (+ CShuffle, + fused elementwise)
              │                  └─ K-loop mainloop schedule (vN)
              └─ (M,N,K) → grid; gridDim = ceil(M/kM) × ceil(N/kN)
```

- **TilePartitioner** — block tile `kM×kN×kK`; size so `ceil(M/kM)·ceil(N/kN) ≈ k·304` to fill MI300X's
  304 CUs (e.g. M=4864,N=4096 with a 256×256 block → 19×16 = **304** blocks = one per CU).
- **Pipeline name encodes dataflow:** `GemmPipelineAgBgCrCompV3` = **A** from **g**lobal, **B** from
  **g**lobal, **C** in **r**egisters, **Comp**ute-optimized, **V3**.
  - `GemmPipelineAGmemBGmemCRegV1` — single buffer, low VGPR; teaching / memory-bound.
  - `GemmPipelineAgBgCrCompV3` — double-buffer LDS, 2-stage prefetch; **the compute-bound workhorse**.
  - `GemmPipelineAgBgCrMemV3` / `...CompV4` — memory-opt / deeper prefetch for very large / fp8-dense K.
  - `*_async` persistent — `buffer_load` direct-to-LDS (DGL), no VGPR staging; latest LLM GEMM/MoE.
- **Policy** (e.g. `UniversalGemmPipelineAgBgCrPolicy`) is the layout brain: `MakeADramTileDistribution`/
  `MakeBDramTileDistribution` (global-load distributions + per-lane vector width), `MakeALdsBlockDescriptor`
  (LDS layout with an **XOR swizzle** to kill bank conflicts), and `GetWarpGemm()` (selects the MFMA).
- **WarpGemm** is where CK-Tile meets the matrix core. The `operator()` wraps the intrinsic directly,
  e.g. for fp16 32×32×8: `c = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);` (see
  ../asm_mfma/mfma_intrinsics.md).
- **Epilogue (CShuffle):** the MFMA C accumulator is in a scattered per-lane layout not coalescable for
  the global store. CShuffle re-tiles C through LDS (`shuffle_tile`/`store_tile`), optionally fuses a
  bias/act/residual, then writes. Knobs: `CShuffleDataType`, store vector width (8 for bf16), shuffle
  granularity (`MXdlPerWavePerShuffle`).

Build & run (gfx942):
```bash
sh ../script/cmake-ck-dev.sh ../ gfx942
make tile_example_gemm_basic -j && ./bin/tile_example_gemm_basic -m=4096 -n=4096 -k=4096 -v=1
make tile_example_universal_gemm -j && ./bin/tile_example_universal_gemm -m=4096 -n=4096 -k=4096 -v=0
```

For the FMHA mapping and GEMM template details see
[fmha_template.md](fmha_template.md), [gemm_template.md](gemm_template.md),
[codegen_instances.md](codegen_instances.md).

## Pitfalls
- **Dense square GEMM gap:** Issue #1727 — at 4096³ bf16, ck_tile `universal_gemm` ran ~0.382 ms /
  359 TFLOP/s vs classic-CK `DeviceGemmXdlUniversal` v3 at ~0.223 ms / 615 TFLOP/s, *same* 256×256×64
  tile. CK-Tile's edge today is fusion + attention/MoE, not raw square GEMM. Always benchmark vs classic
  v3 before shipping a dense path.
- Repo deprecation: pin `ROCm/rocm-libraries:projects/composablekernel` (or the read-only
  `ROCm/composable_kernel@develop` mirror) — never assume the standalone repo is current.
- `ckProfiler` (the classic-CK sweeper) is absent in some images → no CK instance sweeps there; CK-Tile
  examples each have their own bench harness instead.
- Hand-writing a `tile_distribution_encoding` by hand is error-prone — let the Policy generate it.

## Verify
- Build the matching example (`tile_example_universal_gemm`, `tile_example_fmha_fwd`) and bench at your
  exact shapes; compare against classic-CK v3 (GEMM) or the Triton FMHA backend (attention).
- Disassemble the hot loop and confirm `buffer_load` width, `s_waitcnt lgkmcnt(1)` before `v_mfma`, and
  no `scratch_`/`v_accvgpr` spam (see ../asm_mfma/pitfalls.md).
- Numerics: fp32 accumulate; greedy temp=0 parity vs a reference at ≥10 prompts for attention.

## Sources
- Hands-On with CK-Tile: build & run optimized GEMM on AMD GPUs (ROCm Blog, Apr 2025 — WarpGemm struct, pipeline/policy, gfx942 build): https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
- From Theory to Kernel: FlashAttention-v2 with CK-Tile (ROCm Blog — fmha pipeline mapping): https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- ck_tile component layout (core/host/ops/gemm/ops/fmha): https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/README.md
- ck_tile Tile Window / Tensor Views / Sweep Tile concept docs: https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/tile_window.html ; https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/tensor_views.html ; https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/sweep_tile.html
- A Block GEMM on MI300 (LDS sizing, pipeline stages, 256×256 / 304 CU): https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
- Issue #1727 — ck_tile universal_gemm vs classic CK v3 perf gap (615 vs 359 TFLOP/s): https://github.com/ROCm/composable_kernel/issues/1727
- Repo deprecation/move to ROCm/rocm-libraries: https://github.com/ROCm/composable_kernel (README banner)
