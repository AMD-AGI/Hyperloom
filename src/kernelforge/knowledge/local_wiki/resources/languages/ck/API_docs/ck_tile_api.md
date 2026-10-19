---
title: CK-Tile API — core headers, tile abstractions & kernel composition
kind: api_reference
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, mxfp4]
regimes: [both]
status: sota
updated: 2026-07-09
sources:
  - https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/README.md
  - https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/
---

# CK-Tile API

The tile-programming front-end interface (`include/ck_tile`). This is the API surface; for the perf model
(pipelines, policies, WarpGemm, LDS swizzle) see
[../skills/optimize/ck_levers/ck_frontend_tile.md](../skills/optimize/ck_levers/ck_frontend_tile.md), and for FMHA/GEMM
templates see [../skills/optimize/ck_levers/ck_fmha_stack.md](../skills/optimize/ck_levers/ck_fmha_stack.md) /
[gemm_template](../skills/optimize/ck_levers/ck_gemm_stack.md).

## Headers
```cpp
#include "ck_tile/core.hpp"        // TensorView, TileWindow, TileDistribution, DistributedTensor
#include "ck_tile/ops/gemm.hpp"    // GemmKernel / GemmPipeline* / epilogue
#include "ck_tile/ops/fmha.hpp"    // fmha_fwd / fmha_bwd pipelines
```

## The five core abstractions
| Type | Header | Role |
|---|---|---|
| `TensorView` | `core/tensor/tensor_view.hpp` | strided, optionally padded N-D view over a raw pointer (global/LDS/VGPR) |
| `TileDistribution` | `core/tensor/tile_distribution.hpp` | the thread↔element map (which lane/wave owns which coordinate) |
| `TileWindow` | `core/tensor/tile_window.hpp` | a *moving* sub-view + distribution — the load/store gateway (coalescing, OOB guard) |
| `DistributedTensor` | `core/tensor/...` | in-register result of `load_tile()` — per-lane storage |
| Pipeline / Policy / Epilogue | `ops/gemm/`, `ops/fmha/` | mainloop schedule, its layout policy, and the writeback |

Golden rule: `make_naive_tensor_view` / `make_tile_window` only **declare** addresses; the real
load/store happens inside the **pipeline**/**epilogue**. A window is a cursor, not a copy.

## Tile verbs (on distributed tensors)
```cpp
auto t   = load_tile(window);          // global/LDS → registers (DistributedTensor)
store_tile(window, t);                 // registers → global/LDS
update_tile(window, t);                // accumulate
async_load_tile(window);               // direct global→LDS (buffer_load), skip VGPR staging
auto s   = shuffle_tile(t, ...);       // re-distribute across lanes (e.g. transpose)
sweep_tile(t, [&](auto idx){ ... });   // iterate per-lane Y elements with a lambda
auto r   = block_tile_reduce(t, ...);  // block-wide reduce (FMHA row-max / row-sum)
```

## Kernel composition (GEMM)
```cpp
using Kernel = GemmKernel< TilePartitioner, GemmPipeline, EpiloguePipeline >;
//               TilePartitioner  → (M,N,K) → grid (aim ceil(M/kM)·ceil(N/kN) ≈ k·304 on MI300X)
//               GemmPipeline     → K-loop mainloop (e.g. GemmPipelineAgBgCrCompV3 = A/B from global, C in reg, ComputeV3)
//               EpiloguePipeline → writeback (+ CShuffle, + fused elementwise)
```
The `Policy` (e.g. `UniversalGemmPipelineAgBgCrPolicy`) generates the `TileDistribution`s and picks the
`WarpGemm` (the MFMA) — you rarely hand-write a `tile_distribution_encoding`.

## Build / run an example
```bash
sh ../script/cmake-ck-dev.sh ../ gfx942
ninja tile_example_universal_gemm && ./bin/tile_example_universal_gemm -m=4096 -n=4096 -k=4096 -v=1
ninja tile_example_fmha_fwd       && ./bin/tile_example_fmha_fwd -b=1 -h=8 -s=4096 -d=128 -v=1
```
`generate.py` instantiates per-trait `.cpp` files (prune traits to your shapes — see
[../skills/optimize/ck_levers/ck_instance_codegen.md](../skills/optimize/ck_levers/ck_instance_codegen.md)).

## Sources
- ck_tile component layout (core/ops/gemm/ops/fmha): https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/README.md
- ck_tile concept docs (tile_window / tensor_views / sweep_tile): https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/
