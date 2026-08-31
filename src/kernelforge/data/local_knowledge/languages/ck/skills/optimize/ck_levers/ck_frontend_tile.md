---
title: CK — the ck_tile front-end, and what it is and isn't good at
kind: language
lever: ck_frontend_tile
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
  - https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/README.md
  - https://github.com/ROCm/composable_kernel/issues/1727
---

# The ck_tile front-end

## Route here when
- The kernel is **attention** — FMHA, paged-KV prefill or decode. This is the production path on
  Instinct and there is no close second inside CK.
- The value of the kernel is **fusion**: fused MoE, fp8/mxfp4 GEMM, anything with a non-trivial epilogue.
- You are starting new CK work and want to be where upstream development actually happens.

**Do not route here for dense square bf16/fp16 GEMM.** Measured at the same 256×256×64 tile and the
same 4096³ shape, classic v3 turned in 615 TFLOP/s against ck_tile's 359 — roughly 1.7× (Issue #1727).
That gap is not folklore; benchmark before you ship ck_tile as a dense path. See
`ck_frontend_classic.md`.

> **Repo pin.** The standalone `ROCm/composable_kernel` repository is **deprecated**; the live source is
> `ROCm/rocm-libraries` under `projects/composablekernel/`, and `develop` survives only as a read-only
> mirror. Every path below is relative to the CK source root and is spelled the same either way.

## What you are actually getting
ck_tile puts a CUTLASS/CuTe-style tile-programming surface on top of the compile-time
coordinate-transform engine CK already had. The engine did not change. What changed is what you write:
tiles, windows and distributions, instead of hand-nesting `constexpr` descriptors.

## The five objects
| Object | Header | What it is |
|---|---|---|
| `TensorView` | `core/tensor/tensor_view.hpp` | an N-D strided (optionally padded) view over a raw pointer — global, LDS, or VGPR |
| `TileDistribution` | `core/tensor/tile_distribution.hpp` | the thread↔element map: which lane in which wave owns which coordinate |
| `TileWindow` | `core/tensor/tile_window.hpp` | a movable sub-view plus a distribution; the gateway through which loads and stores get their coalescing, vectorization and bounds guard |
| `DistributedTensor` | `core/tensor/...` | what `load_tile()` hands back — the data, in registers |
| Pipeline / Policy / Epilogue | `ops/gemm/`, `ops/fmha/` | the K-loop schedule, the layout decisions behind it, and the writeback |

> **A window is a cursor, not a copy.** `make_naive_tensor_view` and `make_tile_window` only *describe*
> where data lives. Nothing is read or written until the pipeline or the epilogue does it. Reading the
> declaration and expecting to see memory traffic is the most common way to misread a ck_tile kernel.

`TileDistribution` is simultaneously the most important object here and the least readable. It is a
`tile_distribution_encoding` built from compile-time `sequence` and `tuple` types, stating how the
wavefront's 64 lanes and the block's waves carve up a region and how many elements each lane ends up
holding. The `<Repeat, Warp, Lane, Vector>` shape — `<4,2,8,4>` and friends — is what lands the tile on
MFMA lanes correctly.

**You should almost never write one.** A Policy derives it from your tile sizes and your chosen
WarpGemm. Hand-authoring is possible and is a reliable way to produce a kernel that compiles and
computes the wrong thing.

### The verbs
`load_tile`, `store_tile`, `update_tile`, `async_load_tile` (global straight to LDS, no VGPR staging),
`shuffle_tile` (redistribute across lanes — this is how a transpose happens), `slice_tile`,
`sweep_tile` (run a lambda over the lane's own elements), and `block_tile_reduce` (the primitive FMHA
uses for row-max and row-sum).

## Assembling a GEMM
```
GemmKernel< TilePartitioner, GemmPipeline, EpiloguePipeline >
              │                  │              │
              │                  │              └─ writeback: CShuffle, plus any fused elementwise
              │                  └─ the K-loop mainloop schedule
              └─ (M,N,K) → grid;  gridDim = ceil(M/kM) × ceil(N/kN)
```

**TilePartitioner** fixes the block tile `kM×kN×kK`. Choose it so `ceil(M/kM)·ceil(N/kN)` lands near a
multiple of **256** — that is gfx950's CU count, and missing it leaves a wave-quantization tail where
most of the GPU idles through the last wave.

**Pipeline names spell out their own dataflow.** Decode `GemmPipelineAgBgCrCompV3` left to right: **A**
from **g**lobal, **B** from **g**lobal, **C** held in **r**egisters, **Comp**ute-optimized, version
**3**. Once you can read the name you rarely need to open the header.

| Pipeline | When |
|---|---|
| `GemmPipelineAGmemBGmemCRegV1` | single-buffered, low VGPR pressure — memory-bound work, or learning the structure |
| **`GemmPipelineAgBgCrCompV3`** | **double-buffered LDS with 2-stage prefetch — the compute-bound default** |
| `GemmPipelineAgBgCrMemV3` / `...CompV4` | memory-optimized, or deeper prefetch for very large K and fp8-dense cases |
| `*_async` persistent | direct-to-LDS with no VGPR staging — the current LLM GEMM and MoE path |

**Policy** — `UniversalGemmPipelineAgBgCrPolicy` and relatives — is where the layout thinking lives.
`MakeADramTileDistribution` and `MakeBDramTileDistribution` produce the global-load distributions and
the per-lane vector width; `MakeALdsBlockDescriptor` lays out LDS with an XOR swizzle chosen to avoid
bank conflicts; `GetWarpGemm()` picks the matrix-core instruction.

> **If you customized that swizzle, re-derive it.** It was designed against 32 banks. gfx950 has
> **64**, and a swizzle that was conflict-free at 32 is not automatically conflict-free at 64. See
> `../../../../../hardware/mi350_lds.md`.

**WarpGemm** is the seam where ck_tile touches the matrix core — `operator()` is a thin wrapper over
the intrinsic itself:

```cpp
c = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c, 0, 0, 0);
```

(Shapes and builtins: `../../../../hip/skills/optimize/hip_levers/hip_builtins.md`.)

**Epilogue / CShuffle** exists because of a hardware fact: the MFMA accumulator leaves C scattered
across lanes in a layout that cannot be stored coalesced. CShuffle routes C back through LDS
(`shuffle_tile`, then `store_tile`) into a storable arrangement, folds in bias, activation or residual
on the way, and only then writes. The knobs are `CShuffleDataType`, the store vector width (8 for
bf16), and the shuffle granularity `MXdlPerWavePerShuffle`.

## Build and run
```bash
sh ../script/cmake-ck-dev.sh ../ gfx950
make tile_example_gemm_basic -j     && ./bin/tile_example_gemm_basic     -m=4096 -n=4096 -k=4096 -v=1
make tile_example_universal_gemm -j && ./bin/tile_example_universal_gemm -m=4096 -n=4096 -k=4096 -v=0
```

`-v 1` turns on the example's own reference check. Use it while iterating; turn it off to time.

## Verify
| Check | How | Pass condition |
|---|---|---|
| It beats the alternative | bench at **your** shapes against classic v3 (GEMM) or the Triton FMHA backend (attention) | a real margin, measured per `measure_protocol.md` |
| The mainloop is clean | disassemble | wide `buffer_load`; `s_waitcnt lgkmcnt(1)` ahead of `v_mfma`, not `(0)`; no `scratch_` traffic, no `v_accvgpr` churn |
| Numerics hold | fp32 accumulate; for attention, greedy temp=0 against a reference over ≥10 prompts | bit-parity is not the bar; task output is |

## Failure modes
| Symptom | Cause | Fix |
|---|---|---|
| Dense square GEMM is well below expectation | ck_tile is not the strong path there | benchmark classic v3 (Issue #1727) before shipping |
| Kernel compiles, produces wrong values | hand-written `tile_distribution_encoding` | let the Policy generate it |
| LDS conflicts after porting from MI300X | XOR swizzle derived for 32 banks | re-derive for 64 banks |
| Following docs from the old repo | standalone `composable_kernel` is deprecated | use `rocm-libraries/projects/composablekernel/` |
| `ckProfiler` shows nothing for your kernel | it does not sweep ck_tile at all | use the example's own bench harness |
| Reading the window declaration, seeing no traffic | windows only declare addresses | the loads live in the pipeline and epilogue |

Full trap list, indexed by symptom: `ck_traps.md`.

## Where next
`ck_gemm_stack.md` (the parameter stack and the constraints between its levels) ·
`ck_fmha_stack.md` (FA-2 mapping, paged-KV) ·
`ck_instance_codegen.md` (how kernels get emitted; trimming build time) ·
`ck_tuning_knobs.md` (which knob to turn first)

## Sources
- WarpGemm struct, pipeline/policy structure, build steps (ROCm blog, hands-on CK-Tile GEMM):
  https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
- FMHA pipeline mapping (ROCm blog, FlashAttention-v2 with CK-Tile):
  https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- ck_tile component layout: https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/README.md
- Tile Window / Tensor Views / Sweep Tile concept docs:
  https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/tile_window.html ·
  https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/tensor_views.html ·
  https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/sweep_tile.html
- The dense-GEMM gap versus classic v3: https://github.com/ROCm/composable_kernel/issues/1727
- Repository deprecation and move: https://github.com/ROCm/composable_kernel
