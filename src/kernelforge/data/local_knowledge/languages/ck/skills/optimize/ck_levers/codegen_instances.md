---
title: CK codegen & the instance system — how instances are generated and selected
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, mxfp4]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
  - https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md
---

# CK codegen & instances

## TL;DR
CK ships **pre-instantiated** device ops (classic) and **Python-generated** kernel instances (ck_tile)
so you don't compile a fresh template per call. Two distinct machineries: (1) classic-CK *instance
factory headers* + `DeviceOperationInstanceFactory` (runtime sweep = `ckProfiler`); (2) ck_tile
*`generate.py` codegen* that emits one `.cpp` per kernel trait so the build parallelizes. Understanding
both is how you trim CK build time from hours to minutes and pin a single tuned config per LLM shape.

## Core concepts
### Classic-CK instances (two layers)
1. **Instance factory headers** — `library/src/tensor_operation_instance/gpu/gemm*/...` define lists of
   concrete `DeviceGemmXdlUniversal<...>` specializations (different tiles/pipelines), registered via
   `add_device_gemm_xdl_universal_*_instances(...)`.
2. **`DeviceOperationInstanceFactory<...>::GetInstances(ops)`** — at runtime returns every registered
   instance for a layout/dtype; you sweep, keep the fastest that returns `IsSupportedArgument()==true`,
   pin its index. (See [ck_classic.md](ck_classic.md) for the sweep loop — it *is* `ckProfiler`.)

### ck_tile codegen (`generate.py`)
The ck_tile FMHA/GEMM examples generate kernels with a Python script rather than shipping a fixed
instance DB. For FMHA (`example/ck_tile/01_fmha/`): `generate.py` instantiates the kernel template into
**separate `.cpp` files** "to benefit from parallel building." The kernel itself
(`fmha_fwd_kernel.hpp`, the grid-wise op) takes two template params:
- **`FmhaPipeline`** — one of the `block_tile_pipeline`s, "a performance critical component"
  (e.g. `qr_ks_vs`, `qr_ks_vs_async`, paged-KV variants — see [fmha_template.md](fmha_template.md)).
- **`EpiloguePipeline`** — modifies and stores the result in the last phase.

The step-by-step instantiation lives in the `FMHA_FWD_KERNEL_BODY` blob inside `generate.py`; the
per-trait knobs it sweeps are head-dim, dtype, causal/mask spec, bias/alibi, rotary, paged-KV. The
example dir holds `example_fmha_fwd.cpp`, `example_fmha_bwd.cpp`, `fmha_fwd.hpp`, `fmha_bwd.hpp`,
`mask.hpp`, `bias.hpp`, `rotary.hpp`, `quant.hpp`, plus `codegen/`, `misc/`, `script/`.

## The levers (trim build time + emit only what you need)
- **`GPU_TARGETS=gfx942`** at cmake — CK otherwise builds for every gfx (huge). gfx950 fp4/mxfp4 are
  behind `DTYPES` flags.
- **Build only your instance group**, e.g. `make device_gemm_xdl_universal_f16_instance` or
  `ninja tile_example_fmha_fwd`.
- **CK-Tile dispatcher** — the newer unified codegen / arch-filter front-end (C++ & Python, per CK
  CHANGELOG) that emits only the instances your shapes require.
- For ck_tile FMHA, restrict the codegen trait list in `generate.py` (head-dims, dtypes, masks you
  actually run) so you don't compile the full Cartesian product.

## Pitfalls
- The classic instance DB is large; building the whole library for all archs/dtypes can take an hour+.
  Scope `GPU_TARGETS` + the instance group.
- A hand-copied "winning instance" table is **build-specific** (tile/pipeline IDs drift across CK
  versions) — re-sweep after a CK bump; don't ship a frozen table as portable.
- ck_tile `generate.py` trait combos explode quickly — uncapped, the FMHA codegen emits hundreds of
  `.cpp` files. Prune to your serving shapes.

## Verify
- After codegen, `ls` the generated `.cpp` files to confirm only the wanted traits were emitted.
- Sweep with `ckProfiler` (classic) or the example's own `-v 1` validation flag (ck_tile) at your shapes.

## Sources
- ck_tile 01_fmha example layout + `generate.py` / `FMHA_FWD_KERNEL_BODY` / `FmhaPipeline`+`EpiloguePipeline`: https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha
- ROCm "Optimizing with Composable Kernel" (instance selection / profiler): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- CK CHANGELOG (CK-Tile dispatcher, persistent async input scheduler, fp4 DTYPES gates): https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md
