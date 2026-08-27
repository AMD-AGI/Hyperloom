# FlyDSL Overview

## The Stack (top to bottom)

```
┌────────────────────────────────────────────────────────────┐
│ User code:  @flyc.jit  +  @flyc.kernel  +  flydsl.expr     │
├────────────────────────────────────────────────────────────┤
│ ASTRewriter:  Python for/if/while  →  scf.for/scf.if/...    │
├────────────────────────────────────────────────────────────┤
│ Tracing:  execute body in MLIR Context → MLIR module        │
│           (fly + gpu + scf + arith + memref + vector ...)   │
├────────────────────────────────────────────────────────────┤
│ MlirCompiler pipeline:  fly-* → convert-fly-to-rocdl →      │
│                         convert-gpu-to-rocdl → LLVM IR →    │
│                         rocdl-attach-target → fatbin (HSACO)│
├────────────────────────────────────────────────────────────┤
│ JITCFunction:  MLIR ExecutionEngine wrapper                 │
│                + JitCacheManager (in-memory + ~/.flydsl/cache)│
├────────────────────────────────────────────────────────────┤
│ Runtime: hipModuleLoadData / hipModuleLaunchKernel          │
│         (lib/Runtime/ROCm/FlyRocmRuntimeWrappers.cpp)       │
└────────────────────────────────────────────────────────────┘
```

## Two Decorators You Always Write

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu

@flyc.kernel                       # device function
def k(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr[int]):
    tid = gpu.thread_id("x")
    ...

@flyc.jit                          # host launcher (also a tracing function)
def launch(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr[int],
           stream: fx.Stream = fx.Stream(None)):
    k(A, B, N).launch(grid=(N//256,), block=(256,), stream=stream)
```

Calling `launch(torch_a, torch_b, 1024)` triggers (first call only):
1. **Cache lookup** by tuple of `(target, compile_hints, per-arg cache_sig)`
2. **AST rewrite** of `launch` AND `k` bodies
3. **Tracing**: execute the rewritten Python in an MLIR context to emit ops
4. **Pipeline**: 20-pass MLIR run → HSACO fatbin
5. **ExecutionEngine** wrap + serialize to `~/.flydsl/cache/`

Subsequent calls with the same type signature replay only the JITCFunction.

## Parameter Types at the Boundary

| Type | Meaning | At host |
|---|---|---|
| `fx.Tensor` | memref descriptor | torch.Tensor → DLPack → `TensorAdaptor` |
| `fx.Constexpr[T]` | compile-time constant; **affects cache key** | Python int/str → baked into IR |
| `fx.Int32` / `fx.Int64` / `fx.Index` | runtime scalar | Python int auto-converted |
| `fx.Float32` / `fx.BFloat16` / `fx.Float16` | runtime scalar | constant or runtime |
| `fx.Stream` | HIP stream (async dispatch) | `fx.Stream(torch.cuda.Stream())` |

## The Layout Algebra in 30 Seconds

- **Shape** + **Stride** → **Layout**: `Index = sum(coord_i * stride_i)`
- **Coordinate ↔ Index**: `fx.crd2idx`, `fx.idx2crd`
- **Composition**: `(A ∘ B)(x) = A(B(x))` via `fx.composition`
- **Divide** (partition): `fx.logical_divide(layout, tiler)` splits into
  (tile-interior, tile-id) modes
- **Product** (multiply): `fx.raked_product(thr_layout, val_layout)` distributes
  values across threads with interleaved access
- **Slice**: `fx.slice(divided, (None, bid))` selects this block's tile

This is the same algebra as NVIDIA CuTe (CUTLASS `cute/layout.hpp`) but
implemented for AMD GPUs with wave64/MFMA primitives.

## The Atom Catalog

**Copy atoms** = one hardware copy instruction:

| Atom | Width | Backend |
|---|---|---|
| `fx.UniversalCopy(N)` / `UniversalCopy{8,16,32,64,128}b` | configurable | dialect-agnostic |
| `fx.UniversalAtomic{Add,Max,Min,And,Or,Inc,Dec}(val_ty)` | scalar | dialect-agnostic |
| `fx.rocdl.BufferCopy{32,64,128}b` | 32/64/128b | CDNA3 buffer descriptor |
| `fx.rocdl.cluster_load_async_to_lds_b{8,32,64,128}` | 8–128b | CDNA4 async G→LDS |

**MMA atoms** = one hardware matrix instruction:

| Atom | Hardware | Available |
|---|---|---|
| `fx.rocdl.MFMA(m, n, k, elem_ty, ...)` | CDNA3/4 MFMA | gfx942+ |
| `fx.rocdl.WMMA(m, n, k, elem_ty, ...)` | RDNA WMMA | gfx120x, gfx1250 |
| `fx.rocdl.UniversalFMA(ty)` | scalar FMA fallback | all |

MFMA variants wired through `rocdl.*` (see [dsl_api.md](dsl_api.md) §4):
`mfma_f32_{16x16,32x32}x{8,16,32,64}_{f16,bf16,fp8,bf8}`,
`mfma_i32_*_i8`, `mfma_scale_f32_16x16x128_f8f6f4` (CDNA4 MXFP4/FP6/FP8).

## The Compilation Pipeline

`MlirCompiler._pipeline_fragments()` runs (in order):

1. `fly-rewrite-func-signature` — pack DSL types into LLVM-friendly structs
2. `fly-canonicalize` — fly-specific canonicalization
3. `fly-layout-lowering` — lower `fly.crd2idx`, `fly.logical_divide`, etc. to arith
4. `fly-int-swizzle-simplify` — peel addends out of canonical swizzle bit patterns
5. `canonicalize`
6. `fly-convert-atom-call-to-ssa-form` — memref → SSA atom calls
7. `fly-promote-regmem-to-vectorssa` — `fly.make_ptr(register)` → vector SSA
8. `convert-fly-to-rocdl` — emit ROCDL intrinsics (MFMA, buffer_rsrc, ds_*)
9. `canonicalize`
10. `gpu.module(convert-scf-to-cf, cse, convert-gpu-to-rocdl{chipset=…}, fly-rocdl-cluster-attr)`
11. `rocdl-attach-target{O=2 abi=600 chip=<arch>}`
12. `convert-scf-to-cf` → `convert-cf-to-llvm`
13. `gpu-to-llvm{use-bare-pointers=...}`
14. `convert-vector-to-llvm` → `convert-arith-to-llvm` → `convert-func-to-llvm`
15. `reconcile-unrealized-casts`
16. (optional, when `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`) `ensure-debug-info-scope-on-llvm-func`
17. `gpu-module-to-binary{format=fatbin opts="…"}`

Compile hints injected via `CompilationContext.compile_hints()`:
`waves_per_eu`, `maxnreg`, `fast_fp_math`, `unsafe_fp_math`,
`enable_debug_info`.

## Target Architectures

| Arch | Chip | Wave | MMA | LDS/CU | Notes |
|---|---|---|---|---|---|
| `gfx942` | MI300X / MI308X | 64 | MFMA | 64 KB | CDNA3 baseline; preshuffle GEMM, PA decode |
| `gfx950` | MI350 / MI355X | 64 | MFMA | 160 KB | CDNA4; FP4, MFMA scale, `ds_read_b64_tr_b16/8/4`, `ds_read_b96_tr_b6`, 64 banks |
| `gfx1201` | Radeon AI PRO R9700 | 32 | WMMA | 64 KB | RDNA4 |
| `gfx1250` | MI450 | 32 | WMMA + TDM | 320 KB | FP8/FP4, async/TDM copy helpers |
| `gfx90a` | MI250X | 64 | MFMA | 64 KB | CDNA2 baseline (verified platform) |

Detection in `python/flydsl/runtime/device.py:get_rocm_arch()`:
1. `FLYDSL_GPU_ARCH` env var
2. `HSA_OVERRIDE_GFX_VERSION` (supports `9.4.2` → `gfx942`)
3. `rocm_agent_enumerator` system tool
4. Default: `gfx942`

## What FlyDSL Is For (And Not For)

**FlyDSL** = maximum control over data layout in registers / LDS / global, with
explicit thread-value mapping and explicit MFMA atom placement. The right tool
when you need to hit close to peak MFMA utilization or beat the CUTLASS/CK
pattern catalog on a specific shape.

**Not the right tool** for fast iteration on novel research kernels — use
Triton or Gluon for that. FlyDSL's verbosity is proportional to the precision
of layout control it gives you.

## Comparison with Triton/Gluon

| Aspect | FlyDSL | Triton | Gluon |
|---|---|---|---|
| Layout control | Explicit algebra | Implicit (block ptrs) | Implicit |
| Tiling | `divide`/`product` ops | `tl.program_id` auto | auto |
| Memory | Copy atoms + TiledCopy + buffer_ops | `tl.load/store` | `gluon.load/store` |
| MFMA | Direct `rocdl.mfma_*` intrinsics | `tl.dot` | `gluon.dot` |
| Shared mem | `SmemAllocator` explicit | implicit scratchpad | implicit |
| Abstraction | low (near hardware) | medium | medium-high |
| Compilation | MLIR Fly → ROCDL → HSACO | MLIR Triton → LLVM | MLIR |
