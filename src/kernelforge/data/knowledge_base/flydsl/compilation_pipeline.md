# Compilation Pipeline + JIT Cache + Autotune

## 1. End-to-end flow

```
@flyc.jit / @flyc.kernel
       │
       │  (1) Cache check (in-mem → disk)
       │      key = (target, hints, per_arg_cache_sig)
       │
       ▼
   ASTRewriter.transform()  (compiler/ast_rewriter.py)
       │  Python for/if/while → scf.for/scf.if/scf.while
       │  range_constexpr → unrolled
       │
       ▼
   MLIR module setup
       │  gpu.container_module + target attribute
       │
       ▼
   Tracing: execute body in MLIR Context
       │  fly + gpu + scf + arith + memref + vector ops emitted
       │
       ▼
   MlirCompiler.compile()  (compiler/jit_function.py)
       │  Runs the pass pipeline (see §2)
       │  Emits fatbin (HSACO)
       │
       ▼
   JITCFunction (ExecutionEngine wrapper)
       │  Thread-safe, lazy engine init
       │  Pickle-serializable for disk cache
       │
       ▼
   Disk write: ~/.flydsl/cache/<func>_<fingerprint>/<key>.pkl
```

## 2. Pass pipeline (`compiler/backends/rocm.py`)

Pre-binary lowering fragments:

1. `fly-rewrite-func-signature` — pack DSL types to LLVM struct
2. `fly-canonicalize`
3. `fly-layout-lowering` — `crd2idx`/`logical_divide`/… → arith
4. `fly-int-swizzle-simplify`
5. `canonicalize`
6. `fly-convert-atom-call-to-ssa-form`
7. `fly-promote-regmem-to-vectorssa`
8. `convert-fly-to-rocdl`
9. `canonicalize`
10. `gpu.module(convert-scf-to-cf, cse, convert-gpu-to-rocdl{chipset=ARCH, use-bare-ptr-memref-call-conv=true …}, fly-rocdl-cluster-attr)`

Binary preparation:

11. `rocdl-attach-target{O=2 abi=600 chip=ARCH fast=... unsafe-math=... has-debug-info=...}`
12. `convert-scf-to-cf`
13. `convert-cf-to-llvm`
14. `gpu-to-llvm{use-bare-pointers-for-...}`
15. `convert-vector-to-llvm`
16. `convert-arith-to-llvm`
17. `convert-func-to-llvm`
18. `reconcile-unrealized-casts`
19. *(if `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`)* `ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}`
20. `gpu-module-to-binary{format=fatbin opts="..."}`

### Compile hints (injected via `CompilationContext.compile_hints()`)

| Hint | Effect |
|---|---|
| `waves_per_eu` | `--amdgpu-waves-per-eu=N` LLVM flag |
| `maxnreg` | `--amdgpu-num-vgpr=N` LLVM flag |
| `fast_fp_math` | `fast=true` on `rocdl-attach-target` |
| `unsafe_fp_math` | `unsafe-math=true` on `rocdl-attach-target` |
| `enable_debug_info` | adds the DI scope pass and `-g` opt to `gpu-module-to-binary` |

Hints flow through `Autotune.Config`:
```python
Config(BLOCK=128, num_warps=4, waves_per_eu=2, maxnreg=128, ...)
# num_warps/waves_per_eu/maxnreg do NOT become kernel kwargs;
# they're forwarded to compile_hints().
```

## 3. JIT cache (`compiler/jit_function.py`)

### In-memory cache (`_mem_cache`)
- Keyed by the tuple constructed at every call
- Lives for the process lifetime
- Always active (independent of `FLYDSL_RUNTIME_ENABLE_CACHE`)

### Disk cache
- Directory: `~/.flydsl/cache/<func>_<fingerprint>/<sha-of-key>.pkl`
  (override via `FLYDSL_RUNTIME_CACHE_DIR`)
- `enable_cache` (default `True`) controls disk read+write only
- `FLYDSL_RUNTIME_RUN_ONLY=1` — skip compilation entirely; raise on cache miss
  (for AOT-only / no-GPU compile gates)

### Cache key construction (`compiler/jit_function.py:1110-1147`)
Tuple of:
1. `("_target_", GPUTarget(backend, arch, warp_size))`
2. `("_hints_", tuple(sorted(hints.items())))` if hints non-empty
3. For each parameter: `(param_name, _arg_cache_sig(arg))`

### `_arg_cache_sig` priority (`compiler/jit_function.py:1079`)
1. `arg.__cache_signature__()` — adapter-defined (TensorAdaptor includes
   `assumed_align`, `mark_layout_dynamic` directives)
2. Python scalar:
   - annotated runtime type (`Int32`, `Stream`, …) → **type only** (value
     doesn't affect compile)
   - else → **value + type**
3. `JitArgumentRegistry.raw_cache_signature(arg)`
4. Type only

### Fingerprint (whole-package invalidation)
SHA-256 of:
- All Python source in `flydsl.compiler.*`, `flydsl.expr.*`,
  `flydsl.runtime.*`, `flydsl.utils.*`
- Native libraries: `_mlirDialectsFly*.so`, `libFly*.so`,
  `libfly_jit_runtime.so`, `libmlir_rocm_runtime.so`
- `flydsl.__version__`

A new fingerprint means a fresh subdirectory; old cache entries persist but
are inert.

### Closure capture caveat
If a helper function used inside `@flyc.jit` / `@flyc.kernel` changes its
behavior but isn't a closure value tracked by the rewriter (e.g. a top-level
helper imported from another module), the disk cache does **not** invalidate.
Workaround: `FLYDSL_RUNTIME_ENABLE_CACHE=0` or `rm -rf ~/.flydsl/cache`.

## 4. Autotune (`python/flydsl/autotune.py`)

### `Config` class (lines 18–72)
```python
Config(
    num_warps=N,
    waves_per_eu=N,
    maxnreg=N,
    pre_hook=callable,     # called before each config's benchmark
    **kwargs,              # become Constexpr kernel kwargs
)
```

Methods: `all_kwargs()` returns `kwargs + num_warps`; `compiler_opts()` returns
`{waves_per_eu, maxnreg}`; `to_dict() / from_dict()` for cache I/O.

### `@autotune` decorator (lines 262–296)
```python
@autotune(configs=[...], key=["n", "k"], warmup=5, rep=25,
          prune_configs_by=fn, reset_to_zero=["acc"],
          pre_hook=fn, post_hook=fn, do_bench=fn)
@flyc.jit
def my_jit(...): ...
```

### Cache (lines 139–161, 243–259)
- Cache key tuple = `(shapes/dtypes of key-args, dtypes of all args)`
- Disk path: `~/.flydsl/autotune/{fn_name}.json`
  (override via `FLYDSL_AUTOTUNE_CACHE_DIR`)
- JSON map: string-encoded key → `Config.to_dict()`

### Benchmarking (`_bench_one` at line 181)
- Default `do_bench`: `torch.cuda.Event` (requires PyTorch)
- Custom: `do_bench=lambda fn, warmup, rep: …`
- Configs merged into the `@jit` call via `all_kwargs()`; compiler opts
  injected via `compile_hints()` on `CompilationContext`

### ⚠ Autotune caveats
- `waves_per_eu` may not propagate to the final LLVM attr through
  `gpu-module-to-binary opts=`; rely on the `Config(waves_per_eu=N)` channel
  (which goes through `rocdl-attach-target`).
- Never mix `flyc.from_dlpack()` adaptors with varying `Constexpr` configs in
  the same call: `DLTensorAdaptor` caches MLIR types from the first
  `ir.Context` and segfaults on context recreation. Pass raw `torch.Tensor`
  instead.

## 5. IR dump workflow

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python my_kernel.py
```

Produces in `dumps/<func_name>/`:
```
00_original.mlir                          # post-tracing, pre-passes
01_gpu-kernel-outlining.mlir
02_fly-canonicalize.mlir
03_fly-layout-lowering.mlir
04_convert-fly-to-rocdl.mlir
05_canonicalize.mlir
06_convert-scf-to-cf.mlir
07_rocdl-attach-target.mlir
08_convert-scf-to-cf.mlir
09_convert-cf-to-llvm.mlir
10_gpu-to-llvm.mlir
11_convert-arith-to-llvm.mlir
12_convert-func-to-llvm.mlir
13_reconcile-unrealized-casts.mlir
14_gpu-module-to-binary.mlir
final_isa.s                                # AMD ISA (best-effort)
```

Additional knobs (see [env_vars.md](env_vars.md) for full list):
- `FLYDSL_DEBUG_PRINT_AFTER_ALL=1` — print IR after each MLIR pass
- `FLYDSL_DEBUG_PRINT_ORIGIN_IR=1` — print IR before any pass
- `FLYDSL_DEBUG_AST_DIFF=1` — show before/after AST diff per transformer
- `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` — emit `.debug_line` in HSACO for rocprofv3 ATT
  (Triton's `add_di_scope` equivalent — needed for source-to-ISA mapping;
  the `-g` flag alone is useless without the DI scope pass)

## 6. AOT / compile-only mode

```bash
COMPILE_ONLY=1 python my_kernel.py
```

Compiles each `@jit` function as it would be called, but skips ExecutionEngine
creation. Returns `None`. Useful for CI gates that don't have a GPU.

For airgapped deploys: `FLYDSL_RUNTIME_RUN_ONLY=1` reads-only from
`~/.flydsl/cache` (raise on miss). Combine with a populated cache from a
build host.
