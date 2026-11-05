# FlyDSL Environment Variables

> Authoritative list from `python/flydsl/utils/env.py` (lines 219–279).
> EnvManager subclasses: `CompileEnvManager`, `DebugEnvManager`, `RuntimeEnvManager`.

## Compile (`FLYDSL_COMPILE_*`) — `CompileEnvManager`

| Var | Default | Effect |
|---|---|---|
| `FLYDSL_COMPILE_OPT_LEVEL` | `2` | MLIR optimization level (0–3) |
| `COMPILE_ONLY` | `0` | If `1`, compile only — skip ExecutionEngine creation; returns `None`. Useful for CI gates without a GPU. |
| `ARCH` | auto-detect | Override target GPU arch (e.g. `gfx942`, `gfx950`, `gfx1250`). Wins over `FLYDSL_GPU_ARCH` / `HSA_OVERRIDE_GFX_VERSION` for the compile target. |
| `FLYDSL_COMPILE_BACKEND` | `rocm` | Backend ID (`rocm` for HIP/ROCDL — the only backend) |
| `FLYDSL_COMPILE_LLVM_DIR` | `""` | External LLVM/MLIR install prefix for final code generation (override the build's bundled LLVM) |

## Debug (`FLYDSL_DEBUG_*` / `FLYDSL_DUMP_*`) — `DebugEnvManager`

| Var | Default | Effect |
|---|---|---|
| `FLYDSL_DEBUG_DUMP_ASM` | `false` | Dump assembler output to disk |
| `FLYDSL_DUMP_IR` | `false` | Dump IR at each pipeline stage to `FLYDSL_DUMP_DIR` |
| `FLYDSL_DUMP_DIR` | `~/.flydsl/debug` | Output directory for IR dumps |
| `FLYDSL_DEBUG_AST_DIFF` | `false` | Print before/after AST diff for each AST rewriter pass |
| `FLYDSL_DEBUG_LOG_LEVEL` | `WARNING` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `FLYDSL_DEBUG_LOG_TO_FILE` | `""` | Path for file logging (empty = disabled) |
| `FLYDSL_DEBUG_LOG_TO_CONSOLE` | `false` | Enable console logging |
| `FLYDSL_DEBUG_PRINT_ORIGIN_IR` | `false` | Print pre-pipeline IR |
| `FLYDSL_DEBUG_PRINT_AFTER_ALL` | `false` | Print IR after every MLIR pass |
| `FLYDSL_DEBUG_ENABLE_DEBUG_INFO` | `false` | Generate DWARF debug info; adds `ensure-debug-info-scope-on-llvm-func` pass + `-g` to `gpu-module-to-binary`. Required for rocprofv3 ATT source-to-ISA mapping. |
| `FLYDSL_DEBUG_ENABLE_VERIFIER` | `true` | Run MLIR module verifier between passes |

## Runtime (`FLYDSL_RUNTIME_*`) — `RuntimeEnvManager`

| Var | Default | Effect |
|---|---|---|
| `FLYDSL_RUNTIME_KIND` | `rocm` | Device runtime (must match `FLYDSL_COMPILE_BACKEND`) |
| `FLYDSL_RUNTIME_CACHE_DIR` | `~/.flydsl/cache` | Disk cache directory for compiled kernels |
| `FLYDSL_RUNTIME_ENABLE_CACHE` | `true` | Enable disk cache. **In-memory cache is always active** regardless of this flag — only disk r/w is gated. |
| `FLYDSL_RUNTIME_RUN_ONLY` | `false` | Skip JIT compilation; load AOT cache only. Raise `RuntimeError` on cache miss. Useful for airgapped deploys. |

## Autotune (`autotune.py`)

| Var | Default | Effect |
|---|---|---|
| `FLYDSL_AUTOTUNE_CACHE_DIR` | `~/.flydsl/autotune` | JSON cache for best-config-per-key |

## GPU Arch Detection (`runtime/device.py:get_rocm_arch()`)

Resolved in this priority order:
1. `FLYDSL_GPU_ARCH` env var (string `gfx942` etc.)
2. `HSA_OVERRIDE_GFX_VERSION` env var (e.g. `9.4.2` → `gfx942`)
3. `rocm_agent_enumerator` system tool
4. Default: `gfx942`

`ARCH` (no prefix) overrides #1 but only for the compile pipeline. Set both
together if you also want runtime detection to follow suit.

## Common Workflows

```bash
# Dump all IR + final ISA
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python my_kernel.py

# Print IR after every pass (very verbose)
FLYDSL_DEBUG_PRINT_AFTER_ALL=1 python my_kernel.py

# Force recompile (disk cache off) — needed after C++ pass / non-closure helper change
FLYDSL_RUNTIME_ENABLE_CACHE=0 python my_kernel.py

# Wipe disk cache
rm -rf ~/.flydsl/cache

# AOT-only: load cache; fail on miss
FLYDSL_RUNTIME_RUN_ONLY=1 python my_kernel.py

# Compile-only (no GPU needed; verifies that everything traces+lowers)
COMPILE_ONLY=1 python my_kernel.py

# Override arch for compile pipeline
ARCH=gfx950 python my_kernel.py
# Or simulate a different arch end-to-end
HSA_OVERRIDE_GFX_VERSION=9.5.0 ARCH=gfx950 python my_kernel.py

# Enable rocprofv3 ATT source mapping
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 python my_kernel.py
```

## Important Caveats

1. **In-memory cache is always on.** `FLYDSL_RUNTIME_ENABLE_CACHE=0` does NOT
   disable `_mem_cache` — it only stops disk reads/writes. Within a single
   process, the first compile result is cached and reused.
2. **Closure captures don't invalidate disk cache.** Helper code outside the
   traced function (imported, top-level) is NOT part of the fingerprint or
   cache key. Set `FLYDSL_RUNTIME_ENABLE_CACHE=0` or wipe the cache after
   such edits.
3. **`-g` alone is useless.** Source-to-ISA mapping needs
   `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` so the `ensure-debug-info-scope-on-llvm-func`
   pass runs; this converts MLIR `loc()` metadata to LLVM `DISubprogramAttr` /
   `DICompileUnitAttr` before binary emission. Without the pass, location
   metadata is silently dropped during MLIR-to-LLVM-IR translation, and `-g`
   has nothing to preserve.
