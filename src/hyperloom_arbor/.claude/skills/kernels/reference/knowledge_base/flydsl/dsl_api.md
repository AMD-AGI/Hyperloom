# FlyDSL Python API Reference (`flydsl.expr.*` and decorators)

> Distilled from the Python frontend at `python/flydsl/`. Source lines reference
> the upstream FlyDSL repo at commit `2812676`.

## 1. Decorators

### `@flyc.kernel` — GPU device kernel  (`compiler/kernel_function.py:628-671`)

```python
@flyc.kernel
def k(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr[int]): ...
```

Parameters: `name=` (profiler-visible label), `known_block_size=[x,y,z]`
(required when block > 256 threads on AMDGPU). Returns a `KernelFunction`;
**calling it returns a `KernelLauncher`**, on which you call `.launch(grid=, block=, smem=, stream=)`.

`.launch(...)`:
- `grid`, `block`: tuple of 1–3 `int` or `ir.Value` (missing dims default to 1)
- `smem`: dynamic shared memory bytes (separate from `SmemAllocator`-managed static LDS)
- `stream`: `fx.Stream` (default = null stream)

### `@flyc.jit` — Host launcher  (`compiler/jit_function.py:1405-1409`)

```python
@flyc.jit
def launch(A: fx.Tensor, ..., stream: fx.Stream = fx.Stream(None)): ...
```

- Lazy signature analysis on first call
- `_mem_cache` keyed by `(target, compile_hints, per_arg_cache_sig)`
- Disk cache in `~/.flydsl/cache/<func>_<flydsl_fingerprint>/`
- When called inside an existing MLIR context, it composes (acts as a normal function)

### `@autotune` — Triton-style benchmark+pick  (`autotune.py:262-296`)

```python
from flydsl.autotune import autotune, Config

@autotune(
    configs=[
        Config(BLOCK=128, VEC=4, num_warps=4, waves_per_eu=2),
        Config(BLOCK=256, VEC=8, num_warps=8),
    ],
    key=["const_n"],   # re-tune when these arg shapes/dtypes change
    warmup=5, rep=25,
    prune_configs_by=None, reset_to_zero=None,
    pre_hook=None, post_hook=None, do_bench=None,
)
@flyc.jit
def my_jit(...): ...
```

- `Config` kwargs become `Constexpr` args at `@jit` call time
- `Config.num_warps` / `waves_per_eu` / `maxnreg` are special compiler hints
  (NOT kernel kwargs)
- Disk cache: `~/.flydsl/autotune/{fn_name}.json` (set
  `FLYDSL_AUTOTUNE_CACHE_DIR` to override)
- ⚠ Known limitation: `waves_per_eu` does **not** propagate via
  `gpu-module-to-binary opts=`; it must be set via LLVM function attribute or
  `rocdl-attach-target`. Use the `Config(waves_per_eu=N)` channel.
- ⚠ `DLTensorAdaptor` caches MLIR types from the first `ir.Context`; do NOT use
  `flyc.from_dlpack(t)` when calling a `@jit` with varying `Constexpr` values
  (causes segfault on re-context). Pass raw `torch.Tensor` instead.

## 2. Construction Primitives  (`expr/primitive.py`)

| Function | Purpose | Line |
|---|---|---|
| `make_shape(*dims)` | IntTuple shape (variadic; supports nested tuples) | 332 |
| `make_stride(*strides)` | IntTuple stride | 346 |
| `make_coord(*coords)` | IntTuple coord (supports `None` for free axes) | 360 |
| `make_layout(shape, stride)` | Layout from (Shape, Stride) | 374 |
| `make_layout_like(ref)` | Clone layout type | 393 |
| `make_ordered_layout(shape, order)` | Compact layout following stride order | 398 |
| `make_identity_layout(shape)` | Identity layout | (see source) |
| `make_int_tuple(elems)` | Build from list of `int` / `ir.Value` | 318 |
| `static(result_type, …)` | Materialize a static value at a static type | 304 |
| `make_tile(*shapes)` | Tile descriptor (used by TiledCopy / TiledMma) | 187 |

## 3. Layout Algebra  (`expr/primitive.py`)

### Coordinate mapping
| Function | Semantic |
|---|---|
| `crd2idx(layout, coord)` | `coord → linear idx` (`Index = Σ c_i·d_i`) |
| `idx2crd(layout, idx)` | inverse |
| `get_flat_coord(layout)` | flatten nested coord |
| `get_1d_coord(layout)` | linearize coord |

### Inspection
`size`, `cosize`, `rank`, `depth`, `get_shape`, `get_stride`, `get_layout`,
`get_iter`, `coalesce`.

### Algebra
| Function | Meaning |
|---|---|
| `composition(A, B)` | `result(x) = A(B(x))` |
| `complement(tiler, target_size)` | orthogonal complement; building block for `logical_divide` |
| `right_inverse(L)` / `left_inverse(L)` | pseudoinverses |
| `recast_layout(L, old_bits, new_bits)` | adjust for type-width change (e.g. f16→f8) |

### Products and Divides (all take `(layout, tiler)`)
| Family | Variants |
|---|---|
| Products | `logical_product`, `zipped_product`, `tiled_product`, `flat_product`, `raked_product`, `block_product` |
| Divides | `logical_divide`, `zipped_divide`, `tiled_divide`, `flat_divide` |

### Structural
`select(it, indices)`, `take(it, indices)`, `group(it, begin, end)`,
`append(it, elem)`, `prepend(it, elem)`, `zip(a, b)`, `slice(layout, coord)`,
`dice(layout, tiling)`.

### Tuple/Integer arithmetic
`int_tuple_{add,sub,mul,div,mod}`, `int_tuple_product`,
`int_tuple_product_each`, `int_tuple_product_like`, `shape_div`, `ceil_div`,
`elem_less`, `equal`.

## 4. Atoms (Hardware Instruction Wrappers)

### CopyAtom subclasses
| Atom | Source |
|---|---|
| `UniversalCopy(N)` / `UniversalCopy{8,16,32,64,128}b` | `expr/primitive.py:191-196` |
| `UniversalAtomic{Add,Max,Min,And,Or,Inc,Dec}(val_ty)` | `expr/primitive.py:198-205` |
| `rocdl.BufferCopy(bitsize)` / `rocdl.BufferCopy{32,64,128}b` | `expr/rocdl/__init__.py:30-33` (CDNA3) |
| `rocdl.cluster_load_async_to_lds_b{8,32,64,128}(...)` | `expr/rocdl/__init__.py:21-32` (CDNA4 G→LDS) |

### MmaAtom constructors
| Atom | Description |
|---|---|
| `rocdl.MFMA(m, n, k, elem_ty_ab, elem_ty_acc=None)` | CDNA3/4 MFMA descriptor (line 36) |
| `rocdl.WMMA(m, n, k, elem_ty_ab, elem_ty_acc=None)` | RDNA wave32 WMMA (line 62) |
| `rocdl.UniversalFMA(ty)` | scalar FMA fallback |

### MFMA instruction variants (`expr/rocdl/__init__.py:94-195`)
Each is `rocdl.<name>(result_type, [a, b, acc, cbsz=0, abid=0, blgp=0])`.

| Instruction | Arch | Notes |
|---|---|---|
| `mfma_f32_32x32x8f16` | gfx942+ | line 95 |
| `mfma_f32_32x32x8bf16_1k` | gfx942+ | 103 |
| `mfma_f32_32x32x16_f16` | gfx950+ | 111 — doubled K |
| `mfma_f32_32x32x16_bf16` | gfx950+ | 119 |
| `mfma_f32_16x16x16f16` | gfx942+ | 127 |
| `mfma_f32_16x16x16bf16_1k` | gfx942+ | 133 |
| `mfma_f32_16x16x32_fp8_fp8` | gfx942+ | 141 |
| `mfma_i32_16x16x32_i8` | gfx942+ | 147 |
| `mfma_f32_16x16x32_f16` | gfx950+ | 153 (doubled K) |
| `mfma_f32_16x16x32_bf16` | gfx950+ | 161 |
| `mfma_scale_f32_16x16x128_f8f6f4` | gfx950 | 169 — MXFP4/FP6/FP8 with per-block scale operand |

For SMFMAC (2:4 sparse) variants and WMMA (gfx1250) variants see the rocdl module
directly; they follow the same `(result_type, operands)` signature.

## 5. Tiled Operations

### TiledCopy / TiledMma construction
```python
# CopyAtom + per-thread layout + value layout + tile
tiled_copy = fx.make_tiled_copy(
    copy_atom,
    fx.raked_product(thr_layout, val_layout),
    fx.make_tile(M_thr, N_val),
)

# MmaAtom + thread layout (M_rep, N_rep, K_rep)
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2,2,1), (1,2,0)))

# TiledCopy variants matched to a TiledMma
fx.make_tiled_copy_A(copy_atom, tiled_mma)
fx.make_tiled_copy_B(copy_atom, tiled_mma)
fx.make_tiled_copy_C(copy_atom, tiled_mma)
```

### Per-thread slicing
```python
thr_copy = tiled_copy.get_slice(tid)        # ThrCopy
src_part = thr_copy.partition_S(gtensor)    # source partition
dst_part = thr_copy.partition_D(rmem)       # destination partition
retiled  = thr_copy.retile(frag)            # remap fragment to copy layout

thr_mma  = tiled_mma.thr_slice(tid)         # ThrMma (alias: get_slice)
frag_A   = thr_mma.make_fragment_A(tile_A)  # register fragment
part_A   = thr_mma.partition_A(tile_A)      # spatial partition (different use case)
```

### Execute
```python
fx.copy(copy_atom, src_part, dst_part)               # tiled copy (memref-level)
fx.copy(copy_atom, src_part, dst_part, pred=mask)    # with predicate (boundary)
fx.copy_atom_call(atom, src, dst)                    # single-thread atom invoke
fx.gemm(mma_atom, D, A, B, C)                        # D = A @ B + C
```

## 6. GPU + ROCDL Intrinsics

### `flydsl.expr.gpu`
```python
tid = gpu.thread_id("x")            # raw i32
bid = gpu.block_id("x")
gpu.thread_idx.x  / .y / .z         # Tuple3D wrappers (legacy spelling still works)
gpu.block_idx.x / .y / .z
gpu.block_dim.x / .y / .z
gpu.grid_dim.x / .y / .z
gpu.barrier()                       # s_barrier (workgroup sync)
gpu.smem_space()                    # #gpu.address_space<workgroup>
gpu.lds_space()                     # alias
gpu.SharedAllocator(base_alignment=16, static=True)
```

### `flydsl.expr.rocdl` (sched + wave + math + cluster)
```python
# Scheduling barriers (gfx942+):
rocdl.sched_mfma(cnt)        # allow N MFMA before next barrier
rocdl.sched_vmem(cnt)        # allow N global mem ops
rocdl.sched_dsrd(cnt)        # allow N LDS reads
rocdl.sched_dswr(cnt)        # allow N LDS writes
rocdl.sched_barrier(0)       # full fence

# Wave/lane:
rocdl.workitem_id_x/y/z()
rocdl.workgroup_id_x/y/z()
rocdl.wave_id()
rocdl.readfirstlane(val)
rocdl.ds_bpermute(idx, src)

# Async (gfx950):
rocdl.cluster_load_async_to_lds_b{8,32,64,128}(...)
rocdl.s_wait_asynccnt(cnt)
rocdl.cluster_workgroup_id_x/y/z()

# Wait counters:
rocdl.s_waitcnt(0)                   # CDNA3 combined waitcnt
rocdl.s_wait_loadcnt(0)              # CDNA4 fine-grained
rocdl.s_wait_storecnt(0)
rocdl.s_wait_dscnt(0)

# Math (1-cycle VALU, lower precision than math.*):
rocdl.exp2(T.f32, x)                 # v_exp_f32
rocdl.rcp(T.f32, x)                  # v_rcp_f32

# Raw memory:
rocdl.raw_ptr_buffer_load(rsrc, offset, soffset, aux)
rocdl.raw_ptr_buffer_store(data, rsrc, offset, soffset, aux)
rocdl.raw_ptr_buffer_load_lds(rsrc, lds_ptr, size_i32, off, soff, off_imm, aux)
```

### `flydsl.expr.buffer_ops` (AMD buffer descriptor abstraction)
```python
rsrc = buffer_ops.create_buffer_resource(memref, max_size=True)
# offset is in ELEMENTS of `dtype` (NOT bytes)
data = buffer_ops.buffer_load(rsrc, offset, vec_width=4, dtype=fx.Int32)
buffer_ops.buffer_store(data, rsrc, offset)
```

### `fx.rocdl.make_buffer_tensor(t, alignment=4)` — wrap a Tensor with a buffer
descriptor (preferred over raw `buffer_ops.create_buffer_resource()` in new code).

## 7. Numeric and Vector Types (`expr/typing.py`, `expr/numeric.py`)

```python
# Scalar constants (preferred over arith.constant)
fx.Int32(42), fx.Int64(0), fx.Index(0), fx.Float32(0.0), fx.BFloat16(1.0)

# Casting (preferred over arith.index_cast / arith.trunc_f / ext_f)
fx.Index(some_i32_val)
fx.Int32(some_index_val)
fx.Float32(some_bf16_val)

# Vector wrapper (replaces raw vector.* ops)
Vec = fx.Vector
v = Vec(raw_vec)                       # wrap raw vector<NxTy>
v[i]                                   # vector.extract
v.bitcast(fx.Float32)                  # vector.bitcast
v.to(fx.BFloat16)                      # arith.trunc_f / ext_f
Vec.filled(N, val, fx.Float32)         # splat
Vec.from_elements([a,b,c,d], fx.Float32)
v.store(rmem, [idx])                   # vector store
v + v2, v * v2, -v, v.maximumf(other)
cond.select(t, f)                      # arith.select via predicate
```

### Operators table (work on scalar and vector)
| Op | Helper |
|---|---|
| add/sub/mul/div | `+ - * /` |
| negate | `-v` |
| compare | `==`, `<`, `>=`, `arith.cmpf(...)` if you need fastmath |
| select | `cond.select(t, f)` |
| max | `a.maximumf(b)` |
| abs | no helper — use `is_neg = v < 0; out = is_neg.select(-v, v)` |
| FMA | `a*b + c` |

## 8. Control Flow

### Compile-time
```python
from flydsl.expr import range_constexpr, const_expr

for i in range_constexpr(N):       # unrolled at trace time
    ...

if const_expr(USE_FAST_PATH):      # static branch (no MLIR scf.if)
    ...
```

### Runtime (lowered to scf.* by `ASTRewriter`)
```python
for i in range(runtime_n):         # scf.for
    ...

if tid == fx.Index(0):             # scf.if
    ...

while cond:                        # scf.while
    ...
```

### Runtime loop with loop-carried state (software pipelining)
```python
start, stop, step = fx.Index(0), fx.Index(N-1), fx.Index(1)
for iv, state in range(start, stop, step, init=[acc0, tile0_flat...]):
    acc_in = state[0]
    ...
    next_tile = prefetch(iv + 1)
    new_acc = compute(acc_in, ...)
    results = yield [new_acc, *next_tile]
acc_final = results[0]
```

3 critical pitfalls:
1. **Bounds must be DSL `fx.Index(...)`** values. Plain `range(0, 15, 1, init=...)`
   is treated as Python `range` and unrolled (silently ignoring `init=`).
2. Use FlyDSL internal types in `init` and `state`. Unwrap with `v.ir_value()`
   only at hard boundaries.
3. **Clear `SmemPtr._view_cache = None`** before any epilogue access if the
   SmemPtr was used inside the loop body (else SSA dominance error from cached view).

## 9. ASTRewriter Transforms (`compiler/ast_rewriter.py`)

| Python construct | MLIR target | Notes |
|---|---|---|
| `for i in range_constexpr(N)` | unrolled (no MLIR loop) | Python `range`; pure macro |
| `for i in range(start, stop, step)` | `scf.ForOp` | dynamic bounds |
| `for i, state in range(..., init=[...])` | `scf.ForOp` with iter args | software pipeline |
| `if cond: ... else: ...` | `scf.IfOp` | cond must be `ir.Value` or `const_expr`-wrapped |
| `x = a if c else b` | `arith.select` (when single SSA) or `scf.if` | dispatcher chooses |
| `const_expr(x)` | no MLIR op | marks compile-time constant |
| `while cond: ...` | `scf.WhileOp` | |
| `yield x` (in `scf.for` body) | `scf.YieldOp` | |

## 10. Shared Memory: `SmemAllocator`  (`utils/smem_allocator.py`)

```python
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir

# Allocate at module scope (BEFORE @flyc.kernel)
allocator = SmemAllocator(None, arch="gfx942", global_sym_name="smem0")
lds_a = allocator.allocate_array(fx.T.f16, 8192)   # SmemKey for typed access

@flyc.kernel
def k(...):
    lds_base = allocator.get_base()         # get LDS base ptr inside kernel
    lds_a_ptr = lds_a(lds_base)             # SmemPtr (typed view)
    val = lds_a_ptr.load([idx])
    lds_a_ptr.store(val, [idx])

# Finalize the global symbol (BEFORE launch, inside the GPU module body)
comp_ctx = CompilationContext.get_current()
with ir.InsertionPoint(comp_ctx.gpu_module_body):
    allocator.finalize()
```

Two allocators with different `global_sym_name` → two independent LDS regions
(used for ping-pong double buffer — see [gemm_optimization.md](gemm_optimization.md)).

## 11. JIT Cache Key  (`compiler/jit_function.py:1110-1147`)

Cache key tuple:
1. `("_target_", GPUTarget)` — backend + arch + warp size
2. `("_hints_", tuple(sorted(compile_hints.items())))` if present
3. For each parameter: `(name, _arg_cache_sig(arg))`

`_arg_cache_sig(arg)` priority order (`compiler/jit_function.py:1079`):
1. `arg.__cache_signature__()` if implemented (e.g. `TensorAdaptor` includes
   `assumed_align`, `mark_layout_dynamic` directives)
2. Python scalar with non-runtime annotation → value+type; with runtime
   annotation → type only
3. `JitArgumentRegistry.raw_cache_signature(arg)`
4. Type only

Disk cache lives in `~/.flydsl/cache/<func>_<flydsl_fingerprint>/`.
**Fingerprint** = SHA256 of `flydsl.compiler.*` + `flydsl.expr.*` +
`flydsl.runtime.*` + `flydsl.utils.*` Python sources + native .so files
(`_mlirDialectsFly*.so`, `libFly*.so`, `libfly_jit_runtime.so`,
`libmlir_rocm_runtime.so`) + `flydsl.__version__`.

Closure capture is **not** automatically detected; if a helper outside the
traced function changes, set `FLYDSL_RUNTIME_ENABLE_CACHE=0` or
`rm -rf ~/.flydsl/cache`.

## 12. Argument Adaptors (`compiler/jit_argument.py`)

```python
from flydsl.compiler import JitArgumentRegistry, from_dlpack

# Built-in: torch.Tensor → TensorAdaptor (DLPack-based memref)
# Custom registration:
@JitArgumentRegistry.register(MyType, dsl_type=MyDslType)
class MyAdaptor:
    def __get_ir_types__(self): ...
    def __get_c_pointers__(self): ...

# Tensor alignment hints (affect cache key):
adaptor = from_dlpack(tensor).mark_layout_dynamic(leading_dim=0, divisibility=4)
```

## 13. Output Files of Interest

- `python/flydsl/compiler/__init__.py` — public API (`jit`, `kernel`, `from_dlpack`)
- `python/flydsl/compiler/jit_function.py` — `MlirCompiler`, `JitCacheManager`
- `python/flydsl/compiler/kernel_function.py` — `@kernel`, `KernelLauncher`,
  `CompilationContext`
- `python/flydsl/compiler/backends/rocm.py` — pass pipeline construction
- `python/flydsl/compiler/jit_executor.py` — `JITCFunction`
- `python/flydsl/expr/primitive.py` — layout primitives
- `python/flydsl/expr/derived.py` — `CopyAtom`, `MmaAtom`, `TiledCopy`, `ThrCopy`, `ThrMma`
- `python/flydsl/expr/rocdl/__init__.py` — MFMA + WMMA + buffer + cluster
- `python/flydsl/runtime/device.py` — `get_rocm_arch()`
- `python/flydsl/utils/env.py` — `EnvManager` typed env vars
- `python/flydsl/utils/smem_allocator.py` — `SmemAllocator`, `SmemPtr`
- `python/flydsl/autotune.py` — `Config`, `Autotuner`
