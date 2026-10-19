---
name: debug-flydsl-kernel
description: >
  Diagnose a FlyDSL kernel that is wrong, NaN, zero, hanging, or refusing to compile.
  Symptom-indexed: start from what you observed, land on the cause. Covers the stale-cache
  false negative and what the JIT cache key does not see, shape-dependent bounds baked from
  host integers, softmax -inf arithmetic, partition/stride addressing, FP8 error budgets,
  the three loop forms and what each carries, non-short-circuit and/or in traced code,
  buffer_load offset units, and barrier divergence. Use when a FlyDSL kernel produces
  incorrect output or fails to build.
  Usage: /debug-flydsl-kernel
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Debugging a FlyDSL kernel

## Before anything else: clear the cache
FlyDSL caches compiled kernels aggressively, and a stale cache is the single most common reason a
correct fix appears not to work. **Do this before you believe any result**, including a failing one:

```bash
rm -rf ~/.flydsl /tmp/flydsl*
```

If the launch wrapper is memoized, clear that too:

```python
compile_my_kernel.cache_clear()      # any @functools.lru_cache on the compile path
```

Everything below assumes you have done this. A debugging session that skips it can burn hours
"fixing" a bug that was already fixed.

### What the cache key sees
Clearing is the blunt fix; knowing the key tells you when a result was stale without one. In FlyDSL
v0.3.2 (`python/flydsl/compiler/jit_function.py`) the key of a `@flyc.jit` call is built from:

- the FlyDSL toolchain — the Python sources of `flydsl.{compiler,expr,extension,runtime,utils}` and the
  native libraries (C++ passes) — plus the `.py` files of every directory in `FLYDSL_EXTRA_SOURCE_DIRS`;
- the launcher's source, and the source of the functions it names — `@flyc.kernel` / `@flyc.jit`
  functions always, **plain helper functions only if defined in the same directory**;
- the methods it calls on the class the launcher is defined on;
- closure values that are `int`, `float`, `bool`, `str`, `None`, `tuple` or `Enum`, and module globals
  the launcher reads (scalars and builtin containers by value);
- each argument's signature — for a raw `torch.Tensor`: dtype, rank and which axis has unit stride,
  **not its shape** (raw tensors are wrapped layout-dynamic; `flyc.from_dlpack(t)` makes the shape
  static and part of the key);
- five environment variables: `FLYDSL_COMPILE_OPT_LEVEL`, `FLYDSL_COMPILE_BACKEND`,
  `FLYDSL_COMPILE_LLVM_DIR`, `FLYDSL_DEBUG_ENABLE_DEBUG_INFO`, `FLYDSL_EXTRA_SOURCE_DIRS`.

So these changes do **not** invalidate a compiled kernel: methods of a helper class you instantiate
inside the kernel, plain helpers in another directory (a kernel in `kernels/gemm/` calling into
`kernels/common/` is one), functions reached through a module attribute (`buffer_ops.buffer_load`), a
`dict` / `list` / object captured by closure (or anything inside it), and environment variables your
own code reads. With one cache directory and two processes, a kernel that
stores `cfg["v"]` from a closure `dict` returned 1.0 when the second process asked for 2.0 and no new
cache entry appeared, while the same kernel reading a closure `float` recompiled and returned 2.0; a
raw-tensor kernel launched at 8, 16 and 24 rows produced one cache entry. Edits there need `FLYDSL_EXTRA_SOURCE_DIRS` pointing at the
helper directory, `FLYDSL_RUNTIME_ENABLE_CACHE=0` (disables the disk cache; the in-memory cache stays),
or a clean cache.

A captured `dict` is never in the key: not in one process, not through a second call of the factory, not
across processes sharing the cache directory. A captured scalar enters the key when the `@flyc.jit` object is
created: changing it on an existing jit object reuses the old binary, while a new jit object or a new process
recompiles.

Two more cases:
- A module global read through `const_expr` is safe when it lives in the kernel's file or directory:
  changing it after the first compile raises `RuntimeError: global '…' changed since first compile`. A
  global in a helper module **from another directory** is not in the key, so the same jit object silently
  keeps the old branch. Aliasing it to a local of the factory function makes new jit objects and new
  processes recompile; the existing jit object still keeps the old binary.
- A module-level dict that is first filled during tracing (e.g. a cache of generated asm strings) makes the
  next call of the same jit object, `flyc.compile` included, raise `RuntimeError: global '<dict>' … changed
  since first compile`. Fill it before the first call, or keep the cache inside the builder.

Two checks that settle "did my change run?":
- **Break it on purpose.** Make the kernel compute something wrong (scale by 2) without clearing the
  cache. If the old correct result comes back, every earlier measurement of that edit was stale.
- **Identical numbers are a signal.** Variants that should differ but report byte-identical timings, or
  identical `VGPR_Count` / `Accum_VGPR_Count` / `SGPR_Count` in a rocprofv3 kernel trace, are almost
  always one cached binary.

## Start from the symptom
| What you observe | Most likely cause | Section |
|---|---|---|
| Output is all NaN | `-inf` minus `-inf` in softmax, or divide by zero | [§1](#1-all-nan) |
| Output is all zeros | wrong output address, or an unwritten intermediate | [§2](#2-all-zeros) |
| More than half the elements are wrong | missing partitions, or a layout/addressing mismatch | [§3](#3-mostly-wrong) |
| Some shapes partly unwritten, depending on the order shapes ran | a bound baked from a host integer | [§3](#a-bound-or-address-baked-from-a-host-integer) |
| 1–5% of elements are slightly off | FP8 quantization, or a scale applied twice | [§4](#4-slightly-off) |
| Won't compile, or crashes at trace time | `range` vs `range_constexpr`, loop-state typing, scalar/vector mismatch | [§5](#5-wont-compile) |
| GPU hangs | loop bounds, or a divergent barrier | [§6](#6-hang) |

The distinction that matters most is **§3 versus §4**. A large mismatch is structural — you have the
wrong data. A small mismatch is numeric — you have the right data at the wrong precision. They have
disjoint cause sets, so classify before you dig.

---

## 1. All NaN

### `-inf` minus `-inf`
When every token in a partition is masked out (past the end of the context), `qk_max` stays at `-inf`.
Then `exp(s - qk_max)` becomes `exp(-inf - (-inf))` = `exp(NaN)` = NaN, and it propagates through the
whole reduction.

```python
safe_diff = (qk_max > NEG_INF).select(diff, ZERO_F)
```

### Divide by zero in normalization
If every probability is zero, `exp_sum` is zero and `1/exp_sum` is `inf`.

```python
safe_sum = (running_sum > ZERO_F).select(running_sum, fx.Float32(1.0))
inv_sum  = fx.Float32(1.0) / safe_sum
```

### Locate it from the host
Do not guess which buffer went bad — print them. Probes like these last one turn: put them in the
kernel, or in a scratch script under `forge_experiments/`, and remove them before you end the turn.
A submitted kernel carries no `print`, and a probe left behind reads to the next session as
something the operator is supposed to emit:

```python
torch.cuda.synchronize()
print(f"exp_sums   nan={exp_sums.isnan().sum()}  inf={exp_sums.isinf().sum()}")
print(f"max_logits nan={max_logits.isnan().sum()}  range=[{max_logits.min():.4f}, {max_logits.max():.4f}]")
print(f"temp_out   nan={temporary_output.isnan().sum()}")
```

The first buffer in the chain that contains NaN is where to look. A NaN in `max_logits` and a NaN in
`temp_out` are different bugs. When the bug is found, the fix goes into the kernel and the probes do
not.

---

## 2. All zeros

Zeros almost always mean *the write went somewhere else*, not *the compute produced zero*.

### Wrong stride
A wrong `stride_out_seq` or `stride_out_part` sends every store to the wrong address:

```python
print(f"out strides: {output.stride()}, temp strides: {temporary_output.stride()}")
```

### Partition slot versus partition index
For multi-partition kernels, output must be written to the **`part_z` slot** (`0 .. grid_z-1`), not the
absolute partition index. The reduce kernel reads slots. Writing by absolute index scatters results
outside the range the reducer looks at.

### The intermediate was never written
If the main kernel never writes `exp_sums` / `max_logits`, the reduce kernel faithfully reduces
uninitialized memory. Prove it with a sentinel:

```python
exp_sums.fill_(-999.0)
# ... launch ...
torch.cuda.synchronize()
print(f"exp_sums[0,0,0,:4] = {exp_sums[0,0,0,:4]}")   # must NOT still be -999
```

This distinguishes "wrote zeros" from "wrote nothing", which look identical otherwise.

---

## 3. Mostly wrong

### Missing partitions
If `grid_z < total_partitions` and the kernel handles exactly one partition per CTA with no loop, most
of the context is silently skipped:

```python
total_parts = math.ceil(context_len / KV_COMPUTE_BLOCK)
print(f"grid_z={grid_z}, total_parts={total_parts}")
assert grid_z == total_parts or kernel_has_multi_partition_loop
```

### The all-1s isolation test
Fill every input with `1.0`. All softmax probabilities become equal and the PV output is exactly
`1.0`, so any deviation is a layout or addressing bug rather than a data-dependent one:

```python
query.fill_(1.0); key_cache.fill_(1.0); value_cache.fill_(1.0)
```

**Know its blind spot.** Uniform inputs give the correct answer regardless of operand ordering, so
this test **cannot** catch V/P operand misalignment in the MFMA. Passing all-1s narrows the search; it
does not clear the layout.

### Single-partition isolation
Force `max_context_partition_num=1` (one-shot mode) to bypass the reduce kernel entirely. If it passes
here and fails with multiple partitions, the bug is in partitioning or reduction, not in the main
compute.

### A bound or address baked from a host integer
A Python `int` folded into the IR — a `num_records_bytes`, a stride, an address constant — belongs to
the one compiled artifact. If it was derived from a shape and reached the kernel through a channel the
cache key does not see (a `dict` in a closure, a helper object's attribute; see
[What the cache key sees](#what-the-cache-key-sees)), every later shape with the same dtype reuses the
first shape's value.

Signature: part of the output is never written — rows of zeros past a fixed byte count, or a gap spread
evenly across one dimension — and only for some shapes. It hides when shapes run largest first, so a
single-shape test passes. **The decisive experiment is to run the same shapes in reverse order:**
failures that follow the shape are shape-specific; failures that follow the position are a stale bound.

Fix: derive every quantity that varies with the shape from the kernel's traced arguments
(`fx.Int32` sizes, the tensor's layout). Keep host constants for what genuinely changes the code —
branch choice, unroll count, vector width — which changes the IR anyway.

### Differential against another backend
```python
torch.testing.assert_close(flydsl_output, gluon_output, atol=5e-3, rtol=5e-3)
```
An element-wise comparison against a Gluon or Triton implementation of the same math localizes the
divergence far faster than reasoning about the layout.

---

## 4. Slightly off

Before treating a small mismatch as a bug, check whether it is the expected error budget.

| Source | Expected magnitude | Verdict |
|---|---|---|
| FP8 PV MFMA vs a bf16 reference | ~0.03 max error, `atol=5e-3` | **not a bug** — inherent to the FP8 data path |
| Reference uses per-row Q quant, kernel uses per-tensor | ~1–3% | quantization mode mismatch, fix the kernel or the reference |
| `_scale` composition | arbitrary | verify `_scale = softmax_scale * q_scale * k_scale` |

The recurring real bug in this class is **applying `v_scale` twice** — once while scaling the
probabilities and again after the PV product. It produces a small, plausible, uniformly-scaled error
that is easy to mistake for quantization noise.

---

## 5. Won't compile

### The three loop forms
Only `range_constexpr` unrolls. Both `range` forms become a runtime `scf.for`, so the induction
variable is an SSA value and cannot index a Python list:

```python
for i in range(4):            # WRONG if i indexes a Python list: i is an SSA value
    result[i] = ...

for i in range_constexpr(4):  # CORRECT: i is a Python int
    result[i] = ...
```

What each form carries across iterations (`compiler/ast_rewriter.py`):

| Form | Lowers to | Loop-carried state | Bounds |
|---|---|---|---|
| `range_constexpr(n)` | unrolled at trace time | none needed | Python `int` |
| `range(a, b, s)` | `scf.for` on `i32` | **inferred**: every name bound before the loop and reassigned in the body becomes an `iter_arg`, so `data = next_data` is carried | Python `int` or `fx.Int32`; `fx.Int64` raises `TypeError: for-loop start must be i32, got i64` |
| `range(a, b, s, init=[...])` | `scf.for` on `index` with explicit `iter_args` | only what you `yield`; rebinding a Python name in the body is not carried | Python `int` is fine (becomes an index constant), as is `fx.Int32` / `fx.Int64` |

Older guidance — that Python-int bounds unroll the loop and silently drop `init=`, or that a Python
rebinding is never a phi — is wrong (it comes from text bundled with FlyDSL's own skills, not from its
source); a probe with `range(0, 15, 1, init=[acc])`
emits `scf.for %arg2 = %c0 to %c15 step %c1 iter_args(...)`. A name that is `None` before a runtime
loop and assigned inside it raises `TypeError`: give it a typed initial value.

### Runtime versus compile-time conditionals
Runtime comparisons in a Python `if` are supported — the rewriter lowers dynamic conditions to
`scf.IfOp`. Write them with DSL operators, not hand-built MLIR predicates:

```python
tid  = gpu.thread_id("x")
lane = tid % fx.Int32(64)

if lane == fx.Int32(0):                       # lowered to scf.IfOp
    fx.printf("lane zero")

val = (lane < fx.Int32(8)).select(good, zero) # runtime predicate for select
```

Only reach for `arith.cmpi(arith.CmpIPredicate.slt, …)` when you are deliberately constructing
low-level MLIR. Passing a condition straight to `scf.IfOp` requires unwrapping the DSL boolean:

```python
cond   = arith.unwrap(partition_idx >= visible_tile_count)
if_op  = scf.IfOp(cond, has_else=False)
```

`const_expr(...)` is for **compile-time** decisions only:

```python
if const_expr(trans_v):
    ...
```

**Do not write `const_expr(lane == 0)`.** Even with `known_block_size`, `gpu.thread_id("x")`, `lane`,
and `warp_id` are runtime SSA values — the compiler knows their *range*, not which lane is executing.

**`and` / `or` do not short-circuit in a traced body.** The rewriter turns `A and B` into a call,
`dsl_and_(A, B)`, so both sides are evaluated first. `if const_expr(S and K % S == 0):` raises
`ZeroDivisionError` at trace time when `S == 0`, although the same line is fine in plain Python. When the
right side is only legal if the left is true (modulo, division, indexing, `.shape[i]`), nest two `if`s.

**Only `@flyc.kernel` / `@flyc.jit` functions are rewritten.** A plain Python helper the kernel calls
runs as ordinary Python at trace time, so `if <runtime value>:` inside it raises
`RuntimeError: cannot evaluate dynamic 'Boolean' as Python bool during tracing`. Helpers may emit
branch-free code (`range_constexpr` loops, `.select()`, masked loads/stores); a runtime `if` belongs in
the decorated body. Settle this before planning a refactor that moves code into helpers.

**An `if` body is rewritten into its own function.** A value assigned inside the branch is not seen by a
closure that was defined before the `if` and is called inside the branch: the closure reads the old
value. This holds for a runtime `if` and also for a literal `if True:`. With `if const_expr(...)`, a
closure defined inside the branch, or the value passed as an argument, the closure saw the new value. A
name first assigned inside the branch raises `NameError` at trace time. Pass values into closures as
arguments.

### `AssertionError: unexpected ast node <ast.ClassDef ...>`
Raised by `ASTRewriter.transform`. The rewriter re-reads the kernel's source from disk with
`inspect.getsource`, using the line numbers recorded when the module was imported; if the file changed
since (an edit or sync under a long-running server or benchmark process), the lookup lands on whatever
now sits there — a decorated class, in the case that produced this message. It says nothing about the
kernel — restart the process, and keep sources
frozen for the lifetime of a process that compiles lazily.

### Loop-carried state typing
Pass FlyDSL values (`fx.Int32`, `fx.Float32`, `Vector`) straight into `init=` and `yield`: the rewriter
converts them with `as_ir_value` itself. Unwrap to a raw `ir.Value` only where a low-level helper
demands one. Every slot keeps one MLIR type across `init`, the per-iteration `state`, `yield` and the
post-loop results.

### `buffer_load` offset units
The offset is counted in units of `dtype`, not bytes. For FP8 data whose addresses you computed in
bytes, divide:

```python
k_addr_bytes = ...   # for FP8, elements == bytes
k_4xi32 = buffer_ops.buffer_load(k_rsrc, k_addr_bytes // 4, vec_width=4, dtype=T.i32)
```

Getting this wrong reads from a 4×-off address — which usually lands *inside* the buffer, so it fails
as garbage rather than as a fault.

### Vector stores need vector values
```python
Vec(scalar_i32).store(lds_ptr, [idx])                     # WRONG
Vec.from_elements([scalar_i32], fx.Int32).store(lds_ptr, [idx])   # CORRECT
```

---

## 6. Hang

### Loop bounds
`stop < start` under unsigned comparison, or `step == 0`, hangs the GPU. Print the bounds on the host
before launching — it costs nothing:

```python
print(f"loop: start={part_start}, stop={part_end}, step={cpb}")
```

### Divergent barrier
`gpu.barrier()` requires **every** thread in the workgroup to reach it. If a runtime `if` sends some
threads down another path, the barrier deadlocks. FlyDSL does not support divergent barriers — hoist
the barrier out of the conditional, do not try to make the condition uniform.

### Recovery
```bash
rocm-smi                # 100% busy with no progress confirms the hang
sudo amdgpu-reset       # or reboot
```

---

## The workflow, in order
1. Clear `~/.flydsl`. (Everything below is meaningless without this.)
2. All-1s input. Passes? The layout is probably fine and it is a data-dependent bug.
3. Single partition (one-shot). Passes? The bug is in partitioning or the reduce.
4. Host-side prints: shapes, strides, NaN counts.
5. Walk the intermediate buffers in order (`exp_sums` → `max_logits` → `temp_out`) and find the first
   one that is wrong.
6. Still suspecting layout? Trace one thread's addresses by hand — `tid=0` gives `lane16id=0`,
   `rowid=0`, `warp_id=0`, which is tractable on paper.
7. MFMA suspected? Check operand order: `mfma(LHS, RHS, acc)` maps LHS→M and RHS→N. For QK, K is the
   LHS and Q is the RHS.

## Checklist
- [ ] Cleared `~/.flydsl` after the last code change
- [ ] Edits outside the cache key (helper classes, other-directory helpers, closure dicts) re-run uncached
- [ ] Shape-dependent bounds and addresses come from traced arguments, not host integers
- [ ] `range_constexpr()` wherever the induction variable indexes a Python structure
- [ ] No `const_expr` on a runtime GPU value (`lane`, `warp_id`, `thread_id`)
- [ ] `buffer_load` offset units match `dtype` (bytes ÷ 4 for `i32`)
- [ ] Vector stores pass `Vector` values, not scalars
- [ ] Loop-carried state uses FlyDSL types, unwrapped only at hard boundaries
- [ ] Output written to the `part_z` slot, not the absolute partition index
- [ ] `exp_sums` / `max_logits` strides match the real tensor layout
- [ ] Softmax guards `-inf - (-inf)`
- [ ] Division guarded with `select(sum > 0, sum, 1.0)`
- [ ] K/V addressing matches the tensor rank (4-D vs 5-D `trans_v`)
- [ ] MFMA operand order verified — all-1s will not catch this one
