---
name: debug-flydsl-kernel
description: >
  Diagnose a FlyDSL kernel that is wrong, NaN, zero, hanging, or refusing to compile.
  Symptom-indexed: start from what you observed, land on the cause. Covers the stale-cache
  false negative, softmax -inf arithmetic, partition/stride addressing, FP8 error budgets,
  the range vs range_constexpr split, loop-carried state typing, buffer_load offset units,
  and barrier divergence. Use when a FlyDSL kernel produces incorrect output or fails to build.
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

## Start from the symptom
| What you observe | Most likely cause | Section |
|---|---|---|
| Output is all NaN | `-inf` minus `-inf` in softmax, or divide by zero | [§1](#1-all-nan) |
| Output is all zeros | wrong output address, or an unwritten intermediate | [§2](#2-all-zeros) |
| More than half the elements are wrong | missing partitions, or a layout/addressing mismatch | [§3](#3-mostly-wrong) |
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
Do not guess which buffer went bad — print them:

```python
torch.cuda.synchronize()
print(f"exp_sums   nan={exp_sums.isnan().sum()}  inf={exp_sums.isinf().sum()}")
print(f"max_logits nan={max_logits.isnan().sum()}  range=[{max_logits.min():.4f}, {max_logits.max():.4f}]")
print(f"temp_out   nan={temporary_output.isnan().sum()}")
```

The first buffer in the chain that contains NaN is where to look. A NaN in `max_logits` and a NaN in
`temp_out` are different bugs.

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

### `range()` versus `range_constexpr()`
The AST rewriter turns a runtime `range()` into an MLIR loop, so the induction variable becomes an
`ArithValue` and can no longer index a Python list. Compile-time loops need `range_constexpr`:

```python
for i in range(4):            # WRONG: i is an ArithValue
    result[i] = ...

for i in range_constexpr(4):  # CORRECT: i is a Python int
    result[i] = ...
```

### Runtime versus compile-time conditionals
Runtime comparisons in a Python `if` are supported — the rewriter lowers dynamic conditions to
`scf.IfOp`. Write them with DSL operators, not hand-built MLIR predicates:

```python
tid  = gpu.thread_id("x")
lane = tid % fx.Index(64)

if lane == fx.Index(0):                       # lowered to scf.IfOp
    fx.printf("lane zero")

val = (lane < fx.Index(8)).select(good, zero) # runtime predicate for select
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

### Loop-carried state typing
Keep loop-carried state in FlyDSL's own types (`fx.Int32`, `fx.Float32`, `Vector`, `ArithValue`) and
unwrap only where a low-level helper demands a raw `ir.Value`:

```python
def _unwrap(v):
    return v.ir_value() if hasattr(v, "ir_value") else v

init_state = [_unwrap(v) for v in [val1, val2, vec_val]]
```

Supported state types: `f32` scalar, vector values, `i32`, `i64`, `index`.

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
- [ ] `range_constexpr()` for every compile-time loop
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
