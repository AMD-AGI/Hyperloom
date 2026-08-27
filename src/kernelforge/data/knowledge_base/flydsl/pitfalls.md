# FlyDSL Pitfalls & Gotchas

> Collected from the upstream `.claude/skills/debug-flydsl-kernel/SKILL.md` plus
> empirical lessons from earlier ingest. **Always step 0**: clear caches.

## Step 0 (always): Clear caches first

FlyDSL aggressively caches compiled kernels. Stale cache is the #1 cause of
"my fix didn't work":

```bash
rm -rf ~/.flydsl /tmp/flydsl*
```

Disk cache auto-invalidates on traced-function source and closure-value
changes, but NOT on:
- Helper functions imported from another module
- C++ pass implementations (`lib/Conversion/...`)
- MLIR `.so` updates

When in doubt: `FLYDSL_RUNTIME_ENABLE_CACHE=0 python ...`. In-memory cache
remains active.

## Compilation Errors

### `range()` vs `range_constexpr()` inside `@flyc.kernel`
- `range_constexpr(N)` — Python `range`, fully unrolled at trace time; `i`
  is a Python `int`
- `range(N)` — lowered to `scf.ForOp` by the AST rewriter; `i` is an
  `ArithValue` (cannot index Python lists)

```python
# WRONG: i is ArithValue, can't index pylist
for i in range(4): result[i] = ...

# CORRECT: i is Python int
for i in range_constexpr(4): result[i] = ...
```

### `range(start, stop, step, init=[...])` silently ignored
If bounds are **Python ints**, the AST rewriter classifies as Python `range`
and unrolls — `init=` is dropped without warning.

```python
# WRONG: bounds are Python ints
for iv, state in range(0, N-1, 1, init=[...]): ...

# CORRECT: DSL Index values
for iv, state in range(fx.Index(0), fx.Index(N-1), fx.Index(1), init=[...]): ...
```

### Loop-carried state types
Prefer FlyDSL internal types (`fx.Int32`, `fx.Float32`, `Vector`,
`ArithValue`) for state. Unwrap only at hard MLIR boundaries:

```python
def _unwrap(v):
    return v.ir_value() if hasattr(v, "ir_value") else v

init_state = [_unwrap(v) for v in [val1, val2, vec_val]]
```

Supported state types: `f32` scalar, vector, `i32`, `i64`, `index`.

### `buffer_load` offset units
Offset is in elements of `dtype`, NOT bytes.

```python
# WRONG: passing byte offset when dtype=i32
data = buffer_ops.buffer_load(rsrc, byte_offset, vec_width=4, dtype=T.i32)

# CORRECT: divide by element size
data = buffer_ops.buffer_load(rsrc, byte_offset // 4, vec_width=4, dtype=T.i32)
```

### Vector stores need vector values
```python
# WRONG: scalar passed to Vector.store
Vec(scalar_i32).store(lds_ptr, [idx])

# CORRECT: build a 1-element vector
Vec.from_elements([scalar_i32], fx.Int32).store(lds_ptr, [idx])
```

### Runtime vs compile-time conditionals
Use plain Python operators (`==`, `<`, `>=`) for runtime SSA conditions — the
AST rewriter lowers to `scf.IfOp`:

```python
tid = gpu.thread_id("x")
lane = tid % fx.Index(64)

# Correct: readable DSL comparisons
if lane == fx.Index(0):
    fx.printf("lane zero")

in_range = lane < fx.Index(8)
val = in_range.select(good, zero)
```

Use `const_expr(...)` ONLY for compile-time decisions:

```python
if const_expr(trans_v):
    ...
if const_expr(max_partitions <= WARP_SIZE):
    ...
```

**Do NOT** wrap GPU runtime values in `const_expr`. Even with
`@flyc.kernel(known_block_size=(256, 1, 1))`, `gpu.thread_id("x")`, `lane`,
`warp_id` are runtime SSA — the compiler knows their range, not the current
lane:

```python
# WRONG
if const_expr(lane == 0): ...

# CORRECT
if lane == fx.Int32(0): ...
```

### Frontend semantic restrictions (single-value flow)

These look valid in plain Python but break MLIR construction:

1. **No def-in-branch-use-after-branch.** Hoist or merge:
   ```python
   # WRONG
   if cond:
       dst = a
   else:
       dst = b
   use(dst)

   # OK: ternary (single SSA value)
   dst = a if cond else b
   use(dst)
   ```

2. **No mutation of captured outer vars in nested helpers.** Pass+return:
   ```python
   def kernel():
       acc = fx.Float32(0.0)
       def helper(a):
           return a + fx.Float32(1.0)
       acc = helper(acc)
   ```

3. **No early `return` / branch-local `return`/`yield`.** Single exit:
   ```python
   if cond:
       out = v0
   else:
       out = v1
   return out
   ```

4. **Runtime branches with side effects** must be wrapped in a local
   `@flyc.jit` helper to keep MLIR result types well-defined:
   ```python
   @flyc.jit
   def dispatch():
       if runtime_cond:
           then_path()
       else:
           else_path()
   dispatch()
   ```

### `SmemPtr._view_cache` SSA dominance bug
`SmemPtr.get()` caches the view it creates. If the SmemPtr is used inside a
`range(...)` body, the cached view is defined in the loop scope. Using the
same SmemPtr in the epilogue (outside the loop) raises an SSA dominance
error. Fix:

```python
# After the runtime loop, before epilogue compute:
my_smem_ptr._view_cache = None
```

## MFMA / GEMM Layout Bugs

### MFMA operand order
`mfma(LHS, RHS, acc)` — LHS → M dimension, RHS → N dimension. Swapping
silently produces a transposed result. **Always** verify by tracing one
thread's address calculation.

### Output layout depends on operand assignment
For `mfma_f32_32x32x16_bf16`, the per-lane `C[reg_idx]` maps to:
```
C_col = lane_mod_32       (fixed)
C_row = lane_div_32 * 4 + (reg_idx // 4) * 8 + (reg_idx % 4)
```
BUT each kernel's operand assignment (A, B, C) changes interpretation.
fwd's `mfma(V, P_transposed, O)` is different from bwd's `mfma(K, Q, S)`.
**DO NOT copy fwd's labeled-transpose pattern to bwd** — derive from first
principles.

### LSE domain mismatch
Triton stores LSE in scaled-log2 domain: `qk * scale * log2e`.
FlyDSL's `m_running` is raw-qk domain.
Epilogue must compute: `lse = (m * scale * log2e) + log2(sum)`.
Mismatch produces "close but wrong" output — looks like 20–25 dB SNR
instead of >30.

## NaN / Inf / Zero Output Debug

### Softmax NaN: `-inf - (-inf)`
When ALL tokens in a partition are masked, `qk_max = -inf`; then
`exp(s - qk_max) = exp(-inf - (-inf)) = exp(NaN) = NaN`.

```python
safe_diff = (qk_max > NEG_INF).select(diff, ZERO_F)
```

### Division by zero in normalization
```python
safe_sum = (running_sum > ZERO_F).select(running_sum, fx.Float32(1.0))
inv_sum = fx.Float32(1.0) / safe_sum
```

### All-zeros output → addressing or strides
- Wrong `stride_out_seq` / `stride_out_part`
- Partition slot mismatch (output written to absolute partition index
  instead of `part_z` slot — reduce kernel reads `part_z = 0..grid_z-1`)
- `exp_sums` / `max_logits` uninitialized → reduce kernel produces zeros.
  Init sentinel values on host before launch:
  ```python
  exp_sums.fill_(-999.0)
  ```

### Large mismatch (>50%) → missing partitions or layout
- All-1s isolation: `query.fill_(1.0); key_cache.fill_(1.0); value_cache.fill_(1.0)` — uniform input should give uniform output regardless of math; deviation reveals layout/addressing bug.
  Caveat: does NOT catch V/P operand misalignment.
- Single-partition test: force `max_context_partition_num=1` (one_shot) to
  bypass reduce kernel and test main kernel in isolation.
- `grid_z` vs `total_partitions`: if `grid_z < total_partitions` and the
  kernel doesn't loop over partitions, most context is skipped.
- Compare element-wise with a Gluon or Triton reference:
  `torch.testing.assert_close(flydsl_out, ref, atol=5e-3, rtol=5e-3)`.

### Small errors (1–5%) → FP8 / per-tensor scale issues
- FP8 PV MFMA introduces ~0.03 max error vs bf16 reference — inherent, not a bug. Tolerance: `atol=5e-3`.
- Per-tensor vs per-row Q quantization: ~1-3% mismatch if mismatched.
- Scale factor: verify `_scale = softmax_scale * q_scale * k_scale` matches
  reference. Common bug: applying `v_scale` twice (once in prob, once after PV).

## GPU Hang

### Infinite runtime loop
If `stop < start` (unsigned cmp bug) or `step=0`, the GPU hangs. Verify on
host before launch:

```python
print(f"loop: start={part_start}, stop={part_end}, step={cpb}")
```

### Barrier deadlock
`gpu.barrier()` requires ALL threads in the workgroup to reach it. If some
threads take a different branch in a runtime `if`, the barrier deadlocks.
FlyDSL doesn't support divergent barriers.

### Recovery
```bash
rocm-smi                    # check GPU state
sudo amdgpu-reset           # or reboot
```

## WAVES_PER_EU Tuning

- `waves_per_eu` must be **re-tuned per kernel**:
  - Dense flash-attn: `wpe=3` optimal
  - Sparse SLA fwd: `wpe=2` optimal (`wpe=3` is **1.389× slower**)
- **NEVER inherit `wpe`** from another kernel — always measure both.
- Anti-composition: `wpe=3 + ds_bpermute` regresses on BOTH dense and sparse.
  Either knob alone fine; test combinations explicitly.
- Set via `Config(waves_per_eu=N)` in `@autotune` — going through
  `gpu-module-to-binary opts=` may not propagate (known limitation; use
  the `rocdl-attach-target` channel which the Config does).

## Tensor Layout Gotchas

### `.clone()` preserves non-contiguous strides
`.transpose().clone()` does NOT produce a contiguous tensor. `.clone()`
defaults to `preserve_format`.
- **Symptom**: hidden `.contiguous()` HBM copy at first use (900 µs on MI355X)
- **Fix**: `tensor.contiguous()` or `tensor.clone(memory_format=torch.contiguous_format)`

### Mark dynamic dims explicitly when known
```python
adaptor = flyc.from_dlpack(tensor).mark_layout_dynamic(
    leading_dim=0, divisibility=4
)
```
But: see Autotune caveat — don't pass pre-wrapped `from_dlpack` adaptors
across `ir.Context` switches with varying `Constexpr` (segfault).

## Build / Module Switch Gotchas

### Stale MLIR artifacts after pass-source edits
FlyDSL generates MLIR from Python DSL. Editing `kernels/*.py` requires re-import
to retrigger compilation. Stale `.o` artifacts persist across ninja runs.
**Fix**: `rm -f` stale objects, then re-import.

### Module-switching cache eviction
A CK kernel measured 92 µs in pure-CK chain but 99 µs in FlyDSL→CK chain due
to instruction cache eviction at the backend boundary.
- **Impact**: killed the hypothesis that mixing FlyDSL+CK would save 7 µs
- **Rule**: never trust cross-backend chain decomposition from homogeneous baselines.

## Diagnostic Workflow

```
1. Clear caches:  rm -rf ~/.flydsl
2. All-1s input → passes?       Layout OK; data issue
3. Single partition (one_shot) → passes? Multi-partition/reduce bug
4. Add host prints (shapes, strides, NaN counts)
5. Compare intermediate buffers (exp_sums, max_logits, temp_out)
6. Layout bug suspected: trace one thread manually
   (tid=0: lane16id=0, rowid=0, warp_id=0)
7. MFMA bug: verify operand order (K=LHS, Q=RHS for QK gemm)
```

## Checklist

- [ ] Cleared `~/.flydsl` cache after code change
- [ ] `range_constexpr()` for all compile-time loops (not `range()`)
- [ ] No Python `if` on `const_expr`-wrapped GPU runtime values
- [ ] `buffer_load` offset units match dtype (bytes/4 for i32)
- [ ] Vector stores use `Vector` values (not scalars)
- [ ] `range(..., init=...)` bounds are `fx.Index(...)` (not Python int)
- [ ] State in `range(..., init=...)` uses internal types, unwrapped only at hard boundaries
- [ ] Output written to correct partition slot (`part_z`, not absolute index)
- [ ] `exp_sums` / `max_logits` strides match actual tensor layout
- [ ] Softmax guards against `-inf - (-inf) = NaN`
- [ ] Division by zero guarded (`select(sum > 0, sum, 1.0)`)
- [ ] K/V address calculation matches tensor layout (4D vs 5D `trans_v`)
- [ ] MFMA operand order: `mfma(LHS, RHS, acc)` — LHS→M, RHS→N
- [ ] Single explicit definition path through if/else (no def-in-branch)
- [ ] `SmemPtr._view_cache = None` before epilogue if used in loop body
