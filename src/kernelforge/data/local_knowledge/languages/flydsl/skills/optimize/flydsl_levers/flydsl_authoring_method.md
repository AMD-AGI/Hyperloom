---
title: FlyDSL — authoring a kernel, structure before parameters
kind: language
lever: flydsl_authoring_method
gens: [gfx950]
updated: 2026-08-28
---

# Authoring method

> **Reference (how-to), not a verdict.** The winner is decided by on-box measurement against the
> kernel's own oracle.

**Path B: you are writing a new `@flyc.kernel`.** For *using* the shipped library see
`flydsl_kernel_library.md` and `flydsl_knob_space.md`.

## The one rule

**Parameter tuning alone yields marginal gains. Do structural work in early patches and fall back to
tuning in later ones.** A kernel whose structure is wrong cannot be tuned into a good one — you will
just find the best member of a bad family.

## Route here when
Optimizing an authored FlyDSL kernel and the bottleneck is still broad. If the discussion has already
narrowed to `tile_m`/`tile_n`/`tile_k`, MFMA-loop ISA counts, or epilogue strategy on a clearly
GEMM-like kernel, go to `flydsl_gemm_authoring.md`.

## Step 1 — read before you optimize

1. The **full kernel source** — every `@flyc.kernel`, its algorithm, data flow, loop structure.
2. The **`@flyc.jit` host wrapper** — how many kernels launch per call and what data each receives.
   **Multiple kernels sharing data is a fusion opportunity.**
3. **Imported helpers** (`flydsl.utils`, `flydsl.expr`) — reusable building blocks.
4. The **test harness** — which shapes, dtypes and modes are actually benchmarked.
5. If you plan to rewrite loops or memory paths, review `range_constexpr()` vs `range(..., init=...)`,
   `buffer_ops`, and `SmemAllocator` semantics **before** editing.

## Step 2 — classify, and know when to stop

Check the arch via `get_hip_arch()` — LDS size, MFMA variants and wavefront width are arch-dependent
(**gfx950: 160 KiB LDS, 64 banks, 256 CUs, wave64**).

| Class | Lever |
|---|---|
| Memory-bound **with** cross-thread reuse (tiled GEMM operands, attention K/V tiles) | LDS staging, prefetch, vectorization |
| Memory-bound **without** cross-thread reuse (elementwise, RoPE, cache-write-only) | **vectorization and coalescing only** |
| Compute-bound | MFMA selection, software pipelining |
| Latency-bound (small shapes) | reduce launch count — fusion |

**Two stop conditions that save whole sessions:**

- **No cross-thread reuse → do not add LDS staging.** Each thread owns its slice; LDS adds
  synchronization and address math with zero reuse benefit.
- **Already fused** (name contains `fused_` / `*_2stage` / `*_multistage`, or the source already
  combines the ops) → **the structural optimum is likely reached.** Skip to Tier 2/3. Further fusion
  attempts here are the classic wasted patch.

## Step 3 — the four tiers, in order

### Tier 1 — structural (highest impact, highest regression risk)

**Guard: confirm the kernel is not already at the structural optimum before touching it.**
Single-kernel, single-pass, or already `fused_*` → skip to Tier 2.

- **Kernel fusion** — if the `@flyc.jit` wrapper launches 2+ kernels sharing input data, merge them.
  Removes launch overhead and redundant HBM reads.
- **Fast-path relaxation** — look for over-restrictive guards on optimized paths (disabled branches,
  alignment checks stricter than necessary). Relaxing them lets more shapes take the fast path.
- **Loop restructuring** — convert constexpr unrolling to `scf.for` with loop-carried state **when**
  unrolling causes measurable code bloat or register pressure. **Do not** convert when unrolling is
  your main source of ILP or the trip count is small and fixed.
- **Redundant work elimination** — repeated loads, recomputed indices, overlapping branches.
- **Algorithm replacement** — reduce pass count (online softmax vs two-pass; fused attention vs
  separate QKᵀ → softmax → ×V). **A single-pass elementwise kernel is already optimal at the pass
  level — do not add stages.**

**FlyDSL refactor guardrails**
- `range_constexpr()` is for compile-time unrolling only. Runtime-carried state needs
  `range(..., init=...)` so FlyDSL lowers to `scf.for`.
- `scf.for` bounds must be `arith.index()` values, not Python ints; `init`/`yield` values must be raw
  MLIR `ir.Value`s.
- Keep loop-carried state **positionally aligned** across `init`, per-iteration `state`, `yield`, and
  post-loop results — same slot, same meaning, same MLIR type throughout.
- If an `SmemPtr` view created inside a loop body is reused in the epilogue, **clear `_view_cache`**
  before reusing it outside the loop, or you get SSA dominance errors.

### Tier 2 — memory hierarchy

- **LDS staging** — only when the same global data is read multiple times **across threads in the same
  workgroup**. Use `SmemAllocator` / `SmemPtr` from `flydsl.utils.smem_allocator`.
- **Vectorized access** — widest loads/stores matching the element type (`vec(8, ...)`, `vec(4, ...)`).
- **Overlap loads with compute** — move global loads earlier; `sched_barrier` to control interleaving.
  **Only helps when there is MFMA or non-trivial ALU work to overlap with.**
- **Pre-load across passes** — load later-pass data during earlier passes.
- **Coalescing** — restructure loop ordering if the access pattern is not coalesced.
- **Register pressure** — balance registers against LDS spilling.

**Prefetch: when not to.** If the loop body is dominated by global loads with minimal compute, **do not
add prefetch** — there is nothing to hide behind, and the extra carried state raises register pressure
and can cut occupancy for no latency benefit.

**The prefetch shape**: prologue preloads iteration 0 → `scf.for` carries the prefetched values → the
body unpacks current state and issues next-iteration loads **immediately** → epilogue consumes the
final carried values. **Carry everything** needed to materialize the next iteration: not just tensor
payloads but block-table entries, page IDs, scale values, running accumulators. Re-check the register
budget before adding buffers.

### Diagnose LDS from the trace, not from intuition

Three different problems that look alike and want different fixes:

| Signal | Problem | Fix |
|---|---|---|
| Stall on `ds_read_*` / `ds_write_*` themselves | **bank conflicts** | swizzle or padding |
| `s_waitcnt lgkmcnt(0)` spikes right after `ds_write` | **write-read latency exposed** | increase the distance |
| `s_barrier` dominates a reduce/broadcast region | **cross-wave serialization** | fewer barrier stages, cheaper cross-lane primitives |

**Do not treat all three as "a swizzle problem."** Use these metrics twice: before the rewrite to
classify, after the rewrite to confirm the targeted stall actually moved.

**gfx950 LDS: 160 KiB, 64 banks.** A layout that fully aliased banks on a 32-bank part may only
partially conflict here — **swizzle masks and padding must be arch-aware, and inherited ones are
unverified.** The extra headroom also makes padding affordable where it was not before.

**Swizzle vs padding**: XOR swizzle when the access pattern is regular, read/write transforms can stay
consistent, and LDS headroom is tight. Padding when the swizzle math would hurt maintainability and a
small stride change breaks the pattern cleanly. **Either way, keep producer and consumer consistent —
a swizzled store with a linear load is a correctness bug, not a perf trade.**

**Increase write-read distance before adding structure.** If the stall is `lgkmcnt` right after
`ds_write`, first try moving independent work in between:

```python
# BEFORE: write immediately followed by barrier/read
lds_ptr.store(data, [offset])
fx.gpu.barrier()
value = lds_ptr.load([offset])

# AFTER: independent work before the synchronization point
lds_ptr.store(data, [offset])
next_offsets = compute_next_offsets()
next_data = buffer_ops.buffer_load(next_rsrc, next_offsets, vec_width=4, dtype=fx.T.f32())
fx.gpu.barrier()
value = lds_ptr.load([offset])
```

Do not insert work that depends on the just-written value, or extra LDS traffic competing for the same
bottleneck.

### Tier 3 — compute
- **MFMA selection** — the most efficient variant for the arch, via `flydsl.expr.rocdl`.
- **Software pipelining** — ping-pong buffers + scheduler barriers.
- **Scheduler tuning** — match `sched_mfma` group counts to the **actual** MFMA ops per iteration;
  verify `sched_dswr`/`sched_dsrd` timing. Copied constants from another kernel are worse than none.
- **Loop unrolling** — expose ILP; merge loops over the same range.

### Tier 4 — parameters (lowest impact)
Block size, tile dimensions, unroll factors, `known_block_size` hints.

**Tune only after structure stabilizes.** Treat old tuning conclusions as **stale** after any
codegen-affecting refactor. When varying `Constexpr` values across recompiles, pass raw
`torch.Tensor` objects rather than reusing cached `flyc.from_dlpack()` wrappers.

## Modification rules
- **Fusion**: you may create new `@flyc.kernel` functions, remove old ones, and modify the `@flyc.jit`
  wrapper to launch the fused kernel.
- **Everything else**: modify code inside `@flyc.kernel` functions and their kernel-internal helpers.
- The `@flyc.jit` **external signature** (as called by the harness) must not change.
- **Do not modify** the build system, compilation flags, test harness, or benchmark framework.

## Correctness constraints — violations corrupt silently

| Constraint | Why |
|---|---|
| **LDS limit** per `get_hip_arch()` (gfx950: 160 KiB) | exceeding it **silently corrupts results** |
| `tile_k_bytes % 64 == 0`; `tile_m · tile_k · elem_bytes` divisible by thread count | tile divisibility |
| **fp8 `0x80` is NaN in the FNUZ encoding** — sanitize loads with byte AND `0x7F` | gfx950 is OCP; this applies when handling FNUZ-encoded data |
| f32→f16: **clamp to ±65504 first** | otherwise Inf |
| Vector/tile alignment | matches the kernel's access patterns |

**FlyDSL memory contracts**
- `buffer_ops.buffer_load` / `buffer_store` offsets are in **elements, not bytes.** Recompute address
  units whenever a rewrite changes dtype, packing or vector width.
- Packed FP8/INT4 reinterpreted through `dtype=T.i32` — divide byte addresses by the new element width.
- New or resized LDS allocations go through `SmemAllocator`, and `allocator.finalize()` must still
  happen in the GPU module body.
- Moving `SmemPtr` views across loop/region boundaries — re-check cached-view lifetime and dominance.

## Step 4 — validate

1. **Correctness first.** Never trade it for speed.
2. Confirm speedup across **all** tested shapes, not one.
3. **Report speedup ratio AND absolute optimized time (ms).** A better ratio with a *worse* absolute
   time means the baseline shifted, not that the kernel improved — **treat an absolute regression as a
   failure even when the ratio looks better.**
4. For structural rewrites, dump with `FLYDSL_DUMP_IR=1` and inspect the relevant `.mlir` stage plus
   `final_isa.s`.
5. **Verify the specific effect you wanted**: `scf.for` survived tracing, wide loads stayed vectorized,
   the MFMA variant matches dtype/arch, the loop shape reflects the intended schedule.
6. **If the generated form did not change, the optimization did not land** — no matter how right the
   Python looks.
7. If the speedup is marginal or the absolute time regresses, move to the **next structural strategy**
   rather than re-tuning the same approach.

## Common mistakes
- Starting from scheduler constants before proving the bottleneck.
- Copying tile sizes from another kernel without checking work decomposition.
- Multi-stage LDS buffering that destroys occupancy.
- Treating every LDS issue as a swizzle problem instead of checking wait distance.
- Overfitting to one benchmark shape.
- Assuming a trace/ISA pattern from another repository matches this kernel.

## Key APIs
- Device kernel `@flyc.kernel` · host launcher `@flyc.jit`
- Control flow: `range_constexpr()` · `range(..., init=...)` · `arith.index()`
- Intrinsics: `flydsl.expr.rocdl` — MFMA, exp2, rcp, `sched_barrier`, `sched_mfma`
- Shared memory: `SmemAllocator` / `SmemPtr` from `flydsl.utils.smem_allocator`
- Types: `T.f16`, `T.bf16`, `T.f32`, `T.i32`, `T.vec(...)` from `flydsl.expr.typing`
- Buffer ops: `fx.rocdl.make_buffer_tensor`, `fx.make_copy_atom`, `buffer_ops.buffer_load`/`buffer_store`
- IR/ISA: `FLYDSL_DUMP_IR=1`, `FLYDSL_DUMP_DIR=...`, `final_isa.s`
- Autotune: `flydsl.autotune.autotune`, `Config`, `do_bench`

## Related
`flydsl_gemm_authoring.md` (GEMM-specific follow-on) ·
`../../../API_docs/flydsl-tile-programming.md` (first-time authoring) ·
`../../bottleneck/debug-flydsl-kernel.md` (correctness bugs) ·
`hardware/mi350_lds.md` · `hardware/mi350_execution.md`
