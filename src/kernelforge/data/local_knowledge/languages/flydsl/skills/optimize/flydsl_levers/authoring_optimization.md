---
title: "FlyDSL — authored-kernel optimization workflow (structure-first)"
kind: language
gens: [gfx942, gfx950]
updated: 2026-06-01
---

> **Reference (how-to), not a verdict.** Ingested from the FlyDSL authoring skill. It gives a
> prioritization workflow for optimizing an *authored* `@flyc.kernel`; the winner is still decided by
> on-box measurement against the kernel's own oracle. For *using* aiter's built-in flydsl GEMM library
> (knobs of `flydsl_hgemm`, dispatch, preshuffle) see [`knobs.md`](knobs.md) / [`patterns.md`](patterns.md).

# FlyDSL authored-kernel optimization workflow

Parameter tuning alone yields marginal gains. **Prioritize structural optimizations in early patches**, then fall back to tuning in later patches.

## Step 1: Read before you optimize

1. Read the **full kernel source** — all `@flyc.kernel` functions, their algorithm, data flow, and loop structure
2. Read the **`@flyc.jit` host wrapper** — count how many kernels are launched per call and what data each receives. Multiple kernels sharing data = fusion opportunity
3. Read **any imported helper modules** (e.g. `flydsl.utils`, `flydsl.expr`) — they contain reusable building blocks and may reveal optimization opportunities
4. Read the **test harness** — know what shapes, dtypes, and modes are benchmarked
5. If you plan to rewrite loops or memory paths, quickly review the relevant FlyDSL semantics for `range_constexpr()` vs `range(..., init=...)`, `buffer_ops`, and `SmemAllocator` before editing

## Step 2: Classify the bottleneck

- Check the **target GPU arch** via `get_hip_arch()` — LDS size, MFMA variants, and wavefront width are arch-dependent (e.g. gfx942: 64 KB LDS, 304 CUs, wavefront 64)
- **Memory-bound with cross-thread reuse** (e.g. tiled GEMM operands, attention K/V tiles) → LDS staging, prefetch, vectorization
- **Memory-bound without cross-thread reuse** (e.g. pure elementwise ops, RoPE position encoding, cache-write-only passes) → vectorization and coalescing **only**; LDS staging adds overhead without benefit because each thread processes its own independent slice. Do not add LDS staging or structural rewrites to this class.
- **Compute-bound** → improve instruction throughput (MFMA selection, software pipelining)
- **Latency-bound** (small shapes) → reduce kernel launch count (fusion)
- **Already-fused kernel** (kernel name contains `fused_`, or the source already combines multiple ops in one `@flyc.kernel`) → the structural optimum is likely already reached; prioritize Tier 2 (vectorization, coalescing) and Tier 3 (MFMA, scheduling) instead of further fusion or loop restructuring
- **GEMM-like kernel** → if the next optimization question is mainly about `tile_m` / `tile_n` / `tile_k`, GEMM-specific LDS staging, epilogue strategy, or MFMA-loop ISA counts, use this guide for generic prioritization and then switch to [`authoring_gemm_levers.md`](authoring_gemm_levers.md)

## Step 3: Optimize — high impact first

### Tier 1: Structural (highest impact)

**Guard**: Tier 1 rewrites carry the highest regression risk. Before applying any structural change, confirm that the kernel is NOT already at the structural optimum: single-kernel, single-pass, or already named `fused_*` / `*_2stage` / `*_multistage`. For those, skip to Tier 2.

- **Kernel fusion**: if the `@flyc.jit` wrapper launches 2+ kernels that share input data, merge them into a single `@flyc.kernel`. Eliminates launch overhead and redundant HBM reads. Do not attempt further fusion on a kernel that is already named `fused_*` or already combines all relevant ops.
- **Fast-path relaxation**: look for overly-restrictive conditions guarding optimized code paths (e.g. disabled branches, alignment checks stricter than necessary). Relaxing these lets more shapes use the fast path
- **Loop restructuring**: if the kernel uses constexpr loop unrolling that causes measurable code bloat or register pressure for large iteration counts, convert to `scf.for` with loop-carried state. Do not convert when unrolling is the primary source of ILP or the iteration count is small and fixed — converting then adds overhead rather than saving it.
- **Redundant work elimination**: identify repeated loads, recomputed indices, or overlapping branches, and hoist/cache them
- **Algorithm replacement**: if the current algorithm has unnecessary passes over data, restructure to reduce pass count (e.g. online softmax vs two-pass, fused attention vs separate Q×K then softmax then ×V). For single-pass elementwise kernels, the algorithm is already optimal at the pass level; do not add passes or stages.

#### FlyDSL refactor guardrails

- Use `range_constexpr()` only for compile-time unrolling. If the optimization needs runtime-carried state, use `range(..., init=...)` so FlyDSL lowers the loop to `scf.for`
- For `scf.for` rewrites, loop bounds must be `arith.index()` values, not Python ints, and `init` / `yield` values must be raw MLIR `ir.Value`s
- Keep loop-carried state positionally aligned across `init`, per-iteration `state`, `yield`, and post-loop results so each slot keeps the same semantic meaning and MLIR type through the whole loop
- If an `SmemPtr` view created inside a loop body is reused in the epilogue, clear `_view_cache` before reusing it outside the loop to avoid SSA dominance errors

### Tier 2: Memory hierarchy (medium impact)

- **LDS utilization**: if the kernel reads the same global data multiple times **across threads within the same workgroup**, stage through LDS for reuse. Use `SmemAllocator` / `SmemPtr` from `flydsl.utils.smem_allocator`. **Do not add LDS staging for kernels where each thread processes its own independent slice with no block-level data sharing** (e.g. pure elementwise, RoPE position encoding, cache-write-only passes) — LDS adds synchronization and address overhead with no reuse benefit.
- **Vectorized access**: use the widest vector loads/stores (`vec(8, ...)`, `vec(4, ...)`) that match the element type for maximum HBM bandwidth
- **Overlap loads with compute**: move global loads earlier so they complete while ALU/MFMA work is in progress. Use scheduler barriers (`sched_barrier`) to control interleaving. This is only beneficial when MFMA or non-trivial ALU work exists to overlap with; for load-dominated kernels this produces no gain.
- **Pre-load across passes**: if the kernel makes multiple passes, load data needed in later passes during earlier ones to avoid redundant HBM reads
- **Data layout / coalescing**: ensure memory access patterns are coalesced; restructure loop ordering if needed
- **Register pressure management**: balance between keeping data in registers vs spilling to LDS

#### FlyDSL prefetch rewrite pattern

- Use prefetch when the loop has enough independent MFMA/ALU work, or a later barrier-heavy phase, to hide the next global load. **If the loop body is dominated by global loads with minimal MFMA or ALU compute, do not add prefetch** — there is no compute to hide behind, and the extra loop-carried state increases register pressure and can reduce occupancy without any latency benefit.
- Follow a real loop-carried structure: prologue preloads iteration 0, `scf.for` carries the prefetched values, the loop body unpacks current state and issues next-iteration loads immediately, and the epilogue consumes the final carried values.
- Carry every value needed to materialize the next iteration together — not just tensor payloads, but also block-table entries, page IDs, scale values, and running accumulators.
- Keep the swap/prefetch path simple: unpack from `state`, issue the next loads as early as legality allows, and leave the load-to-consume distance for compute or barrier wait to hide.
- If a later phase already spends time in `gpu.barrier()` or reduce synchronization, hoist the next phase's global loads into that region when legality allows; barrier wait is often the easiest place to hide VMEM latency.
- Re-check register budget before adding prefetch buffers. Prefetch only helps when the extra carried state does not cause spills or unacceptable occupancy loss.

#### Diagnose LDS bottlenecks before rewriting

- Diagnose from trace shape, not intuition: high stall on `ds_read_*` / `ds_write_*` themselves points to bank conflicts; high `s_waitcnt lgkmcnt(0)` or barrier immediately after `ds_write` points to exposed write-read latency; barrier-heavy reduce chains point to cross-wave serialization.
- On gfx942, think in 32 LDS banks; on gfx950, think in 64. A stride/layout that fully aliases banks on gfx942 may only partially conflict on gfx950, so swizzle masks and padding choices must be arch-aware.
- Prefer XOR swizzle when you need zero LDS overhead and can keep read/write address transforms consistent. Use padding when swizzle is awkward to integrate and LDS headroom is available.
- If the hotspot is write-read latency, increase the distance between `ds_write` and the dependent `ds_read` / wait by moving independent address computation, global loads, or MFMA work between them.
- If the hotspot is barrier/reduce serialization, reduce unnecessary `gpu.barrier()` stages, merge LDS reduce phases when possible, or switch to cheaper cross-lane / cross-wave primitives before adding more LDS traffic.

#### LDS quick architecture reference

| Arch | LDS size per CU | Banks | Full-conflict stride heuristic |
|------|------------------|-------|-------------------------------|
| `gfx942` | 64 KB | 32 | multiples of 128 bytes often fully alias banks |
| `gfx950` | 160 KB | 64 | multiples of 256 bytes often fully alias banks |

Two practical consequences:

- the same layout may conflict badly on `gfx942` but only partially on `gfx950`
- padding choices that are too expensive on `gfx942` may be acceptable on `gfx950`

#### Classify the LDS problem precisely

Use the trace or ISA shape to separate these cases:

- **Bank conflicts**: `ds_read_*` / `ds_write_*` instructions themselves carry the stall
- **Write-read latency exposed**: `s_waitcnt lgkmcnt(0)` spikes immediately after `ds_write`
- **Cross-wave serialization**: `s_barrier` dominates a reduce or broadcast region

Treat these metrics in two stages:

- **Before the rewrite**: use them to classify which LDS problem dominates
- **After the rewrite**: use the same metrics to verify that the targeted bottleneck actually moved

Do not treat all three as the same issue. Each one wants a different fix.

#### Swizzle vs padding

Use **XOR swizzle** when:

- the read and write paths both use a regular logical row/column mapping
- you need to reduce bank conflicts without increasing LDS footprint
- the address transform is still easy to audit for correctness

Use **padding** when:

- the swizzle math would make the code materially harder to maintain
- a small stride change breaks the conflict pattern cleanly
- LDS headroom is available on the target arch

For either strategy, keep producer and consumer paths consistent. A swizzled
store paired with a linear load is a correctness bug.

#### Increase write-read distance before adding more structure

If the stall is on `lgkmcnt` immediately after `ds_write`, first try reordering
useful work between the write and the dependent read:

- next-phase global loads
- address calculation for the next iteration
- independent MFMA or ALU work
- epilogue preparation that does not depend on the just-written LDS data

At the FlyDSL level, the pattern is:

```python
# BEFORE: write is followed by an immediate barrier/read
lds_ptr.store(data, [offset])
fx.gpu.barrier()
value = lds_ptr.load([offset])

# AFTER: insert independent work before the synchronization point
lds_ptr.store(data, [offset])
next_offsets = compute_next_offsets()
# For example, issue the next global load here if it does not depend on the LDS write.
next_data = buffer_ops.buffer_load(next_rsrc, next_offsets, vec_width=4, dtype=fx.T.f32())
fx.gpu.barrier()
value = lds_ptr.load([offset])
```

Avoid inserting work that depends on the just-written LDS value, or extra LDS
traffic that competes with the same bottleneck you are trying to hide.

#### LDS verification checklist

After an LDS-focused rewrite, verify all of these:

- correctness still holds on the same benchmark shapes
- `ds_read_*` / `ds_write_*` stall decreased if you targeted bank conflicts
- `s_waitcnt lgkmcnt(0)` stall decreased if you targeted write-read latency
- barrier count or barrier stall decreased if you targeted cross-wave serialization
- total LDS bytes still fit the target arch budget
- added address math or prefetch state did not create a register-pressure regression

#### FlyDSL memory rewrite contracts

- `buffer_ops.buffer_load` / `buffer_store` offsets are in **elements**, not bytes. Recompute address units whenever the rewrite changes `dtype`, packing, or vector width
- If packed FP8/INT4 data is reinterpreted through `dtype=T.i32`, divide byte addresses by the new element width before loading
- New or resized LDS allocations must still go through `SmemAllocator`, and `allocator.finalize()` must still happen in the GPU module body
- Any change that moves `SmemPtr` views across loop or region boundaries should re-check cached-view lifetime and SSA dominance

### Tier 3: Compute (medium impact)

- **MFMA instruction selection**: use the most efficient variant available on the target arch (via `flydsl.expr.rocdl`)
- **Software pipelining**: overlap LDS reads/writes with MFMA compute using ping-pong buffers and scheduler barriers
- **Scheduler tuning**: match `sched_mfma` group counts to actual MFMA ops per iteration; verify `sched_dswr`/`sched_dsrd` timing
- **Loop unrolling**: unroll inner loops to expose ILP; merge loops that iterate over the same range

### Tier 4: Parameter tuning (low impact)

- Block size, tile dimensions, unroll factors, `known_block_size` hints

#### Tune after structure stabilizes

- Do structural fixes first: control-flow rewrite, memory-path rewrite, LDS strategy, MFMA selection, and correctness
- Only after semantics and codegen stabilize should you tune block size, tile shape, unroll factors, `known_*` hints, or expose knobs to `@autotune`
- Treat old tuning conclusions as stale after codegen-affecting refactors
- When varying `Constexpr` values across recompiles, prefer passing raw `torch.Tensor` objects rather than reusing cached `flyc.from_dlpack()` wrappers

## Modification rules

- **For kernel fusion**: you MAY create new `@flyc.kernel` functions, remove old ones, and modify the `@flyc.jit` wrapper to launch the fused kernel
- **For all other optimizations**: modify code inside `@flyc.kernel` functions and their kernel-internal helper functions
- The `@flyc.jit` function's **external signature** (parameters as called by the test harness) must remain unchanged
- Do NOT modify: build system, compilation flags, test harness, or benchmark framework

## Correctness constraints

Always verify after each patch — violations often cause silent corruption:
- **LDS limit**: check `get_hip_arch()` for the arch-specific limit (e.g. 64 KB on gfx942) — exceeding it silently corrupts results
- **Tile divisibility**: `tile_k_bytes % 64 == 0`; `tile_m * tile_k * elem_bytes` divisible by thread count
- **FP8 (E4M3FNUZ)**: value `0x80` is NaN — sanitize loads with byte AND `0x7F`
- **f32→f16 truncation**: clamp to ±65504 first to avoid Inf
- **Alignment**: ensure tile/vector dimensions satisfy any alignment requirements in the kernel's memory access patterns

## Step 4: Validate

1. Run correctness tests first — never sacrifice correctness for speed
2. Confirm speedup across **all** tested shapes, not just one
3. **Report both speedup ratio and absolute optimized time (ms)**. A higher speedup ratio paired with a worse absolute optimized time means the baseline shifted, not that the kernel improved. Treat any case where the absolute optimized time regresses as a failure even if the ratio looks better.
4. For structural rewrites, dump IR/ISA with `FLYDSL_DUMP_IR=1` and inspect the relevant `.mlir` stage plus `final_isa.s`
5. Verify the specific effect you wanted: `scf.for` survives tracing/lowering, wide loads/stores stay vectorized, the MFMA variant matches the target dtype/arch, and the final loop shape reflects the intended schedule
6. If the generated form did not change, assume the optimization did not land yet, even if the Python source looks right
7. If speedup is marginal or absolute time regresses, move to the next structural strategy rather than re-tuning the same approach

## Key FlyDSL APIs

- Device kernel: `@flyc.kernel` | Host launcher: `@flyc.jit`
- Control flow: `range_constexpr()` | `range(..., init=...)` | `arith.index()`
- Intrinsics: `flydsl.expr.rocdl` — MFMA, exp2, rcp, sched_barrier, sched_mfma
- Shared memory: `SmemAllocator` / `SmemPtr` from `flydsl.utils.smem_allocator`
- Types: `T.f16`, `T.bf16`, `T.f32`, `T.i32`, `T.vec(...)` from `flydsl.expr.typing`
- Buffer ops: `fx.rocdl.make_buffer_tensor`, `fx.make_copy_atom`, `buffer_ops.buffer_load` / `buffer_store` (offsets are in elements)
- IR/ISA inspection: `FLYDSL_DUMP_IR=1`, `FLYDSL_DUMP_DIR=...`, `final_isa.s`
- Autotune: `flydsl.autotune.autotune`, `Config`, `do_bench`
