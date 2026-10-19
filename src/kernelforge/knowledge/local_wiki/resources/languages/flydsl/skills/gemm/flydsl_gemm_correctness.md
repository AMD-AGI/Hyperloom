---
title: FlyDSL GEMM — correctness gates that actually exercise the change, races, and silent-wrong traps
kind: language
lever: flydsl_gemm_correctness
gens: [gfx950]
dtypes: [fp8, bf16, mxfp4, mxfp8]
regimes: [pipelined, split-k, inline-asm]
updated: 2026-10-19
sources:
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_fp8_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp4_kernel.py
  - ROCm/FlyDSL@dd8373349:kernels/gemm/gemm_a16w16_gfx950.py:362
  - ROCm/FlyDSL@dd8373349:kernels/gemm/preshuffle_gemm.py:263
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/gpu.py:157
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:142
---

# Correctness for a FlyDSL GEMM

> A GEMM that is fast and wrong usually passed its gate. Each case marked *(field)* below went through
> a gate that reported clean. Use them to design the gate before trusting it. Numbers marked *(field)*
> are measured values. Use them for direction and order of magnitude, not as targets. Debugging a
> kernel that is already known wrong starts in `../bottleneck/debug-flydsl-kernel.md`. "PS" is FlyDSL's
> `kernels/gemm/preshuffle_gemm.py` and "A16" is `kernels/gemm/gemm_a16w16_gfx950.py`.

## Route here when
- You changed the pipeline, a wait, a barrier, the LDS layout, register classes or inline asm.
- A gate passes but you have not shown it runs the code you changed.
- Outputs differ between runs, or only on some shapes, or only after the first call.

**Static checks first; they cost nothing.** LDS stays within 160 KiB per workgroup on gfx950; tile packing and
vector widths match the operand layout; the swizzle or padding is applied identically by the writer and the
reader (`../optimize/gemm-optimization.md` §3.4); accumulator and output conversions cannot overflow; edges hold
for shapes that do not divide the tile.

## 1. A gate must execute the path you changed

- **Coverage first, repetition second.** A race shows up as a jittering error. A path the gate
  never runs shows up as a stable, clean number that looks exactly like a pass, and running the gate
  more times cannot change that. *(field)* A gate whose shapes gave the kernel only one or two
  K blocks made the runtime dispatcher take the short branch, so the steady-state loop never ran. A
  steady-state bug passed repeatedly at 55.09 dB; with four times the K the same code read
  19.40 dB.
- Before trusting a gate, compute for its shapes the trip counts the kernel derives (`k_iters`,
  phases per tile, groups that reach the steady loop). A 0 or 1 is an alarm. Keep a second probe
  that reuses the gate's reference and scales the shape up.
- Include the operands production feeds: padded or strided rows, unequal strides, ragged M/N/K.
  A probe that only uses contiguous tensors never reaches the strided branch.
- A previously kept change is not a trusted baseline until a gate that covers it has passed.

**Shapes every GEMM gate needs.** Each one forces a path that only runs at an edge:
1. **The minimum trip count.** Take a K where the main loop runs zero times (e.g. `K = 2·BLOCK_K` for a
   `range_constexpr(K_ITERS-2)` loop). A loose prologue wait that relies on "the loop will drain it later"
   then has nothing after it, and the epilogue reads LDS that is still being filled. `s_barrier` orders
   waves; it does not drain DMA. Turbo's dense fp8 8-wave kernel special-cases exactly this
   (`gemm_fp8_kernel.py:257-262`).
2. **More tiles than CUs.** Only the second tile on a CU can read what the first one left in LDS; the
   first tile finds an empty slot and computes the right answer. Turbo's MXFP4 kernel leaves the k=1 prologue
   fill in flight only for a caller whose loop wait was computed with that fill outstanding; a caller
   written against a drained prologue reads k=1 before it lands. Skipping the drain is an explicit opt-in
   flag (`k1_watermark`), and the flag is part of the cache key (`gemm_mxfp4_kernel.py:1510-1525`, 765).
3. **The deepest branch of every runtime choice** (the largest split factor, the widest tile, the
   persistent path). Pick a shape large enough to force it.
4. **Ragged M, N and K, and K that is not a multiple of the K tile**, for every layout the kernel
   accepts.
5. **The same gate for kernels you did not write.** A shipped kernel's default configuration is also an
   input to verify, not a reference.

Run each gate with at least two calls, fill C with NaN between calls, compare against an fp32
reference built from the dequantized operands, and compare repeated outputs bitwise (§3).

## 2. Call twice; compare from the second call

*(field)* A kernel was clean on its first call after compilation (55.54 dB) and 33 dB from the
second call on, with different elements wrong on each call. The harness scored the first call, so the race
stayed hidden. Every gate calls the kernel at least twice and checks the second and
later results. Work that depends on a first call (atomic counters, semaphores, lazily initialized
workspaces) must be warmed before a probe starts counting.

## 3. Races: bitwise repetition, at the timing shape

- Detect a race by repeating the same inputs and comparing outputs **bitwise**, not by SNR: a race
  that flips 0.17% of elements leaves the SNR almost unchanged.
- Repeat enough: *(field)* 200 runs looked clean where 500 exposed a 1.5e-5 race. Use fresh random
  inputs per pass, and for rates near 1e-4 tens of thousands of runs.
- Repeat **at the shapes you time**. *(field)* Narrow shapes passed a determinism check while the
  timing shapes had 134–389 bad rows; M = 32768 failed every time while M ≤ 8192 was clean 8/8.
- Deeper prefetch fails first: a race that only appears on the largest K is pointing at a
  pipeline-drain seam.

## 4. Waits, barriers and the scheduler

- **Read-write distance decides how much may stay in flight.** With two buffers the next phase reads
  what was just written (distance 1): drain fully. With three (distance 2) a partial drain is safe,
  and only then.
- **A partial `vmcnt(N)` must count every pool still in flight**, not the first one. *(field)* Two
  three-deep pools left 8 writes in flight while the wait accounted for 4: a 1-in-30000 race.
- **A longer in-flight tail is less safe even when the issue-order count says it is enough.**
  *(field)* A wait sized to leave a whole iteration in flight raced on the wide shapes; keep the tail
  below one iteration's issue count minus its first group, and prove a longer one by repetition.
- **`spill ≠ 0` voids every hand-written partial `vmcnt`.** Scratch loads and stores share `vmcnt`
  with your DMA, so `vmcnt(1)` no longer means "one DMA left". Read `.private_segment_fixed_size`
  and the spill counts first. When spills are zero and it still races, show those numbers
  (`llvm-readelf --notes`) as the evidence: the race has another channel, and more `vmcnt(0)` will not fix
  it.
- **Waits written as instruction counts break when the wave count changes.** Recompute them per tile,
  not per instruction, after changing the CTA's waves; the failure signature is "all outputs right,
  determinism passes, every shape passes" until it is not.
- **An inline-asm `s_waitcnt vmcnt(0)` does not stop LLVM from moving an earlier load below it,** where
  the wait no longer covers it. *(field)* Two drains before an epilogue that reused the operand LDS still
  let the next tile's global→LDS prefetch sink past them and land on staged data, only on the deepest-K
  shape. `rocdl.sched_barrier(0)` after the
  drains fixed it (0/10 bad reads vs 3/30) at about 0.7% of time. A mask that let ALU/MFMA cross
  (`0x40F`) made it worse (6/11 bad).
- `s_barrier` does not drain `vmcnt`; `lgkmcnt` is 4 bits (max 15), `vmcnt` 6 bits (max 63).

## 5. LDS placement and DMA

- **An LDS-DMA `M0` base must never go negative.** *(field)* A ring that pre-decremented one slot's base
  by a K step (so one immediate could serve both the global and LDS address) rotated that role onto
  the slot at LDS offset 0. Wave 0's `M0` underflowed and its writes were dropped — not wrapped — so
  the result was deterministic and looked unlike a race: exactly one tile index wrong (55.5 → 14.8 dB),
  or, with a static rotation, rows 0–31 of every tile wrong. Keep any skewed role off the slot at
  offset 0.
- **Reordering `@fx.struct` fields does not move LDS.** With `SharedAllocator(static=True)` every leaf
  is its own symbol placed by the backend (`expr/gpu.py:157`); an arm that stops using a leaf lets
  the others be re-packed. Read `group_segment_fixed_size` before interpreting an LDS-placement result.
- **Buffers that must be contiguous go in one field, or use `static=False`.** The default static
  allocator places fields by size, largest first, and equal sizes by symbol name (declared
  `a4k, b16k, c8k, d256, e1k` land as `b16k, c8k, a4k, e1k, d256`). `SharedAllocator(static=False)` kept
  declaration order in every case tested. A pool addressed as one base plus immediates must not rely on the static order.

## 6. Bounds

- **MFMA ignores EXEC.** With EXEC set by inline asm to lanes 0–31, 32–63, lane 0 only, or none,
  `v_mfma_f32_16x16x32_bf16` writes full results in all 64 lanes, while a `v_mov` in the same asm block
  is masked as expected. A divergent `if tid < 32:` around an MFMA in FlyDSL
  therefore overwrites the accumulators of the "inactive" lanes; when the operand loads were only used
  inside the branch, LLVM sank them into it and every lane came out as garbage. Bound ragged tiles through
  the buffer descriptor (operands read as zero) or select whole tiles; never mask an MFMA with a branch.

- Bound ragged edges in the buffer descriptor (`make_buffer_tensor(..., max_size=False,
  num_records_bytes=...)`, as PS does for A and C at PS:263): out-of-range loads return zero and
  stores are dropped at no instruction cost. *(field)* A software predicate in the store loop cost 2.8%.
- The default `max_size=True` sets `num_records = 0xFFFFFFFF` — no masking. Indices derived from data need the
  real size.
- Prove a grid or index decode is an in-bounds permutation offline before timing it: an
  out-of-bounds write can look like a speedup.
- Derive every shape-dependent bound from traced arguments; a host integer can be baked into a cached
  kernel and reused for the next shape (`../bottleneck/debug-flydsl-kernel.md` § A bound or address
  baked from a host integer).

## 7. Register classes and inline asm

- **asm that writes an MFMA accumulator into AGPRs hides the hazard from the compiler.** gfx950 needs
  18 wait states between an MFMA writing an AGPR and a `v_accvgpr_read` of it. *(field)* Two of 64
  accumulators lost the last K iteration's contribution (SNR still 37–45 dB). Put the wait states in
  the asm string of the tail iteration only.
- **The reverse too:** asm that *reads* an MFMA destination loses the compiler's `s_nop`. A drop in
  `s_nop` count after moving code into asm is a red flag, not a saving.
- **An untied asm pin (`"=a,v"`) silently deletes the operand.** It compiles, runs, removes spills and
  runs faster, because it emits no copy and the output is uninitialized.
- **Moving an asm output from AGPR to arch VGPR needs `&` (early clobber).** Without it the register
  allocator may legally place an input (an LDS base, an offset) inside the output range. Signature:
  some shapes right, others collapse, correlated with tile count rather than K; every ISA count stays
  the same. `&` is incompatible with a tied (`"0"`) operand.
- In `$n` numbering, tied inputs count: in `"=v,0,v,v"`, `$1` is the tied input.
- The constraint rules for a whole-loop asm region (literal `src2=0` for a zero accumulator, `~{scc}`
  around an asm loop, every `=&s` output written) and how to set the `amdgpu-agpr-alloc` attribute
  are in `flydsl_gemm_mainloop.md` §4.

## 8. Reproducibility and tolerance

- **Split-K with atomics is not bitwise reproducible.** FlyDSL's bf16 GEMM accumulates splits with
  `BufferAtomicPkAdd` into the bf16 output (A16:362): the order is arrival order and each partial is
  rounded to bf16 as it lands. Gate it on a calibrated tolerance against an independent reference,
  not on bit equality with the previous build.
- A changed reduction order is a changed result: compare against a third reference rather than
  old-vs-new maximum difference.
- Timed autotuning that picks configurations per run makes outputs differ across runs; pin the
  configuration for determinism checks.
- Compare only the region the kernel contract writes, using the harness's own input tensors.

## 9. Low precision: validate in order

1. conversion and packing; 2. scale handling; 3. the math against a dequantized reference; only then
4. performance. Block-scaled probes need random E8M0 exponents (e.g. `torch.randint(122, 132)`), not
scales quantized from `randn` data: N(0,1) blocks almost always share one power-of-two exponent, so a
kernel applying the wrong scale set reads the same as a correct one. *(field)* A tail phase that kept
reusing the scales from two phases earlier read 49.6 dB with `randn`-derived scales and 11–14 dB the
moment the exponents were random.

## Related
`../bottleneck/debug-flydsl-kernel.md` · `flydsl_gemm_mainloop.md` · `flydsl_gemm_datapath.md` ·
`flydsl_gemm_schedule_and_locality.md` · `flydsl_gemm_measurement.md`
