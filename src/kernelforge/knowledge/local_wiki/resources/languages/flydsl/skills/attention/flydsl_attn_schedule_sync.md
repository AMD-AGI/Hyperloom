---
title: FlyDSL attention — scheduling hints, fences, priorities, waits, barriers and DMA pipelining
kind: language
lever: flydsl_attn_schedule_sync
gens: [gfx950]
dtypes: [bf16, fp8]
regimes: [prefill, backward, decode, causal]
updated: 2026-10-26
sources:
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:107
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:810
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/inline_asm.py:55
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/universal.py:96
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:744
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/kernel_function.py:157
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:74
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:1754
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_gfx950.py:1018
---

# Scheduling, synchronisation and pipelining in a FlyDSL attention kernel

> An attention body interleaves MFMAs with a transcendental-heavy softmax, LDS reads and DMA, all between
> barriers. The hint vocabulary (mask bits, greedy group filling, mask rules for unrolled epilogues), the
> single `vmcnt` and LDS-DMA waits are in `../gemm/flydsl_gemm_schedule_and_locality.md` §3–§4 and
> `../gemm/flydsl_gemm_datapath.md` §4; how to rank scheduling arms is in `flydsl_attn_attribution.md` §2
> and §5. This card adds what attention adds. Regime first: when co-resident waves already fill the
> MFMA shadow (no bare VALU between MFMAs), reordering rarely pays; where bare VALU remains it still can, and it
> paid more at 2 waves/SIMD than at 3 (§4). At one workgroup per CU, interleaving inside the workgroup is the only
> cover (`flydsl_attn_registers_occupancy.md` §8).
>
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as targets.

## Route here when
- You are writing or changing `sched_group_barrier` recipes, `sched_barrier` fences, `iglp_opt`, `s_setprio`.
- You want to move work across a barrier, delete a wait or change a wait depth.
- You are adding a prefetch, a second LDS buffer, or cache hints to K/V/Q/dO loads.
- A scheduling change came back numerically wrong on some shapes.

## 1. Primitives in FlyDSL v0.3.2
- `rocdl.sched_group_barrier(mask, size, group_id)` (mask as an int or a keyword such as `"transcendental"`),
  `rocdl.sched_barrier(mask)`, `rocdl.iglp_opt(id)`, `rocdl.s_setprio(p)`.
- **exp is TRANS, not VALU.** `0x400` selects transcendentals including `v_exp_f32`; `0x002` (VALU) does
  not; `0x008` matches both 16x16 and 32x32 MFMAs. A `{1 MFMA, 2 VALU} × 4` recipe left every `v_exp`
  after the last MFMA; only `{1 MFMA, 2 TRANS}` placed exps between MFMAs.
- FlyDSL's gfx950 forward uses `0x008 / 0x002 / 0x400` groups of `{1 MFMA, n VALU}` and
  `{1 MFMA, 3 EXP}`, halved at D=64, and wraps every rendezvous as
  `sched_barrier(0); s_barrier; sched_barrier(0)` (`_dualwave_sync_barrier`).

## 2. `sched_group_barrier` recipes
- **Classes not named in a region's groups go to the end of the region.** *(field)* A recipe naming only MFMA and
  DS-read groups cut its read stalls as intended but pushed the region's unnamed exp chain behind every group:
  −7.7%, VGPR, AGPR and LDS unchanged. Name every class the region contains.
- **Naming a class pins it.** *(field)* A 1-read-per-MFMA group placed each LDS read beside its consumer:
  read-to-wait distance 32.4 → 19.7, 3.48% slower. To issue reads early, lead with a group larger than the
  region's read count.
- **Count every MFMA in the region, row-sum MFMAs included.** *(field)* A D=64 forward QK region held 8
  `32x32x16` plus 4 `16x16x32` row-sum MFMAs; 8 anchors let the row sums take half and serialized the QK
  chain after the exps. With 9 registers freed first, 10 anchors gave +1.3% and let LLVM pack 38 VALU ops
  into 32 `v_pk_fma_f32`; the same anchors without the freed registers −0.3%; 12 broke the packing. D=128
  sat at 16, ±2 costing 0.5%. Sweep per head_dim, on the current register budget.
- **A recipe governs only its own region.** `sched_barrier`, `s_setprio` and `s_barrier` cut regions.
  *(field)* A recipe placed after an internal `sched_barrier(0)` steered a region holding 2 MFMAs; moving
  or deleting it produced byte-identical ISA. After restructuring (e.g. loop unswitching) whole recipe
  sweeps can go inert.
- **Hash the opcode stream before benchmarking an arm:**
  `grep -o "^\s*[a-z_0-9]*" 21_final_isa.s | md5sum` on the `FLYDSL_DUMP_IR=1` dump. Equal hashes cost no
  GPU time.

## 3. `iglp_opt`
- One strategy per region. Position inside the region and repeated calls compile byte-identically; the
  only choices are which regions get a call and which id. *(field)* A software-pipelining change that
  merged two regions destroyed the exp interleave of `iglp_opt(3)`: 32 exps became one bare run,
  `lgkmcnt(0)` 53 → 119.
- `iglp_opt` and `sched_group_barrier` drive the same LLVM mutation. *(field)* Where a hand recipe plus
  fences already interleaved exps, adding `iglp_opt(2)` gained nothing; replacing a working `iglp_opt` with
  hand groups started 0.9% behind.

## 4. `sched_barrier` fences: price each site
- **A fence wrapping a real `s_barrier` is a tax.** With mask 0 nothing crosses, so the matrix pipe idles
  through the barrier wait. MFMAs touch no LDS: relaxing both fences to `0x008` keeps DS ordering across the
  rendezvous. *(field)* +0.6–0.84% at D=128, ~0 at D=64. On another forward, removing both fences
  +0.61–0.96% (16 exps moved ahead of the barrier); removing either one alone changed nothing or lost.
  Prove safety by counting `ds_*`/`buffer_load` per barrier-delimited region: equal counts mean no memory
  op crossed.
- **Cluster-boundary fences do work.** *(field)* Removing the fence at the head of a memory cluster: −1.5%
  (spills 31 → 35); a DS-read-only mask (`0x100`) was flat. Deleting (not relaxing) the fence before the row-sum
  MFMAs inside a compute cluster cost 2.5% on another forward. Relaxing such a fence to `0x008` paid where bare VALU
  remained between MFMAs (next bullet); on the other forward partial masks read −0.15 to −0.50%: price each site.
- **A register fence inside a compute cluster may relax to `0x008`.** *(field)* Three such fences relaxed
  together: +3.1% at 2 waves/SIMD (D=128), noise at 3 (D=64), `.vgpr_count` and spills unchanged. Count `naked_work` (VALU
  and TRANS between adjacent MFMAs) in the ISA: when it reaches 0 the lever is spent.
- An MFMA-only fence makes MFMAs close ranks on both sides, leaving the fenced block with no cover.

## 5. `s_setprio`
- Only bits [1:0] count: priorities 0–3. *(field)* 1 vs 2 or 3 is byte-identical apart from the immediate
  and measured flat; all waves running one stream compare only in-region against out-of-region.
- Raise priority only where this wave has an MFMA run and its co-resident siblings do not, at ≥ 2 waves per
  SIMD on an MFMA-pipe-serialised body *(field: removing it there −2.8%)*. In a dual-wave forward protect the
  VALU-dense window instead *(field: priority on the pure-MFMA QK region −0.45 to −1.9%; releasing the P·V
  window early −3.5%)*. Always drop back to 0.
- **It is a scheduling anchor.** *(field)* Deleting 4 `s_setprio` at occupancy 1 (inert by hardware
  arbitration): −65 instructions, two regions merged, `lgkmcnt(0)` 44 → 88, −4.7%, and a dS fence race
  (NaN). Price a removal together with the reschedule.
- Sign depends on regime. *(field)* `s_setprio(1)` around a backward GEMM1 at occupancy 2: −3.3% (it
  starves the co-resident workgroup); forward sweeps with priority off: −0.8% and −2.4%.

## 6. LLVM scheduler strategy through `compile_hints`
```python
jitfn.compile_hints["llvm_options"] = {"amdgpu-sched-strategy": "max-ilp"}   # changes the ISA
```
- Option names are LLVM `cl::opt`s; an unknown name or a wrong value type raises. Hint layers merge
  shallowly and a `CompilationContext.compile_hints(...)` context wins over `jitfn.compile_hints`: the
  context's `llvm_options` replaces yours whole. FlyDSL's forward launcher sets its own in such a context, so
  for that kernel add the option to the launcher's hint dict and keep
  `{"enable-post-misched": False, "lsr-drop-solution": True}`. Hints are part of the cache key.
- Confirm the opcode hash changed (§2), then rerun SNR and determinism.
- On a body not yet hand-scheduled: *(field)* `max-memory-clause` with post-RA scheduling +0.6–0.7%,
  `max-ilp` +0.4%; `iterative-ilp` was fastest but reordered hand fences and hazard guards into NaN.
- On a hand-scheduled body only the downside is left: *(field)* backward −0.59% / −0.85%, forward −0.1% to
  −1.1%.

## 7. Hazards
- Inline-asm results are invisible to the hazard recognizer (`../gemm/flydsl_gemm_correctness.md` §7).
  In v0.3.2 `rocdl.cvt_pk_bf16_f32` is inline asm. An f32 → bf16 `.to(fx.BFloat16)` on a scalar or a
  2-element vector lowers to the same `v_cvt_pk_bf16_f32` as an ordinary instruction; prefer it.
- *(field)* MFMAs consuming the asm pack directly got `s_nop 0` instead of the required wait states:
  non-deterministic, half the q rows (alternate groups of 4) biased, 24.99 dB. Fix: `sched_barrier(0)` +
  `s_nop 1` before those MFMAs, or the arith truncation.
- **A clamp can be the hazard carrier.** When an inline-asm instruction reads an MFMA result, a
  compiler-visible VALU op on that value (a `min(s, 0)` that never fires on real data) is what makes LLVM
  insert the wait states. *(field)* Every variant without it failed the SNR gate; reducing it from one per
  element to one per 4-element vector: +3.6%.
- **An anchor must read the exact value it protects.** *(field)* One anchor per group instead of per
  vector: 0.31–2.53% faster, 16.9–19.7 dB and non-deterministic, only at q_split 4 with BLOCK_KV 128 or
  at BLOCK_KV 64; the main shapes passed. Gate hazard edits on a shape grid with bitwise determinism
  (`flydsl_attn_correctness.md`).

## 8. Waits and barriers
- **Before moving an instruction across a barrier, ask which write-after-read its `s_waitcnt` protects.**
  `gpu.barrier()` is only `s_barrier`. *(field)* MFMAs moved past a single-slot LDS barrier took the waits
  that retired the previous head's reads with them; an `s_waitcnt lgkmcnt(0)` before the barrier restored
  it at zero cost.
- **A hand-written `lgkmcnt(0)` right before a barrier is a tax** when every read it drains already has a
  consumer before the barrier: the consumers' graded waits retire them and the backend re-emits its own drain
  at `s_barrier` (check the ISA). That drain gives visibility, not write-after-read safety: delete the
  hand-written one only after checking, per buffer, that the LDS each read targets is overwritten no earlier
  than two barriers later. *(field)* +0.54%; the same removal in prologue and
  epilogues another +0.19–0.32%.
- **A full drain can sink to the cluster that consumes it** when the buffer it guards is overwritten two
  barriers later. *(field)* +1.42%; without a stagger gate an 8-wave configuration degraded (rel_l2
  0.0027 → 0.0294) while the scored SNR did not move.
- **Count depth, not waits.** `lgkmcnt` ≤ 15, `vmcnt` ≤ 63, `expcnt` ≤ 7 on gfx950. *(field)* Moving a
  prefetch ahead of partial stores: `vmcnt(0)` 16 → 1, wait depths (7..0) → (11..4), +3.9%; grouping DMA
  cut `vmcnt` waits 104 → 18 but turned graded waits into full drains: −5.3%.
- A barrier is also a scheduling boundary and, in a dual-wave body, the phase lock. Price a
  barrier family at N, N/2 and 0 and re-price after nearby edits (`flydsl_attn_attribution.md` §5); the
  rendezvous pool tracks co-resident workgroups and barrier-domain width
  (`flydsl_attn_grid_dispatch.md` §9).

## 9. Moving work into exposed segments
- Cover the first uncovered wait of an exposed segment with MFMAs whose operands are already in registers.
  *(field)* Handing 8 trailing MFMAs to sit right after a transposed-read burst, ahead of its first wait: +0.9%
  raw, +0.74% corrected.
- **Dose ≤ the target body's ring depth.** *(field)* Within the depth: 0.96%, then 1.20–1.35% fewer wave
  cycles; one past it 14.4% more, two past 74.3% more. Take the dose from the target body, not the source.
- **Place the hand-off right after the reads it covers.** *(field)* The same 8 MFMAs after the whole next
  GEMM bought more cover and cost 23.3% more wave cycles: no other MFMA segment may sit between hand-off and
  reads.
- Ask who covers the wait left behind. *(field)* The same move on a body whose MFMAs were already covering
  the approach to a fence: 1.98% more wave cycles (on another D=64 build the same move read +0.9% wall; the two
  readings are unreconciled).
- Batching reads loses to covering waits. *(field)* Grouped reads −1.4% against the covered hand-off.

## 10. Pipelining and DMA
- **One in-order `vmcnt` for loads and stores:** issue a prefetch before heavy partial stores, never after.
  *(field)* Same prefetch, before the stores +3.4%, a few instructions after them −2.4%.
- **An LDS DMA in flight forces `vmcnt(0)` before every `ds_read` of the same allocation** (mechanism in
  `../gemm/flydsl_gemm_datapath.md` §4). *(field)* With a second K/V buffer, drains 16 → 85 and 109. With
  compiler-visible reads, DMA overlaps only register-resident work; asm reads that carry their own waits avoid the
  drain (`../gemm/flydsl_gemm_datapath.md` §4) at the price of hand-placed `lgkmcnt` (`flydsl_attn_lds.md` §2).
- **K/V double buffering usually loses: it hides latency; measured cost was throughput.** *(field)* dq
  `BLOCK_KV` 96/64/32 double-buffered: −2.9/−3.2/−2.3%; a 256-VGPR body duplicated hoisted LDS addresses
  into 424 B of scratch: −16.5%. The winning form is a single buffer with a hand-off: issue the next DMA
  where the transposed-read registers die (`SQ_WAIT_ANY` −41%). At one workgroup per CU a second slot that
  removes a write-after-read barrier can pay *(field: +2.4%, +3.8% with a prefetch retune on one dQ kernel)*,
  but it is not a rule: *(field)* on another such backward it lost 6.0% because the second slot moved the
  next head's `ds_write` burst into a transposed-read burst (`flydsl_attn_lds.md` §9); on a dual-wave forward,
  deepening the ring to drop one barrier lost 2.2% (LDS doubled, spills 4 → 9; the deeper ring alone cost 1.65%).
- One barrier per iteration needs ≥ 3 buffers; a deeper ring relaxes write-after-read only, never
  read-after-write.
- Register prefetch of cold per-head loads (LSE, delta): `flydsl_attn_grid_dispatch.md` §10.
- `buffer_load … lds` gets no cross-workgroup vL1D reuse; judge request pressure by `TCP_TCC_READ_REQ`,
  not by DMA instruction count (`flydsl_attn_grid_dispatch.md` §2, §9).
- **Cache bits:** the LDS-DMA atoms (`BufferCopyLDS*`, `cdna4.BufferLoadAsyncLDS*`, `GlobalLoadAsyncLDS*`)
  always emit `aux = 0`. For `nt`/`sc` bits call
  `rocdl.raw_ptr_buffer_load_lds(rsrc, lds_ptr, size, voffset, soffset, offset, aux=<python int>)`; any other
  type is silently dropped (`../gemm/INDEX.md` § FlyDSL version).
- **Non-temporal hints on attention operand streams were negative.** *(field)* Q `nt` −0.95% (waves of one
  workgroup share Q lines); dkdv Q/dO DMA `nt` −7.6% (reused across workgroups); O store `nt` +0.11%
  (noise). Before marking a stream `nt`, ask whether lanes or waves of one workgroup share its lines.
  Workspace partials are in `flydsl_attn_backward.md` §5.
- **Decode prefetch depth:** a prologue depth of 2 pays only at ≤ 2 CTAs per CU. *(field)* It was 1.7–2.7%
  faster up to 512 CTAs (2 per CU on 256 CUs) and 2.0–2.5% slower from 576 up; a steady-state depth ≥ 2
  was 4.6–6.6% slower. Account the extra prologue tile so the first iteration consumes it, or it becomes
  the steady-state depth (`flydsl_attn_decode_split_kv.md`).

## Verify
- Every recipe arm has a changed opcode hash, and every region's MFMAs and TRANS ops are named.
- Fence relaxations keep per-region memory-op counts; hazard edits pass a shape grid with bitwise determinism.
- Wait changes are judged by `vmcnt(0)` count and depth distribution, then by `SQ_WAVE_CYCLES` /
  `SQ_WAIT_INST_LDS`, then by wall time.

## Related
`flydsl_attn_registers_occupancy.md` · `flydsl_attn_lds.md` · `flydsl_attn_attribution.md` ·
`flydsl_attn_correctness.md` · `flydsl_attn_grid_dispatch.md` · `flydsl_attn_decode_split_kv.md` ·
`../gemm/flydsl_gemm_schedule_and_locality.md` · `../gemm/flydsl_gemm_correctness.md` ·
`../optimize/prefetch-data-load.md` · `../bottleneck/debug-flydsl-kernel.md`
