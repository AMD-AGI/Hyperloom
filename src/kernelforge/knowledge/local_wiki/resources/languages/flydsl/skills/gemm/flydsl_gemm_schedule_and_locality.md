---
title: FlyDSL GEMM — wave placement, XCD order, scheduling fences and epilogue cost that are easy to get backwards
kind: language
lever: flydsl_gemm_schedule_and_locality
gens: [gfx950]
dtypes: [bf16, fp8, mxfp4]
regimes: [compute-bound, register-saturated]
updated: 2026-10-19
sources:
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:99
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:142
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:877
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/universal.py:266
  - ROCm/FlyDSL@dd8373349:kernels/gemm/fp8_gemm_4wave.py:135
---

# Wave placement, XCD order, scheduling and epilogue in a FlyDSL GEMM

> Each section here is a place where the obvious move measured in the wrong direction. Read the rule,
> then check whether your kernel is in the regime it was measured in. "GH" is Primus-Turbo's
> `primus_turbo/flydsl/utils/gemm_helper.py` at main `7615a5a2`; "PS" and "A16" are FlyDSL's
> `kernels/gemm/preshuffle_gemm.py` and `kernels/gemm/gemm_a16w16_gfx950.py`.
>
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as
> targets.

## Route here when
- A change that should be neutral (re-labelling waves, a new swizzle, a relaxed fence) moved wall time.
- You are choosing a tile order across XCDs, or porting one kernel's order to another.
- You are writing `rocdl.sched_*` hints or `sched_barrier` fences.
- You want to fuse work into the epilogue of a register-heavy GEMM.
- Accumulators no longer fit, or buffer loads sit inside a `v_readfirstlane` loop.

**Somewhere else if:** the grid, tile size, split-K or a boundary variant is the question →
`flydsl_gemm_grid_pricing.md`; the general order of operations → `flydsl_gemm_workflow.md`.

## 1. Co-resident waves must not read the same heavy fragment

A workgroup's waves stripe across the CU's 4 SIMDs; with 8 waves, wave `w` and wave `w + 4` share a SIMD.
If those two read the same LDS fragment of the heavier operand at the same moment, their LDS requests are
in phase and neither hides the other's latency.

*(field)* 8-wave, 2-waves-per-SIMD kernel, 2(M)×4(N) wave grid. Writing the same `(wave_m, wave_n)`
bijection three ways — identical tile split, LDS bytes, instruction counts:

| What the co-resident pair shares | Result |
|---|---|
| `wave_n` only (same B fragment, the lighter side) | baseline |
| neither | same as baseline |
| `wave_m` only (same A fragment, 8 of the wave's 12 `ds_read`s) | **~25% slower** |

Transposing the wave grid to 4(M)×2(N) cost another ~16%. Cache, LDS, bank-conflict and MFMA counters
were bit-identical between arms; only `SQ_WAIT_ANY / SQ_WAVE_CYCLES` moved (31.7% → 50.0%).

So: any rewrite of the wave grid — new MFMA atom, different `n_tiles` split, changed boundary granularity
— must state the `wave_id → (wave_m, wave_n)` mapping explicitly and recheck which waves co-reside. It is
index algebra that can cost a quarter of the kernel, and ordinary counters will not show it.

## 2. Tile order across XCDs

Topology and the basic swizzle are in `../../../../common_methodology/optimization/lever_xcd_locality.md`.
The GEMM-specific rules:

**Decide by what co-resident workgroups reuse.**
- Each group of tiles streams **a slab of its own** (shared inside the group, not across the grid) that is larger
  than one XCD's L2: keep the group's workgroups on the same XCD at the same time — cluster across all 8 XCDs.
- All tiles read a **shared slice**: clustering by XCD only fragments the
  sharing; the hardware round-robin (`block_id % 8`) with a narrow group-major band wins.

*(field)* Porting one kernel's winning band/XCD setting to another kernel made it 0.8% to 5.7% slower. Predict the direction from the reuse class, then sweep 3–5 points on every shape class
you are scored on.

**FlyDSL's kernels expose the order as an argument:** `xcd_swizzle` in PS (`xcd_remap_bx_by`, in
`kernels/common/mma/mfma_preshuffle_pipeline.py`), `use_xcd_remap` in `fp8_gemm_4wave`, and `group_m`
(`BlockSwizzle`) in A16. A16's swizzle falls back to the plain order for small grids
(`flydsl_gemm_small_m.md` §6).

**How Turbo orders tiles** (Primus-Turbo main `7615a5a2`, `primus_turbo/flydsl/utils/gemm_helper.py` = GH):
- `xcd_remap_pid` (GH:1639) gives each XCD a contiguous slice of logical tiles, for any total. It is a
  bijection, so it cannot change results.
- `block_mn` (GH:1967) is a `GROUP_M` (1D) or `group_n` (2D) super-tile decode, also a bijection.
- `xcd_band_remap_pid` (GH:2694) gives each XCD every `num_xcd`-th run of `band` tiles. That keeps reuse
  inside a run while every XCD samples the whole tile range, which balances tiles of unequal cost.
- **Gate the band on L2 bytes.** Once one band's operand slice outgrows an XCD's L2 slice (4 MiB on
  gfx950), clustering buys no reuse and only costs balance. Turbo's dense fp8 table then switches to the
  plain round-robin (`_NT_BAND_L2_BYTES`, `gemm/gemm_fp8_kernel.py:3186-3192`). The MXFP4 kernel picks
  the widest power-of-two N band whose B slice fits one XCD's L2, and no band once a single N tile fills
  it (`_mxfp4_l2_band`, `gemm/gemm_mxfp4_kernel.py:3131-3139`).
- **Make each XCD's resident set a rectangle.** The dense fp8 TN whole loop sizes `(group_m, group_n)`
  so the tiles one XCD holds pull the fewest operand bytes through its L2. It snaps the group height to a
  divisor of 8 "so its resident set [is] a rectangle instead of a staircase" (`_tn_wave4_band`,
  `gemm_fp8_kernel.py:3280-3297`).
- **A persistent loop strides by the grid, not by 1.** Then one dispatch round's tiles stay inside one
  band and the workgroups co-resident on an XCD share a B tile from L2. A consecutive walk per workgroup
  widened the live band and cost 2.7–4.0% (`gemm_mxfp4_kernel.py:2961-2965`).
- **Keep the tile decode cheap.** It sits in front of the first load at occupancy 1. Use unsigned
  division and modulo, make N a compile-time constant so divisions become multiplies, and when dividing
  by a runtime value, use a fixed-point reciprocal that is proven exact over the reachable range.

**2D bands:** compare `GM · GN` with the per-XCD concurrent window (resident workgroups per XCD: 32 at one
workgroup per CU). `GM · GN` near the window makes the footprint depend on where the window starts inside
the band; a 1D band is steadier there. Use 2D only when `GM · GN` is much larger than the window.

**Mechanics:**
- One locality remap at a time — stacking two usually defeats both.
- Persistent kernels remap the work id, not `block_idx`.
- Prove the remap is a bijection over every grid you actually launch (a chunked launch can be narrower
  than the deployment shape). `x*(G//8) + min(x, G%8) + block_id//8` with `x = block_id % 8` is a
  permutation for any G, so it needs no divisibility gate.
- FlyDSL's own `_xcd_swizzle` (`kernels/gemm/fp8_gemm_4wave.py`) is a contiguous-chunk remap plus
  `GROUP_M = 4`, used only when the grid has ≥ 4·256 workgroups and a multiple of 8; otherwise linear.

**Judge by wall time and stall counters, not L2 hit rate.** *(field)* An XCD order that was 3% faster left
`TCC_HIT`, `TCC_REQ` and `TCP_TCC_READ_REQ` unchanged; the gain showed in `MemUnitStalled` (12.4% →
10.7%), i.e. TCP→TCC request queueing. Elsewhere hit rate rose sharply with no wall change. Use
`SQ_WAIT_ANY` and `MemUnitStalled`. Differences of 1–2% on this axis flipped sign between back-to-back
sweeps; 10%-scale effects reproduced.

## 3. Scheduling hints and fences

FlyDSL's helpers are thin wrappers over `sched_group_barrier`:

| Helper | Mask | Classes |
|---|---|---|
| `rocdl.sched_mfma(n)` | `0x008` | MFMA/WMMA |
| `rocdl.sched_vmem(n)` | `0x020` | **VMEM reads only** |
| `rocdl.sched_dsrd(n)` | `0x100` | DS reads |
| `rocdl.sched_dswr(n)` | `0x200` | DS writes |

`rocdl.sched_barrier(mask)` takes the same bit vocabulary — `0x001` non-memory/non-side-effect (this
**includes VALU**), `0x002` VALU, `0x004` SALU, `0x008` MFMA, `0x010` all VMEM, `0x020`/`0x040` VMEM
read/write, `0x080` all DS, `0x100`/`0x200` DS read/write, `0x400` transcendental, `0x800` LDS DMA. A set
bit means that class **may cross** the fence; `0` pins everything.

**FlyDSL's `hot_loop_scheduler` is the reference use.** `preshuffle_gemm` builds one hint table per tile. In the
sync-copy pattern FlyDSL's guide describes, a prologue places the first LDS reads, then each step is one VMEM
load, a group of MFMAs, one LDS read and a second group of MFMAs, with the next tile's LDS writes at the tail so
they land before the barrier (`../optimize/gemm-optimization.md` §5.3–§5.4). With async copy it spreads the LDS reads and VMEM loads evenly
over all MFMAs (§5.5); the `preload=(dsrd, dvmem)` argument sets how many go ahead of the first MFMA
(PS:450-473).

Fences inside a fully unrolled epilogue usually exist to stop the scheduler from hoisting work and
lengthening live ranges. What must stay pinned is the arithmetic, not only the memory operations:
- *(field)* Relaxing such a fence from `0` to `0x001`: VGPR 476 → 512, spills 0 → 30 — VALU intermediates
  hoisted across the fence.
- *(field)* `0x004 | 0x008 | 0x080` (SALU, MFMA, DS may cross; VALU and VMEM pinned): no spills, unchanged
  VGPR, `s_nop` 79 → 53, faster on every affected shape.

Gate any mask change on the ISA's `vgpr_spill_count` before benchmarking. Hint counts must come from the
loop in front of you; constants copied from another kernel are worse than none.

**A count mismatch is never reported.** `sched_group_barrier` groups are filled greedily in program
order. For 4 independent MFMAs and 4 global loads:
- with no hints the order is `L L L L M M M M`;
- `(mfma 1, vmem 1) × 4` gives `M L M L M L M L`;
- `(mfma 1, vmem 3) × 4` front-loads the loads: `M L L L M L M M`;
- `(mfma 1, vmem 2) × 2` gives `M L L M L L M M`: the groups pin the head and the leftover MFMAs follow;
- the compiler prints no warning in any case.
Read the scheduled ISA after changing a hint table.

**Kernels with hand-placed waits use none of this.** Turbo's dense kernels contain no
`sched_group_barrier`. They use `rocdl.sched_barrier(0)` around hand-written waits, `s_setprio` at two
waves per SIMD, and `s_nop` issue pads inside whole-loop asm (`flydsl_gemm_mainloop.md` §2–§3).

## 4. gfx950 has one memory counter for loads and stores

`s_waitcnt vmcnt(N)` and `s_waitcnt lgkmcnt(N)` assemble for gfx950; `s_waitcnt_vscnt`,
`s_wait_loadcnt`, `s_wait_storecnt` and `s_wait_dscnt` do not (the assembler reports "instruction not
supported on this GPU" — they belong to later ISAs). Stores in flight count in `vmcnt`, so waiting for a
load also waits for every earlier store. Consequences:
- Dripping C stores into the main loop serializes them against the next operand wait.
- MFMA is the only work that occupies neither `vmcnt` nor `lgkmcnt` — it is the medium to hide a wait
  under.

**Write a wait as "what may still be in flight", not as a formula of the block shape.** At a phase
barrier, leave outstanding only the fills that nothing reads before a later phase, and only the
`ds_read`s whose consumers come after the barrier. Then race the pair:
- *(field)* On one whole loop, the autotune race offers `(vmcnt, lgkmcnt)` of `(10, 9)` and `(16, 15)`, the
  deeper pair only for K ≥ 8192 and adopted only at a 2% margin (`_wl_opts`, `_WL_MARGIN`,
  `gemm_mxfp4_kernel.py:3208, 3274`).
- `lgkmcnt` 15 is the encoding limit of the 4-bit field, not a tuning boundary. The depth at which a wait
  stops being bitwise repeatable is per kernel: another whole loop was fastest at 12 and still repeatable at
  15. Prove any deeper setting by repetition (`flydsl_gemm_correctness.md` §3).
- *(field)* The lgkm-side partial drain carried over between kernels and precisions (+0.8%). A one-`s_nop`
  pad after the barrier helped MXFP4 and was worth 0.00% on fp8. Port the lgkm side first.

## 5. Hiding an LDS write when registers are nearly full

The usual advice is to put independent work (the next global load, address math, an independent MFMA
group) between a `ds_write` and the `lgkmcnt` wait that guards its read. When VGPR headroom is small, any
insertion lengthens a live range and loses:

*(field)* Occupancy 1, 440/472 VGPR + 256 AGPR, fully unrolled epilogue, same gap three ways:

| Move | Result |
|---|---|
| issue the `ds_write` itself earlier, as soon as its data is ready | **+0.6%** |
| insert the next stage's `buffer_load`s into the gap | −2.0% |
| move the `ds_write` into the previous MFMA stream | −2.6% |

Rule of thumb from that kernel: below ~15% VGPR headroom, move the write, do not insert work. A real
latency fix barely changes the instruction count (5315 → 5307 here); a large count change means you
changed something else.

## 6. Epilogue fusion competes with the tile that made the GEMM fast

The tile that wins the GEMM often leaves one wave per SIMD. Then nothing hides a dependent VALU chain in
the epilogue, and each dependent instruction costs its full latency.

*(field)* In a 1-wave-per-SIMD GEMM's epilogue: one FMA per element +0.5%, ten dependent FMAs +159%, a
full exact GELU +452%; the same math as a standalone pointwise kernel was ~20× cheaper than in that
epilogue. Work interleaved into main-loop slots (stores issued inside the loop) cost far less:
SwiGLU plus two MXFP4 quantizations and packed writes there added 34–38%.

Before planning an epilogue fusion, read the kernel's occupancy. At one wave per SIMD, price every
element-wise dependency chain by latency, and compare against a separate kernel.

## 7. Accumulators, descriptors and large operands

- **Accumulators beyond the AGPR file.** A wide tile can need more than 256 accumulator dwords per lane.
  Place them yourself (part AGPR, part VGPR). *(field)* Leaving a 128×512 tile's accumulators to the
  compiler through `fx.gemm` produced 613 `v_accvgpr_*` moves and 18 scratch loads in the hot loop;
  explicit placement of a 256×384 tile ran at 500/512 registers with no spill. Check: count
  `v_accvgpr_read/write` and `scratch_` in the hot loop.
- **Descriptor fields must be wave-uniform, and operands past 4 GiB need an i64 rebase.** See
  `flydsl_gemm_datapath.md` §8. Signature of a non-uniform descriptor: `v_readfirstlane_b32` plus an
  `s_and_saveexec_b64` loop around the hot loop's `buffer_load`s.
- **Ragged edges.** `fx.rocdl.make_buffer_tensor(t)` defaults to `max_size=True`, which sets
  `num_records = 0xFFFFFFFF` — no out-of-range masking. To have out-of-range loads return zero and
  out-of-range stores dropped, pass `max_size=False` or `num_records_bytes`, and derive the bound from
  traced values (`../bottleneck/debug-flydsl-kernel.md` § What the cache key sees).

## Verify
- The `wave_id → (wave_m, wave_n)` mapping and its co-resident pairs are written down for the kernel.
- A remap is checked as a bijection over every launched grid width, with the exit on the raw id.
- A fence change shows `vgpr_spill_count` unchanged before any timing.
- An XCD/order change is judged on wall time and stall counters across all shape classes.

## Related
`flydsl_gemm_grid_pricing.md` · `flydsl_gemm_workflow.md` ·
`../../../../common_methodology/optimization/lever_xcd_locality.md` ·
`../../../../hardware/mi350_execution.md` · `../../../../hardware/mi350_chiplet.md` ·
`../bottleneck/debug-flydsl-kernel.md`
