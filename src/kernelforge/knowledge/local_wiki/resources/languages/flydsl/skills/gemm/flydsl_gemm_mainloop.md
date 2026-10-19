---
title: FlyDSL GEMM — the main loop: 2-stage ping-pong, 8-wave pipeline, 4-wave designs
kind: language
lever: flydsl_gemm_mainloop
gens: [gfx950]
dtypes: [fp8, int8, mxfp4, mxfp8, bf16, fp16]
regimes: [prefill, training, large-K]
updated: 2026-10-19
sources:
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_fp8_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp4_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/utils/gemm_helper.py
  - ROCm/FlyDSL@dd8373349:kernels/gemm/preshuffle_gemm.py
  - ROCm/FlyDSL@dd8373349:kernels/gemm/gemm_a16w16_gfx950.py
  - ROCm/FlyDSL@dd8373349:kernels/gemm/fp8_gemm_8wave.py
  - ROCm/FlyDSL@dd8373349:kernels/gemm/fp8_gemm_4wave.py
  - ROCm/FlyDSL@dd8373349:.claude/skills/gemm-optimization/SKILL.md
---

# The FlyDSL GEMM main loop

> Three loop designs are in use for FlyDSL GEMMs on gfx950: FlyDSL's 2-stage LDS ping-pong
> (`preshuffle_gemm`, explained step by step in `../optimize/gemm-optimization.md` §2 and §4–§6), the 8-wave
> pipeline (FlyDSL's `fp8_gemm_8wave.py`; Primus-Turbo's fp8 and MXFP8 GEMMs) and the 4-wave designs (FlyDSL's
> `fp8_gemm_4wave.py` and `fp4_gemm_4wave.py`; Primus-Turbo's whole-loop inline asm). Primus-Turbo's kernels are
> built for training shapes (large M; small M is in `flydsl_gemm_small_m.md`). "FP8" below means Turbo's
> `gemm/gemm_fp8_kernel.py`, "FP4" means `gemm/gemm_mxfp4_kernel.py` and "GH" means `utils/gemm_helper.py`, all
> under `primus_turbo/flydsl/` at main `7615a5a2`; "PS" is FlyDSL's `kernels/gemm/preshuffle_gemm.py`. All of
> them run on FlyDSL v0.3.2. What to know when calling FlyDSL's kernels is in §8.
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as targets.

## Route here when
- You are choosing or writing the K loop of a FlyDSL GEMM: wave count, LDS buffering, where the waits go,
  and whether the loop should be one inline-asm region.
- An inline-asm MFMA or loop compiles and runs but produces wrong numbers on some shapes.
- You are deciding between AGPR and VGPR accumulators, or between MFMA atoms.

Grid, tile and split-K pricing are in `flydsl_gemm_grid_pricing.md`. Wait counts, priority and tile order
are in `flydsl_gemm_schedule_and_locality.md`. LDS layout, global→LDS DMA and M0 are in
`flydsl_gemm_datapath.md`. The C store is in `flydsl_gemm_epilogue.md`.

## The baseline: FlyDSL's 2-stage ping-pong (`preshuffle_gemm`)

FlyDSL's guide walks through this design in detail (`../optimize/gemm-optimization.md`); the essentials:
- **Four waves split `tile_n`; B never touches LDS.** B is preshuffled on the host into the MFMA register
  layout and loaded straight to VGPRs (guide §4.2, §12). LDS holds only A, in two buffers (guide §2).
- **Each loop iteration covers two K tiles, one per buffer.** For each: load the next A tile into the idle
  buffer and the next B tile into registers, run the MFMAs on the current buffer, then `s_waitcnt` + barrier
  (guide §2.3, §4.4).
- **`hot_loop_scheduler` spreads the loads over the MFMAs** with `sched_vmem` / `sched_dsrd` / `sched_dswr`
  / `sched_mfma` groups whose counts come from this tile (guide §5; `flydsl_gemm_schedule_and_locality.md` §3).
- **Async copy (gfx950).** `use_async_copy` moves A global→LDS by DMA and frees A's staging VGPRs (guide §11).
- Budget it with the guide's formulas: MFMAs per tile (§1.5), LDS per stage (§2.4), VGPRs (§10).

The 8-wave and 4-wave designs below put both operands through LDS and use larger tiles (FlyDSL's defaults are
256×256).

## 1. Two large-tile loop designs — race them, do not pick by rule

| | 8-wave software pipeline | 4-wave whole-loop |
|---|---|---|
| Threads / wave grid | 512, waves 2×4 (`wave_m = wid//4`, `wave_n = wid%4`) | 256, waves 2×2 |
| Occupancy | 2 waves per SIMD | 1 wave per SIMD (the whole 512-register file is one wave's) |
| K loop | FlyDSL code, `range_constexpr(K_ITERS-2)` | one inline-asm region with a hardware loop |
| LDS | 8 buffers: A/B × cur/next × two halves | operand pools of mixed depth (A 2-deep, B 3-deep) |
| Where it wins | short K, where the epilogue is a large share of the time and the second wave per SIMD overlaps it | long K, where the steady state amortizes an epilogue that nothing overlaps |
| Turbo source | FP8 `_compile_dense_nt` (:85) | FP8 `_dense_nt_wave4_asm` (:2187), `_dense_tn_wave4_asm` (:1401); FP4 `MfmaScaleFp4` |

FlyDSL's `fp8_gemm_8wave.py` has the same 8-wave structure (512 threads, eight LDS buffers). FlyDSL's
`fp8_gemm_4wave.py` and `fp4_gemm_4wave.py` use the same 2×2 wave split (a 128×128 block per wave, quadrants
c00/c01/c10/c11) but keep the K loop in FlyDSL code with an 8-buffer, depth-2 ping-pong, and use one
inline-asm block per MFMA to keep the accumulator in AGPRs. Turbo's 4-wave design runs all of K as one asm
region (§3). The LDS argument below holds for every 2×2 split.

**Why the 4-wave tile wins on long K.** A 2×2-wave split of a 256×256 tile gives each wave a 128×128
block. That is 64 MACs per operand element per K step, against 42.7 for the 8-wave's 128×64 wave blocks,
so the 4-wave tile issues 1.5× fewer LDS reads per MAC. *(field)* The 8-wave/4-wave LDS instruction ratio
was 1.49×. The 4-wave lead grew with K, from 6.4% at K=8192 to 10.5% at K=28672.
Turbo's own comment gives the mechanism on a power-capped part: "the whole-loop moves a third less LDS
traffic per MFMA, which buys clock" (FP8:3200).

**Why it loses on short K.** At one wave per SIMD nothing overlaps the epilogue. Turbo therefore only
offers the whole loop when `ceil(K/128) >= 16` (`_NT4_MIN_K_ITERS`, FP8:3203). Above that it compiles
both designs and races them on the first call for each shape (`_autotune_nt_dispatch`, FP8:3210; see
`flydsl_gemm_autotune.md`). Do the same in your own kernel: the crossover depends on the shape and on the
epilogue, so no fixed rule works.

## 2. The 8-wave pipeline (FP8 `_compile_dense_nt`)

- **LDS.** One `@fx.struct` with eight `fx.Array` fields: A/B × cur/next × two halves (FP8:134-142).
  A 256×256×128 fp8 tile uses 128 KiB. Allocate it with `fx.SharedAllocator().allocate(...).peek()`.
- **Quadrants.** Each wave owns four accumulator quadrants, c00/c01/c10/c11. One iteration interleaves
  four S2R fragment loads, the next tiles' G2S fills and four MFMA clusters (FP8:264-310):
  ```
  b0 = S2R(b_cur0); a0 = S2R(a_cur0); G2S(a_next1, k+1); s_barrier
  s_setprio(1); c00 = mfma(a0, b0); s_setprio(0); s_barrier
  b1 = S2R(b_cur1); G2S(b_cur0, k+2); s_barrier
  ... c01, c10 ...
  G2S(b_cur1, k+2); wait_barrier(2*nA + nB)          # s_waitcnt vmcnt(n); s_barrier
  s_setprio(1); c11 = mfma(a1, b1); s_setprio(0); s_barrier
  s_waitcnt vmcnt(3)                                   # end-of-iteration drain
  swap cur/next names in Python
  ```
- **Every barrier is load-bearing.** The kernel docstring says "each guards a compiler-reorder race".
  *(field)* Every attempt to drop a barrier produced an operand race; the cross-wave cause is a missing
  LDS rendezvous, not out-of-order `vmcnt`.
- **The end-of-iteration drain is a correctness knob, not a tuning knob.** `nt_vmcnt=3` is documented
  at FP8:92: N=3 gave bitwise-repeatable output at ≤1.1% cost, N≥4 raced, N<3 was slower. If you build
  this pipeline, sweep the drain with a bitwise repetition gate (`flydsl_gemm_correctness.md` §3),
  never with SNR alone.
- **`s_setprio(1)` … `s_setprio(0)` around each MFMA cluster.** At two waves per SIMD this lets the wave
  that reached its MFMAs keep the matrix pipe. Always drop priority afterwards. The whole-loop kernels at
  one wave per SIMD use none.
- **Edge cases the prologue must cover.** The prologue primes k=0 and k=1. When `K_ITERS == 2` the main
  loop is skipped, so the prologue's own wait must drain what the epilogue reads (FP8:257-262).
- **Code size grows with K** because the loop is `range_constexpr`. That is acceptable for an 8-wave tile
  at moderate K. It is one reason the whole loop exists.

## 3. The 4-wave whole loop: one inline-asm region for all of K

The K loop is a Python-generated assembly string passed to one `_llvm.inline_asm` call. The loop label
and back edge are part of the string (`s_add_u32 cnt; s_cmp_lt_u32; s_cbranch_scc1 1b`, FP8:2355-2359).
Results come back through `llvm.extractvalue` on a struct return type.

What this buys:
- No FlyDSL/LLVM boundary per iteration, so the compiler cannot insert a conservative `vmcnt(0)` or
  move instructions across the phases. Every wait is the one you wrote.
- Code size does not grow with K. The trip count can be a compile-time constant or a runtime SGPR.
- FlyDSL's own fp4 kernel avoids full unrolling for that reason: "full unroll -> 111KB > 32KB I-cache"
  (`kernels/gemm/fp4_gemm_4wave.py:651`).
- *(field)* The whole-loop rewrite was the largest single lever.

How Turbo structures it (FP8:1133-1230, 2187-2412):
- **Operand pools.** A macro tile is described as `(side, width, nbuf)` pools. The NT square tile is
  `((A,128,2),(A,128,2),(B,128,3),(B,128,3))` (FP8:2049). The B pools are 3 deep so their loads stay in
  flight across a phase drain. Deep pools are issued last, so their loads are the youngest in flight: a
  partial `vmcnt(N)` drain waits for the shallow pools and leaves the deep pools' loads outstanding
  (assert, FP8:2323).
- **Passes and phases.** One pass is `lcm(nbuf)` phases, here 6 K-blocks, so every pass ends with every
  pool back on buffer 0 and the loop body is identical each pass (`_tn4_phases`, FP8:1187; assert
  FP8:2324).
- **Inside a phase.** MFMAs run in a code-generated order: srcA pool outermost, walking a diagonal
  (`_tn4_mfma_order`, FP8:2165). Each operand fragment is re-read from LDS right after its last
  consuming MFMA. The G2S fills for a later phase are spread evenly over the MFMA slots that have no
  refill (FP8:2306). An `s_nop 0` goes between instruction groups (`_tn4_pad`, FP8:1175): at one wave per
  SIMD, memory issue arriving in bursts back-pressures the matrix pipe.
- **Phase boundary.** `s_waitcnt vmcnt(n_flight) lgkmcnt(drain_lgkm)` then `s_barrier`. `n_flight`
  counts only the fills nothing reads before a later phase. The barrier is required because a wave's
  `ds_read`s consume chunks that every wave's DMA wrote.
- **Peeled first and last phases.** The first pass is peeled so its opening MFMAs take `src2=0` and
  write the accumulators directly. That removes the zero-fill: one VALU per accumulator dword, 256 per
  tile in FP4. The closing phase is handed back as SSA values, so the epilogue can be scheduled between
  its MFMAs (FP8:2366-2368, 2703-2708).
- **Persistent carry.** When a workgroup walks several tiles, the closing phases prefetch the next tile's
  first K-blocks into buffers the next tile does not re-prime (`_nt4_carry_bufs`, FP8:2512).
- **Buffer select is free.** Each pool keeps one set of per-lane addresses. The buffer and tile deltas
  go into the 16-bit `ds_read` offset field (`_Tn4Pool` asserts the sum stays below 65536, FP8:1208).

*(field)* This emit order is a sharp local optimum. Every perturbation lost:

| Perturbation | Effect |
|---|---|
| Refill two MFMAs later (to avoid a read-after-write) | −0.9 pp |
| All G2S fills at the head of the phase | −15% (a run of `buffer_load … lds` stalls the MFMA stream) |
| Spread fills by instruction count instead of by MFMA slot | −8% |
| Separate each `m0` write from its `buffer_load` by one slot | −0.7 pp |

After the order is right, the next gain comes from changing the work (tile shape, quadrants, boundary
bodies), not from re-ordering.

## 4. Register plan and inline-asm constraints

**The plan.** Accumulators are `"=a"` outputs in AGPRs, up to 64 `vector<4xf32>` (256 AGPRs, FP8:1123).
Any beyond that go to a pinned arch-VGPR window. Operand fragments sit in a pinned VGPR window starting
at `v8` (FP8:1122). The loop counter and the per-pool soffsets are early-clobber SGPR outputs.

**Constraint rules.** Each of these has produced silently wrong output:
1. **A zero accumulator is a literal `src2=0`, never a VALU-zeroed register tied in.** The hazard
   recognizer cannot see an MFMA inside inline asm, so it cannot pad the two wait states a VALU write
   needs before an MFMA reads it. Turbo's `asm_mma_do` checks for a zero-fill vector and emits `0`
   (GH:1597-1635; the comment cites CDNA4 ISA §7.6, Table 38).
2. **An output that lives in arch VGPRs must be early-clobber (`=&v` / `=&{v[a:b]}`).** Without `&` the
   allocator may place an input (a `ds_read` base, a voffset) inside the output range. The asm then
   overwrites it mid-loop. *(field)* Moving accumulators from AGPR to VGPR without `&` passed 3 of 10
   shapes and collapsed the other 7 to 0–2 dB, depending on tile count, not on K. Every ISA check stayed
   green. Adding `&` fixed all of them and changed no ISA byte. `&` and a tied operand (`"0"`) are
   mutually exclusive: a tied output shares its input's register by definition.
3. **An unwritten `=&s` output is a register-allocation hazard** (FP8:1430). Write every declared output.
4. **A loop inside the asm needs `~{scc}` in the clobbers.** Without it LLVM can keep an outer loop's
   exit condition in SCC across the blob. *(field)* The back edge became a bare `s_cbranch_scc1` with no
   `s_cmp`, giving a wrong trip count and a memory fault. Fully unrolled callers looked fine, so the bug
   stayed latent until a loop was rolled.
5. **Waits that order an intrinsic DMA need `~{memory}`**, and asm `ds_read`s must drain their own
   `lgkmcnt` before their consumer (`flydsl_gemm_datapath.md` §4).
6. **An untied "pin" (`"=a,v"`) deletes the operand.** See `flydsl_gemm_correctness.md` §7.

**AGPR or VGPR accumulators.** Judge by the shuttle traffic (`v_accvgpr_read/write/mov` counts in the
hot loop) and by wall time, not by `accum_offset` or register totals.
- A scaled MFMA needs `dst` and `src2` in the same file but takes srcA/srcB from either. FP4 therefore
  swaps register files: the accumulator slices live in arch VGPRs and the A/B fragments in AGPRs, so the
  C store packs straight out of arch VGPRs with no shuttle (`_MXFP4_ARCH_ACC`, FP4:578-582).
- The same trade on the fp8 NT whole loop "measured a wash" (FP8:2045-2047): an MFMA writing an arch
  VGPR that a VALU then reads needs wait states that the single wave per SIMD has nothing to fill.
- *(field)* At unchanged occupancy, freed registers buy nothing on their own. Releasing 72
  VGPRs and 72 AGPRs cost −2.05%. Name the specific gain the headroom is for before building it.

**Setting `amdgpu-agpr-alloc` and friends.** Turbo passes these through `value_attrs`
(`make_value_attrs`, GH:1580). Put them under `"llvm.passthrough": [...]`. A bare `"passthrough": [...]`
key is silently dropped at `convert_gpu_to_rocdl`. A dropped attribute gives no error. An inline-asm AGPR
accumulator then gets the default allocation, and the result is wrong. After setting any kernel attribute,
read `.agpr_count`, `.vgpr_count` and the waves-per-eu fields from the ISA (`FLYDSL_DUMP_IR=1`, §10).

## 5. MFMA atom choice

- FlyDSL's guide lists the atom per dtype (`../optimize/gemm-optimization.md` §6.2). On gfx950, PS switches
  fp8 to the 16x16x128 form when `tile_k % 128 == 0` (`use_mfma_scale_128`, PS:159).
- fp8 and MX formats on gfx950 use `v_mfma_f32_16x16x128_f8f6f4`, or its scaled form for MX. The
  operand formats come from `cbsz` (A) and `blgp` (B); the operand tuple width must match the format
  (fp4 takes 4 dwords).
- **bf16: prefer 16x16x32 to 32x32x16.** Both cover the same block with the same accumulators, operand
  bytes and `ds_read` count, but each MFMA rewrites its whole accumulator tile, so the smaller atom moves
  fewer register-file bytes per MAC. *(field)* At the power cap the clock rose from 1.40 to 1.667 GHz
  and the wall fell 10.8%, while cycles per dispatch rose 8% and MFMA utilization fell from 84% to 77%.
  `SQ_VALU_MFMA_BUSY_CYCLES` was bit-identical across the swap. **Rank atoms on wall time and energy,
  never on utilization counters.**

## 6. K tails inside the kernel

Turbo accepts any K without a padding copy. FlyDSL's shipped kernels assert `K % BLOCK_K == 0`.
An over-read past K is legal only when every term it contributes has a factor that is exactly zero by
construction. The three ways to get that zero:
- **K-contiguous operands** (stored `[·, K]`, as in NT). Run `ceil(K/128)` iterations. A partial last block over-reads into the
  next row, not past the descriptor, so zero A's out-of-range K bytes in registers (`mask_a_tail`,
  GH:1551-1577; FP8:362, 384, 2712-2717). This relies on quantized operands containing no NaN encodings.
- **K-row operands** (stored `[K, ·]`: B in NN, A and B in TN). The buffer descriptor's `num_records` bound makes out-of-range loads return
  zero.
- **MX formats.** The packed E8M0 scale is zero past K, so the tail terms are scaled by 2⁻¹²⁷ twice and
  underflow to exactly zero. Drop a 128-K sub-step that is entirely padding rather than computing it.
- Row strides that are not a multiple of the 128 B cache line cost bandwidth even when the tail is
  correct. See `flydsl_gemm_datapath.md` §10.

## 7. Occupancy and allocation knobs

- 8-wave candidates race `waves_per_eu=2` with different AGPR budgets: Turbo's NT table lists
  32, 48 and 64 (`_NT_CANDIDATES`, FP8:3193).
- The 4-wave whole loop runs at `waves_per_eu=1` and uses the full register file
  (`make_value_attrs(1, 0, "256,256")`, FP8:1992). FP4 sets `amdgpu-agpr-alloc=256` through
  `llvm.passthrough` (FP4:3095).
- Arch VGPRs and AGPRs share one 512-entry file per SIMD. Allocation is in granules of 8, and the
  occupancy tiers are 64/72/80/96/128/168/256 total registers (`../../../../hardware/mi350_execution.md`).
  Arch registers are rounded up to 4 before the AGPRs are added: 101 arch + 21 AGPR is 125 total.
  FlyDSL's guide has the same model with a worked VGPR estimate for a `preshuffle_gemm` tile
  (`../optimize/gemm-optimization.md` §10).

## 8. FlyDSL's own GEMM kernels (v0.3.2): what to know when calling or tuning them

"PS" is `kernels/gemm/preshuffle_gemm.py` and "A16" is `kernels/gemm/gemm_a16w16_gfx950.py`. They handle no
K tail (K must be a multiple of the K tile), and the caller picks the configuration (aiter does it from tuned
tables). PS and the MXFP4 kernels need B pre-shuffled on the host; the fp8 4-/8-wave kernels take either
layout (`b_preshuffled`).

| Kernel | dtypes | Loop | Notes |
|---|---|---|---|
| PS `compile_preshuffle_gemm` | fp8, int8, fp16, bf16 | 4 waves along N; A through LDS, pre-shuffled B straight to VGPR | per-row × per-col scales; `use_async_copy` is gfx950-only |
| A16 `gemm_a16w16` | bf16, fp16 | `m×n×k` waves (default 2×4×1), async DMA for A and B, `stages` deep | split-K by atomics (not bitwise reproducible), `group_m` XCD swizzle, half-tile interleave |
| `fp8_gemm_8wave.py`, `fp8_gemm_4wave.py` | fp8 (row-wise scales) | the §2 pipeline / a 4-wave per-MFMA-asm loop | NT only, `K % 128 == 0` |
| `fp4_gemm_4wave.py`, `mxfp4_preshuffle.py` | MXFP4 (and MXFP8 via `mxfp4_preshuffle`) | 4-wave, scales through LDS | host-side scale and weight preshuffle |

Facts about PS and A16 that matter when tuning them:
- **PS tile algebra.** `tile_K_perm = 128 if mfma_scale else (64 if 8-bit else 32)` (PS:171). Four waves
  split N. The A tile must be a multiple of 4096 B, or its tail is never loaded (PS:190).
- **PS waits.** `rocdl.s_waitcnt(num_b_loads)` passes a raw bitfield, not a `vmcnt`. `s_waitcnt(8)`
  emits `vmcnt(8) expcnt(0) lgkmcnt(0)`. In your own code write `rocdl.s_waitcnt(vmcnt=n)`.
- **PS scheduler hints.** `sched_group_barrier` groups spread VMEM reads (mask `0x020`), DS reads and DS
  writes over the MFMAs. Without `preload=`, 8-bit tiles on gfx950 look up `_TILE_PRELOAD_TABLE`
  (PS:19), and every other dtype or tile gets `(0, 0)`.
- **PS epilogue.** Ragged M is bounded by the descriptor (`max_size=False, num_records_bytes=...`), not
  by a predicate.
- **A16 waits.** A16 waits with `rocdl.asyncmark()` / `rocdl.wait_asyncmark(stages-2)` around
  `cdna4.BufferLoadAsyncLDS128b`, which counts DMA groups instead of `vmcnt`. This is the in-DSL
  alternative to hand-written waits.
- **Configuration space.** The argument space of the aiter-shipped variants is in
  `../optimize/flydsl_levers/flydsl_knob_space.md`.

## 9. Throughput arithmetic

```
TFLOP/s = 2·M·N·K / t
bytes   = M·K·a_bytes + N·K·b_bytes + M·N·c_bytes  (+ scales)
```

The ridge is `peak / 8 TB/s`: about 312 FLOP/B for bf16 and 625 for fp8
(`../../../../hardware/mi350_overview.md`). With N and K much larger than M, a GEMM's intensity is about
M FLOP/B (bf16) or 2M (fp8). Compare against the best kernel on the same box, never against the
datasheet peak. Dense fp8 on realistic data runs at the power cap, so a reduction in register-file or LDS
traffic can show up as clock rather than as fewer cycles (§1, §5).

FlyDSL's guide gives the same formulas with per-dtype byte counts, including MX scales
(`../optimize/gemm-optimization.md` §8.1–§8.2), and the rule of thumb "M ≤ 512 is memory-bound" (§8.3). The ridge
above puts the crossover nearer M ≈ 300 on gfx950 for both bf16 and fp8, and only when N and K are much larger
than M; compute the intensity for your shape rather than using either number.

## 10. Reading the ISA

`FLYDSL_DUMP_IR=1` writes every pass to `<FLYDSL_DUMP_DIR or ~/.flydsl/debug>/<kernel>/`. It skips the
disk-cache lookup, so the dump reflects the current source. Do not time a run with it on.
`ARCH=gfx950 COMPILE_ONLY=1` compiles without launching.
- Count instructions inside the hot loop's label range only. Compare them with the plan: MFMAs per phase,
  `ds_read`s per fragment, `buffer_load … lds` per fill, and one `s_waitcnt`/`s_barrier` per phase.
  FlyDSL's guide gives the expected per-tile counts for `preshuffle_gemm` and target ratios such as "MFMA ≥ 40%
  of the loop" (`../optimize/gemm-optimization.md` §9). Use the ratios to spot a loop body far from its plan, not
  to rank changes: *(field)* deleting 7.4% of a hot loop's instructions, all SALU and `s_nop`, gained 0%
  (`flydsl_gemm_attribution.md` §4).
- Check `.agpr_count`, `.vgpr_count`, `.amdhsa_next_free_vgpr` and the scratch size after every
  register-plan or attribute change (§4).
- For a whole-loop kernel, check that the back edge has its `s_cmp` and that no compiler `s_waitcnt`
  sits between the asm phases.

## Related
`flydsl_gemm_grid_pricing.md` · `flydsl_gemm_schedule_and_locality.md` · `flydsl_gemm_datapath.md` ·
`flydsl_gemm_epilogue.md` · `flydsl_gemm_autotune.md` · `flydsl_gemm_correctness.md` ·
`../bottleneck/debug-flydsl-kernel.md` · `../../../../hardware/mi350_matrix_core.md`
