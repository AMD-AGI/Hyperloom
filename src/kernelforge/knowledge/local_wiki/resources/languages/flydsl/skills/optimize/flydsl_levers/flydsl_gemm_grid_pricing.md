---
title: FlyDSL GEMM — price the grid, the tile and each structural idea before building it
kind: language
lever: flydsl_gemm_grid_pricing
gens: [gfx950]
dtypes: [bf16, fp8, mxfp4, mxfp6]
regimes: [small-M, ragged-last-round, large-K, padded-K]
updated: 2026-10-19
sources:
  - ROCm/FlyDSL@dd8373349:kernels/gemm/gemm_a16w16_gfx950.py:71
---

# Pricing a FlyDSL GEMM's grid and tile

> Most GEMM wins on gfx950 — and most wasted rounds — are decided by arithmetic you can do before
> writing code: how many workgroups there are, how full the last dispatch round is, what a split
> costs, and what a structural change could return at most. Do that arithmetic first.
>
> Numbers marked *(field)* are measurements from FlyDSL GEMM work on MI355X. They are not reproducible
> from a pinned source; use them for order of magnitude and direction, never as a target for your shape.

## Route here when
- You are choosing `tile_m` / `tile_n`, or deciding whether to widen or narrow a tile.
- A small-M or ragged shape sits far from roofline and the loop itself looks fine.
- You are considering split-K, stream-K, a tail/boundary body, or a K-padding change.
- You are about to build a structural change and want its ceiling first.

**Somewhere else if:** the loop structure is the question → `flydsl_gemm_authoring.md`; wave placement,
XCD order, scheduling hints or the epilogue → `flydsl_gemm_schedule_and_locality.md`; the arguments of a
shipped kernel → `flydsl_knob_space.md`.

## 1. Count workgroups and dispatch rounds — before any profile

```
tiles      = ceil(M/tile_m) * ceil(N/tile_n) * split_k
slots      = NCU * wg_per_cu                      # NCU = 256 on gfx950; query it
rounds     = ceil(tiles / slots)
last_round = tiles - (rounds - 1) * slots
eff        = tiles / (rounds * slots)             # fraction of CU slots busy over the launch
```

`wg_per_cu` comes from the occupancy arithmetic in `../../../../../hardware/mi350_execution.md`; a
register-heavy GEMM is usually 1, which makes `slots = 256`.

How to read it:

| Reading | What it means | Direction |
|---|---|---|
| `tiles < NCU` | every workgroup walks all of K serially; latency is set by K, not M | split K, or shrink the tile |
| `last_round == slots` (`eff ≈ 1`) | no idle slot to recover; intensity is the limit | **widen** the tile |
| `last_round < slots` | CUs idle through the last round | **narrow** the tile (or split) so the last round fills |

Narrowing pays even when it does not remove a round — filling a partial last round is enough.
*(field)* Two campaigns on related MXFP6/MXFP4 GEMM kernels, same GPU, same week: widening `tile_n` 256→384 on
forward shapes whose last round was already full gained +16.9%; narrowing 256→192 gained on exactly the 6
of 12 backward shapes whose last round was partial and on none of the others, and the gain tracked the
idle fraction (last round 144/256 → +16%, 128/256 → +4–6%).

*(field)* Small-M example: bf16 NT, N=6016, K=7168, tile 128×128, `split_k=1` gives
`ceil(6016/128) = 47` workgroups for every M ≤ 128. The kernel ran ~112 µs flat from M=8 to M=112,
while a split-K library kernel ran ~32 µs at M=8. The grid alone explains it — read the launched grid
from a trace before reading any counter.

`../../../../../common_methodology/optimization/lever_grid_sizing.md`'s "≥1024 workgroups and a
multiple of 8" is not sufficient: 1152 tiles at one
workgroup per CU is 4.5 rounds with the last round half idle.

A narrower tile also costs reuse: LDS/global reads per tile scale with `tile_m + tile_n`, MFMA work with
`tile_m · tile_n`. A geometry change that helps one shape moves others the other way, so evaluate it
across the whole shape set you are scored on.

## 2. Split-K: gate on wave quantization, then price the reduce

**Gate.** Propose a split `s` only if `eff(tiles · s) > eff(tiles) + 0.05`; on ties take the smaller
`s`, because partial-sum bytes and reduce traffic scale with `s`. Keep `s = 1` in the candidate set and
choose by measurement with a margin, so a split can never regress a shape. "Few tiles" is only one case
of this gate — a grid that is full but ragged (1.5 or 3.5 rounds) qualifies too.

**Price.** The split recovers at most the idle part of the launch, `t · (1 − eff)`. The reduce costs
roughly `(s · M · N · partial_bytes + M · N · out_bytes) / BW`. If the reduce costs more than the idle
time it recovers, no split wins.
*(field)* M=4096, N=2880, K=32768 on 256×256 tiles (192 tiles, one round): ideal-occupancy time saves
27.9 µs, an `s=4` bf16 reduce costs ~42 µs ⇒ `s=1` is correct; the autotuner that picked it was right.

Refinements, in the order they usually pay:
- **Tail split.** Split only the tiles of the ragged last round. Makespan matches a uniform split, but
  partial bytes cover only the tail. If the whole-wave part and the tail are two launches on one stream
  they run back to back, so **each must be whole rounds by itself** — otherwise the idle round has only
  moved (e.g. 3.5 rounds → 3 + 1 = 4).
- **Stream-K** when you want the idle slots without a full reduce: only boundary tiles need a fixup.
- The split must cut K into whole K-blocks the kernel accepts; shipped kernels assert their own
  constraints (`flydsl_knob_space.md` covers aiter's `flydsl_hgemm`).
- If the scoring harness rejects GPU work from kernels you did not write, the reduce has to be your own
  kernel in the same language, not a framework op.

**The reduce kernel itself.** It has many more workgroups than CUs, so occupancy is not its limit; the
number of waves per workgroup only decides whether the sum crosses waves (barriers and LDS round-trips).
Try one wave per workgroup first. *(field)* A combine kernel went from 4 waves to 1: `s_barrier` 5→0,
time −3.7…−6.7%. A changed reduction tree is not bit-identical to the old one — gate it against an
independent reference, not old-vs-new max diff.

## 3. Asymmetric operand widths: the best tile is not square

Per K element, a workgroup moves `tile_m · A_row_bytes + tile_n · B_col_bytes` to produce
`tile_m · tile_n` MACs:

```
traffic_per_MAC = A_row_bytes / tile_n + B_col_bytes / tile_m
```

Under an accumulator cap `tile_m · tile_n ≤ C`, this is minimized at

```
tile_m : tile_n = B_col_bytes : A_row_bytes
```

With equal operand widths (bf16×bf16, fp8×fp8) that is square, which is where the square-tile habit comes
from. With MXFP6 A (96 B per 128-K row) and MXFP4 B (64 B per 128-K column) the optimum is
`tile_m : tile_n = 2 : 3`, e.g. 256×384. *(field)* 256×384 beat 256×256 by 3.7% on one shape and 5.4%
over four; 128×512 lost 13.9%.

Then check two things before building it:
- **Divisibility.** `N % tile_n == 0` on your shapes. Non-power-of-two widths (192, 384) often divide model
  dimensions of the form `3·2^k` better than 512 does.
- **Accumulator placement.** A wide tile can exceed the 256-dword AGPR file per lane; see
  `flydsl_gemm_schedule_and_locality.md` §7.

*(field)* The same rule predicted gains on `tile_m = 128` shapes that measured as 4–9% regressions —
another term dominated there. The ratio tells you where to look, not what will win.

## 4. Occupancy moves in workgroup-sized steps

The register tier gives waves per SIMD (`../../../../../hardware/mi350_execution.md`). A workgroup's waves
must all be resident on one CU, so the reachable occupancy is quantized by workgroup size:

```
waves_per_cu = floor(4 * occ_simd / W) * W          # W = waves per workgroup
```

With an 8-wave workgroup only 2/4/6/8 waves per SIMD are reachable: cutting from 128 to 96 registers
(per-SIMD tier 4 → 5) buys nothing. Align a register-reduction target to a reachable tier before spending
a round on it.

The compiler's `; Occupancy:` comment in the ISA does **not** apply this quantization. Checked with
`llc -mcpu=gfx950` (AMD LLVM 22.0.0git, ROCm 7.2.1): a kernel using 96 VGPRs with
`amdgpu-flat-work-group-size=512,512` is reported as occupancy 5; the CU can hold only two such
workgroups, i.e. 4 waves per SIMD.

## 5. K granularity: the MFMA decides what a K-tail can save

The MFMA's K is a hard granule. A tail body that skips K padding saves instructions only if

```
ceil(K_real / inst_k) < ceil(K_pad / inst_k)
```

`K_real = 2880`, `K_pad = 2944`, `inst_k = 128` gives 23 and 23 — the tail body saves no MFMA, only the
last iteration's feed bytes. Confirm the granule from counters: `SQ_INSTS_MFMA` should equal
`M·N·K / (m·n·k of the instruction)`.

A bigger MFMA shape is not a throughput lever. *(field)* On gfx950,
`v_mfma_f32_32x32x64_f8f6f4` occupies the matrix pipe exactly twice as long as
`v_mfma_f32_16x16x128_f8f6f4` (`SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA` = 64 vs 32), so pipe time per
FLOP is unchanged; it needs 4× the accumulator registers per instruction and leaves fewer independent
accumulation chains to hide dependencies. Swapping only the main-loop MFMA was 0.8–3.0% slower on 12 of
12 shapes. Use the larger shape to fix K granularity, not to go faster.

Whether an instruction exists on the target is a five-second check:
`echo '<instruction>' | llvm-mc -arch=amdgcn -mcpu=gfx950 -show-encoding`.

## 6. Fit the per-tile fixed cost before polishing the loop

At fixed M and N, sweep K and fit

```
t_tile = a · n_kblocks + F
```

`F` — prologue, epilogue, tail peel, address and descriptor setup, drain — is paid once per tile, and its
share `F / (a · n_kblocks + F)` grows as K shrinks. Across a shape table it usually explains the TF/s
ranking better than any other variable. *(field)* `F` = 2.21 K-blocks: 19% of tile time at K=2944, 1.7%
at K=32768, and the table's TF/s order was essentially the inverse of that share. By the fit, removing
one K-block of `F` is worth ~4–5% over that table — a projection, not a measurement, but already larger
than the 1–2% the swizzle and candidate-table sweeps measured.

Two traps when reading such a table:
- **Compare absolute µs, not TF/s, across padded shapes.** TF/s counts useful FLOPs, so padding waste
  shows up as "efficiency" instead of as work done.
- **"Divisible ⇒ fast" is not a rule.** *(field)* K=2816 (11 K-blocks) ran slower than K=2944 padded to 12
  blocks, because an odd block count fell off an unroll-2 pipeline's store-folding path. Before adding a
  shape, evaluate the kernel's own gates (parity of block counts, tail predicates) for it.

## 7. Bound the gain before building: the illegal arm

For any proposal that shortens a wait, removes a transfer or skips work, first build an arm that removes
it entirely — the results are wrong, the time is the ceiling. Prove the arm really compiled in with an
output checksum (an absurd sum shows it did); a "change" that never reached the binary reads as noise.
If the ceiling is below what you need, skip the idea. *(field)* Removing the whole first prologue wait
bounded the gain at 0.72%; the planned half-wait reorganization was worth at most ~0.35% and was dropped.

## 8. Boundary variants are not free

Each extra body variant (a quarter-N or quarter-M tail, a half-K body) adds code even when it never runs.
*(field)* One variant added 739 instructions (+39.5% of the module) and cost 0.32 points on every shape
compiled into that module, with VGPR, SGPR and LDS identical across arms — code size alone.

```
net = hit_rate × gain_per_tile × shapes_that_trigger − carry_cost × shapes_in_module
```

Measure the hit rate on the scored input distribution first: a runtime ragged tail can be 0% on balanced
inputs. When a variant does pay, prefer re-assigning waves over computing less — a ragged tile wastes
whole waves pinned to an empty region, and re-tiling the waves (e.g. 2×2 → 4×1) serves the N tail, while
the same body rotated 90° serves the M tail. Write the gate as geometry (`m_tail <= rows_per_wave`), not
as shape constants.

## Verify
- The tile/round table for every scored shape exists before the first build.
- A split or tile change is judged per shape, with `s = 1` / the old tile still in the race.
- A structural change has an illegal-arm ceiling, and the arm is proven live by a checksum.
- The win does not come from one shape while its neighbours regress.

## Related
`flydsl_gemm_authoring.md` (order of operations) · `flydsl_gemm_schedule_and_locality.md` (wave
placement, XCD order, scheduling, epilogue) · `flydsl_knob_space.md` (shipped kernels' split-K and tile
arguments) · `../../../../../common_methodology/optimization/lever_grid_sizing.md` ·
`../../../../../hardware/mi350_execution.md` · `../../../../../hardware/mi350_matrix_core.md`
