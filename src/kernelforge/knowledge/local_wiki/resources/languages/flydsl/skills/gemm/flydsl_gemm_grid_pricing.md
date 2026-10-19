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
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp4_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp8_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_fp8_kernel.py
---

# Pricing a FlyDSL GEMM's grid and tile

> Most GEMM wins on gfx950 — and most wasted attempts — are decided by arithmetic you can do before
> writing code: how many workgroups there are, how full the last dispatch round is, what a split
> costs, and what a structural change could return at most. Do that arithmetic first.
>
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, never as a
> target for your shape.

## Route here when
- You are choosing `tile_m` / `tile_n`, or deciding whether to widen or narrow a tile.
- A small-M or ragged shape sits far from roofline and the loop itself looks fine.
- You are considering split-K, stream-K, a tail/boundary body, or a K-padding change.
- You are about to build a structural change and want its ceiling first.

**Somewhere else if:** the loop structure is the question → `flydsl_gemm_mainloop.md`; wave placement,
XCD order, scheduling hints or the epilogue → `flydsl_gemm_schedule_and_locality.md`; the arguments of a
shipped kernel → `../optimize/flydsl_levers/flydsl_knob_space.md`.

FlyDSL's guide starts the tile choice from `preshuffle_gemm`'s constraints and a table by M
(`../optimize/gemm-optimization.md` §1.3–§1.5), and its checklist asks for enough blocks to fill the GPU (§13).
This card is the arithmetic behind "enough": dispatch rounds, last-round fill, and what split-K or a different
tile can return.

## 1. Count workgroups and dispatch rounds — before any profile

```
tiles      = ceil(M/tile_m) * ceil(N/tile_n) * split_k
slots      = NCU * wg_per_cu                      # NCU = 256 on gfx950; query it
rounds     = ceil(tiles / slots)
last_round = tiles - (rounds - 1) * slots
eff        = tiles / (rounds * slots)             # fraction of CU slots busy over the launch
```

`wg_per_cu` comes from the occupancy arithmetic in `../../../../hardware/mi350_execution.md`; a
register-heavy GEMM is usually 1, which makes `slots = 256`.

How to read it:

| Reading | What it means | Direction |
|---|---|---|
| `tiles < NCU` | every workgroup walks all of K serially; latency is set by K, not M | split K, or shrink the tile |
| `last_round == slots` (`eff ≈ 1`) | no idle slot to recover; intensity is the limit | **widen** the tile |
| `last_round < slots` | CUs idle through the last round | **narrow** the tile (or split) so the last round fills |

Narrowing pays even when it does not remove a round — filling a partial last round is enough.
*(field)* On related MXFP6/MXFP4 GEMM kernels, widening `tile_n` 256→384 on
forward shapes whose last round was already full gained +16.9%; narrowing 256→192 gained on exactly the 6
of 12 backward shapes whose last round was partial and on none of the others (+16% where the last round
filled 144 of 256 CUs, +4–6% where it filled 128).

*(field)* Small-M example: bf16 NT, N=6016, K=7168, tile 128×128, `split_k=1` gives
`ceil(6016/128) = 47` workgroups for every M ≤ 128. The kernel ran ~112 µs flat from M=8 to M=112,
while a split-K library kernel ran ~32 µs at M=8. The grid alone explains it — read the launched grid
from a trace before reading any counter.

`../../../../common_methodology/optimization/lever_grid_sizing.md`'s "≥1024 workgroups and a
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

**How Primus-Turbo implements it** (main `7615a5a2`, files under `primus_turbo/flydsl/gemm/`):
- The gate above is Turbo's MXFP4 gate. `_mxfp4_wave_eff = tiles / (ncu · ceil(tiles/ncu))`; no split
  when `eff > 0.95` or `K < 2048`; a split must raise `eff` by more than 0.05; the race gets `s = 1`
  plus the two best splits (`_ksplit_candidates`, `gemm_mxfp4_kernel.py:3488-3512`).
- **Reduce with pairwise `torch.add(out=)` launched after the GEMM**, not inside it. It "runs at the pure write floor where
  `sum(dim=0)` reaches half of it, and the reduce is a sixth of a split launch -- the fold's form decides
  whether a split pays at all" (`_fold_mxfp4_partials`, `gemm_mxfp4_kernel.py:3540-3546`).
- **A bandwidth reduce beats an atomic-fused one** here, "since same-address HBM atomics serialise"
  (`gemm_mxfp4_kernel.py:4536-4538`). Atomic split-K is also not bitwise reproducible.
- **MXFP8 splits only the tail round.** `k_split = NCU / tail` when the tail divides the CU count, with
  at least 16 K iterations per slice; head and tail must both be whole rounds (`_mx_ksplit`,
  `gemm_mxfp8_kernel.py:654-672`). *(field)* A uniform split netted out at zero.
- **Dense fp8 TN splits one window.** Only the last partial round's `rem` tiles are sliced s ways
  (s ∈ {2, 3, 4}), choosing the s with the shortest window time (rounds × K/s; the smaller s on a tie) (`_dense_tn_split`,
  `gemm_fp8_kernel.py:1857-1872`). At s=2 the two slices hand off through a device-scope flag inside the
  GEMM, so no reduce kernel is launched (`_tn4_handoff_wait`, `gemm_fp8_kernel.py:1325`).

Refinements, in the order they usually pay:
- **Tail split.** Split only the tiles of the ragged last round. Makespan matches a uniform split, but
  partial bytes cover only the tail. If the whole-wave part and the tail are two launches on one stream
  they run back to back, so **each must be whole rounds by itself** — otherwise the idle round has only
  moved (e.g. 3.5 rounds → 3 + 1 = 4).
- **Stream-K** when you want the idle slots without a full reduce: only boundary tiles need a fixup.
- The split must cut K into whole K-blocks the kernel accepts; shipped kernels assert their own
  constraints (`../optimize/flydsl_levers/flydsl_knob_space.md` covers aiter's `flydsl_hgemm`).
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

The register tier gives waves per SIMD (`../../../../hardware/mi350_execution.md`). A workgroup's waves
must all be resident on one CU, so the reachable occupancy is quantized by workgroup size:

```
waves_per_cu = floor(4 * occ_simd / W) * W          # W = waves per workgroup
```

With an 8-wave workgroup only 2/4/6/8 waves per SIMD are reachable: cutting from 128 to 96 registers
(per-SIMD tier 4 → 5) buys nothing. Align a register-reduction target to a reachable tier before chasing
it.

The compiler's `; Occupancy:` comment in the ISA does **not** apply this quantization. Example:
`llc -mcpu=gfx950` reports a kernel using 96 VGPRs with `amdgpu-flat-work-group-size=512,512` as
occupancy 5; the CU can hold only two such workgroups, i.e. 4 waves per SIMD.

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
than the 1–2% that swizzle and candidate-table changes returned.

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
bounded the gain at 0.72%, so a half-wait reorganization was worth at most ~0.35% and was not built.

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

## 9. Pricing an alternative tile shape

A second tile shape trades arithmetic intensity for CU fill. Price both sides before adding it to the race.
- **Operand bytes per output cell are `(bm + bn) / (bm · bn)`.** Turbo's dense fp8 whole loop lets its
  narrower 256×192 tile join the race only when its gain in resident workgroups exceeds its loss in
  intensity: `coverage(rect) / coverage(square) > intensity(rect) / intensity(square)` (`_nt4_geoms`,
  `gemm_fp8_kernel.py:2102-2119`).
- **A narrow tile only wins where it removes a whole ragged round.** Turbo's MXFP4 picks BLOCK_N=192 over
  256 only when that changes the round count, pricing a k-block at the pessimistic feed ratio
  `(BM + BN) / (2·BM)` plus the C store the narrow tile can no longer fold (`_mxfp4_pick_block_n`,
  `gemm_mxfp4_kernel.py:3427-3445`). The store penalty was solved from one measured shape; re-derive it
  for your kernel.
- **Dead quadrants still issue their MFMAs.** The descriptor bound drops only their loads and stores. An
  exact tile (e.g. 192 rows when M divides by 192) therefore shortens every workgroup's own path. Turbo
  adds exact tiles only when they fit in one CU pass and cut the per-workgroup MFMAs per K block
  (`gemm_fp8_kernel.py:2066-2070, 2134-2144`).
- **Work freed on only some workgroups returns less than it looks.** On a power-capped part, idle CUs
  come back as clock for the others, not as wall time (`_nt4_wg_path` docstring, `gemm_fp8_kernel.py:2134`).
- **The race decides, not the plan.** An analytic ranking only orders the builds.

## 10. Persistent tiles per workgroup

- With `tiles_per_wg = ceil(tiles / ncu)`, a tile count between two whole CU passes rounds the resident
  workgroup count down, and the rest of the device never gets a wave. "This, not the tile count, is the
  signal" (`_nt4_resident`, `gemm_fp8_kernel.py:2089-2095`).
- Leftover work ids may repeat the last tile. That is harmless for an overwriting store and wrong for
  beta=1, which needs one tile per workgroup (`flydsl_gemm_epilogue.md` §7).
- Turbo's MXFP4 kernel walks up to 4 tiles per workgroup, and only when `tiles % (ncu · tpw) == 0` and
  K/256 ≤ 56; at longer K the per-tile launch is already amortized (`_MXFP4_TPW_*`,
  `gemm_mxfp4_kernel.py:3467-3485`). Stride the persistent loop by the grid
  (`flydsl_gemm_schedule_and_locality.md` §2).

## Verify
- The tile/round table for every scored shape exists before the first build.
- A split or tile change is judged per shape, with `s = 1` / the old tile still in the race.
- A structural change has an illegal-arm ceiling, and the arm is proven live by a checksum.
- The win does not come from one shape while its neighbours regress.

## Related
`flydsl_gemm_workflow.md` (order of operations) · `flydsl_gemm_schedule_and_locality.md` (wave
placement, XCD order, scheduling) · `flydsl_gemm_autotune.md` (racing the candidates) ·
`../optimize/flydsl_levers/flydsl_knob_space.md` (shipped kernels' split-K and tile arguments) ·
`../../../../common_methodology/optimization/lever_grid_sizing.md` ·
`../../../../hardware/mi350_execution.md` · `../../../../hardware/mi350_matrix_core.md`
