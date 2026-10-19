---
title: FlyDSL GEMM — the epilogue: what a C store costs on gfx950 and how to make it cheaper
kind: language
lever: flydsl_gemm_epilogue
gens: [gfx950]
dtypes: [fp8, mxfp4, mxfp8, bf16]
regimes: [prefill, training, beta-accumulate, fused-epilogue]
updated: 2026-10-19
sources:
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/utils/gemm_helper.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_fp8_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp4_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp8_kernel.py
---

# The FlyDSL GEMM epilogue

> "GH" is `primus_turbo/flydsl/utils/gemm_helper.py`, and "FP8"/"FP4"/"MX8" are the dense fp8, MXFP4 and
> MXFP8 kernels under `primus_turbo/flydsl/gemm/`, all at Primus-Turbo main `7615a5a2`. "PS" and "A16" are
> FlyDSL's `kernels/gemm/preshuffle_gemm.py` and `kernels/gemm/gemm_a16w16_gfx950.py`. Numbers marked
> *(field)* are measured values. Use them for direction and order of magnitude, not as targets.

## Route here when
- The C store, a beta=1 read-back or a fused activation/quantization takes a visible share of the kernel.
- You are choosing a store width, a cross-lane reorder, or an LDS-staged (CShuffle) epilogue.
- An occupancy-1 kernel has an epilogue that nothing overlaps.

## 1. The facts that set the price

- **MFMA output layout.** In a 16×16 accumulator tile, a lane holds 4 consecutive rows of one column, and
  neighbouring columns sit in neighbouring lanes. A naive bf16 store therefore writes 16 lanes × 2 B =
  **32 B per request**.
- **The gfx950 write request is 64 B, not the 128 B cache line.** *(field)* `TCP_TCC_WRITE_REQ` matched
  bytes/32 for scalar stores and bytes/64 for paired stores exactly. Widening further to `dwordx2` halved
  the issued store instructions but left the requests unchanged, and wall time got 0.85 pp worse. So
  pairing two columns into one dword already reaches the request floor. Wider stores only save
  instructions.
- **Pairing pays `64 B ÷ (W × 2 B)` for a W-wide fragment.** For 32-wide MFMA fragments, the scalar store
  already covers 64 B, so pairing has no request savings left there. *(field)* It gained +1.3%, against
  +2.8–3.5% on 16-wide fragments.
- **At the power cap, store savings show up as clock, not as wall time inside a tile.** *(field)* Deleting a
  GEMM's whole epilogue at the power cap moved a same-process wall probe by 0 ± 1 pp, while the
  benchmark gained. Price store changes with `TCP_TCC_WRITE_REQ` (the mechanism) plus the real benchmark (the
  size), not with an in-process wall probe.

## 2. Decide whether the store matters before changing it

Run three ablations of the epilogue:
1. **Delete the whole store.** This gives the total cost.
2. **Same bytes, fewer segments.** This measures the request cost.
3. **Same segments, fewer VALU instructions.** This measures the issue cost.

Only (1) moving means the store is byte- or bandwidth-bound, and width or reorder work will not help.
*(field)* Ablation (2) gave +0.8%, (3) gave −0.5% and (1) gave −20.3%, so the store was
byte-bound. Check that an ablation arm writes the same `M×N×2` bytes to the same rows. *(field)* A
probe that collapsed four rows into one segment also cut the written bytes about 6×, and read as a false
request win.

*(field)* The "best" epilogue does not carry over between shape families. A permlane wide store was optimal on
dense 8192²×4096 (16 KB rows, `N % 256 == 0`). The same store was 0.7–1.3% slower on an NT shape with
5888 B output rows.

## 3. Store classes, from free to expensive

FlyDSL's guide describes the two ends (`../optimize/gemm-optimization.md` §7): a **direct store** from the MFMA
layout, and **CShuffle**, which writes the accumulators to LDS row-major, re-reads them as `(MLane=8,
NLane=32)` after a barrier and stores 4-element vectors, for `tile_n ≥ 128`. In FlyDSL v0.3.2, PS stores
directly and can fuse a per-N bias with ReLU/SiLU/GELU (`epilogue=`); A16 and
`kernels/common/mma/mfma_epilogues.py` carry the CShuffle store. Turbo keeps one store class per technique
between those two ends (GH:778-1525). Prefer the cheapest class that reaches the 64 B floor.

| Class | How a lane gets adjacent columns | Cost | Where |
|---|---|---|---|
| Scalar | none: `buffer_store_short` per value, column step in the immediate | baseline, 32 B requests | FlyDSL PS direct store; Turbo `StoreCPerTensor`, GH:778 |
| **Row-permuted feed** | permute the **global rows the B fill fetches** so a lane's two n-fragments become adjacent output columns | **zero instructions**: only the prologue's per-lane global offsets change | `StoreCPerTensorPairCol`, GH:1061; `compute_global_swizzle_pair`, GH:256 |
| DPP pair | `v_cvt_pk_bf16_f32` + one `quad_perm` DPP swap + `v_perm_b32` | 1 cvt + 1 DPP + 1 perm per pair | `StoreCPerTensorPairN`, GH:1113 |
| Row merge | one `v_permlane16_swap_b32` per packed fragment pair moves the second fragment into the other 32-lane half, so one store covers a whole 64 B row run | 1 permlane per pair | `StoreCPerTensorRowN`, GH:1004 |
| Quad/fold | the row permutation folds 4 or 8 columns into one lane: one `dwordx2`/`dwordx4` | offsets only | `StoreCPerTensorQuadN`, GH:1251 |
| Line via LDS | transpose through LDS with `ds_read_b64_tr_b16` into a whole 128 B line per fragment row; scratch borrowed from an operand buffer | LDS traffic + fence | `StoreCPerTensorLineN`, GH:1180 |
| CShuffle | LDS staging, then vector stores | LDS + barriers | FlyDSL A16 `write_cshuffle_vec_to_global`, `c_shuffle_epilog`; Turbo `StoreCPerTensorCShuffle`, GH:1335 |
| Transposed accumulator | swap the MFMA operands so the accumulator holds Cᵀ; two `permlane16_swap`s give 8 contiguous columns, one `dwordx4` | 2 permlanes per 8 columns | FP4 `store_tacc_wide`, FP4:513 |

How to choose:
- **First, see whether the reorder can be an address change.** When the output column is the B operand's
  slow axis (NT: output column = B row), permute B's rows in the G2S fill using the involution
  `W·t + m ↔ 2·m + t` within each `2W`-row block. The fill already takes arbitrary per-lane global offsets,
  and the LDS swizzle key still follows the LDS row, so the reader does not change. *(field)* This gave
  +2.7%: `buffer_store_short` 160→32, `buffer_store_dword` 0→64,
  and `v_cvt_pk` 160→96. The MFMA, barrier, VGPR and LDS counts were unchanged.
- **When the output column is the operand's fast axis (NN/TN), use the row merge.** The transpose read's
  lane→column map is fixed by hardware, and a 16 B fill chunk holds 16 whole columns, so neither can be
  permuted. Pack each row pair (`v_cvt_pk_bf16_f32`) first, then swap the packed dwords. *(field)* Swapping
  unpacked f32 values took 64 permlanes and gained +0.13%; swapping packed pairs took 32 and gained +0.83%.
  **Price a cross-lane primitive by the output values it moves per instruction.**
- *(field)* On the same kernel, every option that paid cross-lane VALU or an LDS round trip for a
  wider store lost: CShuffle −21%, DPP pairing −1.1 to −1.9%, transposed-accumulator wide store −0.7 to
  −1.3%. The free address permutation was the only gain.
- CShuffle only pays at `tile_n ≥ 128`. Its two barriers are pure waste on narrow tiles.

**Instruction-level rules for cross-lane stores**
- A `permlane16_swap` result consumed only by a `buffer_store` (VMEM) needs no `s_nop`. That hazard covers a
  following VALU read (FP4:500-502).
- When a VALU consumes the result, use the `rocdl.permlane16_swap` / `permlane32_swap` intrinsics, not inline
  asm. The hazard recognizer cannot see asm, so asm forces an `s_nop` on both sides. *(field)* Switching to the
  intrinsic cut `s_nop` from 412 to 174.
- Issue all swaps first, then all stores, so the permlanes and the stores are far apart. *(field)* Only the two
  changes together, dropping the hard-coded `s_nop` and the two-phase order, took the count from 80 to 0.
- **Never store the high half of an f32 accumulator as bf16.** That is truncation, not round-to-nearest-even.
  *(field)* It cost 5.8 dB SNR and made the store slower. Convert with `v_cvt_pk_bf16_f32`. FlyDSL's buffer
  store cannot emit `buffer_store_short_d16_hi`, so taking the high half of a packed pair costs one
  `v_lshrrev_b32`.
- On gfx950, a multi-dword MUBUF VGPR tuple must be 64-bit aligned.

## 4. Overlap the store with the last MFMAs

At one wave per SIMD nothing overlaps the epilogue unless you put it inside MFMA shadows.
- **8-wave.** Store quadrants c00/c01 between the c10/c11 MFMAs of the last iteration (FP8:383-400).
  MXFP8 laps two quadrants into the last K iteration. "Past 2 the store temporaries outlive an MFMA pair
  and VGPRs spill" (`_MX_EPI_LAP`, MX8:91-93).
- **Whole loop.** Stop the asm loop one phase early and hand the last phase's MFMAs back as SSA values
  (`flydsl_gemm_mainloop.md` §3). Issue them row-tile by row-tile and put each row-tile's store after the
  *next* row-tile's MFMAs. *(field)* Distance 1 was best; 2 and 4 were worse.
  - The hazard recognizer cannot see asm MFMAs. Put a `sched_barrier(0)` before each store group and
    trailing `s_nop`s after the last row-tile. Otherwise a VALU reads a stale accumulator, and only the last
    row-tile is wrong.
  - The ceiling is one K-block of MFMAs. *(field)* That recovered about 10% of the store cost (+0.7–1.1%).
- **Fold the store into a phase with no G2S.** gfx950 has a single in-order `vmcnt` for loads and stores,
  so a store in flight is serialized against operand DMA waits. FP4 therefore puts its fused C store in a
  peeled tail phase that issues no fills, and paces it at `_CRATE` lines per MFMA (FP4:776, 1095-1233,
  1745). Spent accumulators can be recycled as pack registers when the caller never reads them
  (`acc_dead`, FP4:679).

## 5. Cache policy on the store

- **C is write-once: store it non-temporal** (`nt`, aux=2). Caching it evicts the A/B band that the L2 tile
  order is keeping (FP8:72-73).
- For a beta=1 burst too large for L2, use device scope `sc0|sc1|nt` (aux=19). Turbo switches at 26 MB
  (`_cstore_aux`, FP8:74-77, 1885). That threshold is a measured value.
- **The sign is shape-dependent, so race it.** *(field)* In the MXFP8 GEMM, `nt` cut the wide shapes by 4.6%
  and 3.0% and cost 0.9–1.9% on narrow shapes whose slabs were already L2-resident. It is raced at a 1%
  margin, not pinned (MX8:968-974).
- **Never mark byte-granular scale stores (E8M0) non-temporal.** They rely on L2 merging the write masks.
  *(field)* In the MXFP8 cast kernels that cost 15–40%.
- Keep split-K partials cached, because the reduce reads them back.
- **Pass cache-policy bits (`aux`) as a Python int.** `rocdl.raw_ptr_buffer_load`,
  `raw_ptr_buffer_load_lds` and the buffer atomics replace any `aux` that is neither an `int` nor an
  MLIR attribute with `None`, without an error
  (`python/flydsl/expr/rocdl/__init__.py:663-819`). A traced value there silently means "default
  policy".

## 6. Bounds without per-column masks

- Rebase the C descriptor per row band in i64 and pin base and `num_records` with `readfirstlane`
  (`make_row_band_resource`, GH:2010). A wide store has no per-column mask, so a whole column span is
  dropped by giving its band 0 records. Store classes that span several fragments require
  `N % tile_n == 0` (`col_safe`).
- FlyDSL applies `mask=` after the element→byte multiply, as `select(mask, off, 0x7FFFFFFF)`. Apply the
  predicate to the part of the offset that does not depend on the sub-tile, then add the sub-tile row
  delta. *(field)* Otherwise `v_cndmask_b32` went from 14 to 64.
- One descriptor per tensor per tile is enough when sub-tiles differ only by a compile-time row offset.
  Carry the row delta in the offset instead of building a descriptor per sub-tile.

## 7. beta = 1 (accumulate into C)

- Pipeline the read-back: load quadrant i+1 of C while quadrant i's store is in flight (`_store_quadrants`, GH:1528).
  Add in f32 so the result is rounded once.
- **Compile and race against a scratch output.** Compiling runs the kernel once, which would accumulate
  into the caller's C twice (`compile_with_scratch_out`, GH:86-95). Turbo races at beta=0 and rebuilds the
  winning configuration with the accumulate epilogue (FP8:3008-3027).
- A persistent kernel that repeats its last tile for leftover workgroup ids is fine for overwrite stores
  but double-accumulates at beta=1. Turbo forces one tile per workgroup there.

## 8. Fused epilogues (activation, GLU, quantization)

- **Price the fusion before tuning inside it.**
  - **Tax** = fused kernel minus the same-shape bare GEMM, from one kernel trace. The two kernels often
    share a symbol name; group them by `vgpr_count`, `lds_size` and grid.
  - Then compare the tax with the unfused chain.
  - *(field)* An MXFP4 dGELU + dual-quant epilogue cost 101.9 µs inside the GEMM. The standalone quantizer
    took 100.8 µs, so the fusion broke even. The epilogue inherited the GEMM's one workgroup per CU, which
    dropped its memory efficiency from 4.3 to 2.9 TB/s.
- **A static VALU instruction count is not a bottleneck diagnosis.** Use it to confirm a mechanism landed,
  and test causality by deleting instructions and watching wall time. *(field)* In the case above, deleting
  192 VALU (2.7% of the kernel) changed nothing.
- **Measure how much VALU the main loop can absorb before moving work into it.** Insert N dependent FMAs in
  the K-loop gaps with a side-effect sink. *(field)* At occupancy 1, the first 8 FMAs were 65% absorbed and the
  next 8 only 25%. The free pool was about 38% of the epilogue, about 4.8× less than a static estimate.
- **Two co-resident workgroups (occupancy 2) need `LDS ≤ 80 KiB` per workgroup and
  `arch + AGPR ≤ 256` registers per wave, for 4-wave workgroups.** Check both before planning to hide the epilogue behind a second workgroup.
- **Make fused operands meet in one lane.** For GLU (B = gate‖up, `[2I, K]`), point the second B pool at the
  *up* band, `I` weight rows further in, instead of at the next columns. `gate[m, j]` and `up[m, j]` then land
  in the same lane at the same fragment index, and the epilogue pairs them without a shuffle (FP4:2281-2283,
  `gemm_mxfp4_glu_quant_flydsl_kernel`).
- Turbo's fused GLU/dGLU + MXFP4 quantization epilogues are bit-exact against the standalone quantizer
  (`primus_turbo/flydsl/utils/gemm_epilogue_helper.py`).

## Related
`flydsl_gemm_mainloop.md` (peeled phases, register plan) · `flydsl_gemm_datapath.md` (row-permuted
fills, i64 descriptors) · `flydsl_gemm_schedule_and_locality.md` (L2 order the `nt` policy protects) ·
`flydsl_gemm_autotune.md` (racing the store policy) · `flydsl_gemm_correctness.md`
