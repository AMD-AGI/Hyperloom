---
title: FlyDSL attention backward — dQ/dK/dV accumulation, determinism, workspace, masking and fusion
kind: language
lever: flydsl_attn_backward
gens: [gfx950]
dtypes: [bf16]
regimes: [backward, causal, sliding-window, gqa]
updated: 2026-10-26
---

# The attention backward pass in FlyDSL

> FlyDSL v0.3.2 ships no attention backward kernel; this card is what backward kernels written in FlyDSL
> taught: a KV-outer dK/dV body that emits dQ as split-K partials, a Q-outer dq kernel, a delta (`dO·O`)
> kernel, and the reduce that folds the partials.
>
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as targets.

## Route here when
- You decide how dQ, dK and dV are accumulated, and whether the result must be bit-reproducible.
- The dQ workspace, its reduce, or an atomic alternative dominates the backward time.
- You consider fusing dq with dK/dV, fusing the delta kernel, or splitting a kernel.
- A causal or windowed backward is slower than its FLOPs suggest.

**Somewhere else if:** grid order, XCD placement, GQA merging → `flydsl_attn_grid_dispatch.md`;
occupancy and spills → `flydsl_attn_registers_occupancy.md`; gates → `flydsl_attn_correctness.md`.

## 1. Determinism by construction
- One workgroup owns each output tile, and no float atomic touches an output.
- Reductions across workgroups go through a split-K workspace (one slot per split) summed in fp32 in a
  fixed order (ascending band). The kernel is then bitwise reproducible from run to run on any grid; a different
  band or split partition gives different, still deterministic, bits.
- Determinism also closes "relax it to drop MFMAs". Whether a non-deterministic path is acceptable is the
  owner's decision, not a performance call (§3).

## 2. The D/A law: workspace bytes are set by the register file
Let `A` be the accumulator floats one workgroup holds.
- KV-outer (workgroup holds dK/dV, dQ leaves as partials): `A = BLOCK_KV · D · 2`; partial bytes
  `(Skv / BLOCK_KV) · |dQ| / 2 = B·Sq·Skv·Hq·D² / A`.
- Q-outer (workgroup holds dQ, dK/dV leave as partials): `A = BLOCK_M · G · D` (G = GQA sharers covered);
  partial bytes are the same `B·Sq·Skv·Hq·D² / A`, and dK/dV partials usually need fp32, so ×2.
- So partial bytes per FLOP are `D / A`: no loop order removes them, KV-outer is the cheaper side, and at a
  fixed register file they grow linearly with head dim.
- Only three moves change `A`: a larger accumulator (the 512-register pool is the hardware limit), another
  storage level (LDS is usually already spent: at D128 and `BLOCK_KV = 256` dV alone is 128 KB), or recomputation
  (*(field)* splitting dV across sibling workgroups made each recompute S/dP: +20% MFMA, 0.98 ms spent to save at
  most 0.76 ms).
- *(field)* Measured on a fused body: the D128 compute phase took no more than 2× the D64 one (4.18 ms
  against 2 × 2.12 ms); the D128 excess over 2× D64 sat in the dQ workspace path: partial store 1.82 vs 1.11 ms, reduce
  0.63 vs 0.47 ms, 0.88 ms of a 0.96 ms gap.

## 3. Atomic dQ
- **fp32 atomics on dQ lose by construction.** Each dQ element receives ~`Skv / (2·BLOCK_KV)` contributions (32 at Skv
  = 8192, BLOCK_KV = 128). fp32 read-modify-write moves 2× the bytes of bf16 partials plus their fold, and the MFMA C
  layout gives one lane 8 floats of one q row, so a wave touches 32 cache lines per instruction. *(field)* 105.5 ms
  against 6.34 ms, to replace a 1.229 ms split-K round trip. Transposing the epilogue recovers 8× and is still 5×
  worse; a larger `BLOCK_KV` (fewer contributions) was 17% (D128) and 65% (D64) slower.
- **Packed bf16 atomics (`buffer_atomic_pk_add_bf16`) into a band-less dQ image** remove the workspace and
  the fold. Costs and conditions:
  - dQ is no longer bit-reproducible (dK/dV still are). *(field)* ~1.1 dB lower dQ SNR. This changes
    behaviour for every caller, so it needs the owner's consent; keep it behind an explicit
    `deterministic=False` route with the split-K path as the deterministic one.
  - The 64 lanes of one instruction must cover 256 contiguous bytes (two 128 B lines, four 64 B write
    requests). *(field)* Issued with the store's address layout, one instruction spread over 16 such 64 B
    segments and was 8.6× slower (13.3 vs 1.5 ms).
  - Issue it with the default cache policy. *(field)* With SC0 set it lost three of four updates.
  - Price the exposed fold first. *(field)* D128: fold 0.414 ms = 12.9%, atomics +2.11%, and they made a
    later +12.29% change possible; on another kernel the atomic path took D64 962.9 → 1023.8 and D128
    863.5 → 1008.8 TF/s and cut the dQ workspace from 4.8 to 0.8 GiB. Where the fold is already hidden
    (an idempotent store overlapped with the body), atomics are a pure loss.
- **dK/dV atomic epilogue loses.** The MFMA C layout spreads a wave's lanes over many lines, so each atomic
  degrades to per-line read-modify-write. *(field)* D128 0.419 → 0.554 ms (32% slower), D64 12% slower, in the same
  kernel where coalesced dQ atomics won. Gate an atomic on "does one wave's address set coalesce", not on
  dispatches saved.

## 4. Pricing the workspace round trip
- **Subtract in layers.** A "skip the partial store" probe also lets LLVM delete the GEMM that fed it; run
  the traffic-only, store-and-consumer and next-stage arms together (`flydsl_attn_attribution.md` §5).
- **Price bytes with an alias probe, not a roofline.** *(field)* Aliasing all bands onto one slab saved
  0.071 ms/GB against a roofline 0.161 ms/GB: the roofline overstated the gain 2.3×.
- **Time an auxiliary kernel alone** (`flydsl_attn_measurement.md` §9). *(field)* Widening a fold's tile
  (+10.7–12.9% in the kernel) moved wall 0.07–0.14%.
- **Re-price the fold after every body win.** *(field)* Removing it (diagnostic) cut 4.9% (0.289 ms), and 5.84%
  (0.333 ms) after the body got 4% faster with the same bytes.
- **Each q-split carries a fixed cost.** *(field)* Fixing three chunks, a 1/8 tail and a 1/4 tail both cost
  8%: the tail was ≤ 0.1%, the rest was doubled dK/dV slots and per-band K/V staging.
- **A workspace capacity cap is not free.** *(field)* Capping at 16/4/3 GiB gave 990/908/876 TF/s (D64)
  and 890/783/696 (D128): grouping bands to fit the cap trades throughput 1:1 for capacity. Check that a
  tighter cap does not switch off pair folding (§5).

## 5. Workspace layout and residency
- **Interleave bands by concurrent writer.** Descending q order makes many bands write the same q rows at
  once; in `[band][B][Sq][Hq][D]` those writes land on as many DRAM pages. `[band/ILV][B][Sq][Hq][band%ILV][D]`
  puts them on one page. *(field)* D128 (64 concurrent bands) +1.3%, flat from ILV = 4; D64 (8 bands, page
  already full) flat. A head-major workspace `[band][B][Hq][Sq][D]` was −5.8%.
- **Opposite cache policies on the two ends.** Read the partials non-temporal (they are dead after the
  read; caching them evicts the body's Q/dO); write them cached when one store covers less than a 128 B
  line (two 64 B stores merge in L2). Judge by DRAM transactions (UMC counters), not by rate. *(field)*
  NT writes added ~1.3 GB of DRAM traffic and were 0.45% slower; cached reads cost 0.1 J per pass with DRAM
  traffic unchanged.
- **MALL residency is a step.** *(field)* Clamping the fold's read footprint: 268 MB 7.66 J, 1.07 GB 8.10 J,
  4.29 GB 8.13 J. Above ~1 GB everything prices as a full miss; only a footprint within the 256 MB MALL
  drops. A slightly smaller working set buys nothing.
- **Size every buffer slice under 4 GiB.** `num_records` is 32-bit; a larger slice silently drops the
  out-of-range writes. *(field)* An interleave sweep at ILV = 64 built an 8.6 GB slice and read a fake +18%.
  See `../gemm/flydsl_gemm_datapath.md` §8.
- **Causal pair folding.** Under causal, band group `g` never writes the first `g/ng` of its slab, and group
  `ng − g` needs exactly that many rows, so the two share one slab: `ng → ng/2 + 1` slabs, no extra bytes,
  the minimum for this layout (`Sq · (ng + 1) / 2`). Remove the unpaired branch at compile time.
- **Window: size the workspace as a ring** of `ceil(W / BLOCK_KV) + 1` bands. It is a capacity lever, not
  a speed one. *(field)* 16 → 1.25 GiB at unchanged speed.

## 6. Reducing in the producer: the last-arriving workgroup
- Let the last dK/dV workgroup to finish a q tile fold its partials while they are hot, instead of a
  trailing cold reduce kernel. *(field)* +3.78%.
- Keep it deterministic: an integer counter (`global_atomic_add_u32`) with a release, no float atomics, and
  the sum taken in ascending band order regardless of who arrives last.
- `s_waitcnt` covers only the issuing wave. After a multi-wave store phase, the order is `s_waitcnt(0)` in
  every wave → workgroup barrier → the one atomic announcement.
- It pays when many output tiles reduce in parallel. With few rows per launch (decode) the same shape
  serializes; see `flydsl_attn_decode_split_kv.md` §3.

## 7. Fusion
- **Drop removable GEMMs first.** *(field)* Removing a redundant GEMM and the global rho/R renormalization
  from dq: +9.8% (accuracy cost ~0.2 dB).
- **Fuse the delta `rowsum(dO·O)` into a kernel that already holds dO.** *(field)* +5.3%, largest at long
  Sq; into a Q-outer dq kernel +1.13% for ~6 µs of extra O reads. The cross-lane sum's XOR masks must be
  the lane distances between lanes holding the same q row (16 and 32 for a 16-row MFMA tile); a mask of
  64 or more wraps onto the lane itself, doubles the value, and dropped dQ to 1.45 dB. Cast O and dO to
  the input dtype before the fused kernel reads them.
- **dq + dK/dV fusion: the form decides.**
  - Q-outer fusion reduces dK/dV over q, which takes float atomics (not deterministic) or a dK/dV partial
    workspace, which §2 shows saves no bytes.
  - A KV-outer single pass computes S, dP and the softmax once: 5 GEMM-equivalents instead of 7. It needs a
    wide KV band per workgroup, because `A` sets the dQ partial bytes (§2): the working form used
    `BLOCK_KV = 256` at D64 and 128 at D128, where the dK/dV accumulators cap it.
    *(field)* At `BLOCK_KV = 64` the same fusion was 1.95–2.55× slower (small-tile overhead plus ~2 GB of
    dQ workspace outweighed the 5-instead-of-7 GEMM saving). The wide form started below the split pair (690–785 vs
    840 TF/s) and passed it only after workspace and scheduling work (879, later ~964).
  - The crossover is real: *(field)* at 64k context, D128, fused body + reduce 226.8 ms against 191.8 ms
    split, with zero overlap and the reduce at the DRAM floor.
  - Before calling a fused path slow, A/B its host heuristics (pipeline gate, band width, q-split).
    *(field)* Reported 1.8–3.8× losses came back to 0.86–1.05× after fixing two host constants.
- **Splitting a kernel usually loses.** *(field)* dK/dV split into dV-only and dK-only kernels to reach
  occupancy 3: −36% (Q/dO read twice). Overlapping the reduce with the next band group's body: both
  fought for DRAM, wall 227 → 256 ms.
- **Price an auxiliary kernel by the main kernel's disturbed time.** An occupancy-1 body that fills the
  machine has no idle slots, so a "hidden" kernel only renames CU time.
  - *(field)* A delta kernel's per-call time beside the body spread over 25× (`flydsl_attn_measurement.md` §9); not
    chunking it at all tied (+0.03%).
  - *(field)* Aligning chunks to whole fills still left the D128 body 18% slower and the co-resident reduce
    3× slower: the binding term was the body's register headroom (498/512 → 18–23% tax; 449/512 → 3.6%).
  - Split a knob's effect into body and eviction by running it with and without the co-resident kernel.
    *(field)* A "ring depth regression" of 0.6 ms was 0.15 ms body and 0.46 ms evicted reduce.
  - The co-residency tax scales with fold bytes divided by the body work beside them. *(field)* Spreading
    the same fold over 3 unequal body chunks instead of 2: 1.8% faster; slicing the burst 8 ways: 1.35% faster,
    flat beyond 8.

## 8. Causal and window masking in the q-loop
- **Price diagonal blocks by time, not by visit share.** *(field)* In a causal KV-outer body the diagonal
  q-block was 6.06% of visits; skipping it in every wave (wrong results) bought only 0.37%.
- **Audit the masked q-loop separately**; it is a second, more expensive copy of the body.
  - *(field)* Folding `-log2e · lse` into GEMM1's C-init in the masked loop too: 0.98% faster (masked loop
    6338 → 5402 instructions, hazard `s_nop` 779 → 222 cycles per trip) from 6% of the trips.
  - With `ROWS_PER_WAVE_KV == BLOCK_Q` on square shapes each wave of a diagonal block is entirely below,
    on, or above the diagonal. Classify per wave: below needs no mask; on the diagonal skip the 16-tile
    blocks with `nt > mt`; above publishes zeros. Waves wait for each other at the barrier, so the block
    costs the slowest wave. *(field)* Diagonal MFMAs 128 → 88, critical path 675 → 560, 0.59% faster,
    bit-identical; the dq version 0.34% faster.
  - Skip with a wave-uniform branch. An MFMA ignores EXEC, so lane masks skip nothing.
  - Write the arms as sequential wave-uniform `if`s whose `else` yields the incoming accumulators
    unchanged, not as nested ifs (`flydsl_attn_registers_occupancy.md` §6).
  - Count the values a branch must carry. *(field)* In a fully unrolled dK/dV head loop the branch carried
    40 values: spill 0 → 5, 0.17% slower. Check spill before counting saved work.
- **Sliding window.**
  - Bound the q-loop by the window under a compile-time guard so the full-causal binary is byte-identical.
    *(field)* 216 → 298 effective TF/s.
  - Narrow the band with the window. *(field)* The full-causal band was the worst choice under a window:
    6.4–9.0× slower than the split path, 1.03–1.43× one step narrower; the narrow band also let interleaved
    waves share Q/dO reads; with kv tiles pruned before the grid is fixed (next bullet), that round gained
    17.79% *(field)*.
  - Prune out-of-window kv tiles before the grid is fixed, not inside the body. Keep q-split at 1.

## 9. Head dim
Retune per head dim. *(field)* A GEMM2 prefetch-ring depth of 2 was the D128 setting and 1.9% slower at
D64; band interleave paid at D128 only (§5); the D128 gap is the workspace (§2).

## Verify
- dQ/dK/dV are bitwise identical over repeated runs, or the non-deterministic route is explicit and approved.
- Every buffer slice is under 4 GiB, and the workspace capacity and cache policy were measured at both ends.
- The masked q-loop and the window bounds have their own ISA census and accuracy gate.

## Related
`flydsl_attn_grid_dispatch.md` · `flydsl_attn_decode_split_kv.md` · `flydsl_attn_registers_occupancy.md` ·
`flydsl_attn_attribution.md` · `flydsl_attn_correctness.md` · `../gemm/flydsl_gemm_datapath.md` §8 ·
`../gemm/flydsl_gemm_epilogue.md` §5 · `../gemm/flydsl_gemm_attribution.md` §2 ·
`../../../../hardware/mi350_memory.md`
