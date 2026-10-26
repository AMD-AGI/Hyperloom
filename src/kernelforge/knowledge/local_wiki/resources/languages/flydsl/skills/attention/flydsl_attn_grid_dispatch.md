---
title: FlyDSL attention — grid, dispatch order and XCD placement, audited before the kernel body
kind: language
lever: flydsl_attn_grid_dispatch
gens: [gfx950]
dtypes: [bf16]
regimes: [prefill, backward, causal, sliding-window, gqa]
updated: 2026-10-26
sources:
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:1000
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:4418
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_interface.py:1075
---

# Grid and dispatch for a FlyDSL attention kernel

> Which workgroup gets which `(batch, head, tile)`, in what order, and where the padding falls is index
> arithmetic outside the kernel body. On attention it often pays more than the body: *(field)* three
> index-only, bit-identical changes (XCD-major decode, LPT order, padding origin) gave +7% on a backward
> pass whose body tuning had returned ~1%.
>
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as targets.

## Route here when
- You start on an attention kernel, or the body has stopped paying.
- A causal, windowed or rectangular shape is slower than its FLOPs suggest.
- You are merging GQA heads into one workgroup, or changing the number of waves per workgroup.
- A consumer kernel (combine, reduce) reads what a producer kernel wrote.

**Somewhere else if:** split-KV decode and the combine → `flydsl_attn_decode_split_kv.md`; dQ/dK/dV
workspace and fusion → `flydsl_attn_backward.md`; generic round counting and tile pricing →
`../gemm/flydsl_gemm_grid_pricing.md`.

## 1. The audit, in order
1. What do co-resident workgroups reuse? Make that axis adjacent inside an XCD (§2).
2. Is the work per workgroup uniform? If not, the dispatch order is a list schedule (§4).
3. Where does the tile padding fall — on the most or the least expensive tile (§5)?
4. Which axis does the grid spread over? Key the tile heuristic on it (§6).
5. How many `(batch, kv-head)` slices are live per XCD at once (§7)?

Judge changes at this layer by wall time, `SQ_WAIT_ANY` and `SQ_VALU_MFMA_COEXEC_CYCLES`. An L2 hit
rate can rise sharply and buy nothing.

## 2. XCD-major mapping
- Workgroup `i` runs on XCD `i % 8`, and each XCD has its own 4 MB L2 (`../../../../hardware/mi350_chiplet.md`).
  FlyDSL's launcher states the same mapping for its forward grid (`flash_attn_interface.py:1075`).
- **Give each XCD a whole `(batch, kv-head)` block**, so the GQA workgroups that read the same K/V share one
  L2. *(field)* dq: L2 hit 86.5% → 94.8%, misses −62%, +0.54%; dK/dV kernel +2.81%.
- **The fastest axis inside the block depends on the kernel.** Ask what co-resident workgroups reuse:
  the Q-outer dq kernel wanted the GQA sharers of one kv head adjacent, the KV-outer dK/dV kernel wanted
  adjacent q positions (q-split index fastest). *(field)* The same remap with the inner order swapped:
  −2.5% instead of +2.4%. Do not copy the order from another kernel.
- **Use the remainder-spreading permutation.** `x*(G//8) + min(x, G%8) + bid//8` with `x = bid % 8`
  (`../gemm/flydsl_gemm_schedule_and_locality.md` §2) is a bijection for any grid size `G`, so the
  `num_kv_heads % 8 == 0` gate goes away. *(field)* With 6 kv heads, falling back to the plain decode for head
  counts not divisible by 8 left 9.1% of a windowed dK/dV body on the table.
- **Check bijectivity over every grid you launch.** The same binary is often launched on a narrower chunk
  grid than the deployment shape; a remap validated on `(B, Hkv)` faulted on a one-batch chunk.
- **Compute what `flat_id % 8` already is before changing anything.** A head-fast grid `(Hq, n_q, B)` with
  `Hq % 8 == 0` gives `flat_id % 8 == h % 8`; with `h_kv = h % Hkv` and `Hkv == 8`, every XCD already owns
  one kv head and any XCD remap is the identity. *(field)* An XCD regrouping by kv head on such a kernel
  read −0.19% with memory counters bit-identical. FlyDSL's forward applies its head-slow remap to
  non-causal grids only (`flash_attn_utils.py:1000-1013`).
- **A co-resident stream reverses the sign.** *(field)* A contiguous per-XCD slice for a reduce launched
  alone cut its time 5.4% / 1.8%. The same slice for the same reduce running beside the body made it 12%
  slower, because it pinned
  the reduce's reads onto the L2 sets the body was writing through. Beside a co-resident body, offset the
  phase inside a slice instead of giving the stream a slice of its own.
- vL1D is not a reuse level for sharers. *(field)* Placing the 8 GQA sharers of a forward tile on one CU
  cost 0.57%, with `TCP_TCC_READ_REQ` and `TCC_HIT` unchanged: one workgroup's own 32 KB K/V double buffer
  fills the vL1D.

## 3. A consumer's numbering must match the producer's ownership
A kernel that reads another kernel's output should give the reads of one row to the XCD that wrote it.
*(field)* A split-KV combine with a 1-D row-major grid (`row * blocks_per_row + b`) spread each row's reads
over `min(blocks_per_row, 8)` L2 domains, while the producer owned `split * seq + token`. A 2-D grid
`(seq, blocks_per_row, 1)` makes the flat id `b * seq + row`, the producer's form: 18.6% and 7.0% faster on
two decode shapes, ~0.5% faster where the producer's own traffic had already evicted the partials. It is a pure
permutation. Before splitting any grid finer, check that the low 3 bits of the flat id did not change
with it. The decode grid-size cliff is in `flydsl_attn_decode_split_kv.md` §4.

## 4. Dispatch order is a list schedule
- The dispatcher issues workgroups in block-id order, so the grid order is the schedule. Under a causal
  mask the work of a q-tile grows as `(q_tile + 1) * BLOCK_M / BLOCK_KV`: dispatch in descending q-tile
  order (longest processing time first). *(field)* dq +2.5%; forward +2.34%, but only together with a
  loop-carry split, because the plain reorder raised VGPR 162 → 170 across an occupancy step.
- FlyDSL's fp8 forward does this by reversing the q-block index under causal (`init_causal_lpt_order`,
  `flash_attn_utils.py:4418-4429`).
- **Do not assume in-order is already right.** *(field)* In one code base the dK/dV kernel's in-order grid
  happened to be LPT and the dq kernel's was reversed. Read the decode of every kernel separately.
- **Choose the order from the profile shape.** Monotone → descending. Unimodal → start in the middle and
  walk out (`i → mid ± i/2`, O(1) index math, bit-identical). *(field)* Rectangular causal plus a finite
  window gave a tent of trip counts (2 at the ends, 16 in the middle); ascending and descending were
  equally bad, middle-out cut body time 4.3% and wall 2.7%. Constant → nothing to win.
- **Do not make q-blocks the fastest axis under causal**; it clusters unequal work (FlyDSL records 7% for
  this, `flash_attn_utils.py:1005-1006`).
- **Trust a greedy "N slots per XCD" simulator for the first-order profile only.** *(field)* It predicted
  the LPT gain (2.3% vs 2.5–2.9% measured), but two second-order phasings it priced 3.8% faster measured
  within noise and 0.4% slower. The model has no zero-work workgroups covering others and no cross-XCD DRAM
  contention.

## 5. Padding and empty work
- `num_q_tiles * BLOCK_M - Sq` rows of overshoot anchored at row 0 all land in the last tile, the one with
  the longest causal range. Move the origin down by `floor(pad / BLOCK_KV) * BLOCK_KV` so they land on
  tile 0. *(field)* +0.96% (kv-block visits 7481 → 7396). It needs `BLOCK_M % BLOCK_KV == 0`, the first
  tile clamped to row 0, and the store bounded by the rows the tile owns (shared rows are recomputed and
  written once, so the output stays deterministic).
- **Skip empty segments on the host.** A block-diagonal varlen pattern (context parallel) with most
  segments empty launches workgroups that only exit. *(field)* ~9 µs per empty segment; 64 segments with
  one non-empty: 0.740 → 0.153 ms by launching only the non-empty ones. The crossover was at about 1/8 of
  segments non-empty.

## 6. Key the tile heuristic on the axis the grid spreads over
A KV-outer kernel's grid is `B · Hkv · ceil(Skv / BLOCK_KV) · q_split` (fills the CUs with Skv); a Q-outer
kernel's grid scales with Sq. *(field)* Choosing the KV-outer kernel's tile and split from Sq cost 19% at
Sq = 2048, Skv = 16384. Square shapes never expose this; put rectangular shapes in the acceptance set.

## 7. The first-order term is live `(batch, kv-head)` slices per XCD
At occupancy 1, more ready work per CU is not a gain. What matters is how many operand sets each XCD's L2
must hold at once. *(field)* Running two batch-disjoint chunks concurrently, with bytes, slots and
workgroup count identical, made the dK/dV body 32% slower: a resident window of 256 consecutive block ids
shares ~2 MB of Q/dO plus 0.5–2 MB of K/V, and two chunks doubled that against a 4 MB L2. FlyDSL's forward
launcher comment reports the same term from the other side: the penalty for leaving its remap off tracks
the number of K/V streams per XCD, ~6% at 8 streams, ~3% at 4, nil at ≤ 2 (`flash_attn_interface.py:1075-1085`).
Keep one slice live per XCD; do not overlap chunks on separate streams to "fill" the machine.

## 8. Scan the batch axis before chasing tails
Before writing a persistent or work-stealing scheduler to collect the tail, fix the work per workgroup and
sweep only the batch. *(field)* B = 1..16 read 1146.7 → 1195.7 TF/s with the scored shape already on the
asymptote, so grid quantization plus LPT tail was ≤ 0.2% and that family was closed. A sequence-length
sweep moves the grid and the per-workgroup work together; doubling S bought +0.7%, i.e. ~1.4% of
amortizable fixed cost. Read a pool from two or more points of a curve, never from an end point.

## 9. Bytes per score and GQA sharer merging
- For a kernel that streams K/V once per workgroup, K/V bytes per score are `4D / (NUM_WAVES × ROWS_PER_WAVE)`:
  only the q rows one workgroup covers matter, not `BLOCK_N` or the number of heads it serves. *(field)*
  2 → 4 heads per workgroup with `block_m` 128 → 64 (to keep 8 waves) left sectors and ISA bit-identical.
- Extra K/V bytes can cost cycles rather than energy. *(field)* Doubling them cost 3.69% wall with MFMA-busy
  bit-identical and the clock flat; the cost sat in `SQ_WAVE_CYCLES` (+4.04%) and `SQ_WAIT_ANY` (+6.54%).
  Halving requests by merging sharers (below) paid through the clock instead: read sclk and `SQ_WAVE_CYCLES` for
  every byte change.
- **Two sharers per workgroup.** Run the same q-tile index of two q-heads of one kv-head in an 8-wave
  workgroup (low wave bits pick the row group, high bits the head), sharing one K/V LDS tile and barrier.
  *(field)* Forward at the power cap: 2.04% faster, `TCP_TCC_READ_REQ` −49.5%, clock +8.7% at unchanged
  power, cycles +6.6%. The whole gain came through the clock.
- **Precondition: the merge must not lower workgroups per CU** (`../gemm/flydsl_gemm_grid_pricing.md` §4).
  *(field)* In a VGPR-locked occupancy-2 backward, the same merge left one workgroup per CU with nothing
  to cover barriers and drains: dq −2.7% (the dq kernel itself −6.5%), an 8-wave dK/dV −4.3%, outputs
  bit-identical.
- **Four sharers lose through barrier-domain width.** *(field)* A 16-wave workgroup with registers,
  waves per SIMD, bytes and instruction counts all identical was −3.1% (another reading of the same change
  −4.3%): 16 waves now wait for each other. Where the merge also lowered waves per SIMD (3 → 2) it lost 2.0%.
- **20-second gate:** read `TCP_TCC_READ_REQ` on your kernel and on the reference. A ratio under 1.1 means
  the request path is already level and a merge has nothing to save.
- Changing waves per workgroup changes the rendezvous pool. *(field)* Removing every barrier (diagnostic,
  wrong results) gained 0% with three 4-wave workgroups per CU, +2.4% with two 8-wave ones, and 0% with one
  16-wave one. It also changes how many DMA instructions each wave issues: recompute every `vmcnt` that
  was written as an instruction count.

## 10. Pipeline along the GQA head axis
A backward body that loops over the q-heads sharing one kv-head has a second loop axis to pipeline.
- Issue head `h+1`'s exp2/pack work inside head `h`'s GEMM2 MFMA run. *(field)* +1.4% along the head
  axis; the same pipelining along the d-tile axis was negative.
- Issue the next head's LSE/delta loads at the last d-tile of the current head's GEMM2. *(field)* +1.81%,
  register-neutral and bit-identical. On an occupancy-1 fused backward, carrying it across the q-block
  boundary (the last head prefetches the next q-block's head 0 through the loop's iter_args) added +0.29% /
  +0.55% *(field)*; on the occupancy-2 body it was neutral (the next q-block's own DMA drain already covered
  head 0).
- This hides latency only. If the load's stall is a full VMEM issue queue, moving it earlier does nothing.

## Verify
- The `flat_id → (batch, head, tile)` decode is written down, with `flat_id % 8` evaluated for each shape class.
- Every remap is checked as a bijection over every launched grid width.
- The dispatch order was chosen from the measured trip-count profile.
- Rectangular, windowed and small-batch shapes are in the acceptance set.

## Related
`flydsl_attn_backward.md` · `flydsl_attn_decode_split_kv.md` · `flydsl_attn_attribution.md` ·
`../gemm/flydsl_gemm_schedule_and_locality.md` §2 · `../gemm/flydsl_gemm_grid_pricing.md` ·
`../../../../common_methodology/optimization/lever_xcd_locality.md` · `../../../../hardware/mi350_chiplet.md`
