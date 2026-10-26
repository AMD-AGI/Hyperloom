---
title: FlyDSL attention decode — split-KV grids, the combine, merging them, and pricing gathers
kind: language
lever: flydsl_attn_decode_split_kv
gens: [gfx950]
dtypes: [bf16, fp8]
regimes: [decode, split-kv, mla, sparse]
updated: 2026-10-26
sources:
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_gfx950.py:1010
  - ROCm/FlyDSL@dd8373349:kernels/attention/mla_fwd_decode.py:63
---

# Split-KV decode in FlyDSL

> A decode step has few query rows, so the grid comes from splitting the KV range: a producer writes one
> partial output and LSE per split, and a combine merges them. FlyDSL's forward does this with a second
> launch (`flash_attn_gfx950.py:1010-1016`); its MLA decode takes the partial buffers as arguments
> (`mla_fwd_decode.py:63-64`). The rules below are general; constants from one model's shapes are left out.
>
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as targets.
> Kernel deltas are time (negative = faster); decode-step benchmark deltas are score (positive = better).

## Route here when
- A decode attention call is a producer plus a combine, and the pair is a visible share of the step.
- You choose the number of splits, tiles per workgroup, or the combine's grid.
- You want to merge the combine into the producer.
- A gather-based (sparse or paged) decode kernel does not respond to compute-side changes.

**Somewhere else if:** timing kernels of a few µs → `flydsl_attn_measurement.md`; XCD order of the
combine → `flydsl_attn_grid_dispatch.md` §3; GEMM-side split-K → `../gemm/flydsl_gemm_small_m.md` §4.

## 1. The combine is a fixed cost
- *(field)* One graph node cost 1.5–1.8 µs. Priced by repeated launches, the combine's body was
  0.13–0.53 µs, so ~88% of the combine was dispatch. It was 13–16% of the attention call at the larger row
  counts and 46% at the smallest, not a rounding error.
- An empty kernel's duration is not what removing a launch saves: the producer's tail was overlapped by the next
  launch. Price a merge by launching the combine one extra time in the stream and reading the increase of the total
  (`../gemm/flydsl_gemm_attribution.md` §3).
- Run the combine with one wave per workgroup (`../gemm/flydsl_gemm_grid_pricing.md` §2) and number its
  workgroups like the producer's ownership (`flydsl_attn_grid_dispatch.md` §3).
- Too few combine workgroups: split the value dimension into disjoint column slices (bit-identical).
  *(field)* Gains fell monotonically with the unsliced CTA count (−9.7% at 224, −3.7% at 512, −0.6% at
  1536) and never changed sign. Each slice recomputes the split weights from the LSE column, so more
  slices lose where that recomputation is scalar.

## 2. Merging producer and combine: a persistent, cooperative rendezvous
The merged form that pays has every producer workgroup of a row wait at a rendezvous and then share the
reduce.
1. Write the partials and LSE as write-through stores that are visible across XCDs (`SC1` = aux value 16,
   device scope, is enough across XCDs; values 16 and 17 were both bitwise clean *(field)*. `0x11` adds `SC0`,
   i.e. system scope. Value 4 is DLC and gives no cross-XCD visibility: the reader sees the same stale elements
   as with no bits). The write-through is the release; add no fence.
2. Every wave waits `vmcnt(0)`, then a workgroup barrier (`s_waitcnt` covers only its own wave).
3. One thread per workgroup arrives on a per-row sense-reversing counter: the master adds
   `SENSE − (n − 1)` with `SENSE = −(1 << 31)`, the others add 1, so each launch adds exactly `SENSE` and
   graph replays need no reset. Give each counter its own 128 B line.
4. After the rendezvous, all `waves × n_groups` waves of the row share the reduce in round-robin.

**Opening window, both required.**
- Residency: `rows × n_groups ≤ num_cu`. The rendezvous ends only if all splits of a row are resident
  at once; at most one workgroup per CU guarantees it. Still bound the poll loop, so that a broken
  assumption produces a wrong answer instead of a hung device.
- Column length: `2 ≤ n_groups ≤ 16`. *(field)* 32 workgroups on one counter: +3.25%.

*(field)* Bit-identical outputs; −1.0% to −7.6% against two kernels; +2.47% on the decode-step benchmark.

The poll: keep a nonzero sleep in the loop (*(field)* zero sleep widened the run-to-run spread to 10.2%),
and make the polled load atomic or volatile; a plain load in a store-free loop is loop-invariant to LLVM
and gets hoisted. In the merged epilogue only the number of rounds and the store width per round matter.
*(field)* Cutting the LSE reads per row by 37.5% changed nothing: they issue beside the longer partial loads.

## 3. Why a "last workgroup reduces" ticket fails here
- **Parallelism collapses.** Only the last arrival knows all splits are in, so the reduce of a row runs in
  one workgroup. With rows ≪ CUs that drops one to two orders of magnitude of parallelism. Its floor is
  (reduce work ÷ the largest tile one wave holds) × one memory round trip. *(field)* ~1 µs per loop
  iteration, ~9 µs of tail against the 2–3 µs launch it removed: +79% to +510% even with a correct release.
- **The producer pays for the reduce.** The reduce's live registers sit on the whole producer, and the
  tail that the next launch used to overlap is now exposed. *(field)* Without a fence: kernel +7% and +144%,
  decode-step benchmark −2.5%.
- **No agent-scope fence in a kernel that lives on L2 reuse.** A release/acquire fence writes back and
  invalidates the XCD's L2. *(field)* Decode-step benchmark −62%; release 5.1 µs + acquire 1.3 µs, 3–4× the whole
  epilogue. Get cross-CU visibility from per-instruction scope bits.
- The same pattern pays in a backward pass with thousands of output tiles (`flydsl_attn_backward.md` §6).

## 4. Grid size: fill the CUs once, with fat workgroups
- **Ragged groups.** Allow any `n_groups` with `ceil(n_tiles / n_groups)` tiles per workgroup; in the last
  round, point the short groups' index rows at −1 so the existing padding mask makes them −inf. P = 0 and
  the rescale factor is 1, so accumulators, running max and LSE are bit-identical. Cost: one compare and
  two selects.
- **One workgroup per CU is a cliff.** *(field)* Configurations at ≤ 1.0 CTA/CU won 1–16% (unless tiles
  per CTA grew to 16); every configuration just above 1.0 (1.09–1.31) lost 2.5–22%: some CU then runs two
  fat workgroups and that straggler sets the wall. Take the largest `n_groups` with
  `rows × n_groups ≤ num_cu`; on a tie take fewer groups (less padding). This picked the measured best on
  all nine sequence lengths tested.
- Cap tiles per workgroup and fall back to more groups beyond it. *(field)* 16 tiles per CTA lost 2–20%
  where 11 won 8–11%.
- **The residency limit is a second cliff.** *(field)* Two resident CTAs per CU: 512 → 528 CTAs was +3% work
  and +24% time.
- **The wall is the busiest workgroup's real tiles plus its footprint**, not slots per workgroup. Padding
  tiles move no data (the descriptor bound zeroes out-of-range rows) and run on non-critical workgroups.
  *(field)* The first workgroup to carry an extra real tile cost 0.49 µs, each further workgroup carrying
  one 0.18 µs; making
  dead tiles cheaper (smaller redirect window, zero-traffic out-of-range rows) lost 1–7%.
- **Attribute grid shape only at fixed total work.** *(field)* A sequence-length sweep priced half-filled
  rounds at +11.6%; at fixed total work, 1.5 → 3.0 CTA/CU bought 0.4% while the combine got 32% slower.
  The objective is producer plus combine.

## 5. Split rules
- **At least two tiles per split**, or the pipeline has nothing to overlap. *(field)* Halving the tile and
  doubling the splits at 2 tiles per split: −11%; the same CTA count with 1 tile per split: +1.8%.
- **Tiles per split jump at divisibility boundaries.** `ceil(n / (splits × tile))` steps by a whole round.
  *(field)* Widening the key set by 5 columns past a boundary cost 12–20%; by 64 columns (which divided
  again) 5–12%.
- **Pin every other launch parameter when sweeping splits.** *(field)* A launcher heuristic dropped
  `num_warps` when the split count rose: +21.8%; with `num_warps` pinned the same split read −0.4%.
- **An output that is not a reduction splits with zero merge.** Index-score logits `[rows, Skv]` split
  along KV write disjoint column ranges, bit-identical. *(field)* 1 → 32 splits: 820 → 73 µs; with 1–2
  waves per workgroup the optimum was about 4 workgroups per CU.
- bf16 partials add merge error; gate every split count on accuracy (`../gemm/flydsl_gemm_small_m.md` §4).
- Changing the reduction order changes bits: compare against a reference whose order is shared by neither
  arm (`flydsl_attn_correctness.md`).

## 6. Price the producer
- Fit `t ≈ fixed + bytes / bandwidth`. *(field)* `t ≈ 1.8 µs + bytes / (6.0–6.2 TB/s)`; the marginal bandwidth
  equalled the measured stream floor, so the body had no bandwidth left to buy. The fixed part is the
  dispatch plus the dependent first hop (index → KV address).
- Confirm the fit on a clean axis: fix the group count, CTA count and tiles per CTA, and sweep only the
  number of real tiles. *(field)* Within ±5% of the first fit.
- A plain gather with the same row width and index pattern shows what the pattern allows (random rows 4.5 lines
  wide reached only ~45% of a stream copy *(field)*), but it is not a ceiling: a tuned decode producer ran
  5.9–9.5% faster than the plain gather built to bound it *(field)*. Use the marginal-bandwidth fit above and
  never HBM peak (`flydsl_attn_measurement.md` §5).
- Before compute-side work, run the compute-additivity probe on the producer (`flydsl_attn_attribution.md` §9).
  *(field)* Four times the P·V MFMAs cost +7%. *(field)* Waves spent
  ~54% of their cycles in `SQ_WAIT_ANY` and ~20% issuing; deeper steady-state gather prefetch lost
  4.6–18%: more random 128 B requests in flight only queued.

## 7. Gathers: unique lines and the index distribution
- **The currency is unique 128 B lines per row (L2 misses).** *(field, a sparse gather prefill under uniform
  indices)* All rows L2-resident −52.9%, all
  rows MALL-resident only −10.7%, unique lines per row 5 → 3 −27.3%; cutting TCP requests per row 7 → 5
  with unique lines unchanged: +0.9%.
- **A row whose width is not a multiple of 128 B touches an extra line**: lines per row =
  ⌈(phase + width) / 128⌉, so a 4.5-line row starting at phase 0 or 64 always touches 5. Only a layout change
  reduces that, and it must be measured. *(field)* In a gather-latency-bound kernel whose production indices
  already overlapped heavily, padding rows to whole lines and splitting them into two line-aligned pools were
  all 2–5% slower.
- **Benchmark with production index structure.** *(field)* Adjacent tokens of one request overlapped
  0.69–0.92, different requests 0. With the same 9.4 MB footprint: 12.35 µs for an unstructured
  distribution, 15.44 µs for the production structure. An ownership remap that put a request's tokens on
  one XCD measured up to 10% slower on uniform random indices and 3–15% faster on production indices. Locality
  headroom of 26–29% seen on uniform sampling did not exist at production overlap. Re-measure ownership
  levers after any layout or grouping change.
- **Check a carried-over patch's enabling condition on today's grid.** *(field)* A remap gated on
  `(rows × n_groups) % 8 == 0` returned early on three of four shapes once groups became ragged. The
  remainder-spreading form (`../gemm/flydsl_gemm_schedule_and_locality.md` §2) needs no gate.

## 8. Measuring
Time decode kernels as graph replays with events, arms interleaved in one process. *(field)* A
synchronize-bracketed wall carried ~27 µs of host time and turned a 16.8% kernel gap into 38%. Protocol:
`flydsl_attn_measurement.md`.

## Verify
- Combine numbering matches producer ownership; the combine has one wave per workgroup.
- `rows × n_groups ≤ num_cu` holds wherever the cooperative merge is enabled, and the poll is bounded.
- The grid has ≤ 1 CTA per CU and ≥ 2 tiles per split on every scored shape.
- The benchmark's indices reproduce production adjacency.

## Related
`flydsl_attn_grid_dispatch.md` · `flydsl_attn_backward.md` §6 · `flydsl_attn_measurement.md` ·
`../gemm/flydsl_gemm_small_m.md` · `../gemm/flydsl_gemm_epilogue.md` §5 · `../../../../hardware/mi350_memory.md`
