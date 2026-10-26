---
title: FlyDSL attention — working order: pin the bound, audit the grid, price the MFMA, then hide latency
kind: language
lever: flydsl_attn_workflow
gens: [gfx950]
dtypes: [bf16]
regimes: [prefill, backward, causal, sliding-window, gqa]
updated: 2026-10-26
---

# Working on a FlyDSL attention kernel

> The order in which to look at a flash-attention forward or backward kernel (fwd, dq, dkdv, fused dq+dkdv,
> and their helper kernels). It does not say which configuration wins. Numbers marked *(field)* are measured
> values; use them for direction and order of magnitude, not as targets. The round discipline — one change per
> round, hypothesis before build, escalation after rollbacks — is in `../gemm/flydsl_gemm_workflow.md` and
> applies unchanged.

## Route here when
You are starting on an attention kernel, or a run of rounds has stopped paying and you need to know which
layer you have not audited.

| The open question | Card |
|---|---|
| Where the time goes; which counter or probe answers which question | `flydsl_attn_attribution.md` |
| Grid shape, XCD placement, dispatch order, padding, GQA sharer merging | `flydsl_attn_grid_dispatch.md` |
| MFMA atom per GEMM, operand and accumulator layouts, P/dS as the next B operand | `flydsl_attn_mfma_layout.md` |
| exp2, LSE format, online-softmax rescale, row sums | `flydsl_attn_softmax.md` |
| Transposed LDS reads, row stride, bank-conflict readings | `flydsl_attn_lds.md` (with `../optimize/lds-optimization.md`) |
| VGPR budget, occupancy steps, spills, AGPR | `flydsl_attn_registers_occupancy.md` |
| `sched_group_barrier`, fences, `s_setprio`, waits, barriers | `flydsl_attn_schedule_sync.md` |
| dQ/dK/dV accumulation, determinism, workspaces, fusion, masked q-loops | `flydsl_attn_backward.md` |
| Decode, split-KV, the combine | `flydsl_attn_decode_split_kv.md` |
| Rulers, noise floors, FLOP accounting, host-bound shapes | `flydsl_attn_measurement.md` |
| Gates, spike inputs, mask conventions | `flydsl_attn_correctness.md` |
| The kernel is wrong, not slow | `../bottleneck/debug-flydsl-kernel.md` |

## Step 0 — measure the kernel that ships
- **Audit the dispatch per cell** before trusting any number: record which backend and which code path each
  scored shape really runs (`flydsl_attn_measurement.md` §8). A window wide enough to cover the whole sequence
  may be normalised to full causal; a framework default may not be FlyDSL at all.
- **Have a gate that reaches the changed path** (`flydsl_attn_correctness.md`) before the first timing.
- **Dump the facts per path**, not per code base: the MFMA atom each GEMM issues (fwd and bwd in one code base
  can use different atoms), `.vgpr_count`, `.agpr_count`, spill, waves per SIMD, LDS bytes per workgroup, and
  a per-trip opcode histogram. Every later decision reads from this table.

## Step 1 — pin the bound, per kernel and per head_dim
- **Do not assume throughput-bound.** Backward bodies ran the MFMA pipe at 44–57% duty at the true clock *(field)*;
  the rest was waits, rendezvous and serial VALU chains between GEMMs. But the bound is per cell: re-derive the
  per-pipe census for each head_dim with `v_exp_f32` at 8 cycles (the MFMA-to-exp ratio grows with D: exps fill two
  thirds of the shadow at D=64 and one third at D=128, `flydsl_attn_mfma_layout.md` §1); a reorder that paid on one D
  was noise on the other *(field)*.
- **Do not accept "issue-bound" from a paper count.** Paper issue budgets fail easily: one was off by 4×
  *(field)* because it compared per-SIMD issue slots with per-wave pipe cycles, when all waves of a SIMD share one
  MFMA pipe. Confirm with the PMC split and a deletion probe.
- Order of evidence: true clock → per-kernel additive compute probe → LDS port ratio → stall-pool split →
  subtractive probes. All of them are in `flydsl_attn_attribution.md`.

| What the evidence shows | What it usually means | Next |
|---|---|---|
| MFMA duty well under 80% at the true clock, large share where neither pipe is busy | latency, rendezvous, serial exp/pack chains | price the stall pool (Step 4) |
| MFMA busy + VALU busy near the timeline, low co-execution, occupancy 1 | the two pipes run nearly serially | overlap inside the workgroup (`flydsl_attn_schedule_sync.md`) |
| One pipe saturated in a per-class census (exp at 8 cycles, `flydsl_attn_softmax.md` §1) | throughput on that pipe | delete its work; reordering pays nothing |
| `SQ_VALU_MFMA_BUSY_CYCLES` under ~4× `SQ_LDS_IDX_ACTIVE` | LDS read bound (in fwd, usually the PV transposed reads) | `flydsl_attn_lds.md` |
| sclk far under peak with power at the cap | energy-bound | price in wall and clock (`flydsl_attn_attribution.md` §8) |

## Step 2 — audit the grid and dispatch mapping before the kernel body
Three index/order changes outside the kernel, outputs bit-identical, were worth +7% on one backward; many rounds of
tile, buffering and traversal changes inside the body had been worth ~1% *(field)*. Ask, in order:
1. **Which data do co-resident workgroups share, and does it stay in one XCD's L2?** Before writing a remap,
   compute what `flat_id % 8` already equals for the current grid: many grids make the XCD remap a
   constructive no-op, which three lines of index algebra settle without a benchmark.
2. **Is the dispatch order the list-schedule order?** Take the per-workgroup trip-count profile first; its
   shape (monotone, tent, flat) picks the order.
3. **Where does the padding land** — on the most expensive tile or the cheapest?

- Judge this layer with `SQ_WAIT_ANY` and co-execution, not with L2 hit rate: a hit rate rising from 86.5% to
  94.8% (misses −62%) bought only +0.54% on that kernel *(field)* (`flydsl_attn_grid_dispatch.md` §2).
- Sweep the batch axis before building a persistent or work-stealing scheduler for the tail; on one forward the
  grid quantisation and tail were ≤0.2% *(field)*.
- Mechanics and rules for each question: `flydsl_attn_grid_dispatch.md` §1–§8.

## Step 3 — compute the redundant-MFMA ratio before "reduce MFMA"
Removing MFMAs pays only while the body still issues MFMAs the math does not need. Do the arithmetic first:
```
issued MFMA FLOPs   = Σ over kernels of (MFMA instructions per trip × trips × FLOP per instruction)
causal-exact FLOPs  = the scored convention (5 GEMMs for bwd, attended fraction only)
redundancy          = issued / causal-exact − 1
```
- A causally trimmed D=64 backward issued only 3.1% more than causal-exact *(field)*, the half-triangles of the
  diagonal band at tile granularity. That caps every "fewer MFMA" idea on that body at 3.1%.
- Above ~15%, this is the first lever; under ~5%, go to Step 4.
- With redundancy to remove, audit droppable GEMMs and global correction terms first: dropping one GEMM from a
  dq kernel gained 9.8% *(field)*. Fusion and drop rules are in `flydsl_attn_backward.md` §7.
- Redundancy that buys a constraint is not free. A deterministic two-kernel backward recomputes S and dP in
  both kernels (7 GEMM-equivalents against the 5 the FLOP convention counts, ~29% of MACs) *(field)*; removing
  it needs float atomics or a per-trip dQ partial store (`flydsl_attn_backward.md` §1–§4).

## Step 4 — price the stall pool, then hide latency
- Every route to a speed-up goes through the stall pool; only some routes also need more co-execution. Price
  the pool by source (barrier, wait counters, LDS return, prologue/epilogue) before touching the schedule
  (`flydsl_attn_attribution.md` §3 and §6). Eight rounds of scheduling work on one body returned 0.4–4.0% in
  total while its stall pool had never been priced *(field)*.
- **When co-resident waves already fill the MFMA shadow** (no bare VALU between MFMAs in the ISA, high coexec / MFMA
  busy), reordering inside one wave is worth about zero; only deleting instructions pays *(field: forced exp
  interleave in a backward with two co-resident workgroups per CU, 0.00%)*. Where bare VALU remains between MFMAs,
  fence relaxation still paid at 2 waves/SIMD and was noise at 3 (`flydsl_attn_schedule_sync.md` §4): count
  `naked_work` first. With one workgroup per CU,
  interleaving inside the workgroup is the only cover.
- Then, by what the census names: `flydsl_attn_schedule_sync.md` (anchors, fences, priorities, barriers),
  `flydsl_attn_mfma_layout.md` (atom and chain structure), `flydsl_attn_lds.md` (transposed reads),
  `flydsl_attn_registers_occupancy.md` (what a freed register buys), `flydsl_attn_softmax.md` (exp and rescale
  cost).

## Step 5 — numerics, occupancy, determinism
Check these in order once the structure is settled: exp mode and LSE format (`flydsl_attn_softmax.md`),
whether an occupancy step is reachable without spills (`flydsl_attn_registers_occupancy.md`), and whether the
target requires bit-reproducible outputs (`flydsl_attn_backward.md` §1). A determinism requirement closes
every "relax determinism to drop MFMAs" option; decide it before Step 3, not after.

## Usually dead
Each entry held on the bodies where it was measured; re-open one only with a probe that shows its precondition
changed.
- **LDS swizzle, padding or bank-conflict work** when MFMA busy cycles are ≥4× `SQ_LDS_IDX_ACTIVE`
  (4× on a dq body, 8× on a dkdv body *(field)*): the port is not the limit (`flydsl_attn_lds.md` §5).
- **Raising occupancy by force** (`waves_per_eu`, AGPR allocation) when the code does not already nearly fit: the
  compiler spills (`flydsl_attn_registers_occupancy.md` §2). *(field)* One more wave
  per SIMD made a backward 32% slower and two more 5× slower; forcing AGPR splits took spills from 31 to 52–77.
- **Dual-wave, 8-wave, warp specialisation or stagger in a backward that already runs two workgroups per CU.** Those
  co-resident workgroups already overlap with no barrier; the rebuilt version adds a barrier tax. Its best stagger
  reached 1029 TF/s against a 1116 TF/s baseline *(field)*.
- **Pushing exp into MFMA shadows with two co-resident workgroups per CU**: 0.00% and noise-level results; raising
  MFMA priority to force it, −3.3% *(field)*.
- **Pure reordering on a power-capped kernel whose shadow is already full**: identical instructions in a new order pay
  0 ± 2% *(field, co-execution 44% of SIMD cycles, 52% of MFMA busy)*; at low co-execution reorders still gain
  (`flydsl_attn_attribution.md` §8).
- **K/V LDS double buffering to hide DMA with two co-resident workgroups per CU**: the occupancy already hides most
  DMA latency and the prefetch's LDS writes compete with the compute reads; *(field)* a deletion probe showed 8.7%
  headroom and the built prefetch lost 4.3–4.7% (`flydsl_attn_attribution.md` §5). At one workgroup per CU a second
  slot that removes a write-after-read barrier can pay, but not as a rule (`flydsl_attn_schedule_sync.md` §10).
- **A fixed reference max for the softmax shift** as a general lever: exact only inside a data-dependent window,
  silently wrong outside it (`flydsl_attn_softmax.md`).
- **Lazy (threshold) O-rescale ported on its own**: the branch broke a P-store pipeline's overlap; −31% on one
  path, neutral on another *(field)* (`flydsl_attn_softmax.md` §6.3).
- **Q-outer fused dq+dkdv**: reducing dK/dV over q takes float atomics (not deterministic) or a dK/dV partial
  workspace that saves no bytes (`flydsl_attn_backward.md` §2). A KV-outer fusion with a narrow KV band
  (`BLOCK_KV = 64`) was 1.95–2.55× slower *(field)*; the working fused structure is a KV-outer single pass with a
  wide band (`flydsl_attn_backward.md` §7).
- **fp32 atomic dQ**: each dQ element takes ~Skv/(2·BLOCK_KV) contributions; 105.5 ms against 6.34 ms
  *(field)* (`flydsl_attn_backward.md` §3).
- **Bigger tiles past the register wall**: a dkdv `BLOCK_Q` of 128 lost 67% *(field)*.
- **Tuning hand-written `vmcnt` drain levels**: removing every one of them priced at +0.3% on one forward
  *(field)*; the barrier family, not the wait family, held the time.

## Rules that save rounds
- **Before declaring a lever dead in one regime, align the other levers.** A regime that still runs a different
  softmax, mask or tile path than the one where the lever was measured can flip its sign.
- **A gate condition whose reason nobody can state is usually scope residue.** A `window_left < 0` guard kept
  causal band trimming away from sliding windows; replacing it with a window-aware q-loop bound (compile-time
  guarded) took one window shape from 216 to 298 effective TF/s *(field)*.
- **"The levers are exhausted" usually means one layer is.** List the layers (grid and dispatch, body, waits and
  barriers, helper kernels, host path) and name the last probe that priced each.
- **Re-derive per head_dim.** Anchor counts, tile sizes and which pipe binds change with D; D=64 and D=128
  sharing one file does not share gains (`flydsl_attn_measurement.md` §8).
- **Static proxies do not rank candidates.** −351 instructions was worth 0.0% and a zero-instruction scheduling
  change +0.85% on one body; a 31% longer LDS read-to-wait distance lost 0.65% on an MFMA-bound cell *(field)*.
  Rank with the counter that matches the change (`flydsl_attn_attribution.md` §2).

## Related
`INDEX.md` · `../gemm/flydsl_gemm_workflow.md` (round discipline) · `../gemm/flydsl_gemm_attribution.md` ·
`../optimize/flydsl_levers/flydsl_authoring_method.md` · `../../../../hardware/mi350_matrix_core.md`
