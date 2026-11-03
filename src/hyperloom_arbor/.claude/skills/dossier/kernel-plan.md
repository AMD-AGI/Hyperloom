# Part 11: a kernel plan, for a rewrite

A rewrite is the larger of part 11's two kinds of direction: rebuilding a
kernel or a fused region, rather than tuning or joining what is there. It
takes many iterations before it can serve, so a campaign spent on one needs a
plan it can be held to: what is being built and why it can win; the stages
that each show progress; and the point at which to stop. One plan per
rewrite, written into the dossier as `11.uK-plan-<region>.md` and cited from
11.uK's rewrite table. A campaign spent on the rewrite gets the plan as pack
knowledge.

Write it from parts 3, 5, 7 and 8, and 10.uK's kills. Every number has its
source and label, as everywhere in the dossier.

## 1. Target

- **The region:**
  - the ops it covers, from the first launch to the last;
  - the call sites (part 3, file and method);
  - the kernels that serve it now (part 8);
  - their launches per step.
- **The shapes it serves:**
  - per operating point, the rows per step and the context, as distributions
    (part 5), not the client's caps;
  - which step kinds reach it: decode, prefill, mixed.
- **Its share:** of step time per regime and step kind, from part 7's
  stratified tables, with the capture's window and validation status.
- **What is inside the region and what stays out.** Name the boundary's
  inputs and outputs, their layouts and dtypes, and which collectives sit
  on either side. A boundary that cuts through a collective needs part 2's
  reading of that collective.

## 2. Ceiling

- **The floor at the served shapes,** for each shape the region serves (or
  the weighted cells part 5 gives), from the arithmetic:
  - bytes it must move (weights, activations, KV, scales);
  - FLOPs it must do;
  - launches it cannot avoid.

  Turn each into time with the device's measured bandwidth and peaks: the
  hardware skill's "What our nodes achieve" table
  (`.claude/skills/hardware/SKILL.md`), or a measurement of your own, never
  the datasheet. Its reference pages give the instruction-level limits (MFMA
  rates, LDS banks, direct-to-LDS widths) a kernel design has to respect.
- **The gap:** the region's time now minus the floor, in µs per layer, per
  step and as a share of the run.
- **The gain in the target's unit** (part 4), the gap scaled by how much of
  it a rewrite can plausibly take. Put it beside the noise floor (part 6) and
  the gate. A plan whose plausible gain is under 3× the gate goes back to
  the increments table.
- **What the floor rests on.** Is it measured, reasoned, or measured with the
  kernel under test? A kill that rested on the last of these is reopenable
  (the update round's 10.uK). If part of the region is idle, waiting on a
  collective or the host, say how much: a faster kernel cannot win that time
  back.

## 3. What exists

- Part 8's row for each op in the region:
  - the kernels shipped and unused, and why;
  - AITER's tuned tables and fallbacks;
  - KernelForge's pages;
  - what the reference platform runs (part 9) and the gfx950 gap.
- **What can be reused:** device functions in the tree, AITER, FlyDSL or CK
  primitives, an existing kernel to extend.
- **What has been tried:** 10.uK's kills in this region, each with what it
  rested on. A plan that redoes a killed approach says what is different now.

## 4. Design

- **Algorithm and fusion boundary:** what one launch does; why that boundary
  and not a smaller one.
- **Data layout:** weights (pre-shuffled at load?), activations, scales; what
  the producer and the consumer outside the region expect.
- **Grid and scheduling:** programs per token, per expert or persistent;
  occupancy at the served shapes; behaviour at the edges (one row, the
  largest capture size, empty experts, degenerate routing).
- **Precision:** what must be bitwise against the path it replaces, what may
  move by a stated ulp or SNR, and why. Accuracy trades are declared as such
  and start behind a switch that is off by default.
- **Graphs and compilation:**
  - It must capture under the served graph mode.
  - Counts that vary per step are runtime arguments, not compile-time
    constants. A `tl.constexpr` route count compiled 2,218 variants and
    stalled the engine for 30 s mid-bench (Model M B4, iteration 22).
  - Target zero JIT compiles during inference.
- **gfx950:**
  - wave64;
  - MFMA and scaled-MFMA shapes for the number formats in the region (MXFP4,
    FP8, BF16);
  - LDS per CU (check `rocminfo` on the node);
  - CU count, and the XGMI bandwidth when a collective borders the region.

  Port a reference-platform idea only after naming the gap that could block
  it.
- **The language:** Triton, FlyDSL, CK or HIP, and why. A kernel that belongs
  in AITER (a tuned row, an operator AITER already owns) is written to land
  there, and the plan says so. The `kernels` skill says which tool fits which
  kernel, with its sources; run its `probe-image.sh` before the plan depends
  on a tool the image may not ship.

## 5. Stages, each with its own gate

A rewrite cannot clear the campaign's A/B gate until its last stage, so
each stage gets its own gate. That keeps progress measurable and lets the
campaign stop early. The harness writes the stage records: each attempt,
with its iteration, its numbers and its evidence, lands in the session's
`stages.json`, so the plan does not need updating as it runs.

| Stage | Done when | Stop when |
|---|---|---|
| S1 Reference | A standalone harness reproduces the region's time now, at the served shapes, graph-captured, within a stated tolerance of the in-situ profile (§2) | The re-measured gap is under 3× the gate |
| S2 Correct | The new kernel matches the replaced path over the served shapes and the edge cases, to §4's tolerance, with a negative control that fires | It cannot meet the tolerance without giving up the design |
| S3 Fast | Graph-captured time at the served shapes takes a stated share of the gap (say half), with control drift reported | After its iteration budget, it is under that share |
| S4 In tree | Behind a `VLLM_*` switch that defaults off, declared `SWITCH: NAME=1` beside `DONE:`. The harness checks it at the KEEP, serving the switch on: dispatch logged on every rank, 0 JIT compiles in serving, accuracy gate passed (`arbor/measure/switches.py`) | Dispatch fails at a served shape and the fix breaks S2 or S3 |
| S5 A/B | The campaign's own gate, at every operating point the target names | The gate: KEEP, REVERT or a Pareto point, as part 4 classes it |

Lessons the stages carry, from Model M campaigns B4 and C4:
- **Time graph-captured, not eager:** eager timing is host-bound and read
  one saving at half its size (B4 iteration 47).
- **Gate correctness before timing,** with a negative control. Two checks
  that passed vacuously were caught that way (C4 iteration 34).
- **A fixture can rank right and price wrong.** C4 iteration 31's served
  saving was 334.7 µs per step, against 110–192 projected, so only S5
  books a gain.
- **Prove the arm by an artifact the run owns,** such as a cache file or a
  marker the kernel writes, not by a log line that may be swallowed (C4
  iteration 34).
- **Probe before building:** pricing the design's parts first changed the
  design (B4 iteration 26: the split grid was 2× the interleaved one).

### The `## Items` block

The plan ends this part with its items, in the one line the harness reads
(`arbor/stages.py`). Each design is its own chain of items after `s1`, so a
losing design does not hold up the others:

```
## Items

ITEMS: [
 {"id": "s1", "stage": "S1", "title": "Reference: the chain in situ",
  "gate": {"measure": "region_us", "reference": 1647.8, "tolerance_pct": 10},
  "stop": {"measure": "gap_x_gate", "below": 3, "status": "below_floor"}, "stops_plan": true},
 {"id": "c.s2", "stage": "S2", "after": ["s1"], "gate": {"require": ["correct", "negative_control_fired"], "measure": "snr_db", "min": 30.0}},
 {"id": "c.s3", "stage": "S3", "after": ["c.s2"], "gate": {"measure": "gap_share", "min": 0.5}},
 {"id": "c.s4", "stage": "S4", "after": ["c.s3"], "gate": {"require": ["switch_default_off", "dispatch_all_ranks", "zero_jit", "accuracy_pass"]}},
 {"id": "c.s5", "stage": "S5", "after": ["c.s4"], "attempts": 2}]
```

- **`id`:** lower case, `s1` or `<design>.<stage>`. In a campaign the item's
  ID is the map ID, a dot, and this: `m-finalize-chain.c.s2`.
- **`stage`:** S1–S5 here. An architectural plan uses ARM (one layout at
  every point of a table) and FRONTIER (the dominating layout lands); a port
  of upstream work uses CORRECT, PORT (the commit named) and AB.
- **`after`:** the items it waits for. A pass needs them done; a stop does
  not, so an item can be killed unbuilt.
- **`gate`:** `measure` with `reference` and `tolerance_pct`, `min` or `max`;
  `require`, flags that must be true. S2 always requires
  `negative_control_fired`. S5 carries no gate: the campaign's decides it.
- **`stop`:** `measure` with `below` or `above`, and the `status` it closes
  with (`killed` or `below_floor`). With `stops_plan`, a stop kills every
  later item.
- **`attempts`:** failed attempts that kill the item (S5: REVERTs).

Check it with `python3 -m arbor.stages check <plan.md>`.

**In the campaign, the agent declares each stage's result on one line:**

```
STAGE: {"item": "m-finalize-chain.c.s2", "result": "pass", "measured": {"snr_db": 32.69, "correct": true, "negative_control_fired": true}, "evidence": {"artifacts": ["iter_20/peer_rs_sc_c5.json"], "ledger": ["c5-sc-snr"]}}
```

`result` is pass, fail or stop. The harness judges the line at the next
iteration's start, once this one has its verdict. It checks that:
- the item is ready;
- the iteration's intent fits the stage: LEARN for S1–S3, FIX or stricter
  for S4, WIN for S5 (a stop, which kills, may come from any);
- every artifact was written during the iteration, and every ledger ID is in
  the ledger;
- the numbers clear the gate. A pass that does not is booked as a fail, with
  why.

S4 also needs the iteration to KEEP, and the harness's check of its
switches to pass. At every KEEP, lane or not, the harness serves the tree
again with each `SWITCH:` on and refuses the KEEP when:
- the arm sets the switch on;
- the accuracy check fails with it on;
- a JIT cache gains a file once that server is healthy;
- or the log has no line naming the switch from every rank.

So the kernel logs one line with the switch's name where it dispatches.
S5 needs no STAGE line: a booked KEEP whose DONE line cites the item closes
it. Turning the switch on is held to the accuracy gate.

## 6. Budget and the campaign around it

- **The budget,** in iterations or hours: per stage, and in total. Also say
  what is kept if the budget runs out: the harness, the floor, the kills.
- **The campaign:**
  - one rewrite per campaign;
  - its config sets `"space": {"levels": ["kernel"]}`;
  - its pack raises `policy.max_consecutive_probes` to cover S1–S3, which
    are probes;
  - its pack knowledge is this plan plus the dossier's digest.
- **Checkpoints the operator sees:** the end of each stage, with its
  numbers.
- **Two ways to run a plan:**
  - **As one single-node campaign.** `python3 -m arbor.stages seed --config
    <campaign config> --direction m-<x> --plan <the plan>` makes the
    campaign's session one lane: its map is the plan's items, with a pinned
    copy of the plan. The campaign stops when every item has its verdict,
    with `LANE_CLOSED.json` and a STOP that starts `lane-closed:`.
    `python3 -m arbor.stages show --session <dir>` shows where it is.
  - **As a kernel-plan lane in the parallel mode** (runner decision 0003;
    coming): the coordinator seeds the same lane on its own node, beside the
    round's other lanes.

  Either way, while the next item is S1–S3 the LEARN cap does not demote
  it, and nothing reopens. A direction the work turns up is proposed for the
  next round's map, not added to this plan.

## 7. Landing

- **Where the code lives:** in the served tree behind a switch first. Tuned
  rows and operators AITER owns go upstream to AITER; the plan names which.
- **The switch:** off by default until the A/B and an accuracy check with
  the switch unset have both run. A default-on change keeps its switch.
- **Tests at the production shapes:** they assert the path ran, gated to
  the architecture so they can't pass vacuously.
- **The PR:** the target project's rules (for Model M on mpk, the runner's
  `plans/model-m/MPK_PR_GUIDELINE.md`).

## 8. Risks

Numerics outside the measured shapes; shapes a later workload brings; compile
time and memory at load; collectives that move with the deployment (TP, EP,
sequence parallelism); the maintenance cost of a hand-written kernel against
the library it replaces. Each risk gets the test or the stage that would show
it.

## Worked example

Two plans from the Model M dossier
(`docs/dossier/model-m-mi350x/`), written and run by campaigns'
agents on MI350X, 2026-10-13 to 10-14. One built a slice of its rewrite and
kept it; the other stopped at S1, as it should have. Iterations are the
sessions' own (`session.json` lessons); the gate in both was 0.5 %.

### (a) C5, rewrite R1: the MoE finalize chain

The plan is `u3-c8/11.u3-plan-finalize-chain.md`, written at C5's iteration 1.
Its region is a reduce-scatter, a finalize, a copy and an all-gather per layer
and microbatch on the mixed step: 1647.8 µs per call, which the compute stream
waits on. It carried three designs as one ladder, S1 to S6: S-A (a staged
bf16 peer reduce-scatter), S-B (S-A without its staging copy) and S-C (S-A
with an MXFP8 transport, an accuracy trade).

What each design's stages measured, mapped onto §5:

| The plan's stage | §5 stage | Iteration | Measured | Verdict | Recorded in |
|---|---|---|---|---|---|
| S1 | S1 Reference | 1 | chain 1647.8 µs (reduce-scatter 1472.95); compute stall 55.42 ms/step, 2.31 % of run, 4.6× the gate | done | `u3-c8/07.u3-profile.md`; the plan's §1–2 |
| S-A: S2 | S2 Correct | 2 | 55.42 dB against fp32, RCCL's ring 51.43; both negative controls fire | pass | iteration 2's lesson; `11.u3-addendum-collective-stream-placement.md` §1 |
| S-A: (C4's fixture) | S3 Fast | C4 it 23–25 | 1209.0 µs against RCCL's 1472.95: 264 µs, 0.65 of the 404 µs gap; priced at +0.79 % of run | pass, on a fixture | the plan's §2; `collective-stream-placement` §1 |
| S-A: S3 Dispatched | S4 In tree | 2 | graph capture survives; 4 of 4 ranks installed and dispatching through call #1000 | pass | the same addendum, §1 |
| S-A: S4 A/B | S5 A/B | 2 | REVERT: −1.9 % (125.08 against the 127.54 bar), median TTFT +7.4 % | killed | iteration 2's lesson |
| (S-A explained) | — | 3–6 | wrong stream (3), corrected: already on a side stream (4), a DBO pairing loss (5); in situ (6): the kernel is 186.7 µs faster than the ring, but two rendezvous barriers cost 299.2 µs per call and the staging copy 300.9 µs, 3.1× its idle price | — | addenda `collective-stream-placement`, 2, 3, 4 |
| S-B: S5 | S4, S5 | 6 | not built: without the copy, the chain is still 121.6 µs per call slower than RCCL's, about −0.6 % of run | killed unbuilt | addendum 4 §6 |
| S-C: S6 | S2 Correct | 17, 20 | 17: the transport at 32.5–32.9 dB sits 12.3 dB below the MXFP4 partial's own error (+0.26 dB in all); 20, on the shipped module: 32.69 dB, three negative controls firing | pass | addenda 15; 18 §2 |
| S-C: S6 | S3 Fast | 18–20 | 18: at the shipped grid MXFP8 is no faster (1283.8 against 1239.9 µs), at 32 workgroups 737.5; 19: the local pass priced cold, S-C at +0.63 to +1.71 % of run; 20: the whole link 795.1 µs against RCCL's 1489.8 (bf16 staged 1395.6), +1.01 to +1.14 % of run | pass | addenda 16, 17, 18 §3 |
| S-C: S6 | S4 In tree | 20 | four switches (`VLLM_ROCM_MODEL_M_PEER_RS`, `_STAGED`, `_FMT=mxfp8`, `_LOCAL_BF16`), off by default; vLLM only; declines under capture, so decode is untouched | pass | addendum 18 §1, §5 |
| S-C: S6 | S5 A/B | 20 | KEEP: +1.9 % (137.59 against that node's 135.02 bar), p90 interactivity −0.1 % | done | iteration 20's lesson; runner `plans/model-m/07_C5_RESULTS.md` |
| (after it) | a new direction | 21–23 | the fused form (MoE stage 2 writes MXFP8): priced at +0.85 % of run (22); its first step, stage 2's tile 128 → 256, REVERT −0.1 % (23) | — | addenda 19, 20, 21 |

S-A's S4 and S5 ran in one WIN iteration, and so did S-C's. In this format
they are two: a FIX that lands the code behind its switches, off, and must
KEEP (S4), then the WIN that turns them on (S5).

The `## Items` block the plan would carry (the format of §5). Each design is
its own chain after `s1`; S-B builds on S-A, so it waits for `a.s5`:

```
## Items

ITEMS: [
 {"id": "s1", "stage": "S1", "title": "Reference: the finalize chain and the stall it releases, in situ",
  "gate": {"measure": "region_us", "reference": 1647.8, "tolerance_pct": 10},
  "stop": {"measure": "gap_x_gate", "below": 3, "status": "below_floor"}, "stops_plan": true},
 {"id": "a.s2", "stage": "S2", "title": "S-A: the staged bf16 peer reduce-scatter against fp32", "after": ["s1"],
  "gate": {"require": ["correct", "negative_control_fired"], "measure": "snr_db", "min": 51.44}},
 {"id": "a.s3", "stage": "S3", "title": "S-A: the peer kernel at the served payload", "after": ["a.s2"],
  "gate": {"measure": "gap_share", "min": 0.5}},
 {"id": "a.s4", "stage": "S4", "title": "S-A: in the tree, off by default", "after": ["a.s3"],
  "gate": {"require": ["switch_default_off", "capture_survives", "dispatch_all_ranks", "zero_jit"]}},
 {"id": "a.s5", "stage": "S5", "title": "S-A: the campaign's A/B", "after": ["a.s4"], "attempts": 1},
 {"id": "b.s4", "stage": "S4", "title": "S-B: the restricted hook, no staging copy", "after": ["a.s5"],
  "gate": {"require": ["switch_default_off", "capture_survives", "dispatch_all_ranks", "zero_jit"]}},
 {"id": "b.s5", "stage": "S5", "title": "S-B: the A/B", "after": ["b.s4"], "attempts": 1},
 {"id": "c.s2", "stage": "S2", "title": "S-C: the MXFP8 transport against fp32", "after": ["s1"],
  "gate": {"require": ["correct", "negative_control_fired"], "measure": "snr_db", "min": 30.0}},
 {"id": "c.s3", "stage": "S3", "title": "S-C: the whole link at the served shape", "after": ["c.s2"],
  "gate": {"measure": "gap_share", "min": 0.5}},
 {"id": "c.s4", "stage": "S4", "title": "S-C: four switches, off by default", "after": ["c.s3"],
  "gate": {"require": ["switch_default_off", "dispatch_all_ranks", "zero_jit", "accuracy_pass"]}},
 {"id": "c.s5", "stage": "S5", "title": "S-C: the A/B", "after": ["c.s4"], "attempts": 2}]
```

What it teaches:
- **Designs are separate item chains.** The plan ranked S-C last, behind S-A
  and S-B, so it was not priced until iteration 17, after both had died. As
  its own chain after `s1`, it could have started at iteration 2.
- **Explain a losing A/B in situ before killing its sibling.** Iteration 2's
  −1.9 % was read three ways in three iterations before iteration 6 traced
  it in the server: the barriers and the copy, not the kernel. That killed
  S-B on evidence rather than by association, and it made S-C's case: under
  MXFP8 the copy becomes the format conversion the transport needs anyway
  (addendum 18 §1).
- **The records were scattered.** C5 wrote 21 addenda, 11 on this plan, and
  the plan's own stage table was never updated past S1. Here the harness
  writes the stage records (§5): `stages.json`, each attempt with its
  iteration, numbers and evidence.
- **A lane ends at its last verdict.** Every item had one at iteration 20.
  The fused form came after it; it is a proposal for the next round's map,
  not more of this plan.

### (b) B5, rewrite 3: the decode layer's collective count

The plan is `11.u3-plan-decode-collectives.md`. Its prize was measured
first: 2.06–2.88 % of run per decode collective removed, 9.81 % for all four.
It had no mechanism, so S1 was a dependency census from the source, with a
hard stop: no removal priced at 3× the gate, no rewrite.

| The plan's stage | §5 stage | Iteration | Measured | Verdict | Recorded in |
|---|---|---|---|---|---|
| S1 census | S1 Reference | 7 | the four collectives are totally ordered by data dependence, and the model has no shared expert to hide one behind; the diamond fee, re-measured, is 5.8–6.4 µs, not the 14.8–16.6 µs carried from B4; the best single collective nets +0.83 % of run, 1.66× the gate | stop: below_floor | the plan's "S1 RESULT"; `11.u3-directions.md` §11.11; `07.u3-profile.md` §7.16 |
| S2 overlap, S3 riders | — | — | not run | killed by the stop | the same |

The census is iteration 7's lesson in B5's session; the plan and part 11 call
it iteration 8. The items, with the stop on `s1`:

```
## Items

ITEMS: [
 {"id": "s1", "stage": "S1", "title": "The dependency census of the four decode collectives",
  "gate": {"require": ["census"]},
  "stop": {"measure": "gain_x_gate", "below": 3, "status": "below_floor"}, "stops_plan": true},
 {"id": "o.s3", "stage": "S3", "title": "Overlap: a pair of independent collectives in one diamond, priced", "after": ["s1"],
  "gate": {"measure": "gain_x_gate", "min": 3}},
 {"id": "o.s4", "stage": "S4", "title": "Overlap: in the tree, off by default", "after": ["o.s3"],
  "gate": {"require": ["switch_default_off", "dispatch_all_ranks", "zero_jit"]}},
 {"id": "o.s5", "stage": "S5", "title": "Overlap: the A/B", "after": ["o.s4"]},
 {"id": "r.s2", "stage": "S2", "title": "Rider: AITER's fused all-reduce and RMSNorm express Model M's norm", "after": ["s1"],
  "gate": {"require": ["correct", "negative_control_fired"]}},
 {"id": "r.s5", "stage": "S5", "title": "Rider: the A/B", "after": ["r.s2"]}]
```

At iteration 7, `STAGE: {"item": "<map ID>.s1", "result": "stop", "measured":
{"gain_x_gate": 1.66, "census": true}, ...}` closes `s1` as below_floor and
every later item as killed, and the lane closes.

**A plan stopped at S1 is a success.** One iteration, no build, and the
verdict the plan asked for: the decode tier is at its floor in this region.
The plan also says what would reopen it: a new independent branch at decode,
or a sharding regime that changes the order of dependence.
