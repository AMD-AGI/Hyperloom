# Reference: measurement discipline for knob comparisons

Detail behind steps 5–8 of [SKILL.md](SKILL.md). The cluster-level rules
(where GPUs are, storage policy, container permissions) live in the
repository's `AGENTS.md`; this file covers only what makes two knob states
comparable.

## Both operating points, always

Every result is reported at **C8 and C32**. A single-concurrency number is not
a deliverable, and the two are not the same code path:

- Batch-shaped gates route on request or row count, not on load. With
  spec-decode at 5 speculative tokens, verify rows are
  `concurrency x (1 + spec_tokens)` — 48 rows at C8 against 192 at C32 — and
  ROCm sparse-attention gates branch on exactly that number. A knob can be
  admitted at one point and silently rejected at the other.
- Launch-width and tiling work tuned at high concurrency can invert at low
  concurrency. A prefill-gather knee in programs per CU gave large
  speedups at depth, collapsed on a small input, and **inverted to a
  slowdown** past the knee.
- Components can be entirely absent at one point. At C8, all-reduce,
  shared-expert MoE, sparse prefill and the sparse attention indexer measure
  **zero time with zero calls**. A feature in those components has no C8 time to
  remove, so its gain cannot transfer — the Amdahl denominator is different,
  not merely smaller.

Keep prompt counts identical across the two points, and give every replay its
own output directory so their aggregate files cannot overwrite each other.

## Warm up before every measured run

The first replay against a fresh server pays cold JIT compilation, tuned-config
resolution, and an empty prefix cache. On three back-to-back replays on one
server instance, the cold first replay is a **low outlier**: including it more
than doubles the run-to-run spread.

This matters more in a knob sweep than anywhere else, because each knob state
usually gets a fresh server — so **every measurement is a cold first replay**
unless warmed, biasing every candidate low against a warm baseline.

Run a short warmup replay (60–90 s is enough; it is warming caches, not
measuring), discard it, then run the full measured replay. Give the warmup its
own output directory.

## Set the bar from measured noise

Derive the acceptance threshold from the baseline's own spread, not from a
round number. A gate set below the noise floor books noise as wins.

Compute the baseline's relative spread over at least 3 replays and use it as
the minimum credible gain. Secondary metrics are much noisier than the primary
— a p90 interactivity "win" inside its own spread is not a result.

Re-derive the threshold on the integration branch. It is a property of the
tree and node, not a constant.

## One seed measures one draw

The replay seed selects where inside each conversation every trajectory
starts, which fixes the context-length distribution of the whole run. On a
model whose cost is dominated by context length, that is most of what
determines the number.

- **Compare within a seed.** Knob-on and knob-off must use the same seed.
  That is a paired comparison and it controls the composition difference that
  would otherwise swamp the effect.
- **Generalize across seeds.** A feature is a property of the workload only if
  it holds at three or more seeds. Report per-seed results; a feature that wins
  on two seeds and loses on the third has told you something a mean hides.
- **Record the seed with every number**, alongside the operating point.

A short bench at a fixed seed is a **biased prefix** of one draw, not a small
sample: the requests that complete early are the short-context ones, and the
long-context turns that dominate full-run cost never arrive. A short run may
reject a knob or check liveness, but cannot establish a gain.

## Verify the knob actually switched

A knob that fails to switch yields a null that looks like a measurement. Before
booking either state:

1. Confirm from a dispatch marker (a `logger.info_once` at the dispatch site,
   or a counter) that the intended path executed in the on-state.
2. Confirm the marker is **absent** in the off-state. Without this, the
   comparison may be measuring the same path twice.
3. For a compiled kernel, confirm the build contains it; the knob gates
   dispatch, not presence.

Record the marker counts with the result. A dispatch proof of
"N lines in on-state, 0 in off-state" is the minimum.

### Assert the anchor strings against source first

Derive the grep pattern from the marker strings **in the code**, and assert the
match offline before the campaign starts. It is a two-line test and it has
repeatedly been the difference between a campaign and a wasted one.

- A harness matching `SwiGLU` collected nothing for a knob whose marker reads
  `SiluAndMulWithClamp dispatch`. Checking all 15 anchors found 8 that could
  never match — enough to discard 7 of 14 knobs while every run still reported
  success and every number still looked fine.
- Substring patterns fail on hyphenation and on wording drift between a knob's
  name and its log line: `hidden quant` does not match
  `hidden-state MXFP8 producer`.
- **A marker that fires in both states proves nothing about the knob.**
  `SiluAndMulWithClamp dispatch:` logs an unrelated op's bound method and
  appears whether the fusion knob is on or off; the knob's own marker was a
  different string that also had to be paired with a second line confirming the
  consumer *admitted* the fusion. Prefer markers of the form
  `knob=<0|1> <deciding input> -> <path chosen>`.

### `logger.info_once` reports the first call, not the run

This is the single most misleading instrument in the set. A marker reading

```
DSv4.1 sparse MLA decode: knob=1 batch=1024 -> Triton decode (base path)
```

was read as "this feature declines at C32, so it is inert there". It was not
inert: later decode steps ran smaller batches that fell inside the kernel's
33-72 row band, it dispatched, and it killed the server. The wrong reading
survived two rounds of documentation.

For any knob gated on batch size, sequence length or context length, the first
log line is the *least* informative one. Count dispatches, or log once per
distinct decision, and never conclude "inert" from a single `info_once`.

## Bisect hygiene

When a set of knobs fails as a set, halving the ON set finds the culprit in
`log2(N)` runs. The bisect is only valid if these hold:

- **Re-run the control in the environment's CURRENT state.** Not the state it
  had before the failures. A container degraded over one session from passing
  5-of-5 to 1-of-5 and was fixed by recreating it; without a control re-check
  after the failures, every knob would have been blamed for the container.
- **Confirm each step's environment actually took effect.** Read the marker and
  check the knob you turned off reads `knob=0` and its feature marker is
  absent. A bisect step that silently did not apply points at the wrong half.
- **One failure is not evidence.** This class of fault also occurs
  intermittently on unmodified baselines. Require a repeated, arm-specific
  pattern — 3 deaths in 3 attempts with the culprit on against 0 in 6 without
  is a pattern; one death is an anecdote.
- **Preserve every failed attempt.** Move its directory aside rather than
  letting the next run overwrite it; a server log is the only record of why a
  run died, and it is gone as soon as the next server starts.

Expect the first hypothesis to be wrong. In the run that produced these notes,
the most-suspected knob was cleared by the second bisect step, and the culprit
was one that had been dismissed as "inert at this operating point".

## Prove "one tree, environment only"

A paired A/B claims that the arms differ only in environment. Verify it rather
than assuming it:

```bash
sha256sum $V/envs.py    # identical across every container in the comparison
```

Deploy from a single packaged artifact to every lane, then hash. If the lanes
hold different trees, the comparison measures the tree difference, which is
usually far larger than the effect being chased.

Related: attribute the gain to the right thing. A treatment arm beating an
*older* baseline tells you about the whole tree. Only the paired control —
same tree, knob off — isolates the knob. A lean tree measuring well above a
stock baseline looked like a knob win; its own paired control measured the
same, so the knobs were flat and the merged tree had earned all of it.

## Interaction edges from profiles

Function-set overlap finds edges that share code. Profiles find edges that
share *time*. Two features touching different functions but the same kernel
compete for one denominator and will not sum.

Take a kernel census of the base once, then for each feature ask which census
entries it reduces. Features reducing the same entry get an edge. This costs
one profile, not one run per pair, and it is how a combination can be declined
with arithmetic before it is ever measured.

Price each feature as a percentage of the serving window from that census. A
feature worth 0.1% of the window does not need a 600-second replay to rank,
and should not be allowed to consume one.

## The record each merge leaves

For every candidate, whether kept or **banked**:

- class (NEW / ALT / DUP) and the evidence for it — shared files, shared
  functions, shared new symbols
- knob name and default
- measured result at C8 and C32, per seed, against a baseline from the same
  node and the same session
- dispatch proof for both knob states
- for a banked candidate: which candidate beat it, on what measured grounds,
  and **one sentence on what it does differently from the winner**

That last sentence is what makes the bank usable. A banked side is kept
because it may still be merged into the winner, may inspire a different
optimization, or may be discarded after review — and none of those reviews can
happen from a row that records only a number. Without it the entry is
unreviewable and the next person deletes it.

### The lean configuration

A lean set is a shippable artifact, so it carries its own record:

- which side was chosen at each contended site, and why — state whether the
  lean displaces an existing implementation or adds a capability the base
  lacks
- the required initialisation, if any. A default that needs a build step the
  image does not ship is not a default; it is a trap that silently runs the
  other path.
- the result of running the whole lean set **at every operating point**, not
  just the one it was developed at
- for any knob defaulted OFF after evaluation: the evidence that moved it

For the final tree:

- base commit, and any sibling repos with their branches
- the exact knob configuration, including the knobs left off
- both operating points at three seeds
- node identity
- the list of dropped features with reasons

A merged tree without its knob configuration is not reproducible, and a knob
configuration without its measurements is not a result.
