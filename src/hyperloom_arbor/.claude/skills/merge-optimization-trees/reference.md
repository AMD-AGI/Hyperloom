# Reference: measurement discipline for knob comparisons

Detail behind steps 5–8 of [SKILL.md](SKILL.md). This file covers only what makes two knob states comparable; where GPUs
are, the storage policy and container permissions are the cluster's own rules.

## Every operating point, always

Every result is reported at **every operating point the target names**, at
least a low and a high concurrency. A single-concurrency number is not a
deliverable, and the points are not the same code path:

- Batch-shaped gates route on request or row count, not on load. With
  spec-decode, verify rows are `concurrency x (1 + spec_tokens)`, and ROCm
  sparse-attention gates can branch on exactly that number. A knob can be
  admitted at one point and silently rejected at another.
- Launch-width and tiling work tuned at high concurrency can invert at low
  concurrency. A change with a knee in programs per CU can win large at depth,
  much less on a small input, and **be slower** past the knee.
- Components can be entirely absent at one point. At low concurrency, an
  all-reduce, a shared-expert MoE, a sparse prefill path or a sparse attention
  indexer can make **zero calls**. A feature in those components then has no
  time to remove there, so its gain cannot transfer — the Amdahl denominator
  is different, not merely smaller.

Keep prompt counts identical across the points, and give every replay its
own output directory so their aggregate files cannot overwrite each other.

## Warm up before every measured run

The first replay against a fresh server pays cold JIT compilation, tuned-config
resolution, and an empty prefix cache. On back-to-back replays on one server
instance, the cold first replay is a **low outlier**: including it widens the
run-to-run spread considerably.

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
match offline before the campaign starts. It is a two-line test, and without
it a campaign can be wasted.

- A harness matching `SwiGLU` collects nothing for a knob whose marker reads
  `SiluAndMulWithClamp dispatch`, and discards the knob while every run still
  reports success and every number still looks fine.
- Substring patterns fail on hyphenation and on wording drift between a knob's
  name and its log line: `hidden quant` does not match
  `hidden-state MXFP8 producer`.
- **A marker that fires in both states proves nothing about the knob.** A
  line such as `SiluAndMulWithClamp dispatch:` that logs an unrelated op's
  bound method appears whether the fusion knob is on or off. The knob needs
  its own marker, and may need a second line confirming the consumer
  *admitted* the fusion. Prefer markers of the form
  `knob=<0|1> <deciding input> -> <path chosen>`.

### `logger.info_once` reports the first call, not the run

This is the single most misleading instrument in the set. A marker reading

```
my feature: knob=1 batch=<large> -> base path
```

says only what the first call did. It does not mean "this feature declines at
this operating point, so it is inert there": later decode steps can run
smaller batches that fall inside the kernel's row band, dispatch it, and kill
the server.

For any knob gated on batch size, sequence length or context length, the first
log line is the *least* informative one. Count dispatches, or log once per
distinct decision, and never conclude "inert" from a single `info_once`.

## Bisect hygiene

When a set of knobs fails as a set, halving the ON set finds the culprit in
`log2(N)` runs. The bisect is only valid if these hold:

- **Re-run the control in the environment's CURRENT state.** Not the state it
  had before the failures. A container can degrade over a session from passing
  every check to failing most of them; without a control re-check after the
  failures, every knob is blamed for the container.
- **Confirm each step's environment actually took effect.** Read the marker and
  check the knob you turned off reads `knob=0` and its feature marker is
  absent. A bisect step that silently did not apply points at the wrong half.
- **One failure is not evidence.** This class of fault also occurs
  intermittently on unmodified baselines. Require a repeated, arm-specific
  pattern — a death in every attempt with the culprit on against none in
  several attempts without is a pattern; one death is an anecdote.
- **Preserve every failed attempt.** Move its directory aside rather than
  letting the next run overwrite it; a server log is the only record of why a
  run died, and it is gone as soon as the next server starts.

Expect the first hypothesis to be wrong. The most-suspected knob can be cleared
early in the bisect, and the culprit can be one that had been dismissed as
"inert at this operating point".

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
same tree, knob off — isolates the knob. A lean tree can beat the stock
baseline by a wide margin and look like a knob win while its own paired
control measures the same: then the knobs are flat and the merged tree earned
all of it.

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
- measured result at every operating point, per seed, against a baseline from
  the same node and the same session
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
- every operating point at three seeds
- node identity
- the list of dropped features with reasons

A merged tree without its knob configuration is not reproducible, and a knob
configuration without its measurements is not a result.
