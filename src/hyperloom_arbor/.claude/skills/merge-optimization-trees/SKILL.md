---
name: merge-optimization-trees
description: Merge several independently-discovered optimizations (PRs, campaign trees, forks) into one measurable baseline: branch contended sites in CODE behind per-feature knobs, choose a lean default and test that set, bisect when the lean set is broken, compare contended sites pairwise, stack the winners, and bank the losers. Use when consolidating optimization PRs or Arbor campaign trees, when two trees touch the same files, or when asked which combination of optimizations to ship.
disable-model-invocation: true
---

# Merging optimization trees into one baseline

## The method

```
- [ ] 1. Merge everything onto ONE branch, branching IN CODE at each conflict
- [ ] 2. Choose a lean for each conflict; that set is the baseline -- TEST IT
- [ ] 3. If the lean set is broken, binary-search for the culprit
- [ ] 4. Compare each contended site pairwise, in isolation
- [ ] 5. Stack the winners and check the effects actually stack
- [ ] 6. Bank the losers; do not delete them
```

The underlying rules, unchanged: one feature at a time; every feature gets a
knob; dedup true duplicates; branch contended implementations behind their own
knobs; select with a heuristic, never a brute-force sweep.

> **"Branch" in step 1 means a branch in the CODE, not a branch in git.**
> Two conflicting implementations both land on the **one** integration branch,
> each behind `if envs.KNOB:` / `else:`, with the old path still live. There is
> one git branch at the end, carrying every knob.
>
> This is worth stating because the other reading is natural and expensive. A
> run of this skill produced seven sibling git branches, one per contended
> candidate, and had to fold them back together afterwards — during which seven
> knob declarations were silently lost and the merged branch could not start a
> server.

## Why this process exists

The naive alternative has been tried and its output is unusable. A prior
V5→V6 merge skipped all 9 contended files because "resolving them by hand
would produce code neither campaign ever measured" — and shipped a tree whose
own README begins **"NEVER BENCHMARKED"**. Skipping contention does not avoid
risk; it converts a measurable result into a hypothesis.

Knobs are what make contention measurable instead of unresolvable.

---

## 1. One branch, branching in code at each conflict

Put every candidate on the same base first. Candidates arriving on different
bases measure the base difference, not the feature. Record the base explicitly:
repo, branch, commit, sibling repos, and whether they are modified.

Classify every candidate from what it *touches*, not from its description:

| class | meaning | action |
|---|---|---|
| **NEW** | no counterpart in the base | merge behind a knob |
| **ALT** | base implements the same thing differently | branch per implementation, knob each |
| **DUP** | identical to something already present | dedup, keep one, record why |

```bash
python scripts/overlap.py base.patch candidate.patch --strip-prefix vllm/
```

- **No shared files** → NEW, presumed additive.
- **Shared files, no shared functions** → NEW but co-located; textual conflicts
  without semantic ones.
- **Shared functions, no shared new symbols** → **ALT**. This is the case that
  needs a knob and a lean.
- **Shared new symbols** → likely DUP; confirm by diffing the symbol bodies.

**A clean `git apply --check` is not a working stack.** New kernel files apply
cleanly *because* they are new, but stay inert unless the dispatch site that
calls them also lands — and the dispatch site is usually in the file that
conflicts.

Merge one candidate per commit. Never batch two: when a combined merge
regresses there is nothing to bisect.

### Knob convention

Declare in `vllm/envs.py`, in **both** the annotation block and the lambda
dict:

```python
    VLLM_ROCM_FUSED_SWIGLU_CLAMP: bool = False
```

```python
    # Runs SiluAndMulWithClamp through the fused Triton kernel on ROCm rather
    # than the seven-launch forward_native chain. Off restores the previous
    # behaviour without a rebuild.
    "VLLM_ROCM_FUSED_SWIGLU_CLAMP": lambda: bool(
        int(os.getenv("VLLM_ROCM_FUSED_SWIGLU_CLAMP", "0"))
    ),
```

- **Default OFF while under evaluation.** The integration branch must behave
  exactly like the base until a knob is turned on.
- **Keep the old path live.** A knob whose off-state was deleted cannot be
  compared.
- Gate at **dispatch**, not inside the kernel, so the off-state costs nothing.
- Host-side and capture-time constant. A knob read per-token inside a
  CUDA-graph-captured region is not a knob.
- **Every knob logs a dispatch marker in BOTH states**, naming which
  implementation ran and the input that decided it.

`envs.py` cannot be text-merged across knob branches: every branch inserts
after the same sentinel, so it conflicts every time, and keeping both sides
produces malformed Python. Reset it to the integration branch's version and
re-insert each knob deterministically — then **assert** that the resolver, the
annotation, and the default are all present. A re-insert that silently did
nothing left a branch whose code read `envs.<KNOB>` while `envs.py` no longer
declared it: an `AttributeError` at first access, i.e. a server that cannot
start, pushed in that state.

---

## 2. Choose a lean, and test that lean set as the baseline

For each contended site you now have two or more sides behind knobs. Pick one
to be the **default** — the lean. The set of leans is a named, shippable
configuration (`<branch>-lean`) that someone else can test without setting
anything.

**How to choose a lean, before any measurement:** prefer the side that
**displaces nothing** — a feature the base has no counterpart for — over the
side that replaces a working implementation of a job the base already does.
Displacing an implementation that already works is a bet; adding a capability
that does not exist is not.

Then **test the lean set as a whole, at every operating point, before treating
it as a baseline.** This step is the one that gets skipped, and it is the one
that pays.

Measured here: a six-knob lean set was published as "ready to run, nothing
needs to be set". It ran at C8 and **killed the server at C32 in two of two
attempts** — engine core dead, worker killed by a signal with no Python
traceback, error text pointing nowhere near the cause.

### A default-ON knob is a promise that the configuration works

When a default turns out to crash, **move the default**. Documenting it is not
enough. The crash signature was `Worker proc VllmWorker-N died unexpectedly
(exit code: None)` — anyone hitting it would have spent hours before suspecting
a knob they never set. The fix was one line in `envs.py`; leaving it ON with a
warning in a README was not a fix.

Verify a default change by **importing in the target environment**, not by
reading the diff:

```bash
docker exec $C python3 -c "import vllm.envs as e; print(e.VLLM_ROCM_USE_HIP_SPARSE_MLA)"
```

---

## 3. If the lean set is broken, binary-search for the culprit

The lean set fails as a set. Halve the ON knobs, measure, recurse into the
failing half. `log2(N)` runs, not `N`.

Actual sequence, six knobs, four runs:

| ON set | result |
|---|---|
| all six | died |
| five (minus the most-suspected) | died → suspicion was wrong |
| two known-active only | clean |
| those two + one more | died → **culprit found** |

Note the second row. The first hypothesis was wrong and the bisect said so
immediately — which is the point of bisecting rather than reasoning.

### Before blaming the code, rule out the environment

A bisect is only valid if the environment is constant. Two confounds to kill
explicitly:

1. **Re-run the CONTROL in the environment's current state**, not the state it
   had before the failures. A container here degraded over a session from
   passing 5-of-5 to 1-of-5, and recreating it fixed it. The control re-check
   passed at the current state, which is what licensed blaming the knobs.
2. **Confirm the knob you turned off is actually off** — read its marker
   (`knob=0`, feature marker absent) before drawing a conclusion from the run.

**One failure is not evidence.** This class of fault also occurs
intermittently on unmodified baselines. Require a repeated, arm-specific
pattern: here it was 3 deaths in 3 attempts with the culprit on, 0 in 6
without.

Keep every failed attempt's log. Move it aside rather than overwriting it —
the next run destroys the evidence otherwise.

---

## 4. Compare contended sites pairwise, in isolation

For each contended site, measure side A against side B with everything else
held at the lean. **Deliberately ignore combinatory effects here** — step 5
handles those. The question at this step is only "which side of this one
conflict is better", and answering it for N sites costs N runs rather than
2^N.

Measure at **both C8 and C32**, with a discarded warmup before each measured
run, same seed for both sides, each replay in its own output directory.
Classify each side per axis as win / null / loss against the measured noise
floor, and record both coordinates.

See [reference.md](reference.md) for what makes two numbers comparable.

> #### A short single replay can REJECT a side. It cannot accept one.
>
> Query-to-query variance on an agentic benchmark swamps the effects being
> chased. Measured at C32 on one node:
>
> | | spread |
> |---|---:|
> | 4 control screens, identical config | 0.365% |
> | **same knob, same tree, measured twice** | **0.89%** |
> | effects being chased | 1–1.2% |
>
> A candidate measured **309.130** then **306.399** on a repeat of the
> identical configuration — 1082 vs 1081 requests, prefix-cache hit 0.90674 vs
> 0.90664, so the workload was the same and the machine was not. The apparent
> +1.14% was noise.
>
> **Do not take a tight control spread as the noise floor.** Four controls
> agreeing to 0.365% says nothing about the tail. The honest estimate is a
> *direct replicate* — the same configuration twice — and that was 2.4× wider.
>
> Keep the short screens: they legitimately reject the many candidates that do
> nothing. Anything that survives goes to a full-duration paired A/B at the
> operating point where the effect appeared.

### Attribute the gain to the right thing

A treatment arm beating an *old* baseline does not mean the knobs did it. Here
the lean tree measured ~78 tok/s/chip against a stock baseline of 71.571 and
it was tempting to call the knobs worth +9%. The paired control — same tree,
knobs off — measured 77.968. **The gain belonged to the merged tree, and the
knobs were flat.** Only the paired arm isolates the knob.

### Prove "one tree, environment only" instead of assuming it

Deploy one tree to both lanes and **verify it**:

```bash
sha256sum $V/envs.py    # must match across every container in the comparison
```

If the two lanes hold different trees, the comparison measures the tree.

---

## 5. Stack the winners and check the effects stack

Build the interaction graph first. Nodes are features; an edge joins two that
touch a **common function** (`scripts/overlap.py`) or a **common kernel** in
the profile. No edge → presumed additive. Edge → not expected to sum. ALT
groups are always fully connected.

```
Phase A — screen     : N runs, one per contended side (step 4, already done)
Phase B — prune      : drop every side that loses on both axes -> the bank
Phase C — resolve    : within each ALT group keep the best single side
Phase D — union      : combine survivors from DISJOINT groups; measure it
Phase E — reconcile  : if the union's gain is within noise of the sum of
                       parts, accept. If it falls short, bisect: split the
                       union, measure each half, recurse into the half that
                       underperforms its prediction. ~log2(N) runs.
Phase F — margin     : add remaining features in descending single-gain order,
                       stopping when the next gain is below the noise floor
Phase G — confirm    : re-measure the winning set at 3 seeds, both axes
```

Total roughly `N + log2(N) + a few`, against `2^N`. **Never enumerate the
power set**: fourteen features is 16384 configurations.

Two legitimate shortcuts, and one that is not:

- **Legitimate:** decline a combination with arithmetic when a feature's gate
  provably cannot fire in the configuration under test.
- **Legitimate:** price a feature against the kernel census first. A change
  worth 0.1% of the serving window does not need a 600-second replay to rank.
- **Not legitimate:** inferring a combination's result from its parts and
  booking it as measured. Predicted sums choose what to run; they are never
  the final number.

---

## 6. Bank the losers — do not delete them

A side that loses the pairwise comparison is **not** waste. Move it to a
reviewed bank, with the measurement that demoted it and the question that
would settle it.

Three fates, and all three are real outcomes:

1. **Merged into the winner.** A losing implementation often handles one case
   better than the winner does. Losing overall does not mean losing
   everywhere; the comparison was whole-implementation, and the parts were
   never scored.
2. **Inspires a different optimization.** The losing side is the only other
   perspective anyone has written on that code. It encodes an assumption the
   winner does not make, and that assumption is a lead.
3. **Discarded after review** — but *after*, and with a recorded reason.

What the bank entry must carry: the knob name, the measurement that demoted
it, both operating points, the seed, and one sentence on what it does
differently from the winner. Without that last sentence the entry is
unreviewable and will be deleted by the next person.

Because the losers stay in the tree behind their knobs, banking costs nothing
at runtime — the off-state is a branch not taken. Deleting them is the
expensive option: it destroys the comparison and nobody re-derives it.

---

## Traps that have cost real time

- **`logger.info_once` hides everything after the first call.** A marker
  reading `knob=1 batch=1024 -> base path` was read as "this feature never
  runs at C32". It ran — on later, smaller batches — and crashed the server.
  For any knob whose behaviour depends on batch or shape, the first log line
  is the *least* informative one. Count dispatches, or log per distinct shape.
- **Assert marker anchors against source before the campaign starts.** It is a
  two-line test. A harness grepping for `SwiGLU` collected nothing for a knob
  whose marker says `SiluAndMulWithClamp`; checking all 15 anchors found 8 that
  would never match, which would have discarded 7 of 14 knobs while every run
  still reported success.
- **A marker that fires either way proves nothing.** `SiluAndMulWithClamp
  dispatch:` logs the bound method of an unrelated op and appears with the knob
  on *or* off. The knob's own marker was a different string entirely.
- **`.gitignore` can swallow the thing your build step builds.** A blanket
  `*.hip` rule meant a hand-written kernel source was never committed, so the
  documented init step failed for everyone who cloned — while working locally,
  because the file was in the worktree. Check that new non-source-looking
  artifacts are actually tracked.
- **Editing a script while it runs** under `srun` from NFS corrupts it
  mid-execution; bash reads incrementally and the damage surfaces minutes
  later. Wait for the gap between runs.
- **`pkill -f <pattern>` matches your own command line** when the pattern is
  in an inline `bash -lc`. Put such patterns in a script file.

## Anti-patterns

- **Skipping contended files.** Produces an unmeasured tree.
- **Hand-merging two implementations into a third.** The result is code
  neither side ever measured. Ship one or the other behind a knob.
- **Deleting the losing side.** See step 6.
- **Shipping a lean set you have not run at every operating point.**
- **Merging instrumentation.** Timing hooks from a source tree are overhead,
  not optimization. One campaign spent three iterations removing exactly what
  another had added.
- **Cross-axis credit.** A gain on tok/s/user at C4 is not evidence for
  tok/s/chip at C32.
- **Comparing across nodes.** Internode differences of ~1–4% have been measured
  on identical trees.
- **Trusting a source's self-reported number.** One tree's published +39.64%
  was measured on a cross-contaminated checkout, per its own review.

## Additional resources

- [reference.md](reference.md) — measurement discipline, noise floors, knob
  verification, bisect hygiene, and the record each merge must leave
- `scripts/overlap.py` — per-file changed-function and new-symbol overlap
  between two patches
