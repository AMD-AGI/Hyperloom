---
name: update
description: "Update the dossier in a round after a long campaign or a merge has moved the served tree: 03.uK the code read again, 06.uK the baseline, 07.uK the profile, 10.uK what the campaigns learned (every KEEP, kill and probe, and which kills are reopenable), 11.uK the directions again with a new map. Run by Arbor in the round_end chapter when a map is explored, or by the operator on a stopped campaign."
disable-model-invocation: true
metadata:
  arbor:
    modes: [run]
    chapter: round_end
    gpu: true
---

# Update: a dossier round on a tree that has moved

**In a campaign, Arbor runs this skill in a turn of its own** when every direction of the map is closed. Its argument
is the path of a brief: read the brief first. It names the dossier's directory and the campaign's program; the
program's "MAP EXHAUSTED" or "FRESH START" section says which round this is and what it writes.
**By the operator,** on a stopped campaign, follow "How" below.

**The rules:** [the rules](../../shared/rules.md), relative to this skill's directory. The files of a round are
`NN.uK-<part>.md`, beside the originals ([the template](../../shared/template.md)).

**End your reply:**
- with a `LEDGER:` FACT per file written;
- when the round writes 11.uK, with its map as ONE `DIRECTIONS: [...]` line.

## Updating the dossier: after a long campaign or a merge

A dossier describes one tree. A campaign changes it one KEEP at a time, and a
merge brings in other branches' work at once; after enough of either, the
served code is laid out differently (a path split, a layer re-partitioned, a
collective moved) and the step's time goes elsewhere, sometimes drastically.
Part 7's rule (re-profile after a KEEP) keeps the profile current inside a
campaign. It does not re-read the code or re-rank the directions, so the
campaign goes on optimizing from a map of a tree that no longer exists. One
campaign kept ten changes in forty iterations, a sequence-parallel prefill path
among them, and its stack was then merged onto another team's, which partitions
the MoE pooler another way; the dossier it started from describes neither tree.

**When.** The operator decides, on signs like these:

- the served tree has taken a merge from another branch;
- KEEPs have re-partitioned a layer, a projection or a collective, or moved
  where the step's time goes, not only swapped a kernel for a faster one;
- the directions have run out: the campaign's top leads are killed and what
  is left sits below the floor, or the increments are running out (part 11's
  signs), so 11.uK should rank rewrites;
- the bar has to be measured again anyway (a new tree, a new node class).

An agent that sees these signs says so in its verdict and goes on with its
plan; it does not stop the campaign.

**In the campaign, by the harness: when the map is explored.** The campaign
does not explore aimlessly: it runs the dossier and the profile again, then
continues like new.
- **The trigger.** When every direction of the campaign's map is closed,
  Arbor (`arbor/directions.py`) makes the next iterations this round, in the
  same session and without a STOP.
- **What the round writes:**
  - 07.uK, the profile again;
  - 10.uK, the round's iterations;
  - 11.uK, the directions again, ending in its `## Map` line;
  - 03.uK, brief: what the round's KEEPs changed in the code;
  - 06.uK cites the session's bar, with no new baseline.
- **How each part is run.** As an `INTENT: LEARN` iteration with a LEDGER FACT
  naming the file, with the LEARN cap waived.
- **What the new map starts.** A new map round, with the streak counters
  reset: the campaign carries on like new.
- **When the new map has no open increment,** the campaign starts like a new
  one:
  - **a fresh start**: the next iterations write a new dossier as a new
    campaign would, in the same session: 03.uK, the code read again whole;
    07.uK, the profile again; 10.uK; and 11.uK, a map from 03.uK and 07.uK,
    with the old map as history, not a list to carry over;
  - **then a rewrite**: if the fresh map has no open increment either, the
    harness opens the top planned rewrite (a kernel plan first, then by rank)
    as the next round's work, built stage by stage through the gate.
- **Where the operator still decides.** The harness stops the campaign
  (`MAP_STOP.md`):
  - when no planned rewrite is left after a fresh start;
  - when two rounds in a row end without a KEEP;
  - when the round yields no valid map in its iteration limit.
- **The rest of this section is the operator's version:** a stopped campaign,
  every part, and a new session.

**How.** Stop the campaign first (in Arbor, the session's `STOP` or the
chain's `CHAIN_STOP`), so nothing changes the tree while it is read and
measured. Then, on the served tree as it now is, in order:

1. **03.uK, the code, read again.** Not the diff: the model's code read whole,
   method by method, under part 3's rules (one reader per area, line numbers
   from numbered output, a reader's report is a lead). The model's own files
   and the model-specific ops they call are read; the framework's areas keep
   part 3's reading unless the framework's pins moved, and then they are read
   again too. Each in-path method gets, beside what it does and what it costs:
   - **an opportunity label**: none, or a candidate, with what could change,
     the mechanism, what it would need, and whether the cost behind it is
     measured or reasoned;
   - **its history**: unchanged since part 3, or added or changed by which
     KEEP or merged commit.

   A diff misses what moved around code that did not change: a method left
   alone whose inputs now arrive sharded, or that a new path no longer calls.
2. **06.uK, the baseline again**, as part 6, on the node class the campaign
   will run on: the metric at each operating point and a fresh noise floor.
   It is the restarted campaign's bar.
3. **07.uK, the profile again**, as part 7: plan, then results, on this tree,
   before anything is ranked.
4. **10.uK, what the campaigns since the last round learned.** Read every gated
   iteration of the stopped session, and of any campaign whose work the merge
   brought in, not the ledger alone. The `knowledge-collector` skill stages
   both sources:

   ```bash
   python3 -m arbor.collect knowledge --session <session> --config <config> \
       --types outcomes,findings --no-push
   ```

   This writes `OUTCOMES.md`, every iteration kept, reverted, probed or not
   measured, with why, and `FINDINGS.md`, the ledger's facts and dead ends. Each
   KEEP's `iter_N/patch.diff` says what it changed. Per campaign, the part
   records:
   - **its KEEPs**: what each changed, its gate gain, and where it stands on
     this tree: in it, to re-land, or superseded;
   - **its kills**: each dead end, the reason it died, and what that reason
     rests on, either a measurement (which instrument, which floor, which tree)
     or an argument. Some kills are marked **reopenable**, with the reason: one
     resting on a floor measured with the kernel under test, on a tree this
     round replaced, or on an argument nobody measured;
   - **what its probes established**: the facts that move a ceiling or a cost
     (a launch floor, a roofline, a share of the step), each with its source;
   - **the map against the work**: which of the last round's ranked directions
     were worked, which were never touched, and what was worked off the map;
   - **process lessons**: harness and measurement failures, and the iterations
     they cost;
   - **the counts**: gated iterations, probes and KEEPs, and KEEPs by level
     (config, tuning, framework, kernel), so a round that keeps working one
     level sees it.

   The hand-off a restarted campaign reads is cut from this part. Model M's round
   u2 did this step by hand as `11.u2-c2-ledger.md`. Across several Model M
   campaigns, a fair share of the KEEPs that came from a handed ranking reworked
   something an earlier map had killed. The core kernels there were killed on floors that a later
   campaign found the kernel under test had measured itself.
5. **11.uK, the directions again**, as part 11, from 03.uK, 06.uK, 07.uK and
   10.uK, and the parts still current. A direction the campaign killed stays
   killed unless the new tree removes the reason it died or 10.uK marks it
   reopenable; the entry says which. 10.uK's counts and kills tell whether
   the increments are running out (part 11's signs); if they are, rewrites are
   ranked first, each with a kernel plan.
6. **Restart** the campaign on this tree, as a new session: its bar is 06.uK,
   its pack knowledge is re-digested from the round's parts (10.uK's hand-off
   first), and the stopped session stays as the record. After a merge, "this
   tree" is the merged one, the tree the campaign serves from now on.

Parts 1, 2, 5 and 8 are checked against their "Stale when" as always, and
refreshed if stale; part 8 gains a row for every kernel 07.uK shows that it
lacks.

**Files.** Round K writes `NN.uK-<part>.md` beside the originals
(`03.u1-code.md`, `06.u1-baseline.md`, `07.u1-profile.md`,
`10.u1-lessons.md`, `11.u1-directions.md`). The originals stay as they are, the record of the tree
before. The index's parts table points at each part's newest round, and its
updates table lists every round ([template.md](../../shared/template.md)).

**Done when.** 03.uK labels every in-path method of the model's code; 06.uK's
floor is measured on the campaign's node; 07.uK has results that add up to the
measured step; 10.uK accounts for every gated iteration since the last round
and gives every kill its evidence; 11.uK ranks from them; the index lists the
round; and the evaluation below passes for the round.

## Checking a round

```bash
python3 -m arbor.dossier.check <dossier dir> --round uK                 # 03, 06, 07, 10, 11 .uK and the index
python3 -m arbor.dossier.check <reports dir> --round uK --no-index      # a round staged outside the dossier
```

The round is done when this passes, and the round's "Done when" above holds.
