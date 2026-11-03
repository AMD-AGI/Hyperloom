---
name: plan
description: "Write the plan for an optimization round before the round runs, from the dossier, the ledger and the outcomes of earlier iterations, in the round-plan format (template.md): the directions picked and why, each work item's expected gain against the noise floor, how each is checked, gates with thresholds, stop conditions, sources; then, after the round, settle it item by item. Use when asked to plan a round, turn a dossier's directions into work, decide what a campaign should try next and why, or record what a finished round showed, so it carries over as long-term memory."
---

# Round plans: decide before the round, settle after it

Audience: an agent (or a person) about to start an optimization round, or one
that has just finished. Arbor records each iteration (what changed, what the
gate said); it does not record why the round tried what it tried, what it
expected, or when a failed idea is worth reopening. A round plan carries that,
before and after (the runner's `plans/model-m/01_ARBOR_PERSISTENCE.md`). The
skeleton is [template.md](template.md). A planning campaign (pack `model-m-plan`)
scores drafts with its linter, `packs/model-m-plan/validator/lint_plan.py`: run
`python3 <that file> PLAN.md` to see what a draft is missing.

## Inputs, read in this order

1. **The dossier** (the `dossier` skill finds it). Part 11 (directions) is
   where work items come from; part 4 says what counts as a win; part 6 gives
   the floor; parts 7 and 8 give each direction's share of the step and its
   ceiling; part 10 what was already done. Check each part's status and
   "stale when" first: a stale part is a source for nothing.
2. **The ledger** (`ledger.jsonl`, or the knowledge collector's
   `findings/FINDINGS.md`): FACTs to build on, ASSUMPTIONs and UNKNOWNs a
   work item might close, DEAD_ENDs with their `revisit_if`.
3. **The outcomes** (`session.json`'s lessons, or `outcomes/OUTCOMES.md`):
   what was kept and what was reverted, with the gate's reasons. A REVERT
   that was "not measured" says nothing about its idea.
4. **Earlier rounds' plans and settlements**, and the project's own rules
   for plans (the runner's `AGENTS.md` §8–§11 and `plans/TEMPLATE.md`).

## What each section owes

- **1. Objective**: one paragraph naming the operating point and the metric,
  and what will be true when the round succeeds.
- **2. Evidence**: every claim with a source a reader can open (dossier part,
  ledger id, iteration, profile, job id). A number without a source is
  removed. **2.2** gives each item's largest possible gain (its Amdahl
  ceiling, from parts 7 and 11) beside the measured floor (part 6), in the
  metric's unit. An item whose best case is below the floor is set aside
  there, with its number, unless the plan changes the measurement.
- **3. Fixed baseline**: pins, image digest, benchmark and seed, the flags
  that may not change.
- **4. Work items**, numbered 4.1, 4.2, ... in the order they will run, the
  cheapest decisive one first. Each has:
  - **What**, naming the kernel or code path the evidence shows actually runs;
  - **Produces**: the patch, evaluator, profile or record;
  - **Expected**: the gain in the metric's unit and where the number comes
    from, against the floor; or `enabling: <the item it unblocks>` for
    foundation work, which ships neutral;
  - **Done when**: a check someone else can run, with its threshold;
  - **Sources**.
  Then one item for correctness: what must stay identical, run before any
  timed run. Known dead ends are not work items unless their `revisit_if`
  now holds; say so when it does.
- **5. Gates**, each with a threshold, declared now. Gates are the operator's:
  a plan proposes them, the operator accepts them by reading the plan.
- **6. Stop conditions**, each checkable: when the round ends without a
  candidate.
- **7. Artifacts** and **8. Execution setup**: where things are kept; the
  campaign, node, trees, what to reuse, the report, the hazards.
- **9. Settlement**: "not settled" until the round ends.

## During the round

The campaign that runs the plan cites the work item in every DONE line
(`DONE: 4.2 ...`), so each iteration in `session.json` joins to the item it
served.

## After the round: settle it

Append to section 9, one entry per work item, from `session.json` and the
ledger:

- **tried or not**, and if not, why;
- **verdict**: KEEP, REVERT or not measured, with the iterations;
- **measured effect against the expected one**, in the metric's unit, with
  its noise floor;
- **mechanism level**: measured (a trace or counter shows the cost moving) or
  inferred (only the end-to-end number moved);
- **reopen when**: what would make a REVERT worth trying again.

The settlement is the round's long-term memory. Share it with the campaign's
knowledge (the `knowledge-collector` skill) so the next round on the project
starts from it.
