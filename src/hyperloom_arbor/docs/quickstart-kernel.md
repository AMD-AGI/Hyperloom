# Quickstart: a kernel lane, from a plan to a booked gain

A kernel rewrite runs in Arbor as a **kernel-plan lane**: one campaign works one plan's items, stage by stage, and the
harness gates every stage. This page goes from a plan to the lane's end. How a plan is written is in the dossier
skill's [`kernel-plan.md`](../.claude/skills/dossier/directions/kernel-plan.md); how the harness judges a stage is in
`arbor/stages.py`; every config key is in [reference.md](reference.md).

| Step | Command | Ends with |
| --- | --- | --- |
| Write or pick a plan | the dossier's part 11 (`11-plan-<region>.md`) | a plan with an `## Items` block |
| Check the plan | `python3 -m arbor.stages check <plan.md>` | its items, in an order they allow |
| Make the config | a serving campaign's config, with `"space": {"levels": ["kernel"]}` | `<config.json>` |
| Check the config, no GPU | `python3 -m arbor run --config <config.json> --dry-run` | a summary of verdicts on synthetic numbers |
| Seed the session as the lane | `python3 -m arbor.stages seed --config <config.json> --direction m-<region> --plan <plan.md>` | the lane's map, `stages.json` and `lane_plan.md` in the session |
| Run it | `python3 -m arbor run --config <config.json>` | `LANE_CLOSED.json`, and a `STOP` that starts `lane-closed:` |
| Watch it | `python3 -m arbor.stages show --session <session>` | each item, its attempts, and what waits to be judged |

Names in angle brackets are yours to fill in: `<session>` is the config's `session_root`, and `m-<region>` is the map
ID of the direction the plan belongs to.

## 1. What a lane is

- **One plan per campaign.** The seeded map holds the plan's items and nothing else: the campaign works them, writes
  no map of its own, and opens no other direction. A direction the work turns up goes into the report, for a later map.
- **The stages** (S1–S5), each with its own gate, because the campaign's A/B gate can judge nothing before the last:

  | Stage | The agent's intent | Done when |
  | --- | --- | --- |
  | S1 Reference | LEARN | a standalone harness reproduces the region's time now, at the served shapes, within the plan's tolerance of the in-situ profile |
  | S2 Correct | LEARN | the new kernel matches the path it replaces over the served shapes and edges, with a negative control that fires |
  | S3 Fast | LEARN | graph-captured time at the served shapes takes the plan's share of the gap |
  | S4 In tree | FIX or stricter | behind a switch that defaults off, declared `SWITCH: NAME=1`: the iteration KEEPs, and the harness, serving the switch on, finds dispatch on every rank, zero JIT compiles in serving and the accuracy gate passed |
  | S5 A/B | WIN | the campaign's own gate: a booked KEEP whose `DONE:` line cites the item |

- **S1–S3 do not have to beat the throughput bar:** they run as LEARN, and while the next item is one of them the
  LEARN cap is waived, so a run of measuring iterations is not demoted.
- **The lane ends** when every item has a verdict: passed, failed, or killed by a stop. A stop from an item marked
  `stops_plan` (S1's "the gap is too small", usually) kills every later item.

## 2. The plan

- **From a dossier:** a serving campaign's dossier ranks its rewrites in part 11 and writes a plan for each of the top
  ones, `<session>/reports/11-plan-<region>.md` (or `11.uK-plan-<region>.md` in an update round). You pick one.
- **By hand:** follow [`kernel-plan.md`](../.claude/skills/dossier/directions/kernel-plan.md): the target, the
  ceiling, what exists, the design, the stages and the budget. Every number names its source.
- **The `## Items` block** is what the harness reads. Each design is its own chain after `s1`, so a losing design does
  not hold up the others:

  ```
  ## Items

  ITEMS: [
   {"id": "s1", "stage": "S1", "title": "Reference: the region in situ",
    "gate": {"measure": "region_us", "reference": 400.0, "tolerance_pct": 10},
    "stop": {"measure": "gap_x_gate", "below": 3, "status": "below_floor"}, "stops_plan": true},
   {"id": "c.s2", "stage": "S2", "after": ["s1"], "gate": {"require": ["correct", "negative_control_fired"], "measure": "snr_db", "min": 30.0}},
   {"id": "c.s3", "stage": "S3", "after": ["c.s2"], "gate": {"measure": "gap_share", "min": 0.5}},
   {"id": "c.s4", "stage": "S4", "after": ["c.s3"], "gate": {"require": ["switch_default_off", "dispatch_all_ranks", "zero_jit", "accuracy_pass"]}},
   {"id": "c.s5", "stage": "S5", "after": ["c.s4"], "attempts": 2}]
  ```

  - **`gate`:** a `measure` with `reference` and `tolerance_pct`, a `min` or a `max`; `require` lists flags that must be
    true. S2 always requires `negative_control_fired`. S5 has no gate of its own.
  - **`stop`:** a `measure` with `below` or `above`, and the status it closes with (`killed` or `below_floor`).
  - **`attempts`:** the failed attempts that kill the item (for S5, its REVERTs).
- **Check it:** `python3 -m arbor.stages check <plan.md>` validates the block and prints the items in an order they
  allow. `seed` refuses a plan that does not pass.

## 3. The config

Start from the config of the serving campaign the plan came from: the same container, served tree, model, bench and
operating point, so S1's in-situ numbers and S5's A/B mean the same thing. Then:

- **`"space": {"levels": ["kernel"]}`:** the agent works at the kernel tier only, and never escalates past it.
- **A pack that gives the agent the kernel skills:** `serving-knowledge` loads `kernels`, `hardware` and `profiler`
  beside the dossier family. A pack derived from `serving` adds `kernels` and `hardware` to its `skills:`.
- **A directive** that keeps the agent on the plan, for example: "Work the lane's items in order, S1 first. No GEMM
  table tuning and no server flags: the lane is the rewrite."
- **`time_budget_minutes`:** the plan's budget (its part 6). A plan usually needs many iterations before S4.
- **`session_root`:** a new directory. A session that already has a map is refused by `seed` (`--force` replaces the
  map).
- **The review gate** applies as to any new campaign: approve it with `python3 -m arbor review --session <session>
  approve`, or start with `--no-review-gate`.

```json
{
  "identity": "kernel-lane-<region>",
  "goal": "Rewrite <region> (the plan's target), stage by stage, to a booked gain at the served operating point.",
  "pack": "serving-knowledge",
  "space": {"levels": ["kernel"]},
  "time_budget_minutes": 720,
  "session_root": "<a new directory>",
  "directives": ["Work the lane's items in order, S1 first. No GEMM table tuning and no server flags."]
}
```

The other keys (`docker_container`, `workspace_root`, the server and bench settings) come from the serving campaign's
config, unchanged. `python3 -m arbor run --config <config.json> --dry-run` checks the config on synthetic numbers in a
temporary session, so it can run before or after the seed.

## 4. Seed and run

```bash
python3 -m arbor.stages seed --config <config.json> --direction m-<region> --plan <plan.md>
python3 -m arbor run --config <config.json>
```

- **What `seed` writes** into the session: the lane's map (`directions.json`, scope `lane`), `stages.json`, and a
  pinned copy of the plan, `lane_plan.md`. The agent works from the pinned copy, so editing the dossier's plan later
  does not change the lane.
- **Each iteration works one item.** The agent ends it with a `STAGE:` line, which the harness judges at the next
  iteration's start, once this one has its verdict:

  ```
  STAGE: {"item": "m-<region>.c.s2", "result": "pass", "measured": {"snr_db": "<dB>", "correct": true, "negative_control_fired": true}, "evidence": {"artifacts": ["iter_<n>/<check>.json"], "ledger": ["<ledger ID>"]}}
  ```

  - **`result`:** `pass`, `fail` or `stop`.
  - **`measured`:** the plan's gate measure and flags.
  - **`evidence`:** the artifacts the iteration wrote, and the ledger IDs it recorded.
- **S4** lands the kernel behind a switch that defaults off, declared on a `SWITCH: NAME=1` line beside `DONE:`. The
  kernel logs one line naming the switch where it dispatches, so the harness can see it on every rank.
- **S5** needs no `STAGE:` line: a booked KEEP whose `DONE:` line cites the item closes it.

## 5. Watch it, and its end

- **`python3 -m arbor.stages show --session <session>`:** each item, its attempts with their numbers, and what waits to
  be judged.
- **`<session>/stages.json`:** every attempt, with its iteration, its numbers, its evidence and the harness's reason.
- **The end:** when every item has a verdict, the harness writes `LANE_CLOSED.json` and a `STOP` that starts
  `lane-closed:`, and the driver stops.
- **What you get:** the kernel in the served tree behind its switch (S4), and, when S5 passed, a booked gain with the
  switch on. A killed item says why in `stages.json`: the gap was below the floor, the design could not meet its
  tolerance, or the A/B refused it.

## 6. What goes wrong

- **A pass booked as a fail.** The harness re-checks every `STAGE:` line. It books a fail, with its reason, when:
  - the numbers do not clear the plan's gate;
  - the iteration's intent is more lenient than the stage's (an S4 declared as LEARN);
  - an artifact was not written during the iteration;
  - a ledger ID is not in the ledger;
  - the items the stage comes after are not done.
- **S4 refused at the KEEP.** The harness serves the tree with each declared switch on, and refuses the KEEP when:
  - the arm already sets the switch on;
  - the accuracy check fails with it on;
  - a JIT cache gains a file once the server is healthy (a kernel compiling at serve time);
  - the log has no line naming the switch from every rank.
- **`seed` refuses:** the config names no `session_root` (give `--session`), the direction is not a map ID
  (`m-<component>`), the plan's `ITEMS` do not pass `check`, or the session already has a map (`--force`).
- **Eager timing misleads.** Time S1 and S3 graph-captured, at the served shapes: eager timing is host-bound and can
  misread a saving's size. Only S5 books a gain.

## 7. Other ways to run kernel work

- **In parallel mode:** a run config's lanes may be `kernel-plan` lanes, each seeded the same way on its own node
  beside the round's other lanes (`python3 -m arbor.parallel`; `configs/parallel/example-c6p.json` has a
  `kernel-plan` kind). See [operations.md](operations.md).
- **A kernel on its own, with no server:** the `evolve-asm` pack is a `kernel` workload with its own orchestrator; it
  benches the kernel by itself. See [packs.md](packs.md).
