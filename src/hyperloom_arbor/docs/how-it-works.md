# How Arbor works

Arbor runs a **campaign**: a loop of iterations in which an agent changes a serving tree and the
harness measures the change and keeps or reverts it. This page describes one iteration, how the
gate decides, how the campaign plans its work, and what it leaves behind. File references are to
this repository; [reference.md](reference.md) lists every config key and file.

## The parts

- **The driver** (`arbor/driver.py`) runs the loop: it checks for `STOP` and `ABORTED.md` in the
  session, the iteration and time budget, the round end and the target, then runs one iteration
  and cools down (`--cooldown`, 30 s by default).
- **The harness** (`arbor/policy/claude_loop.py`) runs an iteration. It owns git, the server's
  lifecycle checks, the bench output, correctness, the gate, the records and the revert.
- **Four agents**, all Claude through `claude_agent_sdk`, each a separate turn with its own tools:

  | Agent | Does | Tools |
  |---|---|---|
  | main | reads the campaign program, makes one change, declares its intent | Bash, Read, Edit, the pack's skills |
  | server | launches the server on the changed tree and answers `SERVER_READY` or `SERVER_FAILED` | Bash, Read, Edit |
  | bench | runs the benchmark and reports the metric | Bash, Read, Edit |
  | critic | reviews a result the gate did not revert, and may downgrade it to REVERT | Read, Bash |

  The dossier and profiler are skills of the main agent, not separate agents. The main agent's
  instructions are `arbor/policy/claude_loop_prompt.md` and the campaign program the pack renders
  ([packs.md](packs.md)); the critic's are `arbor/policy/critic_prompt.md`.
- **The pack** says what is optimized and how it is measured; **the config** says where: the
  model, the container, the workspace, the server and the session ([packs.md](packs.md),
  [reference.md](reference.md)).

## Before the first iteration: the review gate

A new campaign (no `session.json` yet) stops once its config and pack load, before anything
executes: no validator is pinned, no workspace commit made, no agent, server or bench started
(`arbor/review.py`). It writes `<session>/REVIEW.md` (the objective and bar, the rules, skills
and policy, the workload and bench, the pack's roadmap rendered and linted, the program the agent
will read at iteration 1, and the files that say so) and `review.json` (`pending`), then waits.

A human reviews and edits any of those files (the pack, its knowledge, its roadmap, the config),
then decides:

```bash
python3 -m arbor review --session <dir> show
python3 -m arbor review --session <dir> approve --note "why"   # checks the edits, then the run goes on
python3 -m arbor review --session <dir> reject --note "why"    # the run ends: review_rejected
```

After the approval the driver loads the campaign again from disk, so the reviewed files are the
ones pinned and measured; `session.json` records who approved, when, and which files changed.
`STOP` ends the wait, and so does a host's deadline in component mode. A session that already has
`session.json` (a resume, a host's relaunch) is not gated again.

The gate is on by default, in both modes. Turn it off with `run --no-review-gate`,
`ARBOR_REVIEW_GATE=0` or `"review_gate": false` in the config (the flag wins over the
environment, the environment over the config). An unattended launcher must use one of them, or
its run waits for an approval that does not come.

## One iteration

1. **Before the agent:** the harness re-seeds `arbor/config.json` in the workspace if it is
   missing, prepares links to binaries outside the tree (below), and stops a server still holding
   memory on the campaign's GPUs (in component mode, only the server it started).
2. **The program:** the harness renders the campaign program: the pack's description and
   knowledge, the objective, the constraints, the map, the ledger, the history and the notices.
3. **The main agent** makes one change and ends with its intent (`INTENT: WIN` and the others
   below), `DONE: <one line>`, and any `LEDGER:`, `DIRECTIONS:`, `DIRECTION:` or `STAGE:` lines.
   An agent that never ran makes the iteration NOT_MEASURED.
4. **Commit:** `git add -A` and a commit named `iter-N: <change>`. No diff is a REVERT, unless it
   is a LEARN that wrote ledger records (then PROBE).
5. **Read-only LEARN:** a LEARN that changed nothing served skips the server and the bench.
6. **Server:** the harness stops the old server; the server agent launches the new one; the
   harness waits up to 120 s for health and checks that the live process carries the flags and
   env the change claims (armcheck, below).
7. **Bench:** the bench agent runs the pack's bench, or the host's `bench_command` in component
   mode, and the harness reads `iter_N/bench_output.json` (the agent's `METRIC:` line is the
   fallback). With `bench_repeats` above 1, a fixed block books the mean.
8. **Correctness:** the harness runs the pack's correctness script (in component mode, under its
   phase cap), then component mode's short-prompt probe ([hosts.md](hosts.md)).
9. **Finalize:** the gate decides; the critic reviews anything not reverted; the harness writes
   the records and reverts a REVERT with a new revert commit.

## Intents and verdicts

The agent declares what its change is for. The intent sets how the gate judges it.

| Intent | For | The gate |
|---|---|---|
| `WIN` (the default) | a faster tree | KEEP only if the metric clears the bar |
| `FIX` | a defect, cleanup or enabling change | KEEP when neutral; no bar |
| `FIX-CORRECTNESS` | a correctness fix | KEEP when neutral; may cost up to `correctness_regression_cap_pct` (5 %) of the metric |
| `LEARN` | a probe: evidence, not a tree | KEEP as PROBE when it records findings; books nothing |

Older names still parse: `CLASS: B` (WIN), `A` (FIX), `A-CORRECTNESS`, `PROBE` (LEARN).

**Verdicts:** KEEP, REVERT and NOT_MEASURED. NOT_MEASURED means the iteration says nothing about
the change: an integrity failure (a protected validator file changed), an arm mismatch under
`arm_check: enforce`, an infrastructure error in correctness, a bench that did not run, or a
missing or non-positive metric. It changes no counter except the invalid streak; three in a row
write `ABORTED.md` and stop the campaign.

**The bar:** the pack's `objective.measurement.threshold_rel` (1.01 by default) over the best
booked result. A change inside the noise band (`noise_band_pct`, 1.0 % by default) is neutral:
KEEP for FIX, REVERT for WIN. The first measurement of a session is its baseline and a KEEP.

**A second metric** (`secondary_metric`) is judged on its own band. A gain on one axis bought with
a loss beyond the band on the other is a **frontier point**: kept, but it moves neither bar.
`secondary_regression_cap_pct` and `secondary_exchange_rate` can cap what the second axis may lose.

**The critic** runs on every result the gate did not revert and can only downgrade it to REVERT.
If it fails or answers nothing it can parse, the gate's verdict stands.

**Optional checks** (pack or config `policy`):
- `accuracy_gate`: when a change moves a numerics knob or the agent writes `NUMERICS: yes`, the
  full GSM8K runs against the baseline's score less `accuracy_tolerance` (0.015).
  - **The baseline's score** is the config's `baseline_record` (`correctness_full.score`).
  - **In component mode, without one,** the driver serves the root once before iteration 1 and
    runs the full set on it: `<session>/baseline_accuracy.json`. A host's own eval reads on
    another scale, so it cannot stand in.
  - **Without either** (the root could not be measured, or the deadline left no room), the pack's
    `GSM8K_MIN` is the floor, with no tolerance.
- `arm_check`: `record` (default) logs whether the live server carries what the change claims;
  `enforce` makes a mismatch NOT_MEASURED; `off` skips it.

**Behaviour to know:** a server that fails to come up (`SERVER_FAILED`, or no health in 120 s) is
booked as a REVERT of the change, not as NOT_MEASURED.

## Tiers, streaks and the LEARN budget

- **Tiers** (`space.levels` in the config; default `config, framework, kernel, feature`): the
  campaign starts at the first and escalates after `config_max_iters` (3) iterations on the config
  tier, or when `null_streak` (iterations without a booked gain) reaches `escalate_null_streak` (3).
- **LEARN budget:** at most `max_consecutive_probes` (2) LEARNs in a row; at `null_streak` 3 the
  program asks for one ("LEARN REQUIRED").
- **Drought:** `barren_streak` counts iterations since the bar last moved. At `barren_notice` (8)
  the program says so; at `barren_act` (16) it tells the agent to change approach, and the LEARN
  budget becomes `learn_budget_barren` (4).
- **All of these** are `policy` fields (`arbor/state.py`): set them in the pack's `policy:` block
  or the config's `policy`, which wins. They are frozen into `session.json` when the session starts.

## Planning: the dossier, the map and rounds

With the `dossier` skill enabled (and a serving workload), a campaign plans before it measures
(using the skills, and giving a campaign its directions: [dossier-and-profiler.md](dossier-and-profiler.md)):

1. **The dossier:** the agent finds this model's dossier in `docs/dossier/` or writes one (the
   model, the system, the code, the target, the workload, the baseline, a profile, the kernels,
   and in part 11 a ranked list of directions). These iterations are LEARNs.
2. **The map:** the agent states the directions with one `DIRECTIONS: [...]` line, each with a map
   ID (`m-<component>`). The harness keeps the map in `<session>/directions.json`; only the
   harness writes it, and a hand edit is undone.
3. **Working the map:** the agent cites the direction it works in its DONE line and closes it with
   `DIRECTION: {"id": ..., "status": "done|killed|below_floor|plan", "why": ...}`.
4. **The update round:** when every direction is closed, the next iterations re-profile, write the
   lessons and re-rank the directions (LEARNs, the cap waived, at most 8). The new map starts the
   next round; a map with nothing new leads to a fresh start, then to the top planned rewrite, then
   to STOP. Two rounds in a row without a KEEP stop the campaign.

**Kernel plans:** a rewrite worth planning becomes a chain of stages with their own gates
(`arbor/stages.py`): S1–S3 run as LEARN (the kernel built and proven on its own), S4 as FIX (wired
in behind its flag, and it must KEEP), S5 needs a booked WIN. `STAGE:` lines are read only in a
lane's map (parallel mode, or `python3 -m arbor.stages seed`).

**Round ends and the target:** a campaign's `target` (a number, or a dict) ends a round when it is
reached, and so does a closed map. With a round-end hook (`ARBOR_ROUND_AB`, or the config's
`round_ab.command`), the driver runs a same-node A/B of the campaign's base against its current
tree: confirmed, the A/B becomes the bar and baseline, the next target is 1.40× the bar, and the
next round starts; not confirmed, the campaign stops with `ROUND_STOP.md`. Without a hook,
reaching the target ends the run.

**The rebased bar:** when a campaign moves to another node, the launcher can rebase its bar to
that node (`scripts/model-m-campaign/rebase_bar.py`), but only for a campaign with a dossier and a map
(`arbor/rebaseline.py`).

## What a campaign leaves behind

In the session directory ([reference.md](reference.md) has the full layout):

- **`session.json`:** the campaign's state: bars and baselines on both axes, counters, policy, and
  `lessons`, one row per verdict.
- **`ledger.jsonl`:** what the agents learned: `FACT`, `ASSUMPTION`, `UNKNOWN` and `DEAD_END`
  records from their `LEDGER:` lines. A REVERT with no record gets a `DEAD_END` written for it.
- **`landed.patch`:** on every KEEP, the cumulative served tree against the workspace's first
  commit: everything but `arbor/config.json`, the agent's probes (`arbor/eval/`, `eval/`, logs and
  the config's `scratch_paths`), bytecode, and any file over 10 MB. It is git's own bytes and
  applies with `git apply`.
- **`scratch.patch`:** the agent's probes and evaluators, on KEEP and PROBE.
- **`landed_packages/` and `landed_packages.json`:** on KEEP, edits to other installed packages
  (`editable_packages`, AITER for example), so a new container can restore them.
- **`outside.json`:** on KEEP, every link the served tree makes outside itself (for example AITER
  JIT modules): target, sha256 and size. Binaries are not stored; AITER rebuilds a missing module
  at its first call, and the harness marks such links so git never commits the rebuilt binary.
- **`iter_N/`:** each iteration's bench and correctness outputs, the server log and effective
  config, and `patch.diff` for KEEPs and PROBEs.

## Modes

- **Standalone** (the default): Arbor owns its node, container and clock. Launchers provision the
  node; see [operations.md](operations.md).
- **Component mode:** Arbor runs inside a host (Hyperloom's multibranch today) that owns the cards,
  the deadline and the merge. See [hosts.md](hosts.md).
- **Rolling chunks** (`ARBOR_ROLL=1`, Cluster A): a campaign longer than one job's wall moves from node
  to node. See [operations.md](operations.md).
- **Parallel mode** (`python3 -m arbor.parallel`): rounds of map, lanes on separate nodes, and an
  A/B per lane at integration. Built and tested on fakes, never run on GPUs. See
  [operations.md](operations.md).
