# Packs

A pack says what a campaign optimizes and how it is measured: the objective, the validators, the
constraints, the knowledge the agent reads and the skills it may use. A config names one pack
(`"pack": "<directory>"`) and says where it runs ([reference.md](reference.md)).

## Layout

```
packs/<name>/
  pack.yaml        the pack
  validator/       bench and correctness scripts (pinned at the session's start)
  knowledge/*.md   background the agent reads every iteration
  eval/            the agent's evaluators (outside the served tree)
  tools/           helper scripts the pack's `tools:` names (not pinned, not served)
```

Only `knowledge/`, `validator/`, `eval/` and `tools/` belong in a pack.

**`results/` needs an opt-in.**
- **By default it is refused.** In component mode that is at start (exit 2), with the reason.
- **Why:** a pack is the contract every campaign of its kind shares. Results kept in it reach every
  campaign that uses the pack, where an agent may read another model's, node's or image's numbers as
  evidence about its own, and they grow the pack and its repository.
- **Where results belong:** in the session, or in a results repository.
- **To keep a pack's `results/` anyway:** pass `--allow-pack-results` to `arbor run`. The run then
  starts with a warning that repeats why.

## `pack.yaml`

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | the pack's name |
| `description` | | what the campaign is; rendered near the top of the program |
| `objective` | yes | `kind` (`maximize_perf` or `find_witness`), `metric`, `direction` (`max`/`min`), `unit`, and `measurement` (below) |
| `validator` | yes | `bench`, `correctness` (pack-relative paths that must exist) and `protected` (paths pinned at start) |
| `workload` | yes | `kind`: `serving`, `kernel`, `repro` or `document` |
| `constraints` | | a list of rules, one bullet each in the program |
| `skills` | | skills the main agent may use ([below](#skills)) |
| `opt-out-skills` | | default skills to drop |
| `policy` | | the campaign's policy ([how-it-works.md](how-it-works.md)); the config's `policy` wins |

**`objective.measurement`** (re-read every iteration, so it can change mid-campaign):

| Key | Default | Meaning |
|---|---|---|
| `threshold_rel` | 1.01 | the bar: a WIN must beat the best by this ratio |
| `noise_band_pct` | 1.0 | inside this band a change is neutral |
| `secondary_metric`, `secondary_direction`, `secondary_noise_band_pct`, `secondary_threshold_rel` | | a second axis |
| `secondary_regression_cap_pct`, `secondary_exchange_rate` | , 1.0 | how much the second axis may lose |
| `correctness_regression_cap_pct` | 5.0 | what a FIX-CORRECTNESS may cost |
| `stability_regression_cap_pct` | 3.0 | what a FIX-STABILITY may cost, its failure on record |
| `stability_collapse_pct` | 10.0 | a cited iteration this far below the best counts as the failure |
| `stability_signals` | collective watchdogs and timeouts, GPU faults, a dead engine | regular expressions for a failure in a server log (`arbor/measure/stability.py`) |
| `bench_repeats` | 1 | replays per measurement |
| `bench_minutes` | | how long one bench runs, in minutes; -1: no end. The bench turn's default cap is never under it plus 15 min, standalone and under a host (`agentx`: 75). A cap the config or the host sets (`phase_minutes`) still wins |

Some packs carry keys nothing reads (`tools:`, `measurement.count`, a `roadmap.yaml`); they are
notes for people.

## The validators

The scripts that run are the config's `bench_script` and `correctness_script`; the pack's
`validator` entries must exist. Both scripts run in the container.

- **The bench** receives `PORT`, `MODEL_PATH`, `MODEL_NAME`, `CONCURRENCY`, `NUM_PROMPTS`,
  `BENCH_OUTPUT`, `BENCH_SERVING` (and `ISL`, `OSL`), plus `server_env`. It writes JSON to
  `$BENCH_OUTPUT` with the objective's metric as a number (above 0, except for `find_witness`),
  and the second metric if there is one. Without it the iteration is NOT_MEASURED.
- **The correctness script** receives `PORT`, `MODEL_PATH`, `MODEL_NAME`, `CORRECTNESS_OUTPUT`, and
  `GSM8K_LIMIT`/`GSM8K_MIN` from the config's `server_env`. It writes `{"passed": ..., "score":
  ..., "error": ...}` to `$CORRECTNESS_OUTPUT` and exits 0.
- **A `document` workload** has no server: the harness runs both scripts on the committed draft,
  with `WORKSPACE_ROOT`, `ITERATION` and `SESSION_DIR` set.
- **Protected paths** are joined to the directory above `bench_script`'s (so the scripts must sit
  at `<pack>/validator/<script>`), hashed at the session's start, and checked at every gate. A
  changed file makes the iteration NOT_MEASURED. Never edit a validator while a campaign runs.

## The program the agent reads

`arbor/pack/render.py` builds it each iteration, in this order: the title (`name` and the config's
`goal`), the regime, `description`, `arbor/knowledge/*.md`, the pack's `knowledge/*.md`, the
objective (baseline, best, target), the editable environment, the config's `directives`, the
constraints, the skills, bundles and the accuracy gate (when on), escalation, the map, the search
space, the ledger, the history, and the notices (a suspect second bar, a drought, the LEARN
budget). The iteration's instructions (`arbor/policy/claude_loop_prompt.md`, with the intent table)
follow.

`packs/_shared/intents.md` is that intent table as a document, generated from
`arbor/measure/intents.py` by `scripts/pack/render_intents.py` and kept in sync by a test.

## Skills

A skill is `.claude/skills/<name>/SKILL.md`. A pack's `skills:` list enables them for the main
agent only (not the server or bench agent); a skill named in a pack must exist.

| Skill | What it is for | Enabled by |
|---|---|---|
| `dossier-v1` | find or write the campaign's dossier and its map | serving, balanced, parallel-dossier |
| `profiler` | profile with rocprofv3, TraceLens and a vendored graph-aware tracer | serving, balanced, parallel-dossier |
| `hardware` | MI350X, MI355X and MI300X facts and achieved limits | balanced |
| `kernels` | whether to write a kernel, in which language, how to prove it (AITER first) | balanced |
| `serving-playbook` | the rules for serving changes, and the knob catalog the harness reads | balanced |
| `plan` | write a round plan | a document campaign that writes one |
| `knowledge-collector` | share and read campaign learnings | parallel-dossier |
| `dashboard`, `realtime-collector`, `merge-optimization-trees` | tools for the people running campaigns | none |

Vendored: the profiler's tracer (`profiler/VENDOR.md` names its source and commit);
`kernels/reference` and `hardware/reference` from KernelForge (MIT licence, `kernels/reference/LICENSE-KernelForge`).

How to use `dossier-v1` and `profiler`, in a campaign and by hand, and why a dossier's directions go
into the dossier rather than the pack: [dossier-and-profiler.md](dossier-and-profiler.md).

### Where skills take part: the load list, chapters and events

Writing a skill, or a family of them: [skills.md](skills.md). A session runs through ordered **chapters**
(`arbor/lifecycle.py`): `review`, `init`, `benchmark`, `accuracy`, `map.discover`, `map.measure`, `map.directions`,
`iterate`, `round_end`, `end`. A runtime skill takes part in up to three ways, which its own metadata declares:

- **`run`:** Arbor runs it in its `chapter`, in an agent turn of its own.
- **`trigger`:** Arbor runs it when one of its `triggers` (events: `keep`, `round_end`, `end`, ...) fires, without
  holding the session up.
- **`use`:** the agent of each of its `chapters` (every chapter when it names none) may use it when it chooses.

A pack only says which skills it loads, instead of `skills:` (not both):

```yaml
lifecycle:
  load: ["dossier:discover", "dossier:read", profiler, knowledge-collector]
```

- **Names:** a family's skill is `family:name` (`.claude/skills/<family>/<name>/`), a flat one `name`.
- **A skill says where it takes part** under `metadata.arbor` in its `SKILL.md` (`type`, `modes`, `chapter`,
  `chapters`, `triggers`, `produces`, `after`, `before`, `reuse_key`, `overlaps`, `gpu`).
- **`validate_pack`** refuses a skill the tree does not hold, a development skill, a loaded skill that fails its own
  check, an `after`/`before` cycle, and the keys this form replaced (`plugins:`, `triggers:`, and chapters with `run`
  and `use` lists).
- **Not `on:`:** YAML reads a bare `on` as true; a trigger skill names its events in its metadata.
- **`use` per chapter:** each agent turn gets its chapter's `use` skills, and the plugins Arbor builds for the families
  loaded (`<session>/skills/plugins/`).
  - **Which chapter:** a dossier round (the map's `bootstrap`) is `map`, all three of its sub-chapters' skills
    together. The dossier's update round is `round_end`. A document campaign is `map`. Every other turn is `iterate`.
  - **Old packs:** a pack without `lifecycle:` reads as before chapters, its `skills:` in every chapter.
  - **The journal:** each chapter entered is written to `<session>/lifecycle.jsonl`, the feed a host follows
    ([hosts.md](hosts.md)).
- **Map rounds** run for a pack that loads a skill running in a map chapter, as for one whose `skills:` has
  `dossier-v1`.
- **`run` in chapters** (`arbor/chapters.py`): a turn whose chapters have `run` skills is theirs, not the main agent's.
  - **The order:** a `map` turn runs the `benchmark` chapter's, then `map.discover`'s, `map.measure`'s and
    `map.directions`'s; a `round_end` turn runs its own. Within a chapter, `after` and `before` in each skill's metadata
    order them, and ties follow the load list.
  - **Each skill's turn:** dispatched as `/<skill> <brief>`, with its chapter's `use` skills. The brief, under
    `<session>/runs/`, names the files it must write and holds the program.
  - **After it:** Arbor checks the files it `produces`. A skill whose files are there, for the same `reuse_key` values
    (config keys) as its last complete run, is skipped. One that leaves a file unwritten stops the chapter; the next
    turn takes it up.
  - **What the iteration reads:** the skills' outputs, then `INTENT: LEARN` and `DONE:`, booked as the map's writing is.
    The last skill's `DIRECTIONS:` line sets the map, so a dossier pass exits `mapped` after `map.directions`.
  - **Placeholders in `produces`:** `{dossier}` (`docs/dossier/<config dossier, else the identity as a slug>`),
    `{benchmark}` (`benchmarks/<config benchmark, else the pack>`), `{session}`, `{workspace}`.
  - **The journal:** each run, skip and stop is written to it (kind `run`).
- **Two loaded skills that write the same thing** (the same file in `produces`, or one `overlaps` the other) conflict
  (`arbor/conflicts.py`). Standalone, a session pauses before anything runs for a person to pick one or let an agent
  merge them (`python3 -m arbor conflicts --session <dir> show|pick|merge`); `"skill_conflicts": "merge"` merges
  without a pause; under a host a conflict refuses the run unless auto-merge is set. See [skills.md](skills.md#when-two-skills-conflict).
- **Under a host** (component mode), the pack's `component.lifecycle.load` replaces the load list:
  ```yaml
  component:
    lifecycle:
      load: ["dossier:discover", "dossier:read", profiler]   # no benchmark:study: the host's rules stand in
  ```
  The other keys proposed for `component:` (`knowledge`, `void`, `checks`, `objective`) are refused until they are
  built.
- **The families Arbor ships:**
  - `dossier` ([dossier-and-profiler.md](dossier-and-profiler.md));
  - `benchmark`, whose `benchmark:study` writes [`benchmarks/<name>/`](../benchmarks/README.md): the rules, the
    reference and what to measure;
  - `pack`, whose `pack:draft` drafts a new pack ([below](#writing-a-pack));
  - `arbor-dev`, for people developing Arbor, which no campaign loads.
- **`trigger` on events** (`arbor/triggers.py`): each loaded trigger skill runs on the events its metadata names.
  - **How each runs:** when its event fires, in an agent turn of its own, in a process of its own, without holding the
    session up.
  - **A blocking skill** (`metadata.arbor.blocking: true`) runs to its end first, within 30 minutes.
  - **Events fired today:** `session_start`, `iteration_start`, `keep`, `revert`, `probe`, `invalid`, `agent_failed`,
    `map_set`, `round_end`, `stretch_end` (a worker's report is written) and `end`.
  - **Events not fired yet:** `review_decided`, `baseline_set`, `map_exhausted`, `escalate`, `rebaseline` and
    `deadline_near`.

**Checking skills and families:** `python3 -m arbor.skillcheck` reports, for every flat skill and family in the tree:
- front matter that is not YAML;
- a name that is not its folder's;
- a missing description;
- unknown `metadata.arbor` keys, types, modes, chapters or events, a run skill without a chapter, and a placement key
  without its mode;
- a family's `family.json`.

It warns on a `SKILL.md` over 500 lines, a description Claude Code would cut, and a `run` skill without
`disable-model-invocation: true`. The tests run it over the repository.

## The packs in this repository

| Pack | What | Config |
|---|---|---|
| `serving` | the serving template: `balanced`'s method for any vLLM or SGLang model; `dossier-v1` and `profiler` | `configs/serving-template.json` (example) |
| `serving-knowledge` | the serving template on the lifecycle (`serving-m10`) with knowledge search: `kernels` and `hardware`, their reference trees behind `knowledge:search`, which also searches KernelForge, KernelWiki and Arbor's notes (`arbor/knowledge/`); the sources synced at the campaign's start, the dossier's knowledge citations checked | `configs/serving-knowledge-template.json` (example) |
| `balanced` | one large MoE model on vLLM, MI350X, TP8, 8k/1k at C32; every harness rule and five skills | its campaign's own |
| `parallel-dossier` | parallel mode's map step | `configs/parallel/` (examples) |
| `agentx` | an InferenceX trace replay on SGLang | its campaign's own |
| `evolve-asm` | assembly kernel evolution (`kernel` workload), its own orchestrator | none |

Other packs in the tree belong to specific campaigns and are not described here.

## Writing a pack

The smallest valid pack is a document campaign:

```yaml
name: my-pack
objective:
  kind: maximize_perf
  metric: score
  direction: max
  measurement: {threshold_rel: 1.01, noise_band_pct: 1.0}
validator:
  bench: validator/bench.sh
  correctness: validator/correctness.sh
  protected: [validator/]
workload: {kind: document}
```

with `validator/bench.sh` writing `{"score": N}` to `$BENCH_OUTPUT`, `validator/correctness.sh`
writing `{"passed": true, "score": 1.0}` to `$CORRECTNESS_OUTPUT`, and a `knowledge/` directory.
Its config needs `pack`, `docker_container` (`"@host"`), `workspace_root`, `session_root`,
`bench_script` and `correctness_script`. `parallel-dossier` is a small real one.

A serving pack starts from the serving template (below). Check a new pack with
`python3 -m arbor run --config <cfg> --dry-run`.

**Drafting one:** the `pack` family's `draft` skill. Build it with `python3 -m arbor.skills plugin pack`, load it
with the `claude --plugin-dir` line that prints, then run `/pack:draft <what the campaign is>`.
- **What it starts from:** the nearest pack that has worked (its `parents.md`), by default the serving
  template.
- **What it takes from the benchmark's study** (`benchmarks/<name>/`): the objective, its bands and the
  constraints.
- **What it leaves out:** the model's facts, which stay in its dossier.
- **What it writes:** the skills in the lifecycle form, a config, a dry run, and a review note. The note
  says what it copied, what it chose and why, and what the person must still fill in.
- **What it does not do:** commit, run a campaign, or change another pack.

## The serving template

`packs/serving` is `balanced`'s method (the v2 line) for any model served by vLLM or SGLang, with
nothing model-specific in it. It is the pack a host's Arbor adapter uses for
a workload with no pack of its own, and the starting point for a new one.

- **Skills:** `dossier-v1` and `profiler` by default. Add `hardware`, `serving-playbook` and
  `kernels` in a derived pack.
- **The model** comes from the config (`model_path`, `framework`, `server_flags`, `server_env`,
  `tp`); its facts and the map come from its dossier. A campaign given its directions (a host's
  plan) works those.
- **The objective:** output tok/s per GPU, gated at 0.5 %, with p90 interactivity as the second
  axis. Change the metric and bands in a derived pack when the benchmark scores something else.
- **The operating point** is the config's: `concurrency`, `num_prompts`, and in `server_env`
  `ISL`, `OSL`, `NUM_GPUS` and `BENCH_SEED` (`knowledge/operating-point.md`).
- **Correctness:** GSM8K through lm_eval. Set `GSM8K_MIN` in `server_env` to the model's reference
  score less about three standard errors of the subset; until it is set, a 0.30 sanity floor
  applies, which fails a broken model and nothing finer.
- **The harness rules** are on: `arm_check: enforce`, bundles, the accuracy gate, and up to 6
  LEARNs in a row for the dossier.

`configs/serving-template.json` is its example config (Qwen3-14B-FP8, vLLM, TP1, 1024/1024 at
C64): copy it and set the `<...>` values.
