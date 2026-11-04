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
```

Only `knowledge/`, `validator/` and `eval/` belong in a pack.

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
| `bench_repeats` | 1 | replays per measurement |

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
| `dossier` | find or write the campaign's dossier and its map | serving, balanced, model-m-2, -2-mi355x, -c8, -c8-tp4, model-m-plan, parallel-dossier |
| `profiler` | profile with TraceLens (the host's profile first), rocprofv3 | serving, balanced, model-m-2, -2-mi355x, -c8, -c8-tp4, parallel-dossier |
| `hardware` | MI350X, MI355X and MI300X facts and achieved limits | balanced |
| `kernels` | whether to write a kernel, in which language, how to prove it (AITER first) | balanced |
| `serving-playbook` | the rules for serving changes, and the knob catalog the harness reads | balanced |
| `plan` | write a round plan | model-m-plan |
| `knowledge-collector` | share and read campaign learnings | parallel-dossier |
| `campaign-archive`, `dashboard`, `realtime-collector`, `merge-optimization-trees`, `geko-hipblaslt-tuning` | operator tools | none |

Vendored: `kernels/reference` and `hardware/reference` from KernelForge (`src/kernelforge`) (MIT).

How to use `dossier` and `profiler`, in a campaign and by hand, and why a dossier's directions go
into the dossier rather than the pack: [dossier-and-profiler.md](dossier-and-profiler.md).

### Where skills take part: chapters, families and triggers

Writing a skill, or a family of them: [skills.md](skills.md). A session runs through ordered **chapters** (`arbor/lifecycle.py`): `review`, `init`, `benchmark`,
`accuracy`, `map.discover`, `map.measure`, `map.directions`, `iterate`, `round_end`, `end`. A skill
takes part in up to three ways:

- **`run`:** Arbor runs it in a chapter, in an agent turn of its own.
- **`trigger`:** Arbor runs it when an event fires (`keep`, `round_end`, `end`, ...), without holding
  the session up.
- **`use`:** the chapter's agent may use it when it chooses, as every skill is used today.

A pack declares them instead of `skills:` (not both):

```yaml
plugins: [dossier]                         # skill families: plugins/<family>/, skills named family:skill
lifecycle:
  map.discover: {run: ["dossier:discover"]}
  iterate:      {use: ["dossier:read", profiler]}
triggers:
  keep: [knowledge-collector]
```

- **A skill says where it can take part** under `metadata.arbor` in its `SKILL.md` (`modes`,
  `chapter`, `triggers`, `produces`, `after`, `before`, `reuse_key`, `gpu`).
- **`validate_pack`** refuses a family that is not there, a skill no flat skill or listed family
  holds, and a skill the pack has take part in a way it does not declare.
- **Not `on:`:** YAML reads a bare `on` as true, so event skills go under `triggers:`.
- **`use` per chapter:** each agent turn gets its chapter's `use` list, and the families' plugins.
  - **Which chapter:** a dossier round (the map's `bootstrap`) is `map`, all three of its
    sub-chapters' lists together. The dossier's update round is `round_end`. A document campaign is
    `map`. Every other turn is `iterate`.
  - **Old packs:** a pack without `lifecycle:` reads as today, its `skills:` in every chapter.
  - **The journal:** each chapter entered is written to `<session>/lifecycle.jsonl`, the feed a host
    follows ([hosts.md](hosts.md)).
- **Map rounds** run for a pack whose map chapters name skills, as for one whose `skills:` has
  `dossier`.
- **`run` in chapters** (`arbor/chapters.py`): a turn whose chapters list `run` skills is theirs, not
  the main agent's.
  - **The order:** a `map` turn runs the `benchmark` chapter's, then `map.discover`'s,
    `map.measure`'s and `map.directions`'s; a `round_end` turn runs its own. A pack lists them in
    order, and `after` and `before` in a skill's metadata are checked against it.
  - **Each skill's turn:** dispatched as `/<skill> <brief>`, with its chapter's `use` skills. The
    brief, under `<session>/runs/`, names the files it must write and holds the program.
  - **After it:** Arbor checks the files it `produces`. A skill whose files are there, for the same
    `reuse_key` values (config keys) as its last complete run, is skipped. One that leaves a file
    unwritten stops the chapter; the next turn takes it up.
  - **What the iteration reads:** the skills' outputs, then `INTENT: LEARN` and `DONE:`, booked as
    the map's writing is. The last skill's `DIRECTIONS:` line sets the map, so a dossier pass exits
    `mapped` after `map.directions`.
  - **Placeholders in `produces`:** `{dossier}` (`docs/dossier/<config dossier, else the identity
    as a slug>`), `{benchmark}` (`benchmarks/<config benchmark, else the pack>`), `{session}`,
    `{workspace}`.
  - **The journal:** each run, skip and stop is written to it (kind `run`).
- **Under a host** (component mode), the pack's `component.lifecycle` replaces the chapters it names:
  ```yaml
  component:
    lifecycle:
      benchmark: {}            # the host's rules stand in for the benchmark's study
  ```
  The other keys proposed for `component:` (`knowledge`, `void`, `checks`, `objective`) are refused until they are built.
- **The families Arbor ships:**
  - `dossier` ([dossier-and-profiler.md](dossier-and-profiler.md));
  - `benchmark`, whose `benchmark:study` writes [`benchmarks/<name>/`](../benchmarks/README.md): the
    rules, the reference and what to measure;
  - `pack`, whose `pack:draft` drafts a new pack ([below](#writing-a-pack)).
- **`trigger` on events** (`arbor/triggers.py`): a pack's `triggers:` lists skills by event.
  - **How each runs:** when its event fires, in an agent turn of its own, in a process of its own,
    without holding the session up.
  - **A blocking skill** (`metadata.arbor.blocking: true`) runs to its end first, within 30 minutes.
  - **Events fired today:** `session_start`, `iteration_start`, `keep`, `revert`, `probe`,
    `invalid`, `agent_failed`, `map_set`, `round_end` and `end`.
  - **Events not fired yet:** `review_decided`, `baseline_set`, `map_exhausted`, `escalate`,
    `rebaseline` and `deadline_near`.

**Checking skills and families:** `python3 -m arbor.skillcheck` reports, for every flat skill and
family in the tree:
- front matter that is not YAML;
- a name that is not its directory's;
- a missing description;
- unknown `metadata.arbor` keys, modes, chapters or events;
- a family's `plugin.json`.

It warns on a `SKILL.md` over 500 lines, a description Claude Code would cut, and a `run` skill
without `disable-model-invocation: true`. The tests run it over the repository.

## The packs in this repository

| Pack | What | Config |
|---|---|---|
| `serving` | the serving template: `balanced`'s method for any vLLM or SGLang model; `dossier` and `profiler` | `configs/serving-template.json` (example) |
| `balanced` | Model M, vLLM, MI350X, TP8, 8k/1k at C32; every harness rule and five skills | `configs/mi350x/model-m-c32-balanced.json` |
| `model-m-c8-tp4` | Model M, MI350X, TP4, C8 | `configs/mi350x/model-m-c8-c2…c5.json`, `-c8-component-test.json` |
| `model-m-1`, `-2`, `-2-mi355x`, `-c8` | Model M campaigns A, B (MI350X and MI355X), C | `configs/mi350x/`, `configs/mi355x/` |
| `agentx-dsv41`, `-2`…`-6`, `-2b`…`-2g`, `-2-replica`, `-2.1`, `-cuda-graph`, `-cg8a`, `-cg8b` | DeepSeek-V4.1-Flash, AgentX replay, vLLM, gfx950: the DSV4.1 campaign line | `configs/mi355x/dsv41flash-*.json` |
| `model-m-plan` | a document campaign: writes a plan, scored by a linter | `configs/mi350x/model-m-plan-c.json` |
| `parallel-dossier` | parallel mode's map step | `configs/parallel/` (examples) |
| `agentx` | GLM-5.2, InferenceX trace replay | `configs/agentx-glm52-mi355x.json` (the dry run's) |
| `evolve-asm` | assembly kernel evolution (`kernel` workload), its own orchestrator | none |

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
`bench_script` and `correctness_script`. `model-m-plan` and `parallel-dossier` are small real ones.

A serving pack starts from the serving template (below). Check a new pack with
`python3 -m arbor run --config <cfg> --dry-run`.

**Drafting one:** the `pack` family's `draft` skill. Load it with `claude --plugin-dir plugins/pack`
from the Arbor tree, then run `/pack:draft <what the campaign is>`.
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

- **Skills:** `dossier` and `profiler` by default. Add `hardware`, `serving-playbook` and
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
