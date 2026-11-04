# Reference

The command line, the campaign config, the environment variables and the session layout, as the
code reads them. How the loop uses them: [how-it-works.md](how-it-works.md). The pack format:
[packs.md](packs.md). Component mode's own keys: [hosts.md](hosts.md).

## Command line

| Command | What it does |
|---|---|
| `python3 -m arbor run --config C [--max-iters N] [--cooldown S]` | run a campaign; the same command resumes it (`--max-iters 0` = no limit; cooldown 30 s). A new campaign first waits at the review gate ([how-it-works.md](how-it-works.md#before-the-first-iteration-the-review-gate)); `--no-review-gate` starts it without one |
| `python3 -m arbor review --session D show\|approve\|reject [--note T] [--by NAME] [--config C]` | review a new campaign before it runs: `show` prints `REVIEW.md` and the status; `approve` checks the edited campaign and lets the run go on; `reject` ends it (`review_rejected`) |
| `python3 -m arbor run --config C --dry-run [--max-iters 10] [--seed 42]` | the loop on synthetic numbers in a temporary git repo: no GPU, no container, no agent |
| `python3 -m arbor --help` | the usage |
| `python3 -m arbor iterate init\|begin\|gate ...` | the harness's internal steps, used by the loop |
| `python3 -m arbor.parallel <cmd> --config RUN.json` | parallel mode: `init`, `tick`, `run`, `status`, `explain`, `graph`, `approve`, `pause`, `resume`, `stop`, `resubmit`, `close-lane` ([operations.md](operations.md)) |
| `python3 -m arbor.stages check <plan.md>` / `seed ...` / `show --session D` | validate a kernel plan's stages; make a campaign one lane of a plan |
| `python3 -m arbor.directions check [file]` | validate a `DIRECTIONS:` block |
| `python3 -m arbor.execute.packages capture\|restore ...` | save or restore a KEEP's edits to other installed packages |
| `python3 -m arbor.execute.outside record\|prepare\|check ...` | record, prepare and check links that leave the served tree |
| `python3 -m arbor.cluster.handoff status --session D` | a rolling chain's state (also `wait`, `ensure-successor`, `selftest`) |
| `python3 -m arbor.collect check\|project\|realtime\|knowledge ...` | the collectors behind the dashboards ([operations.md](operations.md)) |
| `node arbor/dashboard/server.mjs local\|global\|parallel ...` | the dashboard (Node 16 or newer) |

## The campaign config

A JSON file. `pack` is the only key `run` requires; the pack must exist under `packs/` (or under
`$ARBOR_ROOT/packs/`). Everything else depends on the workload. Keys a config may carry that the
code does not read (`gpu_type`, `served_model_name`, `docker_image`, `baseline`) are notes for
people.

**Identity and goal**
| Key | Default | Meaning |
|---|---|---|
| `pack` | required | the pack's directory name under `packs/` |
| `identity` | | a label for the campaign |
| `goal` | | one line, the title of the program the agent reads |
| `target` | none | a number (the metric to reach), or a dict; reaching it ends a round |
| `time_budget_minutes` | none | a wall clock for the run (ignored in component mode) |
| `review_gate` | `true` | a new session waits for a human's approval before it runs anything; `ARBOR_REVIEW_GATE` and `run --[no-]review-gate` override it |
| `baseline_record` | none | a JSON file `{mean, runs, mean_secondary}` with a measured baseline |
| `directives` | `[]` | lines rendered into the program, for the agent |
| `space.levels` | `config, framework, kernel, feature` | the tiers, in order |
| `space.knobs` | | knobs listed for the agent |
| `mode` | `standalone` | `standalone` or `component` ([hosts.md](hosts.md)) |
| `policy` | | overrides the pack's `policy:` ([how-it-works.md](how-it-works.md)) |

**Workspace and container**
| Key | Default | Meaning |
|---|---|---|
| `docker_container` | | the container the server and tools run in; `"@host"` runs them where Arbor runs |
| `workspace_root` | | the served tree (a git repository) the agent edits, as the container sees it |
| `session_root` | the current directory | where the session's files go ([below](#the-session-directory)) |
| `session_mount` | `/workspace/sessions` | where the container sees the sessions |
| `scratch_paths` | `[]` | paths that hold the agent's probes, kept out of `landed.patch` (`arbor/eval`, `eval` and `*.log` always are) |
| `editable_packages` | `[]` | `[{name, path}]`: other installed packages whose edits a KEEP saves |

**Server**
| Key | Default | Meaning |
|---|---|---|
| `framework` | `sglang` | `vllm` or `sglang` (`arbor/frameworks/`) |
| `model_path` | | the weights, as the container sees them |
| `model_name` | | the name the server answers to |
| `port` | 8888 | the server's port |
| `tp` | 1 | tensor parallelism |
| `server_flags` | | the launch flags; the agent edits them in the workspace's `arbor/config.json` |
| `server_env` | `{}` | the launch environment; the same. `GSM8K_LIMIT` and `GSM8K_MIN` here set the correctness check's size and floor |
| `gpu_ids` | | the campaign's GPUs, physical numbers, `"0,1,2,3"` or a list |

**Bench and correctness**
| Key | Default | Meaning |
|---|---|---|
| `bench_script`, `correctness_script` | | absolute paths to the pack's validator scripts, as the container sees them |
| `bench_serving_script` | | an optional serving client the bench script calls |
| `bench_timeout` | 600 | seconds, for the bench and the correctness script |
| `concurrency`, `num_prompts`, `isl`, `osl` | 14, 640, 1024, 1024 | passed to the bench script |
| `bench_repeats` | 1 | measured replays per iteration; the pack's `measurement.bench_repeats` wins |

**Rounds, rolling and component mode**
| Key | Meaning |
|---|---|
| `round_ab.command` | the round-end A/B hook; `ARBOR_ROUND_AB` wins. `{out}` and `{deadline}` are filled in |
| `cluster.roll.*` | rolling chunks: `wall_min` 240, `successor_at_min` 180, `exit_margin_min` 15, `max_chunks`, `resubmit_limit` 3 |
| `component.*` | component mode's block ([hosts.md](hosts.md)) |

Parallel mode reads a separate run config ([operations.md](operations.md)).

## Environment variables

| Variable | Meaning |
|---|---|
| `ANTHROPIC_BASE_URL` | the LLM gateway the agents use. If it does not answer, Arbor tries `localhost:9101`, `9100` and `9102` at import. The gateway's other variables (its token) pass through to the agent's CLI; launchers source them from `~/.arbor/gateway.env`. Never commit them |
| `ARBOR_ROOT` | where `packs/` is found, when the config is not under an Arbor checkout |
| `ARBOR_ROLL=1` | turn on rolling chunks; `ARBOR_ROLL_WALL_MIN`, `ARBOR_ROLL_SUCCESSOR_AT_MIN`, `ARBOR_ROLL_EXIT_MARGIN_MIN`, `ARBOR_ROLL_MAX_CHUNKS`, `ARBOR_ROLL_EST_DEFAULT_MIN`, `ARBOR_ROLL_EST_FLOOR_MIN` override `cluster.roll` |
| `ARBOR_ROUND_AB` | the round-end A/B hook (over the config's `round_ab.command`) |
| `ARBOR_REVIEW_GATE` | `0` starts new campaigns without the review gate, `1` with it; over the config's `review_gate`, under `run --[no-]review-gate` |
| `ARBOR_SHARE_HOST`, `ARBOR_SHARE_ROOT`, `ARBOR_PROJECT`, `ARBOR_USER` | where the collectors push (`arbor-share`, `/shared/arbor`) |
| `ARBOR_AGENT_BACKEND` | leave unset. Its non-Claude value imports a module that is not in the tree |
| `SLURM_JOB_ID`, `SPUR_JOB_ID` | read to label rows and rounds |

## The session directory

`session_root` holds everything a campaign writes. A new container can resume it.

| Path | Written | Holds |
|---|---|---|
| `REVIEW.md`, `review.json` | the review gate, before anything runs | what the campaign will do and which files say so; the status (`pending`, `approved`, `rejected`), who decided, the note, and the files changed during the review (also in `session.json`'s `review`) |
| `session.json` | every iteration | the campaign's state; `lessons` has one row per verdict |
| `ledger.jsonl` | when agents emit `LEDGER:` lines | findings, one JSON record per line |
| `iter_N/` | each iteration | `bench_output.json`, `correctness_output.json`, `scope.json`, `server.log`, `effective_config.json`, KV traces; `patch.diff` on KEEP and PROBE |
| `landed.patch` | KEEP | the cumulative served tree against the workspace's first commit |
| `scratch.patch` | KEEP, PROBE | the agent's probes (`arbor/eval/`, `eval/`, logs, `scratch_paths`) |
| `landed_packages/`, `landed_packages.json` | KEEP | edits to `editable_packages` |
| `outside.json` | KEEP | links that leave the served tree |
| `workspace_config.json` | KEEP | a copy of the workspace's `arbor/config.json` |
| `directions.json` | when the map changes | the map (only the harness writes it) |
| `stages.json`, `lane_plan.md` | kernel-plan lanes | stage attempts and the pinned plan |
| `round_end.json`, `rounds/` | round ends | the A/B and the next target |
| `STOP` | the operator, or the harness | stop after the iteration in flight |
| `ABORTED.md`, `ROUND_STOP.md`, `MAP_STOP.md` | the harness | why it stopped |
| `chain/` | rolling chunks | the lease, handoffs, iteration times, `CHAIN_STOP` |
| `component.json`, `component_exit.json`, `component_refused.json`, `heartbeat.json`, `server/`, `keeps/`, `short_prompts/` | component mode | [hosts.md](hosts.md) |
| `reports/trace/` | the trace hooks | events for `evals/` |
| `collector/` | the collectors | the dashboards' status |

In the workspace: `arbor/config.json` (`server_flags` and `server_env`, the launch the agent
edits; it wins over the config's), commits named `iter-N: <change>`, and `.git/info/exclude` keeping
bytecode out.
