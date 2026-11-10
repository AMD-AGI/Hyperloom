# Reference

The command line, the campaign config, the environment variables and the session layout, as the
code reads them. How the loop uses them: [how-it-works.md](how-it-works.md). The pack format:
[packs.md](packs.md). Component mode's own keys: [hosts.md](hosts.md).

## Command line

Every command's parameters, with their types and defaults, are in [cli.md](cli.md), generated from the commands'
own parsers. Each command also takes them as one JSON or YAML document, `--payload FILE`.

| Command | What it does |
|---|---|
| `python3 -m arbor <command> --payload FILE`, `python3 -m arbor --payload FILE` | the command's parameters as a JSON or YAML document, its keys the flags' names (`max_iters`, `review_gate`); strict about keys and types; a flag on the same line wins. `command:` names the command when the line does not. Also `python3 -m arbor.directions --payload FILE` (`action: generate` or `split`). A campaign's start records it in `<session>/invocation.json` ([cli.md](cli.md)) |
| `python3 -m arbor schema [<command>] [--markdown] [--write]` | the payload's JSON Schema, from the commands' parsers; `--write` regenerates `docs/schema/arbor-cli-payload.schema.json` and [cli.md](cli.md)'s tables |
| `python3 -m arbor run --config C [--max-iters N] [--cooldown S]` | run a campaign; the same command resumes it (`--max-iters 0` = no limit; cooldown 30 s). A new campaign first waits at the review gate ([how-it-works.md](how-it-works.md#before-the-first-iteration-the-review-gate)); `--no-review-gate` starts it without one |
| `python3 -m arbor run --config C --opt-objective TEXT` | optimize another metric than the pack's: a name, a known key as written, or a key the bench writes with its direction (`"tpot_p50:min"`). `"P50 interactivity"` and `"P90 interactivity"` are an AgentX host's end-to-end axis, `e2e_norm_intvty_p50` and `_p90` (1 / the p50 or p90 over requests of E2EL in s / OSL; TTFT counts); the decode-only ones are `"P50 TPOT interactivity"`, or the keys `p50_interactivity` and `p90_interactivity` (1000 / TPOT in ms). Also `"total throughput"`, `"output throughput"`. `version --json` lists the keys and names under `objective`. The pack's second metric, when it is the one asked, changes places with the first. Recorded in `<session>/objective.json`; a resume under another objective is refused. Also `master` and `worker`. The bench must report the metric: a host's client is read for it, but standalone the pack's bench must write its key, or every iteration is NOT_MEASURED (the agentx bench writes the throughputs, and `e2e_norm_intvty_p50` and `_p90` when InferenceX's aggregate has them) |
| `python3 -m arbor run --config C --split processes [...]` | the same campaign, with the master and each worker as a command of its own, in turn ([master-worker.md](master-worker.md#the-commands)) |
| `python3 -m arbor master --config C [--max-iters N] [--cooldown S] [--no-review-gate]` | the master's side: the setup, the map, a round end; stops once a work order is ready (`ordered`) or the campaign ends (`ended`), saying which in `role_exit.json` |
| `python3 -m arbor worker --order W [--config C] [--max-iters N] [--cooldown S]` | one worker's stretch from the master's `work_order.json`, in the master's session, or in C's own; stops with its report (`reported`) |
| `python3 -m arbor.roles split --order W --shares N --out DIR` | a work order in N shares, one worker each |
| `python3 -m arbor.roles deal (--order W \| --map M) [--reports R ...] [--held ID ...] [--held-order O ...] [--size 2] [--format order\|share] --out FILE` | the next order for a free worker, from the directions no report settled and no worker holds ([master-worker.md](master-worker.md)) |
| `python3 -m arbor.ab --config C --out DIR --deadline EPOCH [--base REF] [--current REF] [--rounds 3]` | a same-node A/B of two workspace commits for any pack: the round-end hook, or an integration's A/B; writes `DIR/summary.json` ([how-it-works.md](how-it-works.md)) |
| `python3 -m arbor version [--json]`, `python3 -m arbor --version` | the version; with `--json`, each interface's version, for callers ([master-worker.md](master-worker.md#for-callers-the-version)) |
| `python3 -m arbor review --session D show\|approve\|reject [--note T] [--by NAME] [--config C]` | review a new campaign before it runs: `show` prints `REVIEW.md` and the status; `approve` checks the edited campaign and lets the run go on; `reject` ends it (`review_rejected`) |
| `python3 -m arbor restore --config C [--checkpoint REF] [--force] [--json]` | lay a checkpoint of C's session (the latest by default; REF an iteration, a name or `latest`) into C's workspace, on any node: the exact HEAD from its history, else its patches; verified first, checked byte for byte (docs/operations.md, "Checkpoints, pause and fork") |
| `python3 -m arbor pause (--config C \| --session D) [--reason TEXT] [--cancel]` | pause at the next pass boundary: the server stopped, the checkpoint `pause-NNNN`, exit 76; `run` continues |
| `python3 -m arbor checkpoint list\|show\|verify\|create\|fork ...` | a session's checkpoints; `create --name` one now; `fork --from P --config F` a new session from one (`--fork-policy exact\|exploratory`, `--directions`, `--reset`, `--prompt-profile`) |
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
| `opt_objective` | none | the metric to optimize in place of the pack's, as `--opt-objective` takes it; a host writes it here. Or the structured form, `{metric, direction, bar_pct, guards: [{metric, band_pct} or {metric, max_pct}]}`: the bar replaces `threshold_rel`, and a broken guard is a REVERT ([hosts.md](hosts.md#the-hosts-objective)). The flag and `ARBOR_OPT_OBJECTIVE` win over it |
| `time_budget_minutes` | none | a wall clock for the session (ignored in component mode). A resume counts what the session already spent (`<session>/clock.json`, written at each iteration's start), so a session that dies and resumes still ends at its budget; to give it more, raise the budget |
| `review_gate` | `true` | a new session waits for a human's approval before it runs anything; `ARBOR_REVIEW_GATE` and `run --[no-]review-gate` override it |
| `checkpoints` | `true` | a checkpoint at every pass boundary (`<session>/checkpoints/`); `false` writes none |
| `prompt_profile` | none | a directory with `claude_loop_prompt.md` and/or `critic_prompt.md` in place of the built-ins (`arbor/prompts.py`) |
| `prompt_overrides` | none | `{"main": FILE, "critic": FILE}`, each in place of the built-in; relative to `prompt_profile`, else to the config's directory |
| `baseline_record` | none | a JSON file `{mean, runs, mean_secondary}` with a measured baseline |
| `directives` | `[]` | lines rendered into the program, for the agent |
| `directions_length` | `long` | how many open directions the dossier's map aims at: `short` 4 to 6, `medium` 8 to 12, `long` 15 to 25. Part 11 and an update round's 11.uK read it from their brief or the program; it is part of the directions skill's reuse key, so a change re-runs part 11 |
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
| `session_mount` | `/workspace/sessions` | where the container sees the sessions: the container's path of `session_root`'s **parent** directory, so a session is at `<session_mount>/<session name>` |
| `scratch_paths` | `[]` | paths that hold the agent's probes, kept out of `landed.patch` (`arbor/eval`, `eval` and `*.log` always are) |
| `editable_packages` | `[]` | `[{name, path}]`: other installed packages the agent may edit. Standalone, each is a git tree from the start (a baseline in `<session>/trees.json`; bytecode and JIT outputs excluded), committed and reverted with every iteration and shown to the critic; a KEEP saves every file it changed there |

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
| `gpu_ids` | | the campaign's GPUs, physical numbers, `"0,1,2,3"` or a list. Before each server start a card that holds 10 GB or more with our server stopped is another process's, and the iteration is NOT_MEASURED, its change noted for a revisit. Under a host, only these cards are read |
| `server_host_memory_gb` | | the host memory a server pins (a CPU KV offload such as `SimpleCPUOffloadConnector`): with less `MemAvailable` before a start, the iteration is NOT_MEASURED |

**Bench and correctness**
| Key | Default | Meaning |
|---|---|---|
| `bench_script`, `correctness_script` | | absolute paths to the pack's validator scripts, as the container sees them |
| `bench_serving_script` | | an optional serving client the bench script calls |
| `bench_timeout` | 600 | seconds, for the bench and the correctness script |
| `phase_minutes` | agent 240, server 60, bench 90, critic 20 | standalone: each agent turn's cap in minutes (`{"critic": 30}`; 0 or -1 is none); the bench's is never shorter than `bench_timeout` plus 30 min, nor than the pack's `measurement.bench_minutes` plus 15. Past it the turn is cancelled and the iteration NOT MEASURED. Component mode reads `component.phase_minutes` instead |
| `concurrency`, `num_prompts`, `isl`, `osl` | 14, 640, 1024, 1024 | passed to the bench script |
| `bench_repeats` | 1 | measured replays per iteration; the pack's `measurement.bench_repeats` wins |
| `bench_metric_from_agent` | `false` | book the bench agent's `METRIC:` line when the bench wrote no `bench_output.json`. Off, such an iteration is NOT_MEASURED: the file carries the pack's failure checks and `duration_seconds`, and a missing file is usually a `session_mount` that is not the container's path of the session's parent. Under a host's bench command, the agent's number is always taken |

**Rounds, rolling and component mode**
| Key | Meaning |
|---|---|
| `round_ab.command` | the round-end A/B hook; `ARBOR_ROUND_AB` wins. `{out}` and `{deadline}` are filled in. For any pack: `python3 -m arbor.ab --config <config> --out {out} --deadline {deadline}` ([how-it-works.md](how-it-works.md)) |
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
| `ARBOR_SPLIT` | `processes` makes `arbor run` run the master and each worker as commands of their own, in turn (`--split processes`), so a host that launches `arbor run` needs no other change; `loop` (the default) runs both in one process |
| `ARBOR_OPT_OBJECTIVE` | the metric to optimize in place of the pack's (`--opt-objective` sets it for the run's processes); over the config's `opt_objective` |
| `ARBOR_REVIEW_GATE` | `0` starts new campaigns without the review gate, `1` with it; over the config's `review_gate`, under `run --[no-]review-gate` |
| `ARBOR_SHARE_HOST`, `ARBOR_SHARE_ROOT`, `ARBOR_PROJECT`, `ARBOR_USER` | where the collectors push: the shared directory's host and path, the project and the user. The root has no default: a site sets it (here or in `~/.arbor/settings.env`), and without one the collectors write locally and push nothing |
| `ARBOR_AGENT_BACKEND` | leave unset. Its non-Claude value imports a module that is not in the tree |
| `ARBOR_PROJECTS` | a file of the site's own collector projects, in `arbor/collect/projects.json`'s form, read before the public ones (models not yet public, a customer's); `~` is expanded |
| `ARBOR_SETTINGS` | a file of the site's own `ARBOR_*` values (default `~/.arbor/settings.env`): one `KEY=VALUE` per line, read when Arbor is imported, for each key the environment does not set. Arbor carries no site's values (a share root, a site's projects); a site keeps them here |
| `ARBOR_SITE` | a directory of packages for Arbor alone (`pip install --target`), for a host that cannot install into the interpreter's site-packages. Arbor puts it on its own `sys.path` when imported, after its checkout. Never put it on `PYTHONPATH`: a server the agent starts inherits Arbor's environment, and the directory's pydantic, starlette and uvicorn would shadow the image's |
| `SLURM_JOB_ID` | read to label rows and rounds |

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
| `objective.json` | every start | what the campaign optimizes: `metric`, `direction`, `unit`, `secondary_metric`, `source` (`pack` or `opt_objective`), `asked`, `pack_metric`. A host reads it to check the run optimizes its axis |
| `invocation.json` | every start | how the start was asked for: `command`, its effective `params`, `from_payload` (the keys a `--payload` gave), `argv`, `payload` (`path`, `sha256`, `keys`, or null), `arbor` (the version), `ts` |
| `kept.json` | the start, KEEP, PROBE, each iteration's start | the tree a REVERT or a NOT_MEASURED iteration goes back to: the workspace's commit and each editable tree's |
| `invalids.json` | NOT_MEASURED | each NOT_MEASURED change's commits, intent and reason, and its revisit's outcome |
| `revisit.json` | the revisit phase | the change the next iteration measures again |
| `stages.json`, `lane_plan.md` | kernel-plan lanes | stage attempts and the pinned plan |
| `round_end.json`, `rounds/` | round ends | the A/B and the next target |
| `work_order.json`, `worker_report.json`, `WORKER_REPORT.md`, `roles/` | a worker's stretch | what it starts from and what it did, each stretch's pair kept in `roles/` ([master-worker.md](master-worker.md)) |
| `role_exit.json` | `arbor master`, `arbor worker` | how the command ended: `ordered`, `reported` or `ended` |
| `STOP` | a person, or the harness | stop after the iteration in flight |
| `checkpoints/` | every pass boundary, unless `"checkpoints": false` | `NNNN.json`, the checkpoint after iteration NNNN (`arbor.checkpoint/2`: the state files, the logs' lengths, the served tree and its history, the prompts, the dossier, the fingerprint and bars, its `id`); `objects/`, each file once by its sha256; `names/`, the named ones, frozen; `LATEST`; `restores.jsonl` |
| `PAUSE`, `paused.json` | `arbor pause`; the driver at the boundary | a pause asked; the pause done (its checkpoint, the server, whether the tree was clean) |
| `fork.json`, `prompt_profile/` | `arbor checkpoint fork` | a fork's lineage (parent session, checkpoint and id, policy, resets, map, prompts' digest); its own prompt profile |
| `session.lock` | the driver (shared); `create`, `restore`, `fork` (exclusive) | who holds the session |
| `ABORTED.md`, `ROUND_STOP.md`, `MAP_STOP.md` | the harness | why it stopped |
| `chain/` | rolling chunks | the lease, handoffs, iteration times, `CHAIN_STOP` |
| `component.json`, `component_exit.json`, `component_refused.json`, `heartbeat.json`, `server/`, `keeps/`, `short_prompts/` | component mode | [hosts.md](hosts.md) |
| `reports/trace/` | the trace hooks | events for `evals/` |
| `collector/` | the collectors | the dashboards' status |

In the workspace: `arbor/config.json` (`server_flags` and `server_env`, the launch the agent
edits; it wins over the config's), commits named `iter-N: <change>`, and `.git/info/exclude` keeping
bytecode out.
