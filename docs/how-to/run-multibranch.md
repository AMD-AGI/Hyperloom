---
myst:
    html_meta:
        "description": "Run a multi-branch Hyperloom session with hyperloom-multibranch run: the environment it needs, its options grouped by what they control, and its exit codes."
        "keywords": "Hyperloom, multi-branch, MB, hyperloom-multibranch, run, RSI archive, runner cache, image check, Recipe KB"
---
# Running a multi-branch session

`hyperloom-multibranch` (also `python -m hyperloom_multibranch`) is installed with Hyperloom. `run` starts one
session: PRELUDE, EXPLORE, FINAL and CLOSE. The other subcommands are `review` (see
[Review the agent map](multibranch-review-gate.md)), `final-only SESSION_DIR` (emergency FINAL and CLOSE over a
session that ended before FINAL) and `hosted run` (a session a host injects, measured in the grant's sandboxes or, when
its spec says `placement.measurement: local`, on the app container's own cards; see
[Run hosted multi-branch on a new cluster](hosted-multibranch-new-cluster.md)).

```bash
export HYPERLOOM_RSI_ARCHIVE=/mnt/shared/hyperloom/rsi-archive
hyperloom-multibranch run --model /models/Llama-3-8B --framework vllm --gpu-type mi355x \
  --workload inferencex --tp 2 --conc 16 --isl 1024 --osl 512 --precision fp8 --max-hours 6
```

The session directory is created under `USER_DATA_PATH` and its path is printed when it starts.

## Environment

Every variable below is checked before anything is created: a missing or malformed one fails the command with no
session directory left behind. Only the image check itself runs later, in the new session before PRELUDE.

| Variable | Required | Meaning |
| --- | --- | --- |
| `HYPERLOOM_RSI_ARCHIVE` | yes | Writable shared directory the session reads its predecessors from and archives its search tree in. No default. |
| `HYPERLOOM_RUNNER_CACHE` | with `--measurement hosted` | Absolute path of this host's runner cache directory, on storage the measurement sandboxes mount. |
| `HYPERLOOM_IMAGE_CHECK` | no | `warn` (default) logs a failing fact of the rule pack about the image, `strict` fails the command before PRELUDE (exit 1), `off` checks nothing. Any other value is a usage error. |
| `INFERENCE_SUITE_TOOL_CACHE` | no | The tool cache `install.sh` staged the pinned InferenceX client in, which every test runs. Default `inference-suite-tools` under the dependency cache. Without a staging there that verifies, the command fails, naming `python -m inference_suite stage --tool-cache <dir>`. |
| `KB_STORE_URL`, `KB_STORE_TOKEN` | no | Both set: PRELUDE reads warm-start leads from the Recipe KB and CLOSE writes the final recipe back. Neither set: no Recipe KB. Only one set is refused. |
| `HYPERLOOM_AGENTX_EARLY_STOP` | no | AgentX checkpoint mode: `shadow` (default here), `enforce` or `off`. Read once when the session starts and kept in `MB-manifest.json`, so a resumed session keeps it. Any other value is a usage error. |

The short-prompt and long-context consistency checks are on by default and, when a test runs them, run on its own
server, reading a prompt set staged in this host's runner cache (`HYPERLOOM_RUNNER_CACHE`). `HYPERLOOM_SHORT_PROMPTS`
names the short-prompt set's key and `HYPERLOOM_LONG_CONTEXT` the long-context set's; `off` turns that check off.
Unset, each uses the session model's own set: the local measurement stages it when it is missing, `--measurement
hosted` runs the check only when it is already staged. A named set that is not staged refuses the measurement before
any test starts.

The AgentX checkpoint judges only explore tests of a session graded on `p50_interactivity`: each one's median
interactivity over the first 1800 s of profiling against the median its parent commit recorded, as single-lineage
does. In `shadow` the test runs to its end and its result carries the ruling (`agentx_checkpoint`: both medians, the
drop, and `trails`, whether `enforce` would have stopped it); `MB-report.md` says how many explores it judged, how many
it would have stopped and how those ended (KEEP, REGRESSED, FLAT, ...), and how many ran unjudged for want of a parent
median. In `enforce` a test that trails by more than 2% is stopped, never retried, and judged REGRESSED with the
refusal `checkpoint_trailed`; its gain covers the first 1800 s only and it records no value. Confirm, integration,
FINAL and baseline tests, correctness fixes and retractions are never judged, nor is any test under
`total_token_throughput`, where P50 is only a guard within a 5% band and nothing shows the first half hour predicts a
throughput verdict.

`HYPERLOOM_ACTIVE_IMAGE_REF` and `HYPERLOOM_ACTIVE_IMAGE_DIGEST` must name the image the session runs on, and the
agent CLI the specialists run must be installed. `--measurement hosted` also needs the grant variables its help lists
(see [Hosted run inputs](../reference/environment-variables.md#hosted-run-inputs)).

## Options of `run`

`hyperloom-multibranch run --help` has the full text of each option.

- **Workload:** `--model`, `--framework`, `--gpu-type`, `--workload` (`inferencex` or `agentx`), `--tp`, `--conc`,
  `--precision`; `--isl`/`--osl` for `inferencex` only, `--ep`, `--agentx-duration-sec` (the trace window each
  test replays, the harness's default when not given) and `--agentx-objective` for `agentx` only; `--kernel-optimizer`;
  `--inferencex-commit SHA` to resolve the session's base launch from InferenceX at that commit instead of `main`
  (see [Base launch](../reference/inference-suite.md#base-launch)).
- **AgentX objective:** `--agentx-objective p50_interactivity|total_token_throughput` (default
  `p50_interactivity`) names the round metric an AgentX session keeps on: `e2e_norm_intvty_p50`, or
  `total_token_throughput` (per server; a session's TP is fixed, so it ranks changes as the per-GPU figure does). A
  change is kept on a 2% gain on that metric in one measurement, while P90 interactivity and the other axis hold
  within the noise band (`HYPERLOOM_PERF_NOISE_PCT`, default 5%): total token throughput under `p50_interactivity`
  (refusal `total_regressed`), P50 interactivity under `total_token_throughput` (`p50_regressed`); P90 refuses as
  `p90_regressed`. Output throughput is recorded but guards neither, as it follows the trace window's output-length
  mix. A gain a guard holds back is retested once; the error-rate threshold holds under both. `total_token_throughput`
  suits a high `--conc` only: at a low one the trace's tool and think time paces the lanes, so a server gain barely
  moves it (the same stack on GLM-5.3 TP4/EP4 gained +1.8% total per GPU at conc 4 and +7–14% at conc 8–16). The
  bar and guards are read from the code, so a resumed session judges by the current rule. The manifest records the
  objective, so a resumed session keeps it; the frontier, FINAL, the dashboard, `MB-report.md` and the Recipe KB record
  (`total_throughput_gain_pct`) all read that metric. Each Arbor is told to optimize it (`opt_objective`); a
  P50 session needs Arbor `9c455df` or later, a `total_token_throughput` one `df9b55a` or later.
- **Base launch:** `--reference-script FILE`, a shell launch script or a base launch record frozen elsewhere, as for
  `hyperloom optimize`; the session freezes it as its base, a record without reading any recipe source, and every
  Arbor is served the frozen base. Without it the base comes from the recipe sources.
- **Limits:** `--max-hours` (default 2, 0 for none), `--max-gpu-hours` and `--max-llm-usd` (0 for none),
  `--max-branches` (default 4). A negative or non-finite limit is a usage error.
- **Concurrency:** `--max-parallel-tests` (default: as many as fit at TP), `--max-k-per-branch` (default 2),
  `--dev-leases`, `--judge-interval-sec` (default 600, must be positive), `--measurement local|hosted`.
- **Directions:** `--specialist-prompt FILE` and `--reference NAME=DIR`, each repeatable.
- **Third-party agents:** `--autonomous-agent SOURCE` (repeatable; a URL carrying credentials is refused),
  `--agent-directions`, `--agent-dossier-minutes`, `--agent-share-minutes`, `--agent-pack`, `--agent-launch`,
  `--agent-port`, `--stop-after-map`, `--agent-dossier`, `--no-review-gate`.
  Each agent measures with the gate's own client: on `agentx`, the gate's replay run by
  `python -m inference_suite run` against the agent's server, so an Arbor needs no bench variables in its launch.
- **Scheduling playbook:** `--playbook auto|champion|challenger` (default `auto`).

## Exit codes

0 when the session ran its course, a spent budget included; 1 when it failed or was stopped, or the environment kept
it from starting; 2 on a usage error.
