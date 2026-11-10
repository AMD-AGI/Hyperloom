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
session that ended before FINAL) and `hosted run` (a session a host injects; see
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
| `KB_STORE_URL`, `KB_STORE_TOKEN` | no | Both set: PRELUDE reads warm-start leads from the Recipe KB and CLOSE writes the final recipe back. Neither set: no Recipe KB. Only one set is refused. |

`HYPERLOOM_ACTIVE_IMAGE_REF` and `HYPERLOOM_ACTIVE_IMAGE_DIGEST` must name the image the session runs on, and the
agent CLI the specialists run must be installed. `--measurement hosted` also needs the grant variables its help lists
(see [Hosted run inputs](../reference/environment-variables.md#hosted-run-inputs)).

## Options of `run`

`hyperloom-multibranch run --help` has the full text of each option.

- **Workload:** `--model`, `--framework`, `--gpu-type`, `--workload` (`inferencex` or `agentx`), `--tp`, `--conc`,
  `--precision`; `--isl`/`--osl` for `inferencex` only, `--ep` for `agentx` only; `--kernel-optimizer`.
- **Limits:** `--max-hours` (default 2, 0 for none), `--max-gpu-hours` and `--max-llm-usd` (0 for none),
  `--max-branches` (default 4). A negative or non-finite limit is a usage error.
- **Concurrency:** `--max-parallel-tests` (default: as many as fit at TP), `--max-k-per-branch` (default 2),
  `--dev-leases`, `--judge-interval-sec` (default 600, must be positive), `--measurement local|hosted`,
  `--[no-]make-before-break` (hosted only, default on: where the grant fits at least three tests at TP, a finished
  test's sandbox is kept until the next test's runs, so that the cluster does not give its node to another job, and one
  test fewer runs at once; off runs every test the grant fits).
- **Directions:** `--specialist-prompt FILE` and `--reference NAME=DIR`, each repeatable.
- **Third-party agents:** `--autonomous-agent SOURCE` (repeatable; a URL carrying credentials is refused),
  `--agent-directions`, `--agent-dossier-minutes`, `--agent-share-minutes`, `--agent-pack`, `--agent-launch`,
  `--agent-port`, `--stop-after-map`, `--agent-dossier`, `--no-review-gate`.
- **Scheduling playbook:** `--playbook auto|champion|challenger` (default `auto`).

## Exit codes

0 when the session ran its course, a spent budget included; 1 when it failed or was stopped, or the environment kept
it from starting; 2 on a usage error.
