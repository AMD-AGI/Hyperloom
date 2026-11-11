---
myst:
    html_meta:
        "description": "Run a multi-branch Hyperloom session with hyperloom-multibranch run: the environment it needs, its options grouped by what they control, and its exit codes."
        "keywords": "Hyperloom, multi-branch, MB, hyperloom-multibranch, run, RSI archive, runner cache, image check, Recipe KB"
---
# Running a multi-branch session

`hyperloom-multibranch` (also `python -m hyperloom_multibranch`) is installed with Hyperloom. `run` starts one
session: PRELUDE, EXPLORE, FINAL and CLOSE; `run --resume SESSION_DIR` continues one that stopped before FINAL (see
[Resuming a stopped session](#resuming-a-stopped-session)). The other subcommands are `review` (see
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

Before a test that carries patches is given a sandbox or cards, its toolbox byte-compiles every Python file the patch
stack changes and imports each patched component on CPU, with no GPU visible (the merge smoke): a file that does not
compile, or an import error, missing name or missing attribute raised from a patched file, fails the test as the
candidate's (`merge_smoke`); any other import failure (a device probe that finds no GPU, say) is not judged and the test
is measured. A smoke that cannot run is an infrastructure failure (`smoke_not_run`), retried once. `HYPERLOOM_MERGE_SMOKE=off` in the session's environment turns it off; any
value but `on`, `off` or empty is refused when the toolbox is built.

## Options of `run`

`hyperloom-multibranch run --help` has the full text of each option.

- **Workload:** `--model`, `--framework`, `--gpu-type`, `--workload` (`inferencex` or `agentx`), `--tp`, `--ep`
  (expert-parallel size, default 1), `--conc`, `--precision`; `--isl`/`--osl` for `inferencex` only;
  `--kernel-optimizer`.
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

## Resuming a stopped session

`hyperloom-multibranch run --resume SESSION_DIR` continues a session that stopped before FINAL (signal, kill, crash,
lost machine) with EXPLORE, FINAL and CLOSE, on the settings its manifest holds. It runs no PRELUDE. The baseline,
the profile and the root commit are the ones the first PRELUDE recorded.

- **What it continues:** the earlier run's unfinished tests are withdrawn and the branches it left open are stopped.
  The roster and the integration branch open again (a `reopen` lineage event), and EXPLORE opens new branches, new
  ids and worktrees, on the integration head. Specialist and Arbor branches alike start fresh.
- **Deadline:** the session's deadline moves by its downtime, the time between the last decision the earlier run
  recorded and the resume. The new deadline is written to the manifest and recorded as a `resume` phase in
  `MB-decisions.jsonl` with `ended`, `downtime_seconds` and `deadline`.
- **Knowledge steps:** a CLOSE of a session whose search tree the RSI archive already holds runs none of the knowledge
  steps: no RSI archive, playbook proposal or lesson curation. Each is reported `skipped` with the reason
  `an earlier CLOSE of this session archived its search tree`. This holds for every later CLOSE of that session, a
  second `--resume` or a `final-only` included, and also when the earlier CLOSE's exports failed after it archived.
- **Refusals (exit 2, nothing changed):** `SESSION_DIR` is not a directory or holds no session that ran; the session
  ran on another build, is still running, already chose in FINAL, failed in PRELUDE or recorded no baseline, was
  measured in a grant's sandboxes (`--measurement hosted`), or was to stop once its map was made; or any other `run`
  option is given beside `--resume`.

A session that cannot be resumed but stopped before FINAL can still be finished with `final-only SESSION_DIR`.

## Exit codes

0 when the session ran its course, a spent budget included; 1 when it failed or was stopped, or the environment kept
it from starting; 2 on a usage error.
