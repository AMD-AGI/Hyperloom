---
myst:
    html_meta:
        "description": "Run a multi-branch Hyperloom session with hyperloom-multibranch run: the environment it needs, its options grouped by what they control, and its exit codes."
        "keywords": "Hyperloom, multi-branch, MB, hyperloom-multibranch, run, RSI archive, runner cache, Recipe KB"
---
# Running a multi-branch session

`hyperloom-multibranch` (also `python -m hyperloom_multibranch`) is installed with Hyperloom. `run` starts one
session: PRELUDE, EXPLORE, FINAL and CLOSE; `run --resume SESSION_DIR` continues one that stopped before FINAL (see
[Resuming a stopped session](#resuming-a-stopped-session)). The other subcommands are `review` (see
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
session directory left behind. One step runs later, in the new session before PRELUDE: on a local session the publish
of the inference_suite staging into the runner cache, which fails the command (exit 1) with the session directory left
behind when that directory cannot be written.

| Variable | Required | Meaning |
| --- | --- | --- |
| `HYPERLOOM_RSI_ARCHIVE` | yes | Writable shared directory the session reads its predecessors from and archives its search tree in. No default. |
| `HYPERLOOM_RUNNER_CACHE` | with `--measurement hosted` | This host's runner cache directory, which holds the inference_suite staging every test runs. Hosted: an absolute path on storage the measurement sandboxes mount, no default. Local: `/runner-cache` when unset, and it must be writable. |
| `INFERENCE_SUITE_TOOL_CACHE` | no | The tool cache `install.sh` staged the pinned InferenceX client in, which every test runs. Default `inference-suite-tools` under the dependency cache. Without a staging there that verifies, the command fails, naming `python -m inference_suite stage --tool-cache <dir>`. |
| `KB_STORE_URL`, `KB_STORE_TOKEN` | no | Both set: PRELUDE reads warm-start leads from the Recipe KB and CLOSE writes the final recipe back. Neither set: no Recipe KB. Only one set is refused. |
| `HYPERLOOM_KB_URL`, `HYPERLOOM_KB_TOKEN` | no | The Experience KB: the workspace's loopback service, whose URL and token `hyperloom-setup` writes into the workspace `.env`. Export them as the other variables here: `hyperloom-multibranch` reads no `.env`. Set: the briefs and the learnings tool recall past sessions' tests and learnings from it, and CLOSE writes this session's there; what the service does not take waits in the spool beside its data. A loopback service that is not serving is started before the session and left serving after it; one serving with other settings is not restarted. A service that cannot start, or a URL without its token, logs a warning and the session runs without it. Unset: no Experience KB. |
| `HYPERLOOM_AGENTX_EARLY_STOP` | no | AgentX checkpoint mode: `shadow` (default here), `enforce` or `off`. Read once when the session starts and kept in `MB-manifest.json`, so a resumed session keeps it. Any other value is a usage error. |
| `GITHUB_TOKEN` | no | Not checked when the session starts. The GitHub token this process asks GitHub with when a specialist searches an upstream repository's pull requests. `GH_TOKEN` is read when it is unset. Without either the session runs, logs one WARNING when its toolbox is built and asks GitHub anonymously, which GitHub refuses once the address's small quota is spent. See [Upstream pull-request search](#upstream-pull-request-search). |

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
refusal `checkpoint_trailed`; its gain covers the first 1800 s only and it records no value. The stop is charged to the candidate before any log
is read, so a node that slows down within those 1800 s, or a parent median taken on another node, can make a sound
candidate trail with no retry; that is why `shadow` is the default here. Confirm, integration,
FINAL and baseline tests, correctness fixes and retractions are never judged, nor is any test under
`total_token_throughput`, where P50 is only a guard within a 5% band and nothing shows the first half hour predicts a
throughput verdict.

`HYPERLOOM_ACTIVE_IMAGE_REF` and `HYPERLOOM_ACTIVE_IMAGE_DIGEST` must name the image the session runs on, and the
agent CLI the specialists run must be installed: on `PATH`, or, for `claude`, in `~/.local/bin`, where `install.sh`
puts it (only a shell that sourced the kernel-agent env file has that on `PATH`). `--measurement hosted` also needs the grant variables its help lists
(see [Hosted run inputs](../reference/environment-variables.md#hosted-run-inputs)).

Before a test that carries patches is given a sandbox or cards, its toolbox byte-compiles every Python file the patch
stack changes and imports each patched component on CPU, with no GPU visible (the merge smoke). A file that does not
compile, or an import that fails on a name or attribute a patched module lacks (one patch removing a name another
patched file imports), fails the test as the candidate's (`merge_smoke`), whatever the reason the name is missing: a
patched module that defines a name only under a device check is judged too, since such a stack fails on any other
device as well. Every other import failure (a device probe that finds no GPU, a name an unpatched module defines only
where a GPU is seen, a missing module, a `NameError`) is logged and not judged, and the test is measured. A smoke that cannot run is an infrastructure failure
(`smoke_not_run`), retried once. `HYPERLOOM_MERGE_SMOKE=off` in the session's environment turns it off; any
value but `on`, `off` or empty is refused when the toolbox is built.

## Upstream pull-request search

A specialist of a session measured on this machine's cards (`--measurement local`) can search the pull requests of
its component's upstream repository (`upstream_pr_search`), to learn of a change upstream already has before it
writes one. The repositories are the ones the rule pack names: sgl-project/sglang,
vllm-project/vllm, ROCm/aiter, pytorch/pytorch and ggml-org/llama.cpp. The session's own process asks GitHub, with
the token in its environment.

The token is not in the environment of a specialist's CLI or of a measurement, and no result or error of the search
and no line the session logs about it carries it. A dev sandbox or an agent sandbox that runs without bubblewrap
starts from the session's environment, this token included. Where the session log says at its start that agents run
unconfined on this host, an agent runs as this account and can read what the account can. That is why the token must
be the one of step 1.

To give the session a token:

1. Create a token that can read public repositories and nothing else: a fine-grained personal access token with
   repository access "Public repositories" (read-only) and no permission added, or a classic token with no scope
   selected. Give it an expiry. Do not use the GitHub CLI's own token (`gh auth token`) or any token that reads a
   private repository.
2. Put it in the workspace `.env`, which git ignores, as `GITHUB_TOKEN=<token>`, on one line. Never commit it, and
   never write it into a command: a command line is readable by every account of the machine.
3. `hyperloom-multibranch` reads its process environment, not `.env`. Load `.env` into the shell that starts the
   session:

   ```bash
   export REPO_ROOT="${REPO_ROOT:-$(pwd -P)}"
   INSTALL_SH="$REPO_ROOT/hyperloom/inference_optimizer/assets/install.sh"
   [ -f "$INSTALL_SH" ] || INSTALL_SH="$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh"
   . "${INSTALL_SH%/*}/runtime_env.sh"
   load_dotenv_no_clobber
   ```

   A `.env` another account can write lends no token, and the loader names the variable it refused. Read the token
   into the shell there, without echo: `read -rs GITHUB_TOKEN && export GITHUB_TOKEN`.
4. Check it before a long run, in the shell that will start the session. This prints `5000` with a token GitHub
   accepts and `60` with none, and ends in `HTTP Error 401` for one GitHub rejects; the token is in no command line
   and is not printed:

   ```bash
   python3 - <<'EOF'
   import json, os, urllib.request
   token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
   ask = urllib.request.Request(
       "https://api.github.com/rate_limit", headers={"Authorization": f"Bearer {token}"} if token else {}
   )
   print(json.load(urllib.request.urlopen(ask))["resources"]["core"]["limit"])
   EOF
   ```

   The session's log then says `upstream PR search asks GitHub with this host's token` when its toolbox is built.

With a token GitHub allows 30 searches a minute, for all the tokens of one account together; a search is one
request. A search looks for the agent's keywords, words or phrases, of which only the first six are used, and finds
the pull requests whose title, body or comments hold any one of them, open ones unless the agent names other states;
it does not look at the paths a pull request changes.
The search keeps up to `limit` candidates (10 by default, 20 at most), each with its number, title and URL, in the
job's result record, with how many pull requests matched in all (`matched_prs`) and how many of them the limit left
out (`not_shown`). The specialist is shown the job's summary: how many candidates were kept, of how many that matched
when the limit left some out ("10 of 266"), and the number and title of the first. A GitHub that cannot be reached,
a rejected token, a request the token is not allowed and a spent rate limit each fail the request with that reason,
never with an empty list; a spent rate limit says how long to wait, and a request the token is not allowed says that
asking again will not help.

Without a token the requests are anonymous: GitHub allows an address 10 searches a minute, which a shared
address has usually spent. A refused request then fails saying how long to wait and that the
host has no token. The session logs one WARNING when its toolbox is built. Everything else runs as before.

## Options of `run`

`hyperloom-multibranch run --help` has the full text of each option.

- **Workload:** `--model`, `--framework`, `--gpu-type`, `--workload` (`inferencex` or `agentx`), `--tp`, `--ep`
  (expert-parallel size, default 1), `--conc`, `--precision`; `--isl`/`--osl` for `inferencex` only,
  `--agentx-duration-sec` (the trace window each test replays, the harness's default when not given) and
  `--agentx-objective` for `agentx` only; `--kernel-optimizer`;
  `--boot-budget-sec` (see **Boot budget** below);
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
  objective, so a resumed session keeps it; the frontier, FINAL, the dashboard and `MB-report.md` all read that metric. A
  `total_token_throughput` session writes no Recipe KB record: the KB partitions by workload, not objective.
- **Boot budget:** `--boot-budget-sec SECONDS` (a positive whole number, at most 21600, 6 h) is how long, at least, the
  server of a test may take to be ready, for a model whose checkpoint takes long to load. Without it, a test's timeout
  is 3600 s for `inferencex` and the trace window plus 3600 s for `agentx`, and a boot may take half of it. A budget
  only lengthens both: a test runs for its trace window (`agentx`) plus 1800 s for the warmup, the measured rounds, the
  accuracy evaluation and the drain, plus the budget, when that is longer than its default timeout; and the boot limit
  (`server_boot_timeout`) is the budget when that is longer than half of the timeout. A budget of 1800 s or less
  changes nothing. The server's own start timeout follows the budget: every round sets the framework's ready timeout
  (`VLLM_ENGINE_READY_TIMEOUT_S`, `SGLANG_WARMUP_TIMEOUT`) to 10800 s, or to the budget plus 3600 s when that is
  longer, so the server never gives up on a start the budget still allows.
- **Base launch:** `--reference-script FILE`, a shell launch script or a base launch record frozen elsewhere, as for
  `hyperloom optimize`; the session freezes it as its base, a record without reading any recipe source. Without it
  the base comes from the recipe sources. Every component agent is served the frozen base unless `--agent-launch`
  gives a launch file.
- **Limits:** `--max-hours` (default 2, 0 for none), `--max-gpu-hours` and `--max-llm-usd` (0 for none),
  `--max-branches` (default 4). A negative or non-finite limit is a usage error.
- **Concurrency:** `--max-parallel-tests` (default: as many as fit at TP), `--max-k-per-branch` (default 2; a
  specialist's branch only, the integration branch's K is bounded by `--max-parallel-tests`),
  `--dev-leases`, `--judge-interval-sec` (default 600, must be positive), `--measurement local|hosted`,
  `--[no-]make-before-break` (hosted only, default on: where the grant fits at least three tests at TP, a finished
  test's sandbox is kept until the next test's runs, so that the cluster does not give its node to another job, and one
  test fewer runs at once; off runs every test the grant fits).
- **Concurrency sweep:** `--[no-]enable-conc-sweep` (on by default) and `--conc-sweep-concs C1,C2,...` (see
  [Concurrency sweep after CLOSE](#concurrency-sweep-after-close)).
- **Directions:** `--specialist-prompt FILE` and `--reference NAME=DIR`, each repeatable.
- **Third-party agents:** `--autonomous-agent SOURCE` (repeatable; a URL carrying credentials is refused),
  `--agent-directions`, `--agent-dossier-minutes`, `--agent-share-minutes`, `--agent-pack`, `--agent-launch`,
  `--agent-port`, `--stop-after-map`, `--agent-dossier`, `--no-review-gate`.
  Each agent measures with the gate's own client: on `agentx`, the gate's replay run by
  `python -m inference_suite run` against the agent's server, so an agent needs no bench variables in its launch.
  A component agent is a source whose checkout an adapter recognises. An adapter is a package of its own, installed
  beside Hyperloom and declared under the entry-point group `hyperloom_multibranch.component_agents`; Hyperloom ships
  none. With none installed, every source is onboarded by a model, and the map, pack, launch and dossier options have
  no agent to act on.
- **Scheduling playbook:** `--playbook auto|champion|challenger` (default `auto`).

## Resuming a stopped session

`hyperloom-multibranch run --resume SESSION_DIR` continues a session that stopped before FINAL (signal, kill, crash,
lost machine) with EXPLORE, FINAL and CLOSE, on the settings its manifest holds. It runs no PRELUDE. The baseline,
the profile and the root commit are the ones the first PRELUDE recorded.

- **What it continues:** the earlier run's unfinished tests are withdrawn and the branches it left open are stopped.
  The roster and the integration branch open again (a `reopen` lineage event), and EXPLORE opens new branches, new
  ids and worktrees, on the integration head. Specialist and agent branches alike start fresh.
- **Claw session:** a session started with `CLAW_SESSION_ID` records it as the top-level `claw_session_id` of
  `MB-manifest.json` (`null` without it), so a collector of running sessions can tell whose a directory is before
  CLOSE writes `session_breakdown.json`. The id follows the leg that runs: a `run --resume` or a `final-only` given
  `CLAW_SESSION_ID` writes that id to the manifest and the breakdown, as the LLM attribution header carries it, and
  logs the old and the new id when it changes; a leg given none keeps the recorded id. The manifest is the one record
  of the id: the toolbox's `state.json` and `retest_view.json` do not carry it.
- **Deadline:** the session's deadline moves by its downtime, the time between the last decision the earlier run
  recorded and the resume. The new deadline is written to the manifest and recorded as a `resume` phase in
  `MB-decisions.jsonl` with `ended`, `downtime_seconds` and `deadline`. GPU hours are not given back: the session
  reads what it spent, before the stop included, from its blackboard.
- **Hosted:** a session measured in a grant's sandboxes is resumed by a restart of its hosted run, not by
  `run --resume`. Its budget counts from the original start, for `--max-hours` as for `--max-gpu-hours`: the deadline
  stays the one the first start set and the `resume` phase records it with the downtime. Such a resume is refused when
  the session recorded no baseline, was to stop once its map was made, already spent the GPU hours EXPLORE has under
  `--max-gpu-hours`, or is past that deadline; the restart then finishes it with `final-only`. It is refused too,
  with one warning naming the field, when the baseline was measured on another build, `MEASUREMENT_EPOCH`, image,
  base launch or runner template card count than the restart would measure on, as recorded in
  `MB-toolbox/baseline-environment.json` when the first baseline test was accepted. The application's node and account
  are not compared: a reschedule changes them by design, and the tests run in the grant. The restart ends the
  measurement sandboxes the earlier start left running in the grant before it measures anything, and EXPLORE ends by
  the grant's cap as in a started session. A session file this account may not read finishes the session too.
- **Knowledge steps:** a CLOSE of a session whose search tree the RSI archive already holds runs none of the knowledge
  steps: no RSI archive, playbook proposal or lesson curation. Each is reported `skipped` with the reason
  `an earlier CLOSE of this session archived its search tree`. This holds for every later CLOSE of that session, a
  second `--resume` or a `final-only` included, and also when the earlier CLOSE's exports failed after it archived.
- **Refusals (exit 2, nothing changed):** `SESSION_DIR` is not a directory or holds no session that ran; the session
  ran on another build, is still running, already chose in FINAL, failed in PRELUDE or recorded no baseline, was
  measured in a grant's sandboxes (`--measurement hosted`), was to stop once its map was made, or already spent the GPU
  hours EXPLORE has under `--max-gpu-hours` (all of it less FINAL's share), which would send it straight to FINAL; or
  any other `run` option is given beside `--resume`.

A session that cannot be resumed but stopped before FINAL can still be finished with `final-only SESSION_DIR`.

## Concurrency sweep after CLOSE

`run` (on by default; `--no-enable-conc-sweep`, hosted `multibranch.conc_sweep: false`, turns it off) sweeps the closed session's root and the commit
FINAL chose once CLOSE has run, apart from `--max-hours` and `--max-gpu-hours`; `conc-sweep SESSION_DIR` runs the same
sweep by hand, measured on this machine. It runs only when FINAL chose a commit other than the root whose reproduction
gained over the baseline. The points are every concurrency the InferenceX recipe of the frozen base publishes for the
workload, with `--conc` among them; a `--reference-script` base takes them from `--conc-sweep-concs C1,C2,...`
(hosted: `multibranch.conc_sweep_concs`). Both arms are submitted at once and run side by side where the session runs
two tests at once (`--max-parallel-tests`), else one after the other. An AgentX point is a test of its own; an
InferenceX arm is one sweep test over all its points, whose timeout scales with them. The tests are session
measurements of the commits as they are: no accuracy and no checks.

The sweep logs its estimated length (each point priced as the session's longest root test) and, hosted, what is
left of the grant. It writes `MB-reports/conc_sweep_summary.json`, `conc_sweep_raw.csv` and `conc_sweep_curve.png`,
the summary after every test, with `status` `succeeded`, `partial`, `failed`, `skipped` or `cancelled` (a stop
signal; the measured points stay), and carries the summary each time as `conc_sweep_summary` in
`MB-reports/session_breakdown.json`. A later run measures only the points not yet measured, and a hosted restart of a
session that closed before its sweep finished sweeps it again before it writes `exit.json`.

## Exit codes

0 when the session ran its course, a spent budget included; 1 when it failed or was stopped, or the environment kept
it from starting; 2 on a usage error.
