# Quickstart: a campaign as a master and its workers

From an empty machine to a campaign whose master plans and judges, and whose workers make the changes. What each side
does, and why, is in [master-worker.md](master-worker.md); every config key is in [reference.md](reference.md).

| Step | Command | Ends with |
| --- | --- | --- |
| Check the config, no GPU | `python3 -m arbor run --config <config.json> --dry-run` | a summary of verdicts on synthetic numbers |
| Start the master | `python3 -m arbor master --config <config.json>` | `ordered`: the session's `work_order.json` |
| Run a worker | `python3 -m arbor worker --order <session>/work_order.json` | `reported`: the session's `worker_report.json` |
| Master and worker in turn, to the end | `python3 -m arbor run --config <config.json> --split processes` | the campaign's end |
| Several workers at once, on one node | `python3 -m arbor run --config <config.json> --split workers` | the campaign's end |

Names in angle brackets are yours to fill in: `<config.json>` is the path of your campaign's config (section 2), and
`<session>` is its `session_root`.

## 1. The environment

### 1.1 The machine

- **Linux, Python 3.11 or newer, and `git`.** WSL is enough for a dry run.
- **On the GPU node:** `docker`, AMD Instinct GPUs with ROCm (`amd-smi` or `rocm-smi`), and `curl`. On a cluster, its
  scheduler.
- **Where things run:** the driver runs on the host. The server, the bench and the correctness check run in a container
  that the driver reaches with `docker exec`.

### 1.2 Arbor

```bash
git clone <this repository's URL> Arbor && cd Arbor
```

- **A cluster that cannot reach GitHub** clones from a mirror it can read.
- **Put the checkout on shared, writable storage.** Arbor writes the dossier into `docs/dossier/<name>/` and the
  benchmark study into `benchmarks/<name>/` of this checkout, and later passes read them back.
- **The container must see the pack's validators** at the paths the config gives. Mounting the checkout into the
  container at the same path is the simplest way (1.5).

### 1.3 Python packages

```bash
python3 -m venv ~/venvs/arbor && . ~/venvs/arbor/bin/activate
python3 -m pip install pyyaml claude-agent-sdk
python3 -c "from claude_agent_sdk._cli_version import __cli_version__ as v; print(v)"
```

- **The SDK brings its own Claude CLI** and uses it, not one on `PATH`.
- **That CLI must know the models the agents use.** SDK 0.2.174 (CLI 2.1.310) is known good. SDK 0.2.153 (CLI 2.1.273)
  is too old, and its turns fail with `unrecognized_model`.
- **Arbor has no requirements file yet;** these two packages are all the driver needs.

### 1.4 Credentials for the agents

The agents reach Claude through your LLM gateway. Keep its variables in a file only you can read:

```bash
mkdir -p ~/.arbor && touch ~/.arbor/gateway.env && chmod 600 ~/.arbor/gateway.env
# edit it: ANTHROPIC_BASE_URL=<the gateway's URL>, and the token variable the gateway asks for
set -a; . ~/.arbor/gateway.env; set +a
```

- **Source it in the shell that starts Arbor.** The cluster launchers do the same.
- **A gateway that does not answer:** Arbor then tries `localhost:9101`, `9100` and `9102` at import.
- **Never commit the file.** Rotate a token that shows up in a log or a transcript.

### 1.5 The serving container

Arbor drives a container that runs the serving framework (vLLM or SGLang), sees the weights, the pack's validators and
the session directory, and stays up for the whole campaign. Arbor execs into it and never recreates it.

An example for vLLM on ROCm, from a setup that has run this end to end (adapt the image, paths and names):

```bash
IMAGE=docker.io/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
docker run -d --name arbor-main --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt seccomp=unconfined --ipc=host --shm-size 64g --ulimit nofile=1048576:1048576 --init \
  -v /work:/work -v /models:/models:ro \
  --entrypoint tail "$IMAGE" -f /dev/null
```

- **`/work`** holds the Arbor checkout and the sessions, mounted at the same path inside and out. Then a path in
  the config means the same file on both sides.
- **On NFS with root squash,** run the container as your user (`--user "$(id -u):$(id -g)"`, with a writable `HOME`),
  so the files written inside it are yours and writable.
- **What the template pack's checks need** (`serving-m10`): its correctness check runs GSM8K through `lm_eval`'s API
  model, which needs the `[api]` extra. Without it every check fails with `ModuleNotFoundError: tenacity`:

  ```bash
  docker exec -u 0 arbor-main python3 -m pip install "lm_eval[api]"
  ```

### 1.6 The workspace: the served tree as a git repository

The agent edits the framework's installed source. Arbor commits each change (`iter-N: <change>`) and reverts with git,
so the tree the server imports must be a git repository whose first commit is the image's code:

```bash
WS=$(docker exec arbor-main python3 -c 'import os, vllm; print(os.path.dirname(os.path.abspath(vllm.__file__)))')
docker exec arbor-main bash -lc "cd $WS && git init -q && git config user.email arbor@local && git config user.name Arbor \
  && git config core.fileMode false && git add -A && git commit -q -m 'base: the image vLLM'"
echo "$WS"    # this is workspace_root, e.g. /usr/local/lib/python3.12/dist-packages/vllm
```

- **`landed.patch`,** the campaign's result, is every kept change against this first commit.
- **Run as your user?** First `chown` the tree to your user, as root in the container.

### 1.7 The session directory, and what the container sees of it

- **`session_root`** is where the campaign writes everything. Put it on storage that outlives the job.
- **The bench writes its result from inside the container,** to `<session_mount>/<the session folder's name>/iter_N/`.
  `session_mount` is where the container sees the session folder's parent directory (default `/workspace/sessions`).
- **Example:** `session_root` is `/work/runs/demo/session`, and `/work` is mounted at the same path. Then
  `"session_mount": "/work/runs/demo"`.
- **Get it wrong** and the bench writes into the container's own layer. Arbor then falls back to the number the bench
  agent reports (`METRIC:`), and loses the rest of the result: the second metric and the aggregate file.

### 1.8 The GPUs

- **`gpu_ids`** names the campaign's cards by their physical numbers. Arbor sets each server's visibility from it.
- **Unset `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`** in the shell that starts Arbor, as
  the launchers do.
- **Start on clean cards** (`amd-smi process`). Before each server start, a card that still holds 10 GB or more with
  our server stopped belongs to another process, and the iteration is NOT MEASURED.

## 2. What to provide: the config

A JSON file. Start from `configs/serving-template.json`. This one is from a run of the setup above, with its paths
generalized:

```json
{
  "pack": "serving-m10",
  "identity": "qwen3-8b vllm tp1 c64 mi350x",
  "goal": "Raise Qwen3-8B's output throughput at C64 (random 1024 in, 1024 out) on one MI350X, without losing GSM8K accuracy.",
  "mode": "standalone",
  "docker_container": "arbor-main",
  "workspace_root": "/usr/local/lib/python3.12/dist-packages/vllm",
  "session_root": "/work/runs/demo/session",
  "session_mount": "/work/runs/demo",
  "framework": "vllm",
  "model_path": "/models/Qwen3-8B",
  "model_name": "/models/Qwen3-8B",
  "tp": 1,
  "port": 8006,
  "gpu_ids": "0",
  "server_flags": "--gpu-memory-utilization 0.95 --max-model-len 8192 --trust-remote-code",
  "server_env": {"VLLM_ROCM_USE_AITER": "1", "ISL": "1024", "OSL": "1024", "NUM_GPUS": "1",
                 "GSM8K_LIMIT": "400", "GSM8K_MIN": "<the model's floor>"},
  "concurrency": 64,
  "num_prompts": 320,
  "bench_timeout": 1800,
  "bench_script": "/work/Arbor/packs/serving-m10/validator/bench.sh",
  "correctness_script": "/work/Arbor/packs/serving-m10/validator/correctness.sh",
  "time_budget_minutes": 600,
  "dossier": "qwen3-8b-vllm-mi350x"
}
```

| What | Keys | Notes |
| --- | --- | --- |
| The pack | `pack` | what is optimized, the bar, the rules and the skills it loads ([packs.md](packs.md)). `serving-m10` is the serving template: the benchmark study, the six dossier skills, the profiler and `worker-summary` |
| The campaign | `identity`, `goal`, `target`, `time_budget_minutes` | `goal` heads the agent's program. Reaching `target` ends a round |
| The container and tree | `docker_container`, `workspace_root` | 1.5 and 1.6 |
| The session | `session_root`, `session_mount` | 1.7 |
| The server | `framework`, `model_path`, `model_name`, `tp`, `port`, `gpu_ids`, `server_flags`, `server_env` | the first launch. The agent then edits flags and env in the workspace's `arbor/config.json` |
| The operating point | `concurrency`, `num_prompts`; `ISL`, `OSL` in `server_env` | what the bench sends |
| The instruments | `bench_script`, `correctness_script`, `bench_timeout` | paths as the container sees them. Set `GSM8K_MIN` for the model: unset, only a sanity floor of 0.30 applies |
| Where knowledge goes | `dossier`, `benchmark` | `docs/dossier/<dossier>/` (default: the identity as a slug) and `benchmarks/<benchmark>/` (default: the pack's name) |
| The map's size | `directions_length` | how many open directions the map aims at: `short` 4 to 6, `medium` 8 to 12, `long` 15 to 25 (the default). A short map ends rounds sooner |
| Several workers | `workers` | section 5.1 |

**Check it before spending a node:**

```bash
python3 -m arbor run --config <config.json> --dry-run
```

The dry run loads the pack and its skills, and drives the real loop on synthetic numbers in a temporary repository:
no GPU, container or agent.

## 3. Start the master

```bash
cd /work/Arbor
set -a; . ~/.arbor/gateway.env; set +a
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
python3 -m arbor master --config <config.json>
```

1. **The review gate.** A new session stops before it runs anything, and prints where its `REVIEW.md` is: what it will
   optimize, the bar, the rules and the program the agent will read. Edit the pack or the config if needed, then, from
   another shell:

   ```bash
   python3 -m arbor review --session /work/runs/demo/session approve --note "reviewed"
   ```

   An unattended launch passes `--no-review-gate`.
2. **The setup:** `init` pins the validators, seeds the workspace and measures the baseline; `benchmark` studies the
   benchmark; `accuracy` measures the root's accuracy once.
3. **The map:** `map.discover`, `map.measure` and `map.directions` write the dossier, and end in the map, a ranked
   list of directions (`<session>/directions.json`).
4. **It stops** once the next pass is a worker's. It writes `<session>/work_order.json` and `<session>/role_exit.json`:

   ```json
   {"side": "master", "outcome": "ordered", ...}
   ```

   It says `ended` instead when the campaign ends: the time budget, `STOP`, the target without a round-end A/B, or an
   abort.

- **Keep it alive past your shell.** A master runs for hours: run it in a batch job, `tmux`, or a user unit.
- **Run it again, and it runs nothing twice.** A master whose order still waits for its worker hands the same order
  over.

## 4. Run a worker

```bash
python3 -m arbor worker --order /work/runs/demo/session/work_order.json
```

- **It works in the master's session,** on the order's directions, one change per pass
  ([how-it-works.md](how-it-works.md)).
- **It stops** when the next pass is the master's (every direction closed, or a round end) or the run ends. It writes
  `worker_report.json`, `WORKER_REPORT.md`, `WORKER_SUMMARY.md` (the `worker-summary` skill's account), and
  `role_exit.json` with `"outcome": "reported"`.
- **Then the master again:** `python3 -m arbor master --config <config.json>` ends the round and runs the dossier's update round,
  then writes the next order, or ends the campaign.
- **Alternate them until the master says `ended`,** or let one command do it:

  ```bash
  python3 -m arbor run --config <config.json> --split processes
  ```

- **The round-end A/B (optional):** standalone, a round can end with a same-node A/B of the base tree against the
  current one:

  ```json
  "round_ab": {"command": "python3 -m arbor.ab --config <config.json> --out {out} --deadline {deadline}"}
  ```

## 5. Split the work: several workers

### 5.1 On one node: `--split workers`

The master deals directions to several workers at once, and keeps only what a same-node A/B confirms.

1. **One slot per worker:** a container of its own, started from the same image as the master's, with its workspace
   at the same path and git-initialized (1.5 and 1.6). The master's container is `arbor-main`; name the slots
   `arbor-main-w1`, `arbor-main-w2`, and so on.
2. **The `workers` block** in the config:

   ```json
   "workers": {"count": 2, "gpu_ids": "0,1", "size": 1, "reserve_minutes": 75}
   ```

   | Key | What | Default |
   | --- | --- | --- |
   | `count` | how many slots | `len(gpu_ids) // tp` |
   | `gpu_ids` | the cards the slots share, `tp` each | 0-7 |
   | `container` | each slot's container; `{container}` and `{k}` filled in | `{container}-w{k}` |
   | `port_step` | slot k serves on the config's port plus k times this | 10 |
   | `size` | directions dealt in one order | 2 |
   | `reserve_minutes` | time kept for the integration's A/Bs | 45 |
   | `ab_rounds` | measured pairs in each integration A/B | 3 |

3. **Run it:**

   ```bash
   python3 -m arbor run --config <config.json> --split workers
   ```

- **Each round:**
  - the master maps;
  - each free slot gets an order of up to `size` directions;
  - the workers run at once, each in `<session>/workers/<order>/session`, with its own init and baseline;
  - each report folds back into the master's map, and the slot is dealt again;
  - when nothing is left to deal, the integration merges each worker's change onto the master's tree, A/B-tests it on
    the node, and keeps it only if confirmed.
- **The campaign ends** when only `reserve_minutes` is left.
- **Where to look:** each worker's `worker.log` beside its session, and `<session>/integrate/round-<n>/integration.json`.
- **Refused (exit 2):** slots that share a card, a container or a port, or that use the master's own workspace.
- **Checked on a GPU node:** two TP1 workers on one node, two rounds. The integration kept the changes its A/B
  confirmed and dropped one that measured slower.

### 5.2 Shares by hand, or under a host

A host such as Hyperloom's MB owns the cards and the merge, and drives the two sides itself:

```bash
python3 -m arbor master --config <config.json>                                           # ends ordered
python3 -m arbor.roles split --order <session>/work_order.json --shares 2 --out shares/
python3 -m arbor worker --order shares/work_order-<k>.json --config <share-config.json>   # one per share, at once
```

- **`<share-config.json>` is a config of the share's own,** with its own `session_root`, container, workspace, `gpu_ids` and `port`.
- **What the worker takes from the order:** its directions as its map, and the master's dossier when it has none.
- **What it runs itself:** its own init and baseline on its node. It ends `reported`, or `spent` when it hands its
  directions back.
- **A freed worker's next order:**

  ```bash
  python3 -m arbor.roles deal --order <session>/work_order.json --reports <each worker_report.json> \
    --held-order <each running worker's order> --out next.json
  ```

- **Merging is the caller's job.** A standalone master does not merge shares run by hand. Bringing their reports and
  trees back is the host's work, or `--split workers`'s.

## 6. Watch, stop, resume

- **The log:** one `[arbor] Gate: KEEP (B) — reason` line per iteration, and `[arbor] Master: ...` and
  `[arbor] Worker: ...` lines at each hand-over.
- **The session:**
  - `role_exit.json`: how the last command ended;
  - `lifecycle.jsonl`: the journal of chapters, events and skills;
  - `session.json`: the bars and every verdict;
  - `landed.patch`: the kept changes.
- **The dashboard:** `node arbor/dashboard/server.mjs local --host <host> --session <session>` (port 8766).
- **Stop:** `touch <session>/STOP` stops after the iteration in flight.
- **Resume:** run the same command again.
- **For scripts:** `python3 -m arbor version --json` gives each interface's version and the exit reasons.

## 7. When something goes wrong

| What you see | Why | What to do |
| --- | --- | --- |
| every agent turn fails at once (`unrecognized_model`), then `ABORTED.md` | the SDK's CLI is older than the model | upgrade `claude-agent-sdk`; check the CLI version (1.3) |
| agent turns cannot connect | the gateway's variables are not in the environment | source `~/.arbor/gateway.env` in the shell that starts Arbor |
| nothing runs after the start | the review gate waits | approve it (section 3), or start with `--no-review-gate` |
| the log's `Bench result` line says `"source": "agent_reported"`, with no second metric | the container cannot see the session at `session_mount` | fix `session_mount` (1.7) |
| every correctness check fails: `ModuleNotFoundError: tenacity` | `lm_eval` without its API extra | `pip install "lm_eval[api]"` in the container (1.5) |
| iterations NOT MEASURED: a card holds memory | another process holds one of `gpu_ids` | free the cards, or change `gpu_ids` |
| a worker exits 2: "the order's session is ..." | a worker without `--config`, whose order is another session's | run it with the master's config, or give it `--config <share-config.json>` |
| `--split workers` exits 2 | slots overlap each other or the master's workspace | fix the `workers` block (5.1) |
