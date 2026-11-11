# Quickstart: Arbor inside Hyperloom's multi-branch mode

From a GPU node to a Hyperloom multi-branch session (MB) whose branches are Arbor campaigns. MB owns the node, the
cards, the clock, the measurement of record and the merge; each Arbor runs its loop inside the cards and the time MB
gives it. Which side runs each step is in [who-runs-what.md](who-runs-what.md); the contract key by key is in
[hosts.md](hosts.md); Hyperloom's own pages are `docs/how-to/run-multibranch.md` and
`docs/how-to/multibranch-review-gate.md` in its repository.

| Step | Command | Ends with |
| --- | --- | --- |
| Check the environment | `hyperloom-multibranch run --help`, and section 2.6 | no errors |
| Run a session with Arbor branches | `hyperloom-multibranch run ... --autonomous-agent <arbor> ...` | the session's exit; its directory on stdout's first line |
| Review the map before the Arbors open | `hyperloom-multibranch review --session <dir> show\|approve\|reject` | the share branches open, or none |
| Map now, run later | `run ... --stop-after-map`, then `run ... --agent-directions <map.json> --agent-dossier <dossier-dir> --no-review-gate` | the map in `MB-agent-map/`, then a session on it |
| Finish a session that died before FINAL | `hyperloom-multibranch final-only <dir>` | its report |

## 1. How a session runs

| Phase | MB | Arbor |
| --- | --- | --- |
| PRELUDE | commits the root recipe, measures the baseline, profiles the root (TraceLens) | nothing |
| EXPLORE, the map | a mapping branch leases `--tp` cards and starts one Arbor as the dossier pass | writes the dossier and the map, exits `mapped` |
| EXPLORE, the review | the mapping branch waits for a person; the other branches go on | nothing |
| EXPLORE, the shares | splits the map, opens one branch per Arbor, each with its share (`component.plan`) | iterates on its share, writes a snapshot per KEEP |
| EXPLORE, the merge | tests each KEEP on its own pool, lands the good ones on the branch and into integration | reads MB's notes on what integration holds |
| FINAL and CLOSE | sweeps the integration head and the best branch head, picks one, writes the report | stopped |

With `--agent-directions`, MB skips the mapping branch and splits the map you gave.

## 2. The environment

### 2.1 Where everything runs

- **One serving container on the GPU node** holds MB, its tests and every Arbor. Arbor runs beside MB, as MB's user,
  with `docker_container: "@host"`: it never calls docker.
- **MB leases the cards.** It sets `ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` for each Arbor, and kills a
  process that reaches past its lease.
- **Shared storage** holds the Hyperloom checkout, its venv and user data, the RSI archive, the Arbor checkout, the
  weights and the sessions.

### 2.2 The container

An example container, with the work directory `/work/mb` on shared storage:

```bash
IMAGE=docker.io/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
DIGEST=<the image's sha256 digest>
docker run -d --name mb-session --network host --user "$(id -u):$(id -g)" \
  --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ipc=host --shm-size 64g \
  --ulimit nofile=1048576:1048576 --init -e HOME=/work/mb/home -e USER="$(id -un)" \
  -e HYPERLOOM_ACTIVE_IMAGE_REF="$IMAGE" -e HYPERLOOM_ACTIVE_IMAGE_DIGEST="$DIGEST" \
  -v /work/mb:/work/mb -v /models:/models:ro --entrypoint tail "$IMAGE" -f /dev/null
```

- **`HYPERLOOM_ACTIVE_IMAGE_REF` and `_DIGEST`** must name the image the session runs on.
- **Run it as your user,** with a writable `HOME`, so files on shared storage are yours. Make the user known inside the
  container first (an entry in `/etc/passwd`, as root).
- **Two sessions on one node need two containers.** Magpie's cleanup stops every `vllm serve` in its container.

### 2.3 Hyperloom, its venv and its tools

Inside the container, from the Hyperloom checkout (`AMD-AGI/Hyperloom`, branch `multibranch-dev`):

```bash
cd /work/mb/hyperloom
PY=/opt/python/bin/python3                      # the image's Python, with its vLLM and torch
"$PY" -m venv /work/mb/venv
echo /opt/python/lib/python3.14/site-packages \
  > "$(/work/mb/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/_image_site.pth"
. /work/mb/env.sh                               # section 2.4
bash src/hyperloom/inference_optimizer/assets/install.sh
python -m pip install pyyaml claude-agent-sdk==0.2.174
```

- **The venv sees the image's packages** through the `.pth` file, so what `install.sh` adds goes to shared storage
  and the image stays as it is. Adjust the two paths to the image's Python.
- **`install.sh`** installs Hyperloom and `hyperloom-multibranch`, the toolbox, Magpie, TraceLens and the agent CLI.
  For the vLLM image it also makes `/app/vllm` the git tree of the image's vLLM (section 2.5). Run it in every new
  container: it is quick once its caches are on shared storage.
- **Arbor's own packages** (`pyyaml`, `claude-agent-sdk`) go into MB's venv: Arbor runs on MB's Python.

### 2.4 The environment variables

An `env.sh` you source in every shell that runs MB, and then the two files `install.sh` relies on:

```bash
MB=/work/mb
export VIRTUAL_ENV=$MB/venv PYTHON=$MB/venv/bin/python PATH=$MB/venv/bin:$HOME/.local/bin:$PATH
export USER_DATA_PATH=$MB/userdata                 # sessions are created here
export PYTHONPATH=$MB/hyperloom:${PYTHONPATH:-}
export HYPERLOOM_RSI_ARCHIVE=$MB/rsi-archive       # required, writable, shared
export HYPERLOOM_RUNNER_CACHE=$MB/runner-cache     # its default, /runner-cache, needs root
export GIT_AUTHOR_NAME=mb GIT_AUTHOR_EMAIL=mb@localhost GIT_COMMITTER_NAME=mb GIT_COMMITTER_EMAIL=mb@localhost
mkdir -p "$HYPERLOOM_RSI_ARCHIVE" "$HYPERLOOM_RUNNER_CACHE"
ulimit -Sn 65536
```

```bash
set -a; . $MB/hyperloom/.env; . "$USER_DATA_PATH/runtime/kernel-agent.env.sh"; set +a
```

| Variable | Required | What |
| --- | --- | --- |
| `HYPERLOOM_RSI_ARCHIVE` | yes | a writable shared directory: past sessions are read from it, this one's search tree is archived in it. No default |
| `HYPERLOOM_ACTIVE_IMAGE_REF`, `_DIGEST` | yes | the image the session runs on (2.2) |
| `USER_DATA_PATH` | yes | where sessions are created |
| `HYPERLOOM_RUNNER_CACHE` | for a non-root user | the runner cache (staged prompt sets) |
| the LLM gateway's variables | yes | in Hyperloom's `.env`; MB passes its environment on to every Arbor. Never commit them |
| `FRAMEWORK_REPO_PATH` (`VLLM_REPO_PATH` for vLLM) | for source KEEPs | section 2.5 |
| `HYPERLOOM_IMAGE_CHECK` | no | `warn` (default), `strict` or `off` |
| `KB_STORE_URL`, `KB_STORE_TOKEN` | no | both or neither: the Recipe KB |
| `MAGPIE_REPO`, `MAGPIE_REF` | no | a pinned Magpie, for fixes not yet upstream |
| `INFERENCE_OPTIMIZER_RAY_EXEC=0` | when Ray hangs | the baseline takes its local path instead of Ray (2.6) |

### 2.5 The framework's source, so source KEEPs import

A KEEP that changes framework source imports only if MB pinned the framework at session start:
`FRAMEWORK_REPO_PATH` names a git checkout with a remote, at the commit the image's framework was built from. Without
it, the session runs, but every source KEEP is refused with "vllm is a new component" and only flag and environment
KEEPs import. MB fixes the root at the start, so the fix needs a new session. Check before every run:

```bash
echo "$FRAMEWORK_REPO_PATH"; git -C "$FRAMEWORK_REPO_PATH" log --oneline -1; git -C "$FRAMEWORK_REPO_PATH" remote -v
```

For the vLLM image, `install.sh` fetches the image's vLLM commit into `/app/vllm`. A slow GitHub makes that fetch long;
a `.git` holding that commit at depth 1, copied in before `install.sh`, lets it return at once.

### 2.6 Checks before the first session

```bash
for c in claude TraceLens_generate_perf_report_from_config hyperloom-multibranch; do command -v "$c" || echo "MISSING $c"; done
python -c "from claude_agent_sdk._cli_version import __cli_version__ as v; print(v)"   # 2.1.310 is known good
(cd /tmp && timeout 90 claude -p "Reply with the word ok" --model claude-opus-5-5)
hyperloom-multibranch run --help > /dev/null && echo ok
timeout 60 python -c "import ray; ray.init(address='auto'); ray.shutdown()" || ray stop --force
amd-smi process                                                                     # the cards are free
```

The Ray check: PRELUDE once froze in `ray.init(address="auto")` when the container's Ray head never answered. Reset
it, or set `INFERENCE_OPTIMIZER_RAY_EXEC=0`.

### 2.7 Which Arbor runs

| `--autonomous-agent` | Runs | Use it for |
| --- | --- | --- |
| `arbor` | the copy in Hyperloom, `src/hyperloom_arbor`: Arbor `v0.9.2`, packs `serving` and `agentx` | the release; nothing to fetch |
| a path to an Arbor checkout | that checkout, copied into the session | today's Arbor: `serving-m10`, the master and worker commands, fixes since `v0.9.2` |
| a git URL, `URL@ref` | that ref, fetched once | the same, pinned to a ref; give credentials in the environment, never in the URL |

- **Give the source once per Arbor:** each `--autonomous-agent` opens one Arbor branch.
- **An Arbor checkout** has `arbor/component.py`, `arbor/__main__.py` and `docs/hosts.md`.
- **The pack:** `--agent-pack`, a directory under the checkout's `packs/` (default `serving`).

## 3. What to provide: the flags of `run`

Flags MB sessions have run with: two Arbors from one checkout, on the `serving-m10` pack:

```bash
hyperloom-multibranch run \
  --workload inferencex --isl 1024 --osl 1024 --conc 64 \
  --framework vllm --model /models/Qwen3-8B --tp 1 --gpu-type mi350x --precision bf16 \
  --max-hours 3 --max-branches 2 \
  --autonomous-agent /work/mb/arbor --autonomous-agent /work/mb/arbor \
  --agent-pack serving-m10 \
  -v > mb.out 2> mb.log
SESSION=$(head -1 mb.out)
```

| Group | Flags | Notes |
| --- | --- | --- |
| The workload | `--model`, `--framework`, `--gpu-type`, `--workload`, `--tp`, `--conc`, `--precision`; `--isl` and `--osl` for `inferencex`; `--ep` for `agentx` | one operating point per session; several points are several sessions |
| The limits | `--max-hours` (2), `--max-gpu-hours`, `--max-llm-usd` (0: none), `--max-branches` (4) | `--max-branches` counts the Arbor branches with MB's own |
| The Arbors | `--autonomous-agent` (repeatable), `--agent-pack`, `--agent-launch`, `--agent-port` (31100) | 2.7. Two sessions on one machine need their own `--agent-port` |
| The map | `--agent-directions`, `--agent-dossier`, `--agent-dossier-minutes` (90), `--agent-share-minutes` (63) | sections 4 and 5 |
| The review | `--no-review-gate`, `--stop-after-map` | section 4 |

**The cards decide how many Arbors fit.** Each Arbor branch leases `--tp` cards for its whole life, from the pool MB
measures on, and MB keeps `--tp` free cards for its own tests. On 8 GPUs at TP4 that is one Arbor; at TP1, up to
seven, within `--max-branches`.

**The launch, and `--agent-launch`.** MB serves each Arbor the session's frozen base launch, resolved from InferenceX
or the framework's cookbook (or `--reference-script`), with the root recipe's additions; Arbor checks its launch
against that base at its start. `--agent-launch FILE` serves a launch file instead:

```json
{
  "server_flags": "<the InferenceX recipe's server arguments, with its override_c<CONC> applied>",
  "server_env": {"<NAME>": "<the recipe's environment>"},
  "model_name": "<the name the server answers to>",
  "client": ""
}
```

Template: `examples/multibranch-arbor-launch/` in Hyperloom. `client` empty means the pack's own bench measures; MB's
own client, when the session has one, comes first. MB refuses a file the adapter cannot use before PRELUDE, with every
reason.

**The model** must have a chat template: MB's accuracy check runs GSM8K through chat completions, and a base model
fails every root test with HTTP 400.

## 4. Review the map

With the review on (the default), the mapping branch waits after the map step, holding no cards, while the other
branches go on. MB writes `MB-review/REVIEW.md` (the map, the dossier and the clock) and `MB-review/review.json`.

```bash
hyperloom-multibranch review --session "$SESSION" show
hyperloom-multibranch review --session "$SESSION" approve --note "why"
hyperloom-multibranch review --session "$SESSION" reject --note "why"
```

- **Edit before approving:** the map file and the dossier where `REVIEW.md` names them. MB reads both again, splits the
  map as edited, and copies the dossier into each Arbor branch.
- **The clock does not stop.** Each share must still have `--agent-share-minutes` (63) before its cutoff to open.
  `review show` says how many minutes are left.
- **From a login node,** without MB's venv: the review module needs only the standard library.

  ```bash
  PYTHONPATH=<hyperloom>/src python3 -c 'import argparse, sys
  from hyperloom_multibranch.scheduling import review
  p = argparse.ArgumentParser(); review.add_parser(p.add_subparsers(dest="command"))
  sys.exit(review.main(p.parse_args(["review", *sys.argv[1:]])))' approve --session <dir> --note "why"
  ```

- **Unattended,** pass `--no-review-gate`: nothing waits, and every Arbor starts with its own gate off.
- **A rejection** in EXPLORE opens no share branch; the session goes on.

## 5. Map now, run later

For a review with nobody waiting on a node, run the session in two stages:

```bash
# 1. PRELUDE and the dossier pass only; the map and its dossier go to $SESSION/MB-agent-map/
hyperloom-multibranch run <the flags of section 3> --stop-after-map --agent-dossier-minutes 70 -v > map.out 2> map.log
# 2. review and edit MB-agent-map/directions.json and MB-agent-map/dossier/<name>/ offline, then:
hyperloom-multibranch run <the flags of section 3> --no-review-gate \
  --agent-directions <map session>/MB-agent-map/directions.json --agent-dossier <map session>/MB-agent-map/dossier/<name>
```

- **`--stop-after-map` never pauses:** its stop is the review point.
- **`--agent-dossier-minutes`:** Arbor starts no iteration with under 63 minutes left, so less runs none. 0 skips the
  pass and generates a map from the root's profile instead.

## 6. Watch it

| Where, in the session directory | What |
| --- | --- |
| stdout's first line | the session directory |
| `MB-decisions.jsonl` (or `MB-scheduling/MB-decisions.jsonl`) | every decision: branches opened, the review, KEEPs imported and merged |
| `MB-agent-plans/shares/share-<k>.json` | each Arbor's share of the map |
| `MB-review/` | the review's page and state |
| `MB-branches/<b>/session/agent/src/hyperloom-mb/config.json` | the config MB wrote for that Arbor |
| `MB-branches/<b>/session/agent/src/hyperloom-mb/session/` | that Arbor's session: `lifecycle.jsonl`, `keeps/`, `heartbeat.json`, `component_exit.json` |

**Exit codes:** 0 when the session ran its course, a spent budget included; 1 when it failed or was stopped, or the
environment kept it from starting; 2 on a usage error.

**What MB does with each Arbor exit:**

| Arbor exits | MB |
| --- | --- |
| 0: `done`, `deadline`, `map_exhausted`, `sigterm`, `aborted`, `review_rejected` | reads the reason, learns the ledger, settles the pending KEEP, stops the branch |
| 0 `mapped` (the dossier pass) | takes the map and the dossier |
| 2 (refused) | never restarts it; the branch ends with `component_refused.json`'s reasons |
| 3 (`unreachable`) or any other non-zero | restarts it with the same command and deadline; gives up after 3 crashes in 10 minutes, or 2 stalls in a row |

## 7. What Arbor does differently under MB

- **MB writes its config** (`hyperloom-mb/config.json`): `mode: "component"`, `docker_container: "@host"`, the
  deadline, the bench client as `component.bench_command`, the share as `component.plan`, and MB's rules as
  `directives` (each `Host rule (MB) ...`), which every Arbor agent reads.
- **Accuracy:** MB gives `GSM8K_MIN` 0.5 and `policy.accuracy_tolerance` 0.05. On a GSM8K workload Arbor measures the
  root's full GSM8K once, and holds every KEEP to it less the tolerance. An AgentX launch with simulated acceptance
  runs no GSM8K.
- **No target and no round-end A/B:** MB gates and merges. Arbor's numbers decide only its own KEEPs.
- **Arbor's master and worker commands:** MB starts `python3 -m arbor run`. With `ARBOR_SPLIT=processes` in MB's
  environment, each of those runs as `arbor master`, then `arbor worker` (checked on a GPU node).
  MB calling the two commands itself is proposed to Hyperloom
  ([quickstart-master-worker.md](quickstart-master-worker.md)).

## 8. When something goes wrong

| What you see | Why | What to do |
| --- | --- | --- |
| `run` fails before a session: `HYPERLOOM_RSI_ARCHIVE is not set` | the archive has no default | set it to a writable shared directory (2.4) |
| the runner cache cannot be created | its default is `/runner-cache` | set `HYPERLOOM_RUNNER_CACHE` (2.4) |
| PRELUDE hangs at its start | `ray.init(address="auto")` waits on a Ray head that never answers | `ray stop --force`, or `INFERENCE_OPTIMIZER_RAY_EXEC=0` (2.6) |
| every root test fails with HTTP 400 | a base model without a chat template | serve an instruct model |
| `run` refuses the agent before PRELUDE | an `--agent-launch` file the adapter cannot use | fill it from the template (section 3), or leave it out to serve the session's base |
| an Arbor branch ends at once, `component_refused.json` | Arbor refused its config: an empty `server_flags`, a missing script, no deadline | read the reasons; most come from the launch file |
| KEEPs that change source never land: "vllm is a new component" | `FRAMEWORK_REPO_PATH` was not set at session start | set it and start a new session (2.5) |
| no share branch opens: "too little time left for the shares" | the map and the review took EXPLORE's time | more `--max-hours`, a quicker review, or `--agent-share-minutes 0` |
| the dossier pass runs no iteration | `--agent-dossier-minutes` under 63 | leave it at 90, or 0 for a generated map |
| two sessions on one node stop each other's servers | Magpie's cleanup stops every `vllm serve` in its container | one container and one `--agent-port` per session |
| Arbor exits 3 | the LLM gateway was out for over 45 minutes | MB restarts it; check the gateway |
| an Arbor waits at start, holding its cards | an MB older than its Arbor, which writes no `component.review` | update Hyperloom, or start with `--no-review-gate` |
