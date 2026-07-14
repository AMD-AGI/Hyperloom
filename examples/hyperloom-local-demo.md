# Hyperloom Local Demo — SGLang Inference Optimization on a Single GPU Host

Welcome! This tutorial is a hands-on, copy-paste walkthrough for standing up
**Hyperloom** in Local Mode on a single AMD GPU machine and running an
end-to-end **SGLang inference-optimization** example.

It is written to serve two audiences at once:

- **For you (the reader):** every step explains *what* it does and *why*, so you
  can follow the whole flow from an empty machine to a running optimization.
- **For an AI agent:** the steps are concrete, ordered, and copy-paste-able, so
  you can hand this file to a coding agent and have it execute the demo one step
  at a time.

By the end, you will have Hyperloom running against a model on your own host —
measuring an SGLang serving-throughput baseline and attempting to improve it.

---

## Before you start

If you are reading this file, you already have Hyperloom cloned. Run every
command from the repo root, and drive the demo by handing this document to a
coding agent (e.g. run `claude` inside the repo, or open the Hyperloom folder in
Cursor). All logs for this demo are written under the repo root at
`output/sglang-hyperloom-date-{timestamp}/`.

Not cloned yet? Clone it, then start the agent:

```bash
git clone git@github.com:AMD-AGI/Hyperloom.git
cd Hyperloom
# Start a coding agent — e.g. run `claude` here, or open this folder in Cursor.
# Then give the agent this prompt:
#   Follow the tutorial examples/hyperloom-local-demo.md and run the hyperloom demo.
```

---

## Step 1 — Environment check

Step 1 sets up where logs go, confirms you are on the right machine, and then
verifies the host can run the demo (a supported AMD GPU, a working container
engine, and a sane OS/CPU).

### 1a. Create the demo output folder

All logs for this run live under one timestamped directory in the repo root:

```bash
# Run from the Hyperloom repo root.
export DEMO_OUT="output/sglang-hyperloom-date-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEMO_OUT"
chmod 777 "$DEMO_OUT" 
echo "Demo logs will be saved under: $(pwd)/$DEMO_OUT"
```

Keep `$DEMO_OUT` for the rest of the demo — later steps write their logs here.

### 1b. Confirm the target machine

You might be on a login / jump node while the GPUs live on a separate compute
node. Work out where you are *before* checking hardware. First detect whether
this is a SLURM cluster and, if so, list the nodes allocated to the current
account:

```bash
echo "hostname: $(hostname)"
# SLURM cluster? Show your allocation so you know which node has the GPUs.
if command -v squeue >/dev/null 2>&1; then
  echo "SLURM detected — your allocations:"
  squeue -u "$USER" 2>/dev/null || squeue 2>/dev/null | head
  # Just the node list(s) for the current user's running jobs, for easy selection.
  echo "allocated nodes: $(squeue -u "$USER" -h -t RUNNING -o '%N' 2>/dev/null | paste -sd, -)"
else
  echo "no SLURM (squeue not found) — treat this host as the run target"
fi
```

> **Agent:** decide the run target based on whether this is a SLURM environment,
> then only continue to 1c after the target is fixed:
>
> - **SLURM detected** (`squeue` exists): ask the user which node to run the demo
>   on with a multiple-choice question. Offer these options:
>   1. One of the SLURM node(s) allocated to the current account (from the
>      `allocated nodes:` line above — list each node as its own option).
>   2. The current host machine (`$(hostname)`).
>   3. Let the user type in the node they want to use themselves.
>
>   If the chosen node is **not** the current host, `ssh` into it first and
>   re-run the demo from there so the rest of Step 1 checks the real GPU host.
> - **No SLURM** (`squeue` not found): use the current host machine directly as
>   the run target — no question needed.

### 1c. CPU & OS

```bash
# CPU model, architecture, core counts
lscpu | grep -E "^Architecture|^Model name|^CPU\(s\)|^Core\(s\) per socket|^Socket\(s\)"
# OS + kernel
( . /etc/os-release && echo "OS: $PRETTY_NAME" ); echo "Kernel: $(uname -r) ($(uname -m))"
```

### 1d. Container engine (Docker or Podman)

```bash
# Docker (preferred): installed AND daemon running?
docker --version 2>/dev/null && (docker info >/dev/null 2>&1 && echo "docker: RUNNING" || echo "docker: installed but daemon NOT running")
# Podman (fallback): installed?
podman --version 2>/dev/null && echo "podman: present" || echo "podman: absent"
```

At least one working engine is required. Docker with a running daemon is the
default path used by the rest of this demo.

### 1e. AMD GPU (rocm-smi)

Hyperloom's SGLang runner supports **MI300X / MI325X / MI355X** (MI308X and
MI325X reuse the MI300X scripts). Detect the GPUs and confirm they are idle:

```bash
# GPU model / series (expect MI300X, MI325X, or MI355X)
rocm-smi --showproductname
# VRAM total & used per GPU
rocm-smi --showmeminfo vram
# Any process currently holding a GPU?
rocm-smi --showpids
```

The GPUs should be **idle** before you start (near-zero VRAM used, no foreign
serving processes) so the baseline measurement is not polluted.

### Report — summarize, save, and pause

Build the environment-check report as the table below (fill the "Found" column
from the command output), then do all three:

1. **Show it to the user** in the chat.
2. **Save a copy** under the demo output tree, so every step leaves an artifact:

```bash
mkdir -p "$DEMO_OUT/1-environment-check"
# Agent: write the filled-in report (the table below, with real values) to:
#   $DEMO_OUT/1-environment-check/environment-check.md
```

3. **Pause** — do **not** proceed to Step 2 until the user confirms the
   environment looks good.

| Check | Expected | Found |
|---|---|---|
| Target host | confirmed by user | _fill in_ |
| Output dir | `output/sglang-hyperloom-date-…` created | _fill in_ |
| CPU (model / cores) | x86_64, multi-core | _fill in_ |
| OS / kernel | Linux | _fill in_ |
| Container engine | docker RUNNING (or podman) | _fill in_ |
| GPU model | MI300X / MI325X / MI355X | _fill in_ |
| GPU count | ≥ 1 | _fill in_ |
| VRAM per GPU | idle (~0 used) | _fill in_ |
| Foreign GPU processes | none | _fill in_ |

---

## Step 2 — Start the container & set up Claude

Start the ROCm/SGLang container (repo mounted, host networking), make sure it
can reach Hyperloom's dependency repos on GitHub, and wire up the Claude CLI
that drives the optimizer.

### 2a. Start the container

Pick the image that matches the GPU you found in Step 1:

| GPU | Image |
|---|---|
| MI300X / MI308X / MI325X | `docker.io/primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix` |
| MI355X | `docker.io/primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix` |

Start a long-running container. This is the standard Local Mode command **plus
three changes**: `--network host` (so the container can reach a host-local LLM
proxy later), a bind-mount of this repo, and `-w` set to the repo path:

```bash
# Run from the Hyperloom repo root.
HYPERLOOM_DIR="$(pwd)"
IMAGE="docker.io/primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix"   # ← match your GPU

docker run -d \
  --name hyperloom-local \
  --network host \
  --shm-size 64g \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -v "$HYPERLOOM_DIR:$HYPERLOOM_DIR" \
  -w "$HYPERLOOM_DIR" \
  "$IMAGE" \
  tail -f /dev/null
```

> **Agent:** if a `hyperloom-local` container already exists
> (`docker ps -a --filter name=hyperloom-local`), ask the user which to do —
> do **not** decide on your own:
> 1. **Reuse** the existing container (skip `docker run`).
> 2. **Recreate** it: `docker rm -f hyperloom-local`, then run the command above.

### 2b. Check GitHub connectivity for the public dependency repos

The default demo uses the open GEAK whole-pipeline optimizer (`geak`). Confirm
the container can reach the public repos that `install.sh` and the chained
kernel-agent installer clone:

```bash
docker exec hyperloom-local bash -lc '
for r in AMD-AGI/Magpie AMD-AGI/TraceLens AMD-AGI/GEAK SemiAnalysisAI/InferenceX; do
  h=$(GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/$r.git HEAD >/dev/null 2>&1 && echo OK || echo FAIL)
  echo "$r https=$h"
done'
```

> **Agent:** print the results as the table below. If **any** repo is still
> unreachable, ask the user whether to fix network/proxy access and re-test or
> stop the demo. Do **not** copy SSH keys into the container by default.

| Repo | HTTPS |
|---|---|
| AMD-AGI/Magpie | _fill in_ |
| AMD-AGI/TraceLens | _fill in_ |
| AMD-AGI/GEAK | _fill in_ |
| SemiAnalysisAI/InferenceX | _fill in_ |

### 2c. Set up the Claude & Codex accounts in the container

Hyperloom's orchestrator drives the **Claude CLI** (Anthropic) as its
orchestration model and an **OpenAI Codex** critic agent. The Claude CLI must be
installed in the container, and both accounts must be able to reach their
endpoints.

```bash
# Already installed?
docker exec hyperloom-local bash -lc 'command -v claude && claude --version || echo "claude MISSING"'
```

If missing, install the Claude Code CLI (nvm + npm) in the container:

```bash
docker exec hyperloom-local bash -lc '
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh"
nvm install 22 && npm install -g @anthropic-ai/claude-code'
```

First tell the user that Hyperloom uses **two** coding-agent SDKs — **Anthropic
Claude** (the optimizer's orchestration model) and **OpenAI Codex** (the critic
agent) — so they must provide credentials for **both**. Then configure each
account with the two flows below.

> **Agent:** every `export` in this section must be applied in **two places** —
> the running container's shell **and** the container's `/root/.bashrc` — so all
> future shells and the CLIs pick the values up. Do not skip the `.bashrc` copy.

#### Anthropic Claude account

**setup-claude-1 — choose the auth method.** Ask the user (a question) which kind
of Claude account they will use:
1. **AMD internal** — via the AMD LLM API gateway.
2. **Personal Anthropic account** — direct against `api.anthropic.com`.

**setup-claude-2 — collect the key(s) and set the env vars** for the chosen method.

- **AMD internal (gateway):** ask the user (a question) for `ANTHROPIC_BASE_URL`
  and their `LLM_GATEWAY_KEY`. `ANTHROPIC_BASE_URL` defaults to
  `https://llm-api.amd.com/Anthropic` — tell the user this and use it as-is
  unless they provide their own. `LLM_GATEWAY_KEY` is **required**; if they don't
  have one, they can get it from <https://llm.amd.com/key-management>. Then:

```bash
export LLM_GATEWAY_KEY="xxxxx"   # <- the user's LLM gateway key
export ANTHROPIC_BASE_URL="https://llm-api.amd.com/Anthropic"   # or the user's override
export ANTHROPIC_API_KEY="dummy"
export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: $LLM_GATEWAY_KEY"
```

- **Personal Anthropic account:** ask the user (a question) for their
  `ANTHROPIC_API_KEY`, then:

```bash
export ANTHROPIC_BASE_URL="https://api.anthropic.com"
export ANTHROPIC_API_KEY="xxxxx"   # <- the user's ANTHROPIC_API_KEY
```

**setup-claude-3 — verify connectivity, then pick the model.** From inside the
container (interactive shell so `/root/.bashrc` is loaded), confirm the account
works and list the models it can use:

```bash
docker exec \
  -e ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL" \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -e ANTHROPIC_CUSTOM_HEADERS="$ANTHROPIC_CUSTOM_HEADERS" \
  hyperloom-local bash -lc '
    headers=(-H "anthropic-version: 2023-06-01" -H "x-api-key: $ANTHROPIC_API_KEY")
    [ -n "${ANTHROPIC_CUSTOM_HEADERS:-}" ] && headers+=(-H "$ANTHROPIC_CUSTOM_HEADERS")
    curl -sS -m 30 "$ANTHROPIC_BASE_URL/v1/models" "${headers[@]}"
  '
```

> **Agent:**
> - If connectivity **fails** (non-200 / timeout / empty model list): go back to
>   **setup-claude-1** and let the user re-enter their settings. For the **AMD
>   gateway** path this often means the gateway is not directly reachable from
>   this GPU host — point the user to
>   [Claude Code on Remote Servers (via AMD LLM Gateway)](https://amd.atlassian.net/wiki/spaces/~712020ea4fade82ae94a95b7c0ba1cb554d2a8/pages/1792352515/Claude+Code+on+Remote+Servers+via+AMD+LLM+Gateway)
>   to stand up a proxy/tunnel (that machine needs SSH access to this server),
>   then retry.
> - If connectivity **succeeds**: ask the user (a question listing the returned
>   model IDs) which Claude model to use as the orchestration model, then set
>   `CLAUDE_MODEL`:

```bash
export CLAUDE_MODEL="xxx"   # <- the model the user picked
```

#### OpenAI Codex account

**setup-codex-1 — choose the auth method.** Ask the user (a question) which kind
of OpenAI account they will use:
1. **AMD internal** — via the AMD LLM API gateway.
2. **Personal OpenAI account** — direct against `api.openai.com`.

**setup-codex-2 — collect the key(s) and set the env vars** for the chosen method.

- **AMD internal (gateway):** ask the user (a question) for `OPENAI_BASE_URL` and
  their `LLM_GATEWAY_KEY`. `OPENAI_BASE_URL` defaults to
  `https://llm-api.amd.com/Unified/v1` — tell the user this and use it as-is
  unless they provide their own. `LLM_GATEWAY_KEY` is **required**. Then:

```bash
export LLM_GATEWAY_KEY="xxxxx"   # <- the user's LLM gateway key
export OPENAI_BASE_URL="https://llm-api.amd.com/Unified/v1"   # or the user's override
export OPENAI_API_KEY="dummy"
export OPENAI_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: $LLM_GATEWAY_KEY"
```

- **Personal OpenAI account:** ask the user (a question) for their
  `OPENAI_API_KEY`, then:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="xxxxx"   # <- the user's OPENAI_API_KEY
```

**setup-codex-3 — verify connectivity, then pick the model.** From inside the
container (interactive shell so `/root/.bashrc` is loaded), confirm the account
works and list its models:

```bash
docker exec \
  -e OPENAI_BASE_URL="$OPENAI_BASE_URL" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e OPENAI_CUSTOM_HEADERS="$OPENAI_CUSTOM_HEADERS" \
  hyperloom-local bash -lc '
    headers=(-H "Authorization: Bearer $OPENAI_API_KEY")
    [ -n "${OPENAI_CUSTOM_HEADERS:-}" ] && headers+=(-H "$OPENAI_CUSTOM_HEADERS")
    curl -sS -m 30 "${OPENAI_BASE_URL%/}/models" "${headers[@]}"
  '
```

> **Agent:**
> - If connectivity **fails** (non-200 / timeout / empty model list): go back to
>   **setup-codex-1** and let the user re-enter their settings.
> - If connectivity **succeeds**: ask the user (a question listing the returned
>   model IDs) which model to use as the Codex / critic model, then set
>   `CODEX_MODEL`:

```bash
export CODEX_MODEL="xxx"   # <- the model the user picked
```

### Report — save and pause

Wrap up Step 2 for the user, then **save a copy** and **stop**:

1. **Show** the container status, the repo-connectivity table (2b), and the
   Claude & Codex account setup (2c) — auth method, connectivity result, and the
   chosen `CLAUDE_MODEL` / `CODEX_MODEL` — to the user.
2. **Save** the same summary under the demo output tree:

```bash
mkdir -p "$DEMO_OUT/2-container-claude-setup"
# Agent: write the Step 2 summary (container name/image, repo-connectivity
# table, claude CLI install status, Claude + Codex auth method / connectivity /
# chosen models) to:
#   $DEMO_OUT/2-container-claude-setup/container-claude-setup.md
```

3. **Pause** — do **not** start the next step until the user tells you to
   continue.

---

## Step 3 — Configure Hyperloom

Create the runtime `.env`, point it at this run's workspace and the Claude +
Codex accounts you set up in Step 2c, then bootstrap Hyperloom's public
dependency checkouts with `install.sh`.

### 3a. Create and fill `.env`

Run from the repo root. Compute the run workspace on the host, then write `.env`
from inside the container so the Claude / Codex values come from the container
environment you configured in Step 2c:

```bash
export USER_DATA_PATH="$(pwd)/$DEMO_OUT/workspace"

docker exec -e USER_DATA_PATH="$USER_DATA_PATH" hyperloom-local bash -ic '
set -e
: "${ANTHROPIC_BASE_URL:?ANTHROPIC_BASE_URL missing; redo setup-claude}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY missing; redo setup-claude}"
: "${CLAUDE_MODEL:?CLAUDE_MODEL missing; redo setup-claude}"
: "${OPENAI_BASE_URL:?OPENAI_BASE_URL missing; redo setup-codex}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY missing; redo setup-codex}"
: "${CODEX_MODEL:?CODEX_MODEL missing; redo setup-codex}"

cp .env.template .env
cat >> .env <<EOF

# --- Hyperloom demo settings ---
# Opt out of the AMD-only orchestration-model gate when using a custom gateway
# model id. The current catalog probe also applies custom headers, so this is
# about model selection rather than a workaround for missing auth headers.
export INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1
# Default to the open GEAK whole-pipeline optimizer for the full KERNEL_AGENT phase.
export KERNEL_OPT_BACKEND_ORDER=geak
# Claude (orchestration) — the values you validated in setup-claude-3.
export ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL"
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
export ANTHROPIC_CUSTOM_HEADERS="$ANTHROPIC_CUSTOM_HEADERS"
export CLAUDE_MODEL="$CLAUDE_MODEL"
export ANTHROPIC_MODEL="$CLAUDE_MODEL"
# Codex (critic) — the values you validated in setup-codex-3.
export OPENAI_BASE_URL="$OPENAI_BASE_URL"
export OPENAI_API_KEY="$OPENAI_API_KEY"
export OPENAI_CUSTOM_HEADERS="$OPENAI_CUSTOM_HEADERS"
export CODEX_MODEL="$CODEX_MODEL"
# SGLang serving port — pick one that is free on this host (default 8888 is
# often already taken, e.g. by an observability agent).
export PORT=8899
EOF

sed -i "s#^USER_DATA_PATH=.*#USER_DATA_PATH=$USER_DATA_PATH#" .env
echo "USER_DATA_PATH=$USER_DATA_PATH"
'
```

> **Agent:** `$DEMO_OUT` is the output dir from Step 1. The `ANTHROPIC_*` /
> `OPENAI_*` / model lines are pulled from the container env you set in Step 2c.
> The command fails fast if required values are missing; if that happens, re-run
> the matching setup-claude-* / setup-codex-* step, then regenerate `.env`. Also
> confirm `PORT` is free — `ss -ltn | grep -w ":$PORT"` should print nothing;
> pick another port if it is taken.

### 3b. Bootstrap dependency checkouts

Create the workspace and run the installer **inside the container**. This clones
Magpie / InferenceX and chains the kernel-agent installer, which prepares
TraceLens plus the GEAK runtime. With `KERNEL_OPT_BACKEND_ORDER=geak`, the demo
uses the open whole-pipeline GEAK path.

```bash
docker exec -e USER_DATA_PATH="$USER_DATA_PATH" hyperloom-local bash -lc '
  mkdir -p "$USER_DATA_PATH"
  ulimit -Sn 65536 || true
  bash src/hyperloom/inference_optimizer/assets/install.sh
'
```

`install.sh` is idempotent. It writes
`$USER_DATA_PATH/runtime/kernel-agent.env.sh`, which is the env file to source
before launching or resuming.

### Report — save and pause

Summarize Step 3, save a copy, and stop:

1. **Show** the user: the key `.env` values (**redact** the gateway key), the
   resolved `USER_DATA_PATH` / `PORT`, `KERNEL_OPT_BACKEND_ORDER`, and
   `install.sh`'s result.
2. **Render `kernel-agent.env.sh`** — the resolved kernel-agent dependency paths
   the installer wrote — as a table (one row per relevant `export` line):

```bash
docker exec -e USER_DATA_PATH="$USER_DATA_PATH" hyperloom-local bash -lc 'cat "$USER_DATA_PATH/runtime/kernel-agent.env.sh"'
```

| Variable | Value |
|---|---|
| REPO_ROOT | _fill in_ |
| USER_DATA_PATH | _fill in_ |
| HYPERLOOM_RUNTIME_DIR | _fill in_ |
| MAGPIE_PATH | _fill in_ |
| INFERENCEX_PATH | _fill in_ |
| TRACELENS_ROOT | _fill in_ |
| GEAK_ROOT | _fill in_ |
| GEAK_CONFIG | _fill in_ |

3. **Save** the summary — the `.env` highlights **and** the `kernel-agent.env.sh`
   table above:

```bash
mkdir -p "$DEMO_OUT/3-configure-hyperloom"
# Agent: write the Step 3 summary (.env highlights with the key redacted,
# USER_DATA_PATH, PORT, install.sh outcome, and the kernel-agent.env.sh
# variable table) to:
#   $DEMO_OUT/3-configure-hyperloom/configure-hyperloom.md
```

4. **Pause** — wait for the user to confirm before the next step.

---

## Step 4 — Download the model & launch the optimization

### 4a. Re-check Claude (gate)

The optimizer is driven by Claude, so re-confirm the container's Claude CLI +
configured model still work **before** downloading anything:

```bash
docker exec hyperloom-local bash -ic '
command -v claude >/dev/null || { echo "claude MISSING"; exit 1; }
headers=(-H "content-type: application/json" -H "anthropic-version: 2023-06-01" -H "x-api-key: $ANTHROPIC_API_KEY")
[ -n "${ANTHROPIC_CUSTOM_HEADERS:-}" ] && headers+=(-H "$ANTHROPIC_CUSTOM_HEADERS")
curl -sS -m 30 -o /dev/null -w "opus http=%{http_code}\n" "$ANTHROPIC_BASE_URL/v1/messages" \
  "${headers[@]}" \
  -d "{\"model\":\"${CLAUDE_MODEL:-$ANTHROPIC_MODEL}\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}"' 2>/dev/null
```

> **Agent:** if `claude` is missing or the call is not `http=200`, **stop** and
> tell the user to fix the container's Claude environment (redo Step 2c, or set
> up the proxy from that step's guide). Do not continue.

### 4b. Download the model

Download **Qwen/Qwen3-8B** into this run's model dir
(`output/sglang-hyperloom-date-XXXX/model`):

```bash
export MODEL_PATH="$(pwd)/$DEMO_OUT/model"
docker exec -e MODEL_PATH="$MODEL_PATH" hyperloom-local bash -lc '
  mkdir -p "$MODEL_PATH"
  hf download Qwen/Qwen3-8B --local-dir "$MODEL_PATH"
'
# Sanity check: config + weights present
docker exec -e MODEL_PATH="$MODEL_PATH" hyperloom-local bash -lc 'ls "$MODEL_PATH"/config.json "$MODEL_PATH"/*.safetensors >/dev/null 2>&1 && echo "model OK" || echo "model INCOMPLETE"'
```

> **Agent:** if the download fails on auth / gating / rate limits, ask the user
> for an `HF_TOKEN` and retry, e.g.
> `docker exec -e HF_TOKEN=<token> -e MODEL_PATH="$MODEL_PATH" hyperloom-local bash -lc 'hf download Qwen/Qwen3-8B --local-dir "$MODEL_PATH"'`.

### 4c. Collect the optimization prompt

Ask the user for their optimization request. Offer this reference, pre-filled
with the model path from 4b and the GPU from Step 1 (the user tunes TP / CONC /
ISL / OSL / goal / budget):

```text
Optimize inference for this workload:
- Model: <MODEL_PATH from 4b>
- Framework: sglang
- GPU: <GPU type from Step 1, e.g. MI355X>
- TP: 8
- CONC: 64
- ISL: 1024
- OSL: 1024
- Goal: improve throughput by at least 10%
- Budget: 6 hours
```

> **Agent:** fill `Model` with `$MODEL_PATH` and `GPU` with the Step 1 GPU type;
> `TP` should equal the GPU count. Wait for the user's final prompt before
> launching.

### 4d. Launch the optimization

You are ready to run Hyperloom. The optimization loop
(`baseline → framework → explore → kernel → sweep → close`) is explained here:
<https://github.com/AMD-AGI/Hyperloom/blob/main/docs/conceptual/optimization-loop.md>

Launch by following [`docs/how-to/optimize.md`](../docs/how-to/optimize.md).
Inside the container, re-run `install.sh` (IR-2) **with a raised file-descriptor
limit — run `ulimit -Sn 65536` first** (the install and the optimizer open many
files; the default soft limit can cause "too many open files" errors), source
`.env` plus `kernel-agent.env.sh`, then start `inference_optimizer optimize` in
the background (`setsid nohup` — the run is long). The default backend order is
`KERNEL_OPT_BACKEND_ORDER=geak`, which delegates the whole KERNEL_AGENT phase to
the open GEAK e2e optimizer. Both the Claude and Codex accounts were wired up in
Step 2c, so drive the kernel agent with Claude (`--kernel-claude`) and run the
**real** critic — the in-tree critic-agent backend (`--critic-agent`), which
reviews with your `CODEX_MODEL` over the OpenAI/Unified gateway (point
`CRITIC_AGENT_ROOT` at the in-tree runtime under `src/hyperloom/agents/critic`).
`--robustness-disable-server-probe` avoids false "server unreachable" stops when
the single node restarts the server between benchmarks. **The run parameters
below are a reference example** — set `--gpu-type`, `--tp`, `--conc`, `--isl`,
`--osl`, `--target-gain`, and `--max-hours` from your real setup (the GPU from
Step 1, TP = GPU count, and the values from the 4c prompt), not the literals
shown:

```bash
docker exec -e USER_DATA_PATH="$USER_DATA_PATH" -e MODEL_PATH="$MODEL_PATH" hyperloom-local bash -lc '
  set -e
  set -a; . ./.env; set +a
  export USER_DATA_PATH="${USER_DATA_PATH:?USER_DATA_PATH missing}"
  export KERNEL_OPT_BACKEND_ORDER="${KERNEL_OPT_BACKEND_ORDER:-geak}"
  ulimit -Sn 65536                                                  # raise fd limit (avoid "too many open files")
  bash src/hyperloom/inference_optimizer/assets/install.sh          # IR-2 (idempotent)
  . "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
  # Real critic-agent: reviews with CODEX_MODEL (from .env) over the OpenAI/Unified gateway.
  export CRITIC_AGENT_ROOT="$PWD/src/hyperloom/agents/critic"
  RUN_DIR="$USER_DATA_PATH/optimizer_runs"; mkdir -p "$RUN_DIR"
  TAG="$(basename "$MODEL_PATH")-$(date +%Y%m%d_%H%M%S)"

  # ---- Run parameters: REFERENCE EXAMPLE — set from your real setup ----------
  # GPU_TYPE  : the GPU detected in Step 1 (mi300x / mi325x / mi355x). The
  #             optimizer also auto-probes the GPU and the probe wins, so this is
  #             only a hint, but keep it correct.
  # TP        : tensor-parallel size = GPU count from Step 1.
  # CONC/ISL/OSL, TARGET_GAIN, MAX_HOURS : take from the 4c prompt.
  GPU_TYPE=mi355x
  TP=8
  CONC=64
  ISL=1024
  OSL=1024
  TARGET_GAIN=10
  MAX_HOURS=6
  # ---------------------------------------------------------------------------

  setsid nohup inference_optimizer --verbose optimize \
    --model "$MODEL_PATH" --framework sglang --gpu-type "$GPU_TYPE" \
    --tp "$TP" --conc "$CONC" --isl "$ISL" --osl "$OSL" \
    --target-gain "$TARGET_GAIN" --max-hours "$MAX_HOURS" --tick-interval-sec 30 \
    --claude-model "$CLAUDE_MODEL" --codex-model "$CODEX_MODEL" \
    --kernel-claude --critic-agent --robustness-disable-server-probe \
    --launch-info-file "$RUN_DIR/launch_$TAG.json" \
    > "$RUN_DIR/run_$TAG.log" 2>&1 < /dev/null &
  WRAPPER_PID=$!

  read_json() {
    python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], \"\"))" "$1" "$2" 2>/dev/null || true
  }
  for _ in $(seq 1 15); do
    [ -s "$RUN_DIR/launch_$TAG.json" ] && break
    sleep 2
  done
  REAL_PID="$(read_json "$RUN_DIR/launch_$TAG.json" pid)"
  SESSION_DIR="$(read_json "$RUN_DIR/launch_$TAG.json" session_dir)"
  echo "wrapper_pid=$WRAPPER_PID real_pid=$REAL_PID session_dir=$SESSION_DIR log=$RUN_DIR/run_$TAG.log launch_info=$RUN_DIR/launch_$TAG.json"
'
```

> **Agent:** `GPU_TYPE` / `TP` / `CONC` / `ISL` / `OSL` / `TARGET_GAIN` /
> `MAX_HOURS` are placeholders from an MI355X example — overwrite each with the
> real values (Step 1 GPU + count, and the 4c prompt) before launching. If
> `--critic-agent` aborts with a "critic-agent runtime not found" error, confirm
> `CRITIC_AGENT_ROOT` resolves to `src/hyperloom/agents/critic` (containing
> `runtime/cli.py`).

Then, per `optimize.md`, report the **session ID, log path, PID, and initial
health check**, and monitor every ~300s until the run finishes or fails.

**Watch progress** — everything lands under `$USER_DATA_PATH`:

| What | Path |
|---|---|
| Launcher / orchestration log | `$USER_DATA_PATH/optimizer_runs/run_<tag>.log` |
| Session dir (from launch-info JSON `.session_dir`) | `$USER_DATA_PATH/<model>/<UTC>/` |
| Phase / stop reason / cumulative gain | `<session_dir>/state.json` |
| SGLang server log | `<session_dir>/runs/**/server.log` |
| Final report | `<session_dir>/reports/final.md` |
