---
myst:
    html_meta:
        "description": "Run Hyperloom inside a Docker container or on bare-metal on an AMD GPU machine. Covers installing Hyperloom, configuring credentials, and running a demo."
        "keywords": "Hyperloom, Docker, container, AMD GPU, MI300X, MI325X, MI355X, SGLang, vLLM, Cursor, Dev Containers, quickstart, ROCm"
---
# Hyperloom Quickstart

These instructions allow you to setup and run Hyperloom inside a Docker container
or on bare-metal on an AMD GPU machine. The recommended path is to prepare a
dedicated workspace, open that directory in Cursor, Claude Code, or Codex, and
install the wheel into the current directory with `pip install --target .`. The
source-clone path is kept at the end for developers and manual debugging.

## Recommended Path: Install using wheel

This is the recommended path to install and get started with Hyperloom. The
current directory is both the install target and the agent workspace. Prepare a
dedicated clean directory first, then open that directory in Cursor, Claude Code,
or Codex before running the install command.

### Prerequisites

- Python 3.10+ and `pip`.
- Access to one LLM provider: Anthropic or DeepSeek.
- A dedicated workspace directory opened in the user's agent.

### Install Hyperloom

From the agent terminal in that workspace, install Hyperloom using the following
command:

```bash
python3 -m pip install \
  ./hyperloom_inference_optimizer-0.8.0-py3-none-any.whl \
  --target .
```

It is normal for the current directory to contain many Python package directories
after install. Do not use an existing project directory unless it is acceptable
for Hyperloom to create or update `.env` there.

### Setup Hyperloom

With the agent still opened in the same workspace, run:

```text
/hyperloom-setup
```

In Cursor and Claude Code, use `/hyperloom-setup`; in Codex, use
`$hyperloom-setup`.

This command runs the setup skill installed from
[`src/hyperloom/skills/hyperloom-setup/SKILL.md`](../src/hyperloom/skills/hyperloom-setup/SKILL.md).

The setup skill is interactive. It creates or updates `.env` in the current
workspace, records the selected run scenario, and stops before launching an
optimization. Run `/hyperloom-setup` once per workspace; demo skills reuse the
values already written to `.env`.

It asks for these values with a fixed option order:

1. LLM mode:
   - `Anthropic`
   - `DeepSeek`
2. Anthropic URL, when Anthropic is selected:
   - `Use default (https://api.anthropic.com)`
   - `Use AMD gateway (https://llm-api.amd.com/anthropic)`
   - `Custom`
3. DeepSeek URL, when DeepSeek is selected:
   - `Use default (https://api.deepseek.com/anthropic)`
   - `Custom`
4. Model:
   - `Use default (<provider default>)`
   - `Custom`
5. Secrets:
   - Setup writes placeholders in `.env`.
   - Edit secrets directly in `.env`; never paste API keys into chat.
   - If `.env` already exists, setup preserves unrelated keys but updates the
     Hyperloom setup keys selected in this run.
6. `USER_DATA_PATH`:
   - Default: `<workspace>/session`
   - Custom path
7. Run mode, recorded in `.env` as `HYPERLOOM_RUN_MODE`:
   - `docker`
   - `baremetal`

If Hyperloom is already installed inside a Docker container, choose `baremetal`
because setup should run directly in the current container environment.

## Setup scenarios

Hyperloom supports two local setup scenarios. Pick the one that matches where your 
serving framework will run.

### Scenario A: Bare metal

Use this when the current host is the AMD GPU host where Hyperloom will run
directly.

Requirements:

- ROCm runtime and ROCm torch are already installed.
- `git` is available for dependency checkouts.
- A serving framework is either already installed, or setup may install one.

In this scenario, `/hyperloom-setup` runs the packaged setup backend on the host:

```bash
export REPO_ROOT="$(pwd -P)"
PYTHONPATH="$REPO_ROOT" python3 -m hyperloom.inference_optimizer.setup
```

The backend runs `install_baremetal.sh` in five phases:

1. **Base preflight**: checks ROCm, GPU arch, ROCm torch, torch/triton alignment,
   and serving framework imports.
2. **Framework install**: optionally installs the SGLang or vLLM framework layer.
3. **ROCm hotfix**: applies the profiler hotfix when the ROCm stack is eligible.
4. **Credentials**: resolves LLM gateway credentials into `.env`.
5. **Runtime env**: persists bare-metal runtime vars (framework, ROCm/venv roots,
   etc.) into `.env`.

### Scenario B: Bare metal + Docker

Use this when the agent starts from a bare host or login node, but the workload
will run inside a ROCm container. This is the recommended path when the host
does not have ROCm torch or a serving framework installed, or when the serving
framework should come from a known container image.

Requirements:

- Docker with AMD GPU access (`/dev/kfd`, `/dev/dri`) on the selected target
  host.
- A ROCm container image that already ships the serving framework, such as
  SGLang or vLLM.

In this scenario, `/hyperloom-setup` writes `.env` only and does **not** start a
container. The selected demo skill owns the container lifecycle.

If Slurm is available, setup also checks for allocated nodes. The user chooses
whether Docker should run on:

- the current host;
- one of the current user's allocated Slurm nodes; or
- a custom host.

The chosen host is written to `.env`:

```bash
HYPERLOOM_DOCKER_TARGET_HOST=<hostname>
```

The demo skill reads this value to target the chosen host.

### Environment written by setup

LLM defaults:

| Mode | Required secret | Default base URL | Default model |
|------|-----------------|------------------|---------------|
| Anthropic | `ANTHROPIC_API_KEY` | `https://api.anthropic.com` | `CLAUDE_MODEL=claude-opus-4-8` |
| DeepSeek | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/anthropic` | `DEEPSEEK_MODEL=deepseek-chat` |

Setup creates or updates `.env` in the current workspace and writes the resolved
values there.

Common keys:

- `USER_DATA_PATH`
- `HYPERLOOM_RUN_MODE`
- `HYPERLOOM_DOCKER_TARGET_HOST` (only when `HYPERLOOM_RUN_MODE=docker`)

Bare-metal setup may also write runtime vars such as `FRAMEWORK`, `ROCM_PATH`,
`VIRTUAL_ENV`, and `VLLM_VENV_ROOT`. Kernel-agent paths (`MAGPIE_PATH`,
`INFERENCEX_PATH`, `TRACELENS_ROOT`, `GEAK_ROOT`) are added later by the workload
skill's `install.sh`.

`.env` in the current workspace is the single source of truth; no extra script
needs sourcing. Setup only needs to run again when changing the LLM provider,
base URL, model, `USER_DATA_PATH`, run mode, Docker target host, or bare-metal
framework setup choice.

## Run a demo

Once setup is finished in `baremetal` mode (and `FRAMEWORK` is set), or when `.env`
has been written in `docker` mode, the setup skill offers a model demo run
and hands off to the matching demo skill. Different options are available depending
on the length you would like the demo to run for:

- [`3h`](hyperloom-qwen3-8b-3h/SKILL.md) — Qwen3-8B, short no-kernel run; best
  for a first end-to-end check.
- [`8h`](hyperloom-qwen3-14b-fp8-8h/SKILL.md) — Qwen3-14B-FP8, medium-length FP8 run.
- [`24h`](hyperloom-gpt-oss-120b-24h/SKILL.md) — gpt-oss-120b, long-horizon cyclic
  run.

The demo reuses the values already in `.env`, so nothing needs to be re-entered.

## Use a custom model

For a custom model, still start from one of the example demo skills above. Pick
the demo whose runtime shape is closest to the model and experiment you want to
run, then provide your model path when the skill asks for it. You can also set it
before launching the demo:

```bash
export MODEL_PATH=/path/to/your/model
```

`MODEL_PATH` should point to a model directory that the selected serving
framework can load; a local Hugging Face-style directory should contain
`config.json`. When `MODEL_PATH` is set and valid, the demo skill uses that
directory instead of downloading its default model.

The demo skill is still a preset workload. Replacing the model path does not
automatically retune tensor parallelism, concurrency, input/output lengths,
precision, or the run budget. If the custom model is much larger, smaller, or
uses a different architecture than the preset model, choose the closest preset or
use the source/manual path to adjust the workload parameters explicitly.

## What to expect during a demo

Demo optimizations are long-running background jobs. The agent should not stream
every debug log line, but it should make progress visible before and after
launch.

Before launch, expect a short plan that includes the resolved model path, run
mode, framework, TP, concurrency, ISL/OSL, precision, run budget, and
`USER_DATA_PATH`. After launch, expect the optimizer PID, run log path,
launch-info JSON path, session directory, `state.json` path, and the initial
health check result.

During the run, the agent should report a concise status summary about every
300 seconds. Useful fields include whether the process is still alive, the
current phase, `stop_reason`, baseline throughput, current best throughput,
cumulative gain, latest benchmark result or candidate decision, and the most
relevant recent log lines. Secrets such as API keys, tokens, and custom headers
must never be printed.

## Troubleshooting

- The current workspace contains many package folders after `pip install
  --target .` - this is the expected behavior.
- If `/hyperloom-setup` is not visible, confirm the setup skill exists under
  the current workspace. It is installed to `.claude/skills/hyperloom-setup/`
  (Claude Code), `.cursor/skills/hyperloom-setup/` (Cursor) and
  `.agents/skills/hyperloom-setup/` (Cursor/Codex); restart the agent if needed.
- `ImportError: libamdhip64.so.7` or `libhipblas.so.3` means the installed
  framework torch wheel expects different ROCm user-space libraries; align
  `ROCM_PATH` and `LD_LIBRARY_PATH`.
- `hipDeviceAttributePciChipId` missing during AITER build means `hipcc` is
  using older ROCm headers; put the matching ROCm `bin` first on `PATH`.

## Source checkout / manual installation

These instructions are for advanced users and developer. Use this path only
when developing Hyperloom, testing local source changes, or debugging setup
internals.

Clone the repository:

```bash
git clone https://github.com/AMD-AGI/Hyperloom.git
cd Hyperloom
```

In source mode, the agent workspace is the repository root. Create `.env`
yourself with placeholders and fill in the real values before launching. Never
paste API keys into chat.

```bash
cat > .env <<'EOF'
ANTHROPIC_API_KEY=<PLEASE_FILL_IN>
ANTHROPIC_BASE_URL=https://api.anthropic.com
CLAUDE_MODEL=claude-opus-4-8
# Writable artifact root for runtime files, dependency checkouts, logs,
# optimizer runs, and generated env files. Set an absolute path you own.
USER_DATA_PATH=<PLEASE_FILL_IN>
HYPERLOOM_RUN_MODE=baremetal
EOF
```

### Bare metal (source)

Make sure the host already provides the required base environment:

- ROCm runtime and a ROCm-built torch.
- A serving framework (SGLang or vLLM) importable in the active Python.
- `git` for the dependency checkouts the optimization skill performs.

With that in place, open the repository root in the agent and paste a launch
prompt, filling in your workload and configuration:

```text
@src/hyperloom/inference_optimizer/SKILL.md

Optimize inference for this workload:
- Model: /path/to/your/model
- Framework: sglang
- GPU: MI300X
- TP: 8
- CONC: 64
- ISL: 1024
- OSL: 1024
- Goal: improve throughput by at least 10%
- Budget: 24 hours

Requirements:
1. Report the session ID, log path, PID, and initial health check result.
2. Monitor the process every 300s until the optimization is complete or failed.
```

### Docker (source)

It is recommended that you use a ROCm image that already ships the serving
framework, so nothing needs to be installed inside the container beyond
Hyperloom's runtime deps. The following images are recommended:

- `vllm`: `docker.io/primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix`
- `sglang` MI300X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix`
- `sglang` MI355X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix`

Start a long-running container from the repo root, mounting it at the same path
so `.env`, logs, and session artifacts stay valid:

```bash
export HYPERLOOM_IMAGE=docker.io/primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix
export REPO_ROOT="$(pwd -P)"
docker run -d \
  --name "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" \
  --shm-size "${HYPERLOOM_SHM_SIZE:-64g}" \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -v "$REPO_ROOT:$REPO_ROOT" \
  "$HYPERLOOM_IMAGE" \
  tail -f /dev/null
```

Then run all Hyperloom commands inside that container with
`docker exec -w "$REPO_ROOT" "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" ...`,
using `PYTHONPATH="$REPO_ROOT/src"` so the source checkout is importable. Use the
same launch prompt as bare metal above.