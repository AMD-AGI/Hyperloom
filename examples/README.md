# Hyperloom Quickstart

Follow this quickstart guide to get started using Hyperloom. For more detailed
installation instructions, see the
[Hyperloom ROCm Docs](https://rocm.docs.amd.com/projects/hyperloom/en/latest/)
pages.

The recommended path is to prepare a dedicated workspace, open that directory in
Claude Code, and install the wheel into the current directory with `pip install --target .`.
For source installation instructions, please refer to the [full installation instructions](https://rocm.docs.amd.com/projects/hyperloom/en/latest/install/install.html#source-checkout-manual-installation).

```{note}
If accessing a remote server through ssh, it is recommended to connect remotely through
Claude Code, install Hyperloom on the remote server, and use your local instance to
run the Hyperloom skills.
```

## Pip install Hyperloom

The current directory serves as both the install target and the agent workspace.
Prepare a dedicated clean directory first, then open that directory in Claude
Code before running the install command.

> **Recommended run mode: Docker.** Running the demos inside the provided
> [ROCm container](https://rocm.docs.amd.com/projects/hyperloom/en/latest/compatibility.html#container-images)
> ships a validated ROCm + framework stack, gives reproducible results,
> and keeps your host untouched. Bare-metal mode is for advanced users: it
> depends on your host's existing ROCm/torch and installs framework components
> into your environment, which can cause environment-specific issues or
> conflicts. Docker is preferred for a validated, reproducible stack.

### Prerequisites

- Python 3.10+ and `pip` on the machine where you open the workspace and run
  `pip install --target .`. This covers the Hyperloom wheel only; serving-framework
  Python constraints depend on your setup scenario below.
- Access to the Anthropic LLM provider.
- A dedicated workspace directory opened in the user's agent.

From the agent terminal in that workspace, install the published release wheel
into the current directory:

```bash
pip install hyperloom-inference-optimizer==1.0.0 --target .
```

It is normal for the current directory to contain many Python package directories
after installation; users do not need to inspect them. Do not use an existing project
directory unless it is acceptable for Hyperloom to create or update `.env` there.

## Run `/hyperloom-setup`

With the agent still opened in the same workspace, run:

```text
/hyperloom-setup
```

This command runs the setup skill installed from
[`src/hyperloom/skills/hyperloom-setup/SKILL.md`](../src/hyperloom/skills/hyperloom-setup/SKILL.md).

The setup skill is interactive. It creates or updates `.env` in the current
workspace, records the selected run scenario, and stops before launching an
optimization. Run `/hyperloom-setup` once per workspace; demo skills reuse the
values already written to `.env`.

It asks for these values with a fixed option order:

1. Anthropic URL:
   - `Use default (https://api.anthropic.com)`
   - `Use AMD gateway (https://llm-api.amd.com/anthropic)`
   - `Custom`
2. Model:
   - `Use default (claude-opus-5)`
   - `Custom`
3. Secrets:
   - Setup writes placeholders in `.env`.
   - Edit secrets directly in `.env`; never paste API keys into chat.
   - If `.env` already exists, setup preserves unrelated keys but updates the
     Hyperloom setup keys selected in this run.
4. `USER_DATA_PATH`:
   - Default: `<workspace>/session`
   - Custom path
5. Run mode, recorded in `.env` as `HYPERLOOM_RUN_MODE`:
   - `docker`
   - `baremetal`

```note
If you are performing the Hyperloom setup inside of a Docker container, select
the "baremetal" option as the run mode during setup.
```

## Setup Scenarios

Hyperloom supports two local setup scenarios. Pick the one matching where the
serving framework will run.

### Scenario A: Bare Metal

Use this when the current host is the AMD GPU host where Hyperloom will run
directly.

Requirements:

- ROCm runtime and ROCm torch are already installed.
- `git` is available for dependency checkouts.
- A serving framework is either already installed, or setup may install one.
- **Base Python on this GPU host** — the interpreter `install_baremetal.sh`
  resolves, not a child venv:
  - Python 3.10+ for SGLang and general operation.
  - **Exactly Python 3.12** when setup installs vLLM. vLLM ROCm wheels are
    built for 3.12 only. vLLM defaults to an isolated framework venv, but that
    venv is created from the base interpreter and inherits its version; isolated
    mode does not relax the requirement.

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

### Scenario B: Bare Metal + Docker

Use this when the workload will run inside a ROCm container. This is the
recommended path when the host does not have ROCm torch or a serving framework
installed, or when the serving framework should come from a known container
image.

Requirements:

- Docker with AMD GPU access (`/dev/kfd`, `/dev/dri`) on the selected target
  host.
- A ROCm container image that already ships the serving framework, such as
  SGLang or vLLM.
- Host Python 3.10+ (from [Prerequisites](#prerequisites)) is enough for the
  wheel and agent. Serving-framework Python versions come from the container
  image, not the host.

In this scenario, `/hyperloom-setup` writes `.env` only and does **not** start a
container. The selected demo skill owns the container lifecycle.

If Slurm is available, setup also checks the current user's allocation so Docker
runs on the intended single GPU host instead of a login host. The user chooses
whether Docker should run on:

- the current host;
- one allocated Slurm host;
- a custom host.

The chosen host is written to `.env`:

```bash
HYPERLOOM_DOCKER_TARGET_HOST=<hostname>
```

The demo skill reads this value to target the chosen host.

## Run a Demo

When setup finishes in `baremetal` mode (and `FRAMEWORK` is set), or when `.env`
is written in `docker` mode, the setup skill offers a model demo run
and hands off to the matching demo skill. Pick a preset or the advanced custom
run:

- [`3h`](hyperloom-qwen3-8b-3h/SKILL.md) — Qwen3-8B, short no-kernel run; best
  for a first end-to-end check.
- [`12h`](hyperloom-qwen3-14b-fp8-12h/SKILL.md) — Qwen3-14B-FP8, medium-length FP8 run.
- [`custom advanced`](hyperloom-custom-advanced/SKILL.md) — user-selected model,
  framework, TP/EP, concurrency, ISL/OSL, precision, budget, phase toggles, and
  advanced CLI flags.

The preset demos reuse the values already in `.env`, so nothing is re-entered.
The custom advanced run also reuses setup values, then asks for workload and
phase choices before launch.

### Use a Custom Model

For a custom model with a preset workload, start from one of the fixed demo
skills above. Pick the demo whose runtime shape is closest to the model and
experiment you want to run, then provide your model path when the skill asks for
it. You can also set it before launching the demo:

```bash
export MODEL_PATH=/path/to/your/model
```

`MODEL_PATH` should point to a model directory that the selected serving
framework can load; a local Hugging Face-style directory should contain
`config.json`. When `MODEL_PATH` is set and valid, the demo skill uses that
directory instead of downloading its default model.

The fixed demo skill is still a preset workload. Replacing the model path does not
automatically retune tensor parallelism, concurrency, input/output lengths,
precision, or the run budget. If the custom model is much larger, smaller, or
uses a different architecture than the preset model, use
[`custom advanced`](hyperloom-custom-advanced/SKILL.md) to choose the model,
framework, TP/EP, CONC, ISL/OSL, precision, budget, skip flags, and other
optimizer CLI flags explicitly.

## What to Expect During a Demo

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

- If the current workspace contains many package folders after `pip install
  --target .`, that is expected.
- If `/hyperloom-setup` is not visible, confirm the setup skill exists under
  the current workspace. It is installed to `.claude/skills/hyperloom-setup/`;
  restart the agent if needed.
- `ImportError: libamdhip64.so.7` or `libhipblas.so.3` means the installed
  framework torch wheel expects different ROCm user-space libraries; align
  `ROCM_PATH` and `LD_LIBRARY_PATH`.
- `hipDeviceAttributePciChipId` missing during AITER build means `hipcc` is
  using older ROCm headers; put the matching ROCm `bin` first on `PATH`.
