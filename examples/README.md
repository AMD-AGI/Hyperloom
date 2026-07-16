# Hyperloom Quickstart

This README is the main entry point for setting up Hyperloom and launching the
model demo skills. The recommended customer path is a packaged install with
`pip install --target`; the source-clone path is kept at the end for developers
and manual debugging.

## Recommended Path: Install the Wheel

Use this path for customer demos and clean validation. It installs Hyperloom into
one target directory, and that directory is also the agent workspace you open in
Cursor, Claude Code, or Codex.

### Prerequisites

- Python 3.10+ and `pip`.
- Access to one LLM provider: Anthropic or DeepSeek.
- `gh` access to the published GitHub release asset, or a locally downloaded
  Hyperloom wheel.

Download the release wheel, then install it into a clean target directory:

```bash
gh auth login
gh release download v0.8 \
  -R AMD-AGI/Hyperloom \
  -p 'hyperloom_inference_optimizer-0.8.0-py3-none-any.whl'

rm -rf ~/hyperloom

python3 -m pip install \
  ./hyperloom_inference_optimizer-0.8.0-py3-none-any.whl \
  --target ~/hyperloom
```

`pip install --target` creates the target directory automatically. It is normal
for `~/hyperloom` to contain many Python package directories after install; users
do not need to inspect them.

## Run `/hyperloom-setup`

Open `~/hyperloom` in the user's agent and run:

```text
/hyperloom-setup
```

In Cursor and Claude Code, use `/hyperloom-setup`; in Codex, use
`$hyperloom-setup`.

That command runs the setup skill installed from
[`src/hyperloom/skills/hyperloom-setup/SKILL.md`](../src/hyperloom/skills/hyperloom-setup/SKILL.md).

The setup skill is interactive. It creates `.env`, records the selected run
scenario, and stops before launching an optimization.

It asks for:

- LLM mode: Anthropic or DeepSeek.
- Non-secret LLM settings: base URL and model.
- Secret placeholders in `.env`; edit secrets directly in `.env` and never
  paste API keys into chat.
- `USER_DATA_PATH` (defaults to `<workspace>/session`).
- Setup scenario: `baremetal` or `baremetal + Docker` (recorded in `.env` as
  `HYPERLOOM_RUN_MODE=baremetal` or `HYPERLOOM_RUN_MODE=docker`).

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

In this scenario, `/hyperloom-setup` runs the packaged setup backend on the host:

```bash
PYTHONPATH="$PWD" python3 -m hyperloom.inference_optimizer.setup
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

## Environment Written by Setup

LLM defaults:

| Mode | Required secret | Default base URL | Default model |
|------|-----------------|------------------|---------------|
| Anthropic | `ANTHROPIC_API_KEY` | `https://api.anthropic.com` | `CLAUDE_MODEL=claude-opus-4-8` |
| DeepSeek | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/anthropic` | `DEEPSEEK_MODEL=deepseek-chat` |

Setup writes the resolved values into `.env`.

Common keys:

- `USER_DATA_PATH`
- `HYPERLOOM_RUN_MODE`
- `HYPERLOOM_DOCKER_TARGET_HOST` (only when `HYPERLOOM_RUN_MODE=docker`)

Bare-metal setup may also write runtime vars such as `FRAMEWORK`, `ROCM_PATH`,
`VIRTUAL_ENV`, and `VLLM_VENV_ROOT`. Kernel-agent paths (`MAGPIE_PATH`,
`INFERENCEX_PATH`, `TRACELENS_ROOT`, `GEAK_ROOT`) are added later by the workload
skill's `install.sh`.

`.env` is the single source of truth; no extra script needs sourcing.

## Run a Demo

When setup finishes in `baremetal` mode (and `FRAMEWORK` is set), or when `.env`
is written in `docker` mode, the setup skill offers a model demo run
and hands off to the matching demo skill. Pick a length:

- [`3h`](hyperloom-qwen3-8b-3h/SKILL.md) — Qwen3-8B, short no-kernel run; best
  for a first end-to-end check.
- [`8h`](hyperloom-qwen3-30b-a3b-8h/SKILL.md) — Qwen3-30B-A3B, medium-length run.
- [`24h`](hyperloom-gpt-oss-120b-24h/SKILL.md) — gpt-oss-120b, long-horizon cyclic
  run.

The demo reuses the values already in `.env`, so nothing is re-entered.

## Troubleshooting

- If the target directory contains many package folders after `pip install
  --target`, that is expected.
- If `/hyperloom-setup` is not visible, confirm the setup skill exists under
  the target directory. It is installed to `.claude/skills/hyperloom-setup/`
  (Claude Code), `.cursor/skills/hyperloom-setup/` (Cursor) and
  `.agents/skills/hyperloom-setup/` (Cursor/Codex); restart the agent if needed.
- `ImportError: libamdhip64.so.7` or `libhipblas.so.3` means the installed
  framework torch wheel expects different ROCm user-space libraries; align
  `ROCM_PATH` and `LD_LIBRARY_PATH`.
- `hipDeviceAttributePciChipId` missing during AITER build means `hipcc` is
  using older ROCm headers; put the matching ROCm `bin` first on `PATH`.

## Source Checkout / Manual Path

Use this path only when developing Hyperloom, testing local source changes, or
debugging setup internals. Customers should prefer the wheel install above.

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
prompt, filling in your workload:

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

Use a ROCm image that already ships the serving framework, so nothing is
installed inside the container beyond Hyperloom's runtime deps:

- `vllm`: `docker.io/primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix`
- `sglang` MI300X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix`
- `sglang` MI355X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix`

Start a long-running container from the repo root, mounting it at the same path
so `.env`, logs, and session artifacts stay valid:

```bash
export HYPERLOOM_IMAGE=docker.io/primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix
docker run -d \
  --name "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" \
  --shm-size "${HYPERLOOM_SHM_SIZE:-64g}" \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -v "$PWD:$PWD" \
  "$HYPERLOOM_IMAGE" \
  tail -f /dev/null
```

Then run all Hyperloom commands inside that container with
`docker exec -w "$PWD" "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" ...`,
using `PYTHONPATH="$PWD/src"` so the source checkout is importable. Use the same
launch prompt as bare metal above.
