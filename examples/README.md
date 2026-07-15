# Hyperloom Setup and Examples

This README is the main entry point for setting up Hyperloom and launching the
Qwen3 demo skills. The recommended customer path is a packaged install with
`pip install --target`; the source-clone path is kept at the end for developers
and manual debugging.

## Recommended Path: Install the Wheel

Use this path for customer demos and clean validation. It installs Hyperloom into
one target directory, and that directory is also the agent workspace you open in
Cursor, Claude Code, or Codex.

### Prerequisites

- Python 3.10+ and `pip`.
- Access to one LLM provider: Anthropic or DeepSeek.
- For private Hyperloom releases, `gh` access to the GitHub release asset.

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

The setup skill is interactive. It creates `.env`, records the selected run
scenario, and stops before launching an optimization.

It asks for:

- LLM mode: Anthropic or DeepSeek.
- Non-secret LLM settings: base URL and model.
- Secret placeholders in `.env`; edit secrets directly in `.env` and never
  paste API keys into chat.
- `USER_DATA_PATH` (defaults to `<workspace>/session`).
- Setup scenario: `baremetal` or `baremetal + Docker`.

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

The setup backend does **not** install Magpie, InferenceX, TraceLens, or GEAK.
Those runtime dependencies are installed later by the workload/demo skill through
`install.sh`, just before launching an optimization.

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

In this scenario, `/hyperloom-setup` writes `.env` only and skips host setup. It
does **not** start a container. The selected Qwen3 demo skill later starts the
container, runs setup inside it with `--install-framework none --yes`, installs
runtime dependencies with `install.sh`, and launches the optimization.

If Slurm is available, setup also checks for allocated nodes. The user chooses
whether Docker should run on:

- the current host;
- one of the current user's allocated Slurm nodes; or
- a custom host.

The chosen host is written to `.env`:

```bash
HYPERLOOM_DOCKER_TARGET_HOST=<hostname>
```

The demo skill reads this value. If it is not the current host, the demo skill
SSHes to that host before running `docker run` and `docker exec`.

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
- `HYPERLOOM_DOCKER_TARGET_HOST` (only for `baremetal + Docker`)

Bare-metal setup may also write runtime vars such as `FRAMEWORK`, `ROCM_PATH`,
`VIRTUAL_ENV`, and `VLLM_VENV_ROOT`. Kernel-agent paths (`MAGPIE_PATH`,
`INFERENCEX_PATH`, `TRACELENS_ROOT`, `GEAK_ROOT`) are added later by the workload
skill's `install.sh`.

`.env` is the single source of truth: no combined script is generated and nothing
extra needs sourcing. `PATH` and `LD_LIBRARY_PATH` are derived at launch from
`ROCM_PATH`, `VIRTUAL_ENV`, and `VLLM_VENV_ROOT`.

## Run a Qwen3 Demo

When setup finishes in `baremetal` mode (and `FRAMEWORK` is set), or when `.env`
is written in `baremetal + Docker` mode, the setup skill offers a model demo run
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

In source mode, the agent workspace is the repository root. You must create
`.env` yourself or use the setup skill from the packaged install. A minimal
manual `.env` for Anthropic looks like:

```bash
cat > .env <<'EOF'
ANTHROPIC_API_KEY=<PLEASE_FILL_IN>
ANTHROPIC_BASE_URL=https://api.anthropic.com
CLAUDE_MODEL=claude-opus-4-8
USER_DATA_PATH=/workspace/hyperloom
HYPERLOOM_RUN_MODE=baremetal
EOF
```

For bare-metal source debugging, run the setup backend manually:

```bash
PYTHONPATH="$PWD/src" python3 -m hyperloom.inference_optimizer.setup \
  -- --install-framework none
```

For Docker source debugging, write `.env` on the host, start the container from
the matching demo skill, then run the backend inside the container:

```bash
docker exec -w "$PWD" "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" bash -lc \
  'PYTHONPATH="$PWD/src" python3 -m hyperloom.inference_optimizer.setup -- --install-framework none --yes'
```

Runtime dependencies are still installed later by `install.sh`:

```bash
bash src/hyperloom/inference_optimizer/assets/install.sh
```

## Manual Dry Run

To test the backend without running the agent skill:

```bash
cd ~/hyperloom

cat > .env <<'EOF'
ANTHROPIC_API_KEY=<PLEASE_FILL_IN>
ANTHROPIC_BASE_URL=https://api.anthropic.com
CLAUDE_MODEL=claude-opus-4-8
USER_DATA_PATH=/root/hyperloom
EOF

PYTHONPATH="$PWD" python3 -m hyperloom.inference_optimizer.setup \
  --dry-run -- --skip-base-check --install-framework none
```

The dry run should print the five bare-metal phases without mutating the system.

## Manual Setup Options

The setup skill passes options to the packaged backend. Useful options include:

| Option / environment variable | Description |
|-------------------------------|-------------|
| `--install-framework none` | Use an already-installed SGLang/vLLM stack. |
| `--install-framework sglang` | Install SGLang ROCm framework components. |
| `--install-framework vllm` | Install vLLM ROCm framework components. |
| `--framework-env isolated` | Install vLLM into an isolated venv. |
| `--dry-run` | Print planned actions without changing the system. |
| `--check-only` | Verify only; do not clone or install. |
| `--skip-base-check` | Skip ROCm/framework preflight. Useful only for setup-chain debugging. |
| `ROCM_PATH` / `HIP_PATH` | Point source builds at matching ROCm compiler and headers. |
| `LD_LIBRARY_PATH` | Make matching ROCm user-space libraries visible. |
| `SGLANG_ROCM_EXTRA` | Select AMD SGLang ROCm extra, for example `rocm720`. |
| `AITER_REF` | Override AITER source tag; otherwise setup auto-selects a compatible tag. |
