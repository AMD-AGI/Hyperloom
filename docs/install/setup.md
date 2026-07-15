# Quickstart - Local Setup

Setup runs on an AMD GPU host that already has a ROCm base installed (ROCm
runtime + ROCm torch); Hyperloom does not install ROCm itself. It supports two
run modes:

- `baremetal`: run directly on the host. Setup can optionally install the
  SGLang or vLLM framework layer.
- `docker`: run inside a ROCm container that ships the serving framework. Setup
  only records the mode in `.env`; the container is generated later by the demo
  (workload) skill, so setup installs no framework here.

Setup then configures runtime dependencies, credentials, and `.env`.

The recommended customer flow is:

1. Install the Hyperloom wheel into a target workspace.
2. Open that workspace in an agent and run `/hyperloom-setup`.
3. Let the setup skill create `.env`, then run the packaged setup backend.

The installer stops before launching an optimization.

## Prerequisites

- An AMD GPU host with ROCm runtime and ROCm torch already installed.
- Python 3.10+ and `pip`.
- `git` for dependency checkouts performed by setup.
- Access to one LLM provider: Anthropic or DeepSeek.

For private Hyperloom releases, the default path is to download the wheel from
GitHub Releases with `gh` first. When Hyperloom is published publicly, this can
be replaced by a direct `pip install` URL or PyPI install.

## 1. Install Hyperloom Into a Workspace

Download the release wheel, then install it into the directory the user will
open in Cursor, Claude Code, or Codex:

```bash
gh auth login
gh release download v0.8 \
  -R AMD-AGI/Hyperloom \
  -p 'hyperloom_inference_optimizer-0.8.0-py3-none-any.whl'

rm -rf ~/hyperloom
mkdir -p ~/hyperloom

python3 -m pip install \
  ./hyperloom_inference_optimizer-0.8.0-py3-none-any.whl \
  --target ~/hyperloom
```

The target directory is both the Python install target and the agent workspace.
It is normal for it to contain many Python package directories. Users only need
to open the directory and run the setup skill.

## 2. Run `/hyperloom-setup`

Open `~/hyperloom` in the user's agent and run:

```text
/hyperloom-setup
```

The setup skill is interactive. It:

- asks which LLM mode to use: Anthropic or DeepSeek;
- asks the run mode: `baremetal` (run on this host) or `docker` (record the
  mode only; the demo skill generates the container later);
- creates `.env` with placeholders; you edit secrets directly in `.env` (never
  paste API keys into chat);
- asks for `USER_DATA_PATH` (defaults to `<workspace>/session`);
- in `baremetal` mode, asks whether to install a serving framework: `none`,
  `sglang`, or `vllm`;
- runs the backend with `PYTHONPATH="$PWD" python3 -m hyperloom.inference_optimizer.setup`.

LLM defaults:

| Mode | Required secret | Default base URL | Default model |
|------|-----------------|------------------|---------------|
| Anthropic | `ANTHROPIC_API_KEY` | `https://api.anthropic.com` | `CLAUDE_MODEL=claude-opus-4-8` |
| DeepSeek | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/anthropic` | `DEEPSEEK_MODEL=deepseek-chat` |

During setup, Hyperloom also updates `.env` with runtime paths (`MAGPIE_PATH`,
`INFERENCEX_PATH`, `TRACELENS_ROOT`, `GEAK_ROOT`, `FRAMEWORK`) and the resolved
`HYPERLOOM_RUN_MODE`.

## 3. What Setup Does

The packaged setup backend runs the bare-metal setup phases:

1. **Base preflight**: checks ROCm, GPU arch, ROCm torch, torch/triton alignment,
   and serving framework imports.
2. **Framework install**: optionally installs SGLang or vLLM framework layers.
3. **ROCm hotfix**: applies the profiler hotfix when the ROCm stack is eligible.
4. **Credentials**: validates and persists LLM configuration into `.env`.
5. **Runtime install**: runs packaged `install.sh` to set up Magpie, InferenceX,
   TraceLens, GEAK, Ray, and other runtime dependencies.
6. **Combined env**: writes `runtime/hyperloom.env.sh` and updates `.env`.
7. **Verify**: runs `install.sh --check-only` and prints next steps.

The setup backend no longer downloads or installs the Hyperloom wheel; that is
already done by the `pip install --target` step.

## 4. Run a Demo

When setup finishes and `FRAMEWORK` is set, the setup skill offers a Qwen3-8B
demo run and hands off to the matching demo skill. Pick a length:

- `3h` — short, no-kernel run; best for a first end-to-end check.
- `8h` — medium-length run.
- `24h` — long-horizon cyclic run.

The demo reuses the values already in `.env`, so nothing is re-entered.

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

The dry run should show `Phase 5: install.sh --dry-run` and should not attempt
to download or install the Hyperloom wheel.

## Common Setup Options

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
