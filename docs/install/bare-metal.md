# Quickstart — Bare-Metal (No Docker)

Bare-Metal mode installs and runs Hyperloom **directly on an AMD GPU host that already provides the ROCm base** (ROCm runtime + a ROCm-built torch), with the serving framework either preinstalled or installed by this script — **no Docker required**. It suits hosts that already have a ROCm environment and do not want an extra container layer.

The installer runs: base preflight → optional SGLang/vLLM framework install → resolve credentials → clone dependencies → install runtime → write a single combined env file → verify. **It stops before launching — it never starts optimization automatically.**

---

## Prerequisites

- An AMD GPU host with **ROCm** runtime and a **ROCm-built torch** already installed (**MI300X / MI308X / MI325X / MI355X**; bare-metal preflight maps gfx942 uniformly to MI300X).
- The ROCm user-space libraries, `hipcc` toolchain, torch HIP version, and framework wheel target must be aligned. For example, an `amd-sglang[rocm700]` stack needs ROCm 7.0 user-space libraries and headers on `LD_LIBRARY_PATH` / `ROCM_PATH`, even if the host kernel driver is older but compatible.
- GitHub authentication for the Hyperloom checkout. AMD-AGI KernelForge access
  is only required when you explicitly choose the `forge` kernel backend.

---

## 1. Get the code

For a full source checkout:

```bash
git clone https://github.com/AMD-AGI/Hyperloom.git && cd Hyperloom
```

For a standalone/customer install, copy only `install_baremetal.sh` onto the
host. If the sibling installer scripts are not present, it automatically enters
wheel mode, downloads the Hyperloom release wheel with `gh`, installs it into
`$PYTHON`, and runs the packaged installer assets from site-packages:

```bash
gh auth login
./install_baremetal.sh --install-framework sglang
```

Set `HYPERLOOM_WHEEL=/path/to/hyperloom_inference_optimizer-0.8.0-py3-none-any.whl`
to use a pre-downloaded wheel and skip `gh`.

In wheel mode, the installer prints the installed `SKILL.md` path from
site-packages. Copy that printed `@.../SKILL.md` line into Cursor Chat when
launching an optimization; you do not need to locate it manually.

## 2. Configure LLM Credentials

Export one LLM credential setup before running the installer. The installer validates credentials up front because GEAK/kernel-agent configuration needs them later.

```bash
# Single gateway (default)
export SAFE_API_KEY=ak-your-safe-apikey
export OPENAI_BASE_URL=https://your-openai-compatible-gateway.example.com/v1
```

Or use a split provider setup with separate Anthropic-compatible and OpenAI-compatible entrypoints:

```bash
export ANTHROPIC_BASE_URL=https://api.anthropic.com
export ANTHROPIC_API_KEY=sk-ant-xxxxx
export CLAUDE_MODEL=claude-opus-4-7

export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-xxxxx
export CODEX_MODEL=gpt-4.1

export INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1
```

If you are running from a full source checkout, you can alternatively copy the template and fill in the same values:

```bash
cp .env.template .env
```

The installer and its chained runtime installers accept either the single-gateway pair or a split setup as long as at least one base URL and one API key are present.

> To change the artifact directory (default `/workspace/hyperloom`), `export USER_DATA_PATH=...` in your shell before running the installer, or pass `--user-data-path`. The bare-metal installer reads `USER_DATA_PATH` from the shell environment, **not** from `.env` (only LLM credentials are read from `.env`). See [Authentication and credentials](../reference/authentication.md) for credentials and [Path environment](../reference/authentication.md#path-environment) for path variables.

Bare-metal installs use `KERNEL_OPT_BACKEND_ORDER=geak` by default. To use
the forge backend instead, export an explicit backend order before running the
installer:

```bash
export KERNEL_OPT_BACKEND_ORDER=forge
```

Only the forge backend requires KernelForge access. The installer still
performs the standard LLM/runtime setup for the selected backend.

## 3. Run the bare-metal installer

Use the installer path that matches how you obtained it:

```bash
# Full source checkout
src/hyperloom/inference_optimizer/assets/install_baremetal.sh

# Standalone/customer install
./install_baremetal.sh
```

The script resolves credentials from flags, shell environment, and `.env` when present. Common options and environment overrides:

| Option / environment variable | Description |
|-------------------------------|-------------|
| `--install-framework sglang\|vllm` | Install a missing SGLang or vLLM ROCm framework layer (default `none`: verify only, do not install). |
| `--framework-env isolated` | Install vLLM into an isolated venv (default `/opt/hyperloom/vllm-venv`) to avoid mutating the shared ROCm torch stack (vLLM only). |
| `--dry-run` | Print the planned actions without changing anything. |
| `--check-only` | Verify the environment only; do not clone or install. |
| `PYTHON=/path/to/venv/bin/python` | Use a specific Python environment. If it is a venv, the installer exports `VIRTUAL_ENV` for chained scripts. |
| `ROCM_PATH=/opt/rocm-<version>` / `HIP_PATH=...` | Point AITER and source builds at the matching ROCm compiler and headers. |
| `LD_LIBRARY_PATH=/opt/rocm-<version>/lib:$LD_LIBRARY_PATH` | Make the matching ROCm user-space libraries visible to torch and framework extensions. |
| `SGLANG_ROCM_PYPI_VERSION=7.0.0` | Select the AMD SGLang wheel repository version (default `7.2.0`). |
| `SGLANG_ROCM_EXTRA=rocm700` | Select the AMD SGLang ROCm extra (default `rocm720`). The `rocm700` path is supported only with Python 3.10 AMD wheels. |
| `AITER_REF=v0.1.14.post1` | Override the AITER source tag. By default, `rocm700` selects `v0.1.14.post1`; `rocm720` selects `v0.1.16.post3`. |
| `KERNEL_AGENT_BUILD_GEAK_RAG_INDEX=0` | Skip the install-time GEAK RAG index build. Useful for smoke tests or hosts where model/index build is flaky. |
| `KERNEL_AGENT_RAG_INDEX_STRICT=1` | Fail the install if the GEAK RAG index build fails. By default the installer warns and continues. |
| `HYPERLOOM_INSTALL_SOURCE=wheel` | Force standalone wheel mode even when running from a source checkout. |
| `HYPERLOOM_WHEEL=/path/to/file.whl` | Install Hyperloom from a local wheel or reachable URL instead of downloading with `gh`. |
| `HYPERLOOM_WHEEL_REPO=AMD-AGI/Hyperloom` | GitHub repo used for `gh release download` in wheel mode. |
| `HYPERLOOM_WHEEL_TAG=v0.8` | Release tag used for `gh release download` in wheel mode. |
| `HYPERLOOM_WHEEL_PATTERN=hyperloom_inference_optimizer-*.whl` | Release asset pattern used for `gh release download` in wheel mode. |

> Run `--dry-run` first to preview the actions. The framework layer only installs packages + dependencies + a ROCm check — it does **not** patch framework source.

## 4. Launch with the printed environment

When the install finishes, the script writes a single combined env file and prints the exact `source ...` command to run. Copy that printed command into your current shell before launching:

```bash
source '<path printed by install_baremetal.sh>'
```

The default location is usually `/workspace/hyperloom/runtime/hyperloom.env.sh`, but use the printed path if you exported a custom `USER_DATA_PATH` or passed `--user-data-path`.

Then open the indicated workspace in Cursor, paste the prompt the installer prints into Cursor Chat, and fill in your workload. The workload fields map to the same CLI flags as Local Mode — see [Run a Hyperloom optimization](../how-to/optimize.md).

## Troubleshooting

- `ImportError: libamdhip64.so.7` or `libhipblas.so.3`: the framework torch wheel expects ROCm 7 user-space libraries, but the process only sees an older `/opt/rocm`. Install matching ROCm user-space libraries and set `LD_LIBRARY_PATH`.
- `hipDeviceAttributePciChipId` missing during AITER build: `hipcc` is using older ROCm headers. Set `ROCM_PATH`, `HIP_PATH`, and put the matching ROCm `bin` directory first on `PATH`.
- `aiter gluon kernels require triton>=3.6.0`: the selected AITER tag is too new for the selected framework stack. Use the default `AITER_REF` for the selected `SGLANG_ROCM_EXTRA`, or set a compatible tag explicitly.
- `no such option: --break-system-packages`: a chained script is using an old system pip. Use a venv and set `PYTHON=/path/to/venv/bin/python`; the installer propagates `VIRTUAL_ENV` to chained scripts.
- GEAK RAG index build fails or segfaults during install: set `KERNEL_AGENT_BUILD_GEAK_RAG_INDEX=0` to skip it. The core install can still complete and the index can be built later.
