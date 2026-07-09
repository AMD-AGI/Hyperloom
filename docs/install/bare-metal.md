# Quickstart — Bare-Metal (No Docker)

Bare-Metal mode installs and runs Hyperloom **directly on an AMD GPU host that already provides the ROCm base** (ROCm runtime + a ROCm-built torch), with the serving framework either preinstalled or installed by this script — **no Docker required**. It suits hosts that already have a ROCm environment and do not want an extra container layer.

The installer runs: base preflight → optional SGLang/vLLM framework install → resolve credentials → clone dependencies → install runtime → write a single combined env file → verify. **It stops before launching — it never starts optimization automatically.**

---

## Prerequisites

- An AMD GPU host with **ROCm** runtime and a **ROCm-built torch** already installed (**MI300X / MI308X / MI325X / MI355X**; bare-metal preflight maps gfx942 uniformly to MI300X).
- The ROCm user-space libraries, `hipcc` toolchain, torch HIP version, and framework wheel target must be aligned. For example, an `amd-sglang[rocm700]` stack needs ROCm 7.0 user-space libraries and headers on `LD_LIBRARY_PATH` / `ROCM_PATH`, even if the host kernel driver is older but compatible.
- GitHub authentication and AMD-AGI repository access (the bundled `local_setup.sh` reuses it to clone dependency repos).

---

## 1. Get the code

```bash
git clone https://github.com/AMD-AGI/Hyperloom.git && cd Hyperloom
```

## 2. Configure `.env`

Copy the template and fill in your LLM credentials — this is the only thing you configure by hand:

```bash
cp .env.template .env
```

Edit `.env` and pick one of the two setups:

- **Single gateway (default)** — fill in the two ready-to-use lines:

```text
SAFE_API_KEY=ak-your-safe-apikey
OPENAI_BASE_URL=https://global.primus-safe.amd.com/api/v1/llm-proxy/v1
```

- **Split (native OpenAI / Anthropic)** — uncomment the matching lines in the template, fill them in, and point `OPENAI_BASE_URL` at the GPT-side endpoint:

```text
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-xxxxx
```

The installer and its chained runtime installers accept either the single-gateway pair or a split setup as long as at least one base URL and one API key are present.

> To change the artifact directory (default `/workspace/hyperloom`), `export USER_DATA_PATH=...` in your shell before running the installer, or pass `--user-data-path`. The bare-metal installer reads `USER_DATA_PATH` from the shell environment, **not** from `.env` (only LLM credentials are read from `.env`). See [Authentication and credentials](../reference/authentication.md) for credentials and [Path environment](../reference/authentication.md#path-environment) for path variables.

## 3. Run the bare-metal installer

```bash
src/hyperloom/inference_optimizer/assets/install_baremetal.sh
```

The script reads credentials from `.env` automatically. Common options and environment overrides:

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

> Run `--dry-run` first to preview the actions. The framework layer only installs packages + dependencies + a ROCm check — it does **not** patch framework source.

### ROCm 7.0 user-space on an older compatible host driver

If the host driver cannot be restarted or upgraded but supports ROCm 7.0 user-space, install ROCm 7.0 user-space libraries side by side and point the installer at them. This path requires Python 3.10 because the installer uses the AMD `amd-sglang[rocm700]` wheel; non-3.10 Python would take the source-install path and is rejected to avoid pulling mismatched ROCm 7.2 Triton.

```bash
export PYTHON=/opt/hyperloom/venv-rocm70/bin/python
export ROCM_PATH=/opt/rocm-7.0.2
export HIP_PATH=/opt/rocm-7.0.2
export PATH=/opt/rocm-7.0.2/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.0.2/lib:$LD_LIBRARY_PATH
export SGLANG_ROCM_PYPI_VERSION=7.0.0
export SGLANG_ROCM_EXTRA=rocm700

src/hyperloom/inference_optimizer/assets/install_baremetal.sh \
  --install-framework sglang
```

The `rocm700` SGLang stack uses `triton 3.5.x`, so the installer chooses an older compatible AITER tag unless `AITER_REF` is set explicitly.

## 4. Load the environment and launch

When the install finishes, the script writes a single combined env file and prints its **actual path**. Copy and run the `source ...` command printed by the installer before launching:

```bash
source '<path printed by install_baremetal.sh>'
```

The default location is usually `/workspace/hyperloom/runtime/hyperloom.env.sh`, but use the printed path if you exported a custom `USER_DATA_PATH` or passed `--user-data-path`.

Then open this repo as the workspace in Cursor, paste the prompt the script prints into Cursor Chat, and fill in your workload (referencing `@src/hyperloom/inference_optimizer/SKILL.md`). The workload fields map to the same CLI flags as Local Mode — see [Run a Hyperloom optimization](../how-to/optimize.md).

## Troubleshooting

- `ImportError: libamdhip64.so.7` or `libhipblas.so.3`: the framework torch wheel expects ROCm 7 user-space libraries, but the process only sees an older `/opt/rocm`. Install matching ROCm user-space libraries and set `LD_LIBRARY_PATH`.
- `hipDeviceAttributePciChipId` missing during AITER build: `hipcc` is using older ROCm headers. Set `ROCM_PATH`, `HIP_PATH`, and put the matching ROCm `bin` directory first on `PATH`.
- `aiter gluon kernels require triton>=3.6.0`: the selected AITER tag is too new for the selected framework stack. Use the default `AITER_REF` for the selected `SGLANG_ROCM_EXTRA`, or set a compatible tag explicitly.
- `no such option: --break-system-packages`: a chained script is using an old system pip. Use a venv and set `PYTHON=/path/to/venv/bin/python`; the installer propagates `VIRTUAL_ENV` to chained scripts.
- GEAK RAG index build fails or segfaults during install: set `KERNEL_AGENT_BUILD_GEAK_RAG_INDEX=0` to skip it. The core install can still complete and the index can be built later.
