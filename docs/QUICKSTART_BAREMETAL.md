# Quickstart — Bare-Metal (No Docker)

Bare-Metal mode installs and runs Hyperloom **directly on an AMD GPU host that already provides the ROCm base** (ROCm runtime + a ROCm-built torch), with the serving framework either preinstalled or installed by this script — **no Docker required**. It suits hosts that already have a ROCm environment and do not want an extra container layer.

The installer runs: base preflight → optional SGLang/vLLM framework install → resolve credentials → clone dependencies → install runtime → write a single combined env file → verify. **It stops before launching — it never starts optimization automatically.**

---

## Prerequisites

- An AMD GPU host with **ROCm** runtime and a **ROCm-built torch** already installed (**MI300X** or **MI355X**).
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

```env
SAFE_API_KEY=ak-your-safe-apikey
OPENAI_BASE_URL=https://global.primus-safe.amd.com/api/v1/llm-proxy/v1
```

- **Split (native OpenAI / Anthropic)** — uncomment the matching lines in the template, fill them in, and point `OPENAI_BASE_URL` at the GPT-side endpoint:

```env
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-xxxxx
```

> To change the artifact directory, set `USER_DATA_PATH` in `.env` (default `/workspace/hyperloom`). See [Auth & Environment Guide](ENV_AND_AUTH.md) for credentials and [Path environment](ENV_AND_AUTH.md#4-path-environment) for path variables.

## 3. Run the bare-metal installer

```bash
inference_optimizer/scripts/install_baremetal.sh
```

The script reads credentials from `.env` automatically. Common options:

| Option | Description |
|--------|-------------|
| `--install-framework sglang\|vllm` | Install a missing SGLang or vLLM ROCm framework layer (default `none`: verify only, do not install). |
| `--framework-env isolated` | Install vLLM into an isolated venv (default `/opt/hyperloom/vllm-venv`) to avoid mutating the shared ROCm torch stack (vLLM only). |
| `--dry-run` | Print the planned actions without changing anything. |
| `--check-only` | Verify the environment only; do not clone or install. |

> Run `--dry-run` first to preview the actions. The framework layer only installs packages + dependencies + a ROCm check — it does **not** patch framework source.

## 4. Load the environment and launch

When the install finishes, the script writes a single combined env file and prints its **actual path**. Copy and run the `source ...` command printed by the installer before launching:

```bash
source '<path printed by install_baremetal.sh>'
```

The default location is usually `/workspace/hyperloom/runtime/hyperloom.env.sh`, but use the printed path if you changed `USER_DATA_PATH` in `.env` or passed `--user-data-path`.

Then open this repo as the workspace in Cursor, paste the prompt the script prints into Cursor Chat, and fill in your workload (referencing `@inference_optimizer/SKILL.md`). The workload fields map to the same CLI flags as Local Mode — see [Launch Inference Optimization](QUICKSTART_LOCAL_MODE.md#launch-inference-optimization).
