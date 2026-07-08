---
myst:
    html_meta:
        "description": "Install Hyperloom in Local Mode (Docker) or bare-metal on an AMD GPU environment. Covers prerequisites, cloning, bootstrapping dependency checkouts, and installing runtime dependencies."
        "keywords": "Hyperloom, installation, AMD GPU, Local Mode, Docker, bare-metal, Cursor, ROCm, LLM inference, GEAK, TraceLens, Magpie, SGLang, vLLM"
---
# Install Hyperloom

Choose the mode that fits your environment:

- **Local Mode (Docker)** — Hyperloom runs in a Docker container on your AMD
  GPU machine; Cursor attaches to the container and launches optimization. This
  is the recommended path. Follow the steps below.
- **Bare-metal (no Docker)** — install directly on a ROCm host that already
  provides the runtime. See the
  [bare-metal quickstart](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/QUICKSTART_BAREMETAL.md)
  in the repository.
- **Hosted UI** — no local GPU required. See
  [Quickstart — hosted UI](quickstart.md) instead.

## Prerequisites

- Docker.
- An MI300X, MI325X, or MI355X machine with ROCm-compatible Docker
  access (see [Hyperloom compatibility matrix](../compatibility.md)).
- Cursor, connected to that machine using Remote SSH or Dev Containers.
- An API key from the [LLM Gateway](https://llm.amd.com/). This key provides
  access to TraceLens, GEAK, and OOB services.

## 1. Clone and bootstrap

Clone the repository and run the bootstrap script. This clones or updates
dependency checkouts and writes the `local-setup.env.sh` file used in the next
step.

```bash
git clone https://github.com/AMD-AGI/Hyperloom.git
cd Hyperloom
cp .env.template .env   # then edit credentials (SAFE_API_KEY, OPENAI_BASE_URL)
export USER_DATA_PATH=/path/to/hyperloom-run
bash inference_optimizer/scripts/local_setup.sh
```

## 2. Install runtime dependencies

Run this in the same shell that will start `inference_optimizer optimize`.
This installs Magpie, TraceLens, GEAK, Ray, and CLI auth/config files.

```bash
source "$USER_DATA_PATH/runtime/local-setup.env.sh"
bash inference_optimizer/scripts/install.sh
source "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
```

## Optional — quantization (AMD Quark)

Use `--quantize` when you want to optimize a model that is not already available
in a quantized format (FP8 or MXFP4). Quantizing reduces VRAM consumption and
typically increases throughput on AMD Instinct hardware. The rest of Hyperloom
works without it — only set this up if you need the quantization prelude.

The `--quantize` prelude drives [AMD Quark](https://quark.docs.amd.com/) to
produce a quantized model before the optimization loop runs. Hyperloom does not
bundle Quark — it invokes Quark's published skills end-to-end.

```{note}
The public PyPI package (`pip install amd-quark`) does not include the
`.claude/skills/quark-torch-*` skill entry points that Hyperloom drives. You
must use an internal AMD Quark repository checkout until those skills are
bundled in a public release.
```

Hyperloom resolves the Quark root in this order:

1. The `--quark-root` CLI flag
2. The `QUARK_ROOT` environment variable
3. The built-in default (Core42 only): `/wekafs/hyperloom/Quark`

Outside Core42, set `QUARK_ROOT` explicitly. The resolved path must contain
`.claude/skills/quark-torch-ptq/SKILL.md`. If none of the above resolves to an
existing directory, the run fails fast with `quark_root_missing` rather than
silently optimizing the un-quantized model.

Add this to your `.env` when your checkout lives elsewhere:

```bash
# Only needed for the --quantize prelude
QUARK_ROOT=/workspace/Quark
```

## Related guides

- [Run a Hyperloom optimization](../how-to/optimize.md)
- [Environment variables](../reference/environment-variables.md)
- [Troubleshooting Hyperloom](../reference/troubleshooting.md)
