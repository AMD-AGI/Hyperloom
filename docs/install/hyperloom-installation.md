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

## Related guides

- [Run a Hyperloom optimization](../how-to/optimize.md)
- [Environment variables](../reference/environment-variables.md)
- [Troubleshooting Hyperloom](../troubleshooting.md)
