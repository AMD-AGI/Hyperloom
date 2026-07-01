# Hyperloom vLLM Docker Image — Usage Guide

A Hyperloom local-mode image built on the vLLM ROCm base, designed for
**multimodal (VL) benchmarking** on MI355X using `vllm bench serve` with
synthetic image inputs (`--dataset-name random-mm`).

---

## Prerequisites

- SSH key for AMD-AGI GitHub repos loaded in your agent:
  ```bash
  ssh-add $HOME/.ssh/id_amd
  ssh -T git@github.com   # verify access
  ```
- `SAFE_API_KEY` — your AMD LiteLLM gateway key (`ak-...`). Obtain from
  [core42.example-internal-host.invalid/litellm-gateway](https://core42.example-internal-host.invalid/litellm-gateway).
- `ANTHROPIC_CUSTOM_HEADERS` — your AMD LLM API subscription key, in the form
  `Ocp-Apim-Subscription-Key: <your-key>`.

---

## Build

```bash
cd Hyperloom   # repo root (feat/vl-model-support branch)

DOCKER_BUILDKIT=1 docker build --ssh default \
  -t hyperloom-vl-vllm-local-$USER .
```

**Build ARG overrides** (all optional):

| ARG | Default | Override example |
|-----|---------|-----------------|
| `BASE_IMAGE` | `vllm/vllm-openai-rocm:v0.23.0` | `--build-arg BASE_IMAGE=vllm/vllm-openai-rocm:v0.24.0` |
| `MAGPIE_REF` | `main` | `--build-arg MAGPIE_REF=e1be639` (pin for reproducibility) |

> **Note:** `MAGPIE_REF=main` is intentional — the multimodal benchmark script
> `vllm_mi355x_mm.sh` was added to Magpie after the default Hyperloom pin
> (`b1d4dcd`). Pass a specific SHA to lock the build for reproducibility.

---

## Run

```bash
docker run -d \
  --name hyperloom-vl-vllm-local-$USER \
  --shm-size 64g \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -v $PWD:/workspace/Hyperloom \
  -v /data2/hf_hub_cache:/models \
  -v "$SSH_AUTH_SOCK:/ssh-agent" \
  -e SSH_AUTH_SOCK=/ssh-agent \
  -e USER_DATA_PATH=/workspace/hyperloom \
  -e SAFE_API_KEY="$SAFE_API_KEY" \
  -e OPENAI_BASE_URL="https://core42.example-internal-host.invalid/api/v1/llm-proxy/v1" \
  -e ANTHROPIC_CUSTOM_HEADERS="$ANTHROPIC_CUSTOM_HEADERS" \
  hyperloom-vl-vllm-local-$USER \
  tail -f /dev/null
```

The `-v $PWD:/workspace/Hyperloom` mount overlays your working checkout on
top of the build-time clone, so edits to the branch are picked up immediately
without rebuilding the image.

---

## Bootstrap (first start)

The image bakes Magpie, InferenceX, and bench deps, but kernel-agent (Ray,
TraceLens, GEAK) was skipped at build time because it requires GPU access.
Run the full installer once inside the container:

```bash
docker exec -it hyperloom-vl-vllm-local-$USER bash

# Inside the container:
bash inference_optimizer/scripts/install.sh
source /workspace/hyperloom/runtime/kernel-agent.env.sh
```

This is only needed once per container lifetime. The runtime env file
(`kernel-agent.env.sh`) is written to `USER_DATA_PATH/runtime/` and
persists across restarts as long as the volume is not removed.

---

## Run a multimodal benchmark

Set workload parameters and run the CLI:

```bash
docker exec -it hyperloom-vl-vllm-local-$USER bash -c "
  source /workspace/hyperloom/runtime/kernel-agent.env.sh

  MODEL=Qwen/Qwen2-VL-7B-Instruct \
  FRAMEWORK=vllm \
  GPU_TYPE=mi355x \
  DATASET=random-mm \
  TP=8 \
  CONC=16 \
  ISL=512 \
  OSL=128 \
  MAX_MODEL_LEN=8192 \
  python -m inference_optimizer optimize
"
```

**Key env vars for VL benchmarking:**

| Variable | Description | Default |
|----------|-------------|---------|
| `DATASET` | Set to `random-mm` to activate multimodal mode | — |
| `IMAGE_HEIGHT` | Synthetic image height (px) | `512` |
| `IMAGE_WIDTH` | Synthetic image width (px) | `512` |
| `MM_MAX_IMAGES` | Images per request | `1` |
| `MODEL` | HuggingFace model id or local path | required |
| `FRAMEWORK` | Must be `vllm` for mm mode | required |
| `GPU_TYPE` | `mi355x` (or `mi300x`, `mi350x`) | required |
| `TP` | Tensor parallel degree | required |
| `CONC` | Concurrent requests | required |
| `ISL` / `OSL` | Input / output sequence length | required |
| `MAX_MODEL_LEN` | vLLM `--max-model-len` | `4096` |

When `DATASET=random-mm` is set, Hyperloom automatically switches the Magpie
benchmark script to `vllm_{GPU_TYPE}_mm.sh`, which runs:

```
vllm bench serve --dataset-name random-mm --backend openai-chat ...
```

Results are written to `USER_DATA_PATH/` as `inferencex_result.json`.

---

## Verify the image

```bash
# mm script is present (baked from Magpie main)
docker run --rm --entrypoint bash hyperloom-vl-vllm-local-$USER -c \
  "ls /tmp/hyperloom/open-source-repos/Magpie/Magpie/scripts/benchmark/vllm_mi355x_mm.sh"

# Python deps importable
docker run --rm --entrypoint bash hyperloom-vl-vllm-local-$USER -c \
  "python3 -c 'import Magpie, inference_optimizer; print(\"OK\")'"
```

---

## What is baked vs. runtime

| Component | Baked at build | Installed at runtime |
|-----------|:--------------:|:--------------------:|
| Magpie (from `MAGPIE_REF`) | ✓ | |
| InferenceX | ✓ | |
| `inference_optimizer` package | ✓ | |
| `rocprof-compute` | ✓ | |
| bench serving deps (aiohttp, transformers …) | ✓ | |
| Claude Code CLI | ✓ | |
| Ray (head node) | | ✓ |
| TraceLens | | ✓ |
| GEAK + RAG semantic index | | ✓ |
| `kernel-agent.env.sh` | | ✓ |

Runtime installation is handled by `inference_optimizer/scripts/install.sh`
(see [Bootstrap](#bootstrap-first-start) above).

---

## Claude Code inside the container

The image ships Claude Code pre-installed. To use it:

```bash
docker exec -it hyperloom-vl-vllm-local-$USER bash
claude --dangerously-skip-permissions
```

`IS_SANDBOX=1` is set in the image, which enables `--dangerously-skip-permissions`
as root inside the container.

Override the default model endpoints via `-e` at `docker run` time:
```bash
-e ANTHROPIC_BASE_URL="https://llm-api.amd.com/Anthropic"
-e ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: <your-key>"
```
