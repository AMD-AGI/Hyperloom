# Hyperloom vLLM Docker — Quick Start

Hyperloom local-mode image on the vLLM ROCm base for **multimodal (VL)
benchmarking** on MI355X.  Magpie, InferenceX, `inference_optimizer`, and
`rocprof-compute` are baked in.  Your working checkout is mounted at runtime —
no rebuild needed when you edit the branch.

---

## 1 — Build

```bash
# From the repo root, on the GPU host
ssh-add $HOME/.ssh/id_amd          # key must be loaded in the agent
ssh -T git@github.com               # verify AMD-AGI access

DOCKER_BUILDKIT=1 docker build --ssh default \
  -t hyperloom-vl-vllm-local-$USER .
```

Pin the Magpie ref for a reproducible build (default tracks `main`):

```bash
DOCKER_BUILDKIT=1 docker build --ssh default \
  --build-arg MAGPIE_REF=e1be639 \
  -t hyperloom-vl-vllm-local-$USER .
```

---

## 2 — Run

```bash
docker run -d \
  --name hyperloom-vl-vllm-local-$USER \
  --user $(id -u):$(id -g) \
  --group-add video \
  --group-add render \
  --shm-size 64g \
  --device /dev/kfd \
  --device /dev/dri \
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

`--user $(id -u):$(id -g)` runs as your host UID/GID so files written inside the
container (session dirs, result JSONs, runtime env files) are owned by you, not root.

`-v $PWD:/workspace/Hyperloom` mounts your checkout so edits are live without rebuilding.

---

## 3 — Bootstrap (once per container)

Ray, TraceLens, and GEAK need GPU access, so they are installed at runtime:

```bash
docker exec -it hyperloom-vl-vllm-local-$USER bash

# Inside the container — takes ~3 min on first run
bash inference_optimizer/scripts/install.sh
source /workspace/hyperloom/runtime/kernel-agent.env.sh
```

---

## 4 — Run a VL benchmark

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

`DATASET=random-mm` activates the multimodal path — Hyperloom switches
Magpie's benchmark script to `vllm_mi355x_mm.sh`, which runs:

```
vllm bench serve --dataset-name random-mm --backend openai-chat ...
```

### Key VL env vars

| Variable | Description | Default |
|----------|-------------|---------|
| `DATASET=random-mm` | Activates multimodal mode | — |
| `IMAGE_HEIGHT` | Synthetic image height (px) | `512` |
| `IMAGE_WIDTH` | Synthetic image width (px) | `512` |
| `MM_MAX_IMAGES` | Images per request | `1` |
| `MODEL` | HF model id or local path under `/models` | required |
| `GPU_TYPE` | `mi355x` / `mi300x` / `mi350x` | required |
| `TP` | Tensor parallel size | required |
| `CONC` | Concurrent requests | required |
| `ISL` / `OSL` | Input / output sequence length | required |
| `MAX_MODEL_LEN` | vLLM `--max-model-len` | `4096` |

Results land in `$USER_DATA_PATH/` as `inferencex_result.json`.

---

## 5 — Claude Code

```bash
docker exec -it hyperloom-vl-vllm-local-$USER bash
claude --dangerously-skip-permissions
```

`IS_SANDBOX=1` is set in the image — no extra flags needed.

---

## Verify the image

```bash
# mm script baked in
docker run --rm --entrypoint bash hyperloom-vl-vllm-local-$USER -c \
  "ls /tmp/hyperloom/open-source-repos/Magpie/Magpie/scripts/benchmark/vllm_mi355x_mm.sh"

# Python deps importable
docker run --rm --entrypoint bash hyperloom-vl-vllm-local-$USER -c \
  "python3 -c 'import Magpie, inference_optimizer; print(\"OK\")'"
```

---

## What is baked vs. runtime

| Component | Baked | Runtime (`install.sh`) |
|-----------|:-----:|:----------------------:|
| Magpie | ✓ | |
| InferenceX | ✓ | |
| `inference_optimizer` | ✓ | |
| `rocprof-compute` | ✓ | |
| Claude Code CLI | ✓ | |
| Ray + ray head | | ✓ |
| TraceLens | | ✓ |
| GEAK + RAG index | | ✓ |
| `kernel-agent.env.sh` | | ✓ |

For the full env reference see [`docs/ENV_AND_AUTH.md`](docs/ENV_AND_AUTH.md).
For the full VL benchmark details see [`docs/VLLM_DOCKER.md`](docs/VLLM_DOCKER.md).
