# Hyperloom vLLM Docker — Quick Start

Hyperloom local-mode image on the vLLM ROCm base for **multimodal (VL)
benchmarking** on MI355X.  The full stack — Magpie, InferenceX,
`inference_optimizer`, `rocprof-compute`, plus the kernel-agent stack (Ray,
TraceLens, GEAK) — is baked in.  Your working checkout is mounted at runtime —
no rebuild needed when you edit the branch.

> The one thing NOT baked is GEAK's semantic RAG index: it needs the GPU, and
> `docker build` has no GPU access. It builds automatically on first container
> start (~1 min on an AMD GPU).

---

## Getting started on a new machine

The published image already contains everything — you do **not** need to build.
The `run_hyperloom.sh` launcher pulls the image, starts the container with the
right device/network/mount flags, and drops you into a shell.

### 0 — Prerequisites on the host

```bash
# a) AMD GPU host with docker + ROCm devices (/dev/kfd, /dev/dri)

# b) Docker Hub access to the amdsiloai org (the image is private)
docker login                        # as a user with amdsiloai pull access

# c) SSH key loaded (used for git operations inside the container)
ssh-add $HOME/.ssh/id_amd
ssh -T git@github.com               # verify AMD-AGI access

# d) The AMD LLM API tunnel running on the host (for Claude Code).
#    run_hyperloom.sh auto-detects the port from $ANTHROPIC_BASE_URL.
ss -tlnp | grep -E ':[0-9]+' | grep -i llm   # sanity-check something is listening

# e) Your AMD gateway key exported (or in a .env file next to the script)
export SAFE_API_KEY=ak-...
export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: <your-key>"
```

### 1 — Get the launcher script

Only the launcher is needed on the host (the repo itself gets mounted or lives
inside the image). Clone the branch, or just copy `run_hyperloom.sh`:

```bash
git clone -b feat/vl-model-support-dev git@github.com:AMD-AGI/Hyperloom.git
cd Hyperloom
```

### 2 — Start it (pulls image if missing, then execs in)

```bash
./run_hyperloom.sh
```

On a fresh machine this will `docker pull` the ~51 GB image
(`amdsiloai/vllm-private:mlperf6.1-q3vl-r72-w4a4-fusemoe-20260620-hyperloom`),
start the container, and open a shell. Subsequent runs reuse the local image.

Common flags:

```bash
./run_hyperloom.sh --mount ~/workspace/ml-perf   # mount another repo read-only at /mnt/ml-perf
./run_hyperloom.sh --api-port 9444               # override auto-detected AMD API port
./run_hyperloom.sh --image <ref>                 # use a different image ref
./run_hyperloom.sh --build                        # force a local rebuild (needs --ssh + Dockerfile)
./run_hyperloom.sh --stop                         # stop and remove the container
```

The launcher mounts your host `~/.claude` + `~/.claude.json` (so Claude Code
skips onboarding), sets `--network host` so the container reaches the host's AMD
API tunnel, and adds an `/etc/hosts` entry for `llm-api.amd.com`.

---

## Building it yourself (only if you need to change the image)

Most users should just pull. To rebuild from the Dockerfile:

```bash
ssh-add $HOME/.ssh/id_amd          # key must be loaded in the agent
DOCKER_BUILDKIT=1 docker build --ssh default \
  -t hyperloom-vl-vllm-local-$USER .

# Pin Magpie for a reproducible build (default tracks main):
DOCKER_BUILDKIT=1 docker build --ssh default \
  --build-arg MAGPIE_REF=e1be639 \
  -t hyperloom-vl-vllm-local-$USER .
```

`./run_hyperloom.sh --build` does the same and then starts the container.

---

## 3 — Bootstrap (once per container, inside the shell from step 2)

The whole dependency stack (Ray, TraceLens, GEAK) is already baked in. The only
first-run step is building GEAK's RAG index (needs the GPU) and starting the Ray
head — both handled by re-running `install.sh`, which is now near-instant since
everything else is already installed:

```bash
# You are already inside the container (run_hyperloom.sh dropped you into a shell)
# builds the GEAK RAG index (~1 min on GPU) + starts Ray
bash inference_optimizer/scripts/install.sh
source /workspace/hyperloom/runtime/kernel-agent.env.sh
```

> The CLI preflight also auto-builds the index and starts Ray on first
> `inference_optimizer optimize`, so this step is optional if you go straight
> to a benchmark.

---

## 4 — Run a VL benchmark

Inside the container shell:

```bash
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

Inside the container shell:

```bash
claude --dangerously-skip-permissions
```

`IS_SANDBOX=1` is set in the image and your host `~/.claude` is mounted, so it
skips onboarding — no extra flags or login needed. (To open another shell into a
running container from the host: `docker exec -it hyperloom-vl-vllm-local-$USER bash`.)

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

| Component | Baked | Runtime (first start) |
|-----------|:-----:|:---------------------:|
| Magpie | ✓ | |
| InferenceX | ✓ | |
| `inference_optimizer` | ✓ | |
| `rocprof-compute` | ✓ | |
| Claude Code CLI | ✓ | |
| Ray (package) | ✓ | |
| TraceLens | ✓ | |
| GEAK code + rag-mcp | ✓ | |
| `kernel-agent.env.sh` | ✓ | |
| GEAK RAG index | | ✓ (needs GPU) |
| Ray head (started) | | ✓ |
| framework-agent (`fa`) | | optional |

For the full env reference see [`docs/ENV_AND_AUTH.md`](docs/ENV_AND_AUTH.md).
For the full VL benchmark details see [`docs/VLLM_DOCKER.md`](docs/VLLM_DOCKER.md).
