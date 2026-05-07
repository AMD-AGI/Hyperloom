# Hyperloom Local Mode

> **Local node support for user-owned infrastructure** — run the full Hyperloom inference optimization loop entirely on your own GPU nodes (Docker or K8s), without depending on AMD-hosted PrimusClaw sandboxes or Primus-SaFE authoring pods. See [DESIGN.md](DESIGN.md) for architecture details.

## Prerequisites

- Docker with AMD ROCm support, or K8s cluster with AMD GPU nodes
- Cursor IDE with Remote SSH extension
- LLM API key for GEAK kernel optimization
- OOB API key and base URL for OOB (Claude Code / Codex backends, scheduled via `oob_ray_submit.py` through Ray)

## Quick Start (Docker)

### 1. Launch container

```bash
docker run -d --shm-size=16g \
  --device=/dev/kfd --device=/dev/dri \
  -v /path/to/models:/models \
  -p 20022:22 \
  -e LLM_API_KEY=<your-geak-api-key> \
  -e LLM_API_BASE=https://<your-openai-compatible-endpoint>/v1 \
  -e GEAK_MODEL_NAME=<model-supported-by-that-endpoint> \
  primussafe/hyperloom-local:sglang-mi30x-428-1
```
> vllm images: `primussafe/hyperloom-local:vllm-428-1`

> `LLM_API_KEY` and `LLM_API_BASE` are only used by the `geak` kernel optimization backend. Set `GEAK_MODEL_NAME` to a model that your endpoint actually serves; if omitted, the default is `claude-opus-4-7`. If you use OOB `codex` / `claude` backends, configure `OOB_API_KEY` and `OOB_BASE_URL`.

**Optional env vars** (add as needed):

| Env var | Purpose |
|---------|---------|
| `HIP_VISIBLE_DEVICES=0,1` | Limit to specific GPUs |
| `GEAK_MODEL_NAME=<model>` | Override the GEAK model rendered into the local LiteLLM config |
| `OOB_API_KEY=<key>` | Unified OOB API key (used by both Claude/Codex) |
| `OOB_BASE_URL=<url>` | Unified OOB API endpoint (recommended) |

> `--shm-size=16g` is required for multi-GPU inference (RCCL uses shared memory). Default 64MB will cause errors.

### 2. Configure SSH

#### If Docker runs on your local machine

Add to your `~/.ssh/config` (Linux/macOS) or `C:\Users\<you>\.ssh\config` (Windows):

```
Host hyperloom
    HostName localhost
    Port 20022
    User root
```

#### If Docker runs on a remote GPU server

The container is not directly reachable from your laptop — you must proxy through the host first.

**Linux/macOS** (`~/.ssh/config`):

```
Host hyperloom
    HostName localhost
    Port 20022
    User root
    IdentityFile ~/.ssh/private_key
    ProxyJump <user>@<gpu-server-hostname-or-ip>
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

**Windows** (`C:\Users\<you>\.ssh\config`):

```
Host hyperloom
    HostName localhost
    Port 20022
    User root
    IdentityFile "C:\Users\<you>\.ssh\private_key"
    ProxyJump <user>@<gpu-server-hostname-or-ip>
    StrictHostKeyChecking no
    UserKnownHostsFile NUL
```

Then inject your SSH public key into the container (run once from your local machine):

```bash
PUBKEY=$(cat ~/.ssh/private_key.pub)
ssh <user>@<gpu-server-hostname-or-ip> \
  "docker exec <CONTAINER_ID> bash -c \
  'mkdir -p /root/.ssh && \
   echo \"$PUBKEY\" >> /root/.ssh/authorized_keys && \
   chmod 600 /root/.ssh/authorized_keys && \
   chmod 700 /root/.ssh'"
```

> To find `<CONTAINER_ID>`, run `docker ps` on the GPU server.

Verify the connection before opening Cursor:

```bash
ssh hyperloom echo "Connected successfully!"
```

### 3. Connect with Cursor

1. Open Cursor → Remote SSH → Connect to Host → `hyperloom` (user: `root`, password: `root`)
2. Open folder: `/opt/hyperloom`
3. Skills load automatically

> **Remote server users:** Open the `/opt/hyperloom` folder as the workspace **before** attaching to the container — attaching first and then opening the folder can cause path resolution issues.

> Local mode runs **no persistent MCP services** — TraceLens, GEAK, and OOB are all invoked as in-container CLIs (`tracelens-*`, `geak` via `geak_ray_submit.py` through Ray, OOB via `oob_ray_submit.py` through Ray). No MCP toggles need to be enabled.

### 4. Run optimization

Type in Cursor chat:

```
@inference-optimization Optimize /models/Qwen3-30B-A3B
mode: local
```

The agent auto-detects mode, framework, GPU count, and InferenceX path from the container environment. Specify extra details only when needed:

```
@inference-optimization Optimize /models/Qwen3-30B-A3B
mode: local

TP=8, CONC=64, ISL=1024, OSL=1024
Precision: FP8
GPU type: MI300X
Must optimize at least 5 kernels.
Execute the full skill pipeline (Phase 0-10), including parameter sweep.
Save results to /opt/hyperloom/results/
```

**Kernel optimization backends** — use `KERNEL_OPT_BACKENDS` or prompt instructions to select backends; multiple backends can run in parallel and the best result is kept:

| Backend | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| `geak` | `geak_ray_submit.py` through Ray, GPU isolation, hardware validation | 2-3 hours | `LLM_API_KEY` + matching `GEAK_MODEL_NAME` / `LLM_API_BASE` |
| `codex` | `oob_ray_submit.py` through Ray schedules Codex subprocess + local benchmark | ~1 hour | `OOB_API_KEY` + `OOB_BASE_URL` |
| `claude` | `oob_ray_submit.py` through Ray schedules Claude subprocess + local benchmark | ~1 hour | `OOB_API_KEY` + `OOB_BASE_URL` |

Specify backend in prompt (default `geak`, can also be changed by `KERNEL_OPT_BACKENDS`):

```
@inference-optimization Optimize /models/Qwen3-30B-A3B
mode: local

# Use Codex backend (requires OOB_API_KEY + OOB_BASE_URL)
Use only codex as the kernel optimization backend.

# Use Claude backend (requires OOB_API_KEY + OOB_BASE_URL)
Use only claude as the kernel optimization backend.
```

## Kubernetes Deployment

The same container image can be used as a K8s Pod base image. When K8s overrides the container CMD, `/etc/profile.d/hyperloom.sh` renders the GEAK config, starts Ray on first SSH login, and when `OOB_BASE_URL` is set, starts the local auth-proxy and rewrites `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` to `http://127.0.0.1:4002/...`.

Example Pod spec:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hyperloom
  labels:
    app: hyperloom
spec:
  containers:
  - name: hyperloom
    image: hyperloom-local:sglang-latest
    ports:
    - containerPort: 22
      name: ssh
    env:
    - name: LLM_API_KEY
      valueFrom:
        secretKeyRef:
          name: hyperloom-secrets
          key: llm-api-key
    - name: LLM_API_BASE
      value: "https://<your-openai-compatible-endpoint>/v1"
    - name: GEAK_MODEL_NAME
      value: "<model-supported-by-that-endpoint>"
    - name: OOB_API_KEY
      valueFrom:
        secretKeyRef:
          name: hyperloom-secrets
          key: oob-api-key
          optional: true
    - name: OOB_BASE_URL
      value: "https://<your-oob-endpoint>/v1"
    resources:
      limits:
        amd.com/gpu: 1
    volumeMounts:
    - name: models
      mountPath: /models
  volumes:
  - name: models
    hostPath:
      path: /shared/models
---
apiVersion: v1
kind: Service
metadata:
  name: hyperloom-svc
spec:
  type: NodePort
  selector:
    app: hyperloom
  ports:
  - name: ssh
    port: 22
    targetPort: 22
    nodePort: 30022
```

Connect via Cursor Remote SSH → `<node-ip>:30022`, open `/opt/hyperloom`.

## BYOI (Bring Your Own Image)

If you already have a custom image with your inference stack (specific driver builds, internal registries, etc.), you do not need the Hyperloom prebuilt image — you can run Hyperloom on top of yours.

### Minimal requirements

- Python ≥ 3.10 and ROCm GPU drivers (`/dev/kfd` + `/dev/dri`)
- sglang or vllm installed
- Hyperloom resource bundle mounted in the container (OOB / TraceLens / InferenceX; can live on any shared storage)

> **Core42 cluster**: dependency bundle is pre-staged at `/wekafs/hyperloom` — mount it directly as `HYPERLOOM_BUNDLE`.

### How to launch

```bash
docker run -d --shm-size=16g \
  --device=/dev/kfd --device=/dev/dri \
  -v /path/to/models:/models \
  -v /path/to/hyperloom-bundle:/hyperloom-bundle:ro \
  -v hyperloom-data:/opt/hyperloom \
  -p 20022:22 \
  -e LLM_API_KEY=<your-geak-api-key> \
  -e LLM_API_BASE=https://<your-openai-compatible-endpoint>/v1 \
  -e HYPERLOOM_BUNDLE=/hyperloom-bundle \
  your-custom-image:latest
```

> The image must ship with sshd. `HYPERLOOM_BUNDLE` is the in-container path to the mounted bundle (default: `/wekafs/hyperloom`). Prefer a persistent volume at `/opt/hyperloom` so bootstrap state survives container recreation.

### Workflow

1. Connect with Cursor Remote SSH and open `/opt/hyperloom`
2. On first skill execution the agent detects BYOI and runs `bootstrap.sh`
3. Bootstrap installs GEAK, Ray, TraceLens, OOB, and related deps (~3–5 minutes)
4. After that, behavior matches the prebuilt image

> Bootstrap is idempotent — already-installed components are skipped, and later starts finish in seconds. For the full design, see [section 7 of DESIGN.md](DESIGN.md#7-byoi-design-method-b).

## Container Ports

| Internal Port | Service |
|---------------|---------|
| 22   | SSH (Cursor Remote SSH) — only externally exposed port |
| 6379 | Ray head (GEAK GPU scheduling, internal) |
| 8265 | Ray dashboard (internal) |
| 4002 | OOB auth-proxy — only present when `OOB_BASE_URL` is set (internal) |

> TraceLens, GEAK, and OOB do **not** listen on any port — they are invoked as CLIs (`tracelens-*`, `geak` via `geak_ray_submit.py` through Ray, OOB via `oob_ray_submit.py` through Ray).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | — | LLM API key for GEAK kernel optimization |
| `LLM_API_BASE` | — | LLM API endpoint URL |
| `GEAK_MODEL_NAME` | `claude-opus-4-7` | GEAK model name rendered into the generated LiteLLM config |
| `GEAK_API_KEY` | falls back to `LLM_API_KEY` | Optional GEAK-only API key override |
| `GEAK_BASE_URL` | falls back to `LLM_API_BASE` | Optional GEAK-only endpoint override |
| `FRAMEWORK` | `sglang` | Inference framework (`sglang` or `vllm`) |
| `OOB_API_KEY` | — | Unified OOB API key (shared by Claude/Codex `oob_ray_submit.py run` invocations) |
| `OOB_BASE_URL` | — | Unified OOB API endpoint (when set, an in-container auth-proxy on `:4002` rewrites Bearer auth) |
| `OOB_HOME` | `~/.oob` | Root dir where `oob` stores task workspaces and the SQLite database |
| `HIP_VISIBLE_DEVICES` | — | Comma-separated GPU indices (e.g. `0,1,2`) |
| `GPUS_PER_NODE` | — | Override GPU count for entrypoint display |

## Logs

Service logs are written to `/var/log/hyperloom/`:

```bash
tail -f /var/log/hyperloom/ray-head.log         # Ray (GEAK GPU scheduler)
tail -f /var/log/hyperloom/oob-auth-proxy.log   # OOB auth proxy (only if OOB_BASE_URL is set)
```

> Per-task CLI logs are not written to `/var/log/hyperloom`. `oob_ray_submit.py run` stores files under `${OOB_HOME:-~/.oob}/tasks/cli/<task_id>/workspace/` (for example `execution.log`), while `geak` writes results under its own output directory.

## Security

Default SSH password is `root`. Change it after first login:

```bash
passwd root
```

Or mount your SSH key:

```bash
docker run ... -v ~/.ssh/id_rsa.pub:/root/.ssh/authorized_keys:ro ...
```

## Troubleshooting

**Background services not running (K8s Pod)**

Ray + the optional OOB auth-proxy are initialized on first SSH login. The same startup script also renders `GEAK_CONFIG` and rewrites OOB base URLs to the local `:4002` proxy when `OOB_BASE_URL` is set. If they didn't, run manually:

```bash
source /etc/profile.d/hyperloom.sh
```

Check status:

```bash
ray status                                           # Ray head
ss -tlnp | grep -E ':6379|:8265|:4002' || true       # Listening ports
command -v oob && oob --help | head -5               # OOB CLI present (dependency of oob_ray_submit.py)
command -v geak                                      # GEAK CLI installed
python3 -c "import TraceLens" && echo "TraceLens OK" # TraceLens importable
printf 'OPENAI_BASE_URL=%s\nANTHROPIC_BASE_URL=%s\n' "$OPENAI_BASE_URL" "$ANTHROPIC_BASE_URL"
```

**GPU count shows wrong number**

The entrypoint checks in order: `GPUS_PER_NODE` → `HIP_VISIBLE_DEVICES` → `ROCR_VISIBLE_DEVICES` → `amd-smi` → `rocm-smi`. Set env vars to override hardware scan.
