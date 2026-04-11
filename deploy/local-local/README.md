# Hyperloom Local-Local Mode

Run the full Hyperloom inference optimization loop on your local machine.

## Prerequisites

- Docker with AMD ROCm support, or K8s cluster with AMD GPU nodes
- Cursor IDE with Remote SSH extension
- LLM API key for GEAK kernel optimization
- (Optional) Anthropic / OpenAI API key for OOB Agent MCP (Claude Code / Codex backends)

## Quick Start (Docker)

### 1. Launch container

```bash
docker run -d --shm-size=16g \
  --device=/dev/kfd --device=/dev/dri \
  -v /path/to/models:/models \
  -p 20022:22 -p 20001:8001 -p 20002:8002 -p 20003:8003 \
  -e LLM_API_KEY=<your-api-key> \
  -e LLM_API_BASE=https://api.openai.com/v1 \
  hyperloom-local:sglang-latest
```

**Optional env vars** (add as needed):

| Env var | Purpose |
|---------|---------|
| `HIP_VISIBLE_DEVICES=0,1` | Limit to specific GPUs |
| `ANTHROPIC_API_KEY=<key>` | Enable Claude Code kernel backend |
| `ANTHROPIC_BASE_URL=<url>` | Claude API endpoint (omit to use api.anthropic.com) |
| `OPENAI_API_KEY=<key>` | Enable Codex kernel backend |
| `OPENAI_BASE_URL=<url>` | OpenAI API endpoint (omit to use api.openai.com) |

> `--shm-size=16g` is required for multi-GPU inference (RCCL uses shared memory). Default 64MB will cause errors.

### 2. Configure SSH

Add to your `~/.ssh/config` (Linux/macOS) or `C:\Users\<you>\.ssh\config` (Windows):

```
Host hyperloom
    HostName <gpu-machine-ip>
    Port 20022
    User root
```

> If Docker runs on your local machine, set `HostName localhost`.

### 3. Connect with Cursor

1. Open Cursor → Remote SSH → Connect to Host → `hyperloom` (user: `root`, password: `root`)
2. Open folder: `/opt/hyperloom`
3. Skills and MCP servers load automatically

### 4. Run optimization

Type in Cursor chat:

```
@inference-optimization Optimize /models/Qwen3-30B-A3B
```

The agent auto-detects mode, framework, GPU count, and InferenceX path from the container environment. Specify extra details only when needed:

```
@inference-optimization Optimize /models/Qwen3-30B-A3B

TP=8, CONC=64, ISL=1024, OSL=1024
Precision: FP8
GPU type: MI355X
Must optimize at least 5 kernels.
Execute the full skill pipeline (Phase 0-10), including parameter sweep.
Save results to /opt/hyperloom/results/
```

**Kernel optimization backends** — default is GEAK. To use Codex or Claude instead, say so in the prompt:

```
@inference-optimization Optimize /models/Qwen3-30B-A3B
Use Codex as the kernel optimization backend.
```

> Replace `Codex` with `Claude` for the Claude Code backend. Requires `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` passed to the container.

## Kubernetes Deployment

The same container image can be used as a K8s Pod base image. When K8s overrides the container CMD, MCP services start automatically on first login via `/etc/profile.d/hyperloom.sh`.

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
    - containerPort: 8001
      name: tracelens
    - containerPort: 8002
      name: geak
    - containerPort: 8003
      name: oob-agent
    env:
    - name: LLM_API_KEY
      valueFrom:
        secretKeyRef:
          name: hyperloom-secrets
          key: llm-api-key
    - name: LLM_API_BASE
      value: "https://api.deepseek.com/v1"
    - name: ANTHROPIC_API_KEY
      valueFrom:
        secretKeyRef:
          name: hyperloom-secrets
          key: anthropic-api-key
          optional: true
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
  - name: tracelens
    port: 8001
    targetPort: 8001
    nodePort: 30001
  - name: geak
    port: 8002
    targetPort: 8002
    nodePort: 30002
  - name: oob-agent
    port: 8003
    targetPort: 8003
    nodePort: 30003
```

Connect via Cursor Remote SSH → `<node-ip>:30022`, open `/opt/hyperloom`.

## Container Ports

| Internal Port | Service |
|---------------|---------|
| 22   | SSH (Cursor Remote SSH) |
| 8001 | TraceLens MCP |
| 8002 | GEAK MCP |
| 8003 | OOB Agent MCP (Claude Code / Codex) |

> Map to any host/external port via `docker run -p` or K8s NodePort/LoadBalancer.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | — | LLM API key for GEAK kernel optimization |
| `LLM_API_BASE` | — | LLM API endpoint URL |
| `FRAMEWORK` | `sglang` | Inference framework (`sglang` or `vllm`) |
| `TRACELENS_PORT` | `8001` | TraceLens MCP port |
| `GEAK_MCP_PORT` | `8002` | GEAK MCP port |
| `OOB_MCP_PORT` | `8003` | OOB Agent MCP port |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (for Claude Code backend) |
| `ANTHROPIC_BASE_URL` | — | Anthropic API endpoint (for enterprise LLM gateway) |
| `OPENAI_API_KEY` | — | OpenAI API key (for Codex backend) |
| `OPENAI_BASE_URL` | — | OpenAI API endpoint |
| `HIP_VISIBLE_DEVICES` | — | Comma-separated GPU indices (e.g. `0,1,2`) |
| `GPUS_PER_NODE` | — | Override GPU count for entrypoint display |

## Logs

Service logs are written to `/var/log/hyperloom/`:

```bash
tail -f /var/log/hyperloom/tracelens.log
tail -f /var/log/hyperloom/geak-api.log
tail -f /var/log/hyperloom/geak-mcp.log
tail -f /var/log/hyperloom/oob-mcp.log
```

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

**MCP services not running (K8s Pod)**

Services start on first login. If they didn't, run manually:

```bash
source /etc/profile.d/hyperloom.sh
```

Check status:

```bash
curl -s http://localhost:8001/mcp > /dev/null && echo "TraceLens OK" || echo "TraceLens NOT running"
curl -s http://localhost:8000/health > /dev/null && echo "GEAK API OK" || echo "GEAK API NOT running"
curl -s http://localhost:8002/ > /dev/null && echo "GEAK MCP OK" || echo "GEAK MCP NOT running"
curl -s http://localhost:8003/ > /dev/null && echo "OOB Agent OK" || echo "OOB Agent NOT running"
```

**GPU count shows wrong number**

The entrypoint checks in order: `GPUS_PER_NODE` → `HIP_VISIBLE_DEVICES` → `ROCR_VISIBLE_DEVICES` → `amd-smi` → `rocm-smi`. Set env vars to override hardware scan.

