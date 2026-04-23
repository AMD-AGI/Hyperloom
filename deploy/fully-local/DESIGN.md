# Hyperloom Fully Local Mode Design Document

> Local node support for user-owned infrastructure.

## 1. Overview & Motivation

### 1.1 What is Fully Local Mode

Fully Local mode lets users run the full Hyperloom inference optimization loop on their **own GPU infrastructure**, without depending on AMD-hosted PrimusClaw sandboxes or Primus-SaFE Authoring Pods.

"Fully Local" means **local Agent + local GPU** — the Agent (Cursor IDE) runs locally, benchmarks execute on local GPUs, and **all optimization tooling is invoked as in-container CLIs** (TraceLens, GEAK via Ray, OOB via `oob run`). No persistent MCP/REST services are exposed.

### 1.2 Why Fully Local

| Scenario | PrimusClaw (cloud-hosted) | Fully Local (user-owned infra) |
|----------|--------------------------|-------------------------------|
| User has own GPU cluster | Data uploaded to AMD cloud | Data stays in user's network |
| Air-gapped / private network | Requires public internet | Only LLM API outbound needed |
| Custom images / driver versions | Limited to platform images | User controls the base image |
| Fast iteration & debugging | Job queuing, sandbox spin-up | Persistent container, SSH direct, zero queuing |
| Multi-node / special topology | Platform-scheduled | User-orchestrated |
| Tooling integration | Remote MCP services | Local CLIs (no MCP) |

### 1.3 Relationship with PrimusClaw

```
┌──────────────────────────────────────────────────────────┐
│                  Hyperloom Optimization Engine             │
│           (Skills + Optimization Loop + Knowledge Base)    │
├─────────────────────────┬────────────────────────────────┤
│   PrimusClaw Mode        │      Fully Local Mode            │
│   ────────────────       │    ──────────────────           │
│   Web UI job submission  │    Cursor SSH into container    │
│   AMD cloud sandboxes    │    User-owned GPU nodes         │
│   Remote MCP connections │    MCP on localhost in-container │
│   Minio + Langfuse       │    Local logs + filesystem      │
│   RayJob distributed     │    Docker / K8s Pod             │
└─────────────────────────┴────────────────────────────────┘
```

Both modes share the same Skill files, optimization methodology, and Knowledge Base. The only difference is **deployment topology and resource ownership**.

---

## 2. Architecture

### 2.1 High-Level Architecture

```
User Laptop / Workstation                User GPU Node
┌──────────────────┐                    ┌────────────────────────────────────┐
│                  │                    │  Hyperloom Container               │
│  Cursor IDE      │◄── SSH (22) ──────►│                                    │
│  + Agent         │                    │  In-container CLIs (no servers):   │
│  + Skills        │                    │   • tracelens-* (offline analysis) │
│                  │                    │   • geak         (Ray-scheduled)   │
└──────────────────┘                    │   • oob run      (per-task subproc)│
                                        │                                    │
                                        │  Background processes:             │
                                        │   • sshd                    :22    │
                                        │   • Ray head + dashboard :6379/8265│
                                        │   • OOB auth-proxy (optional):4002 │
                                        │                                    │
                                        │  /opt/hyperloom/                   │
                                        │    ├── InferenceX/                 │
                                        │    └── .cursor/skills/             │
                                        │                                    │
                                        │  GPU: /dev/kfd + /dev/dri          │
                                        └────────────────────────────────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │ LLM API (outbound)│
                                          │ GEAK / OOB backend│
                                          └──────────────────┘
```

### 2.2 Key Design Decisions

**Single self-contained container**: All optimization tooling, inference framework, and benchmark scripts are bundled in one container.
- One `docker run` command to deploy
- Tooling runs as direct subprocesses — no service discovery, no localhost RPC
- Simplifies GPU passthrough — single container binds GPU devices, avoiding multi-container GPU sharing complexity

**SSH access instead of Web UI**: Users connect via Cursor Remote SSH.
- The Cursor Agent needs direct filesystem access for code editing and command execution
- SSH is Cursor Remote's native protocol — zero additional adaptation
- `/opt/hyperloom` inside the container serves as the complete Cursor workspace

**CLI-only tooling, no MCP services**: TraceLens, GEAK, and OOB are all invoked as in-container CLIs by the skill — there are no persistent MCP/REST servers.
- Skills shell out directly: `tracelens-*`, `geak` (via `geak_ray_submit.py`), `oob run`
- GEAK uses the local Ray cluster for GPU allocation (one task per GPU)
- OOB's `oob run` spawns the `claude` / `codex` CLI as a subprocess per task and blocks until completion — replaces the previous `create-task → submit → poll → download` REST flow
- No supervisor loop or health checks for MCP services; only Ray and the optional auth-proxy are kept alive

---

## 3. Container Build Design

### 3.1 Multi-Stage Build Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 0: hyperloom-src (FROM scratch)                           │
│                                                                 │
│ Packages repo artifacts into a runtime-free image so downstream │
│ Dockerfiles need no local build context — just COPY --from.     │
│                                                                 │
│ Artifacts: OOB CLI / TraceLens / InferenceX / Skills / scripts  │
└─────────────────────────────────────────────────────────────────┘
                              │
                    COPY --from=hyperloom-src
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌──────────────────────┐              ┌──────────────────────┐
│ Dockerfile.sglang     │              │ Dockerfile.vllm       │
│ BASE: sglang:v0.5.9   │              │ BASE: vllm:v0.17.0    │
│ FRAMEWORK=sglang      │              │ FRAMEWORK=vllm        │
└──────────────────────┘              └──────────────────────┘
```

### 3.2 Layer Strategy

Image layers are ordered by **change frequency (low → high)** to maximize build cache hits:

| Layer | Contents | Change Frequency |
|-------|----------|-----------------|
| Layer 1 | System deps (openssh, Node.js 20, CA certs) | Very low |
| Layer 2 | GEAK + intellikit (git clone, pinned branch) | Low |
| Layer 3 | TraceLens + OOB Python/Node deps (`@anthropic-ai/claude-code`, `@openai/codex`) | Medium |
| Layer 4 | InferenceX, Skills, OOB sources + `pip install -e` (oob console script), entrypoint | High (lightweight, minimal rebuild cost) |

### 3.3 Dual Framework Support

The same `hyperloom-src` artifacts feed into two inference framework base images (`sglang` / `vllm`). Both Dockerfiles share identical structure — only the `FROM` image and `FRAMEWORK` env var differ.

---

## 4. Service Topology & Lifecycle

### 4.1 In-Container Processes

Persistent processes (managed by `entrypoint.sh` supervisor):

| Process | Port | Role |
|---------|------|------|
| sshd | 22 | Cursor Remote SSH entry point |
| Ray head + dashboard | 6379 / 8265 | GEAK GPU task scheduling |
| OOB auth-proxy | 4002 | Bearer-auth rewrite for AMD LLM gateway (only when `OOB_BASE_URL` is set) |

CLI tools (no port, invoked per task by the skill):

| Tool | How |
|------|-----|
| TraceLens | `tracelens-*` console scripts (offline trace analysis) |
| GEAK | `geak` CLI scheduled through `geak_ray_submit.py` → Ray |
| OOB | `oob run -a {claude,codex} -p ... -f ...` — spawns `claude` / `codex` subprocess and blocks |

### 4.2 Lifecycle

**Docker mode**: `entrypoint.sh` runs as PID 1 — provisions agent CLI auth files, starts sshd / Ray / (optional) auth-proxy → waits for ports to be ready (30s timeout) → enters a supervisor loop (5s interval, restarts crashed Ray or auth-proxy). On SIGTERM/SIGINT, gracefully kills all child processes.

**K8s mode**: When the Pod CMD is overridden, the same background processes are instead started by `/etc/profile.d/hyperloom.sh` on first SSH login (idempotent — uses `ray status` and config-file probes to avoid duplicate work).

---

## 5. Deployment Models

### 5.1 Docker vs Kubernetes

| Dimension | Docker | Kubernetes |
|-----------|--------|------------|
| GPU allocation | `--device=/dev/kfd --device=/dev/dri` | `amd.com/gpu` device plugin |
| Background process startup | entrypoint.sh direct management | autostart.sh idempotent launch on first SSH login |
| Secret management | `docker run -e` | K8s Secret |
| Network exposure | `-p 20022:22` port mapping | Service NodePort (only port 22) |
| Best for | Single node, quick validation | Multi-user, production |

See [README.md](README.md) for deployment examples and full parameter reference.

---

## 6. Networking & Security

### 6.1 Network Model

```
External ──► SSH (:22) ──► Container         (only inbound point)
Container ──► LLM API (GEAK/OOB outbound)   (only outbound point)
Internal: Ray (6379/8265) + optional         (not externally exposed)
          OOB auth-proxy (4002)
No MCP/REST services run; tooling is CLI-only.
```

| Port | Direction | Externally Exposed |
|------|-----------|--------------------|
| 22 | Inbound | Yes (required) |
| 6379 / 8265 / 4002 | Internal | No |

### 6.2 Security Boundary

- **Data stays in user's network**: Model files are volume-mounted; benchmark data and optimization results are stored on the container's local filesystem
- **Only outbound traffic**: LLM API calls (GEAK kernel optimization, OOB `claude`/`codex` subprocesses) — can be restricted via network policies
- **No exposed RPC surface**: No MCP/REST tooling servers exist; only sshd is reachable from outside, Ray and the optional auth-proxy bind to localhost
- **API Key consolidation**: Unified entry points (`LLM_API_KEY` / `OOB_API_KEY`), entrypoint auto-maps to provider-specific variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `AMD_LLM_API_KEY`)

---

## Appendix: File Inventory

```
deploy/fully-local/
├── DESIGN.md                  # This design document
├── README.md                  # User-facing quick start guide
├── Dockerfile.hyperloom-src   # Stage 0: repo artifact bundle image
├── Dockerfile.sglang          # SGLang base image build
├── Dockerfile.vllm            # vLLM base image build
├── entrypoint.sh              # Container entrypoint: service startup + supervisor + health checks
└── hyperloom-autostart.sh     # K8s SSH login auto-start script
```
