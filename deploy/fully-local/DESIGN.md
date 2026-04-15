# Hyperloom Fully Local Mode Design Document

> Local node support for user-owned infrastructure.

## 1. Overview & Motivation

### 1.1 What is Fully Local Mode

Fully Local mode lets users run the full Hyperloom inference optimization loop on their **own GPU infrastructure**, without depending on AMD-hosted PrimusClaw sandboxes or Primus-SaFE Authoring Pods.

"Fully Local" means **local Agent + local GPU** — the Agent (Cursor IDE) runs locally, benchmarks execute on local GPUs, and all MCP services (TraceLens, GEAK, OOB Agent) run inside the same container.

### 1.2 Why Fully Local

| Scenario | PrimusClaw (cloud-hosted) | Fully Local (user-owned infra) |
|----------|--------------------------|-------------------------------|
| User has own GPU cluster | Data uploaded to AMD cloud | Data stays in user's network |
| Air-gapped / private network | Requires public internet | Only LLM API outbound needed |
| Custom images / driver versions | Limited to platform images | User controls the base image |
| Fast iteration & debugging | Job queuing, sandbox spin-up | Persistent container, SSH direct, zero queuing |
| Multi-node / special topology | Platform-scheduled | User-orchestrated |

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
│  + Agent         │                    │  ┌──────────┐  ┌──────────┐        │
│  + Skills        │                    │  │TraceLens │  │  GEAK    │        │
│                  │                    │  │MCP :8001 │  │API :8000 │        │
└──────────────────┘                    │  └──────────┘  │MCP :8002 │        │
                                        │                └──────────┘        │
                                        │  ┌──────────┐                      │
                                        │  │OOB Agent │                      │
                                        │  │MCP :8003 │                      │
                                        │  └──────────┘                      │
                                        │                                    │
                                        │  /opt/hyperloom/                   │
                                        │    ├── InferenceX/                 │
                                        │    ├── .cursor/skills/             │
                                        │    └── .cursor/mcp.json            │
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

**Single self-contained container**: All MCP services, inference framework, and benchmark tooling are bundled in one container.
- One `docker run` command to deploy
- All inter-service communication over localhost — no service discovery needed
- Simplifies GPU passthrough — single container binds GPU devices, avoiding multi-container GPU sharing complexity

**SSH access instead of Web UI**: Users connect via Cursor Remote SSH.
- The Cursor Agent needs direct filesystem access for code editing and command execution
- SSH is Cursor Remote's native protocol — zero additional adaptation
- `/opt/hyperloom` inside the container serves as the complete Cursor workspace

**All MCP services local**: TraceLens, GEAK, and OOB Agent MCP servers all run inside the container, communicating over localhost.
- Eliminates network latency and availability dependency on external MCP services
- GEAK in local mode can directly access GPUs for kernel compilation and validation
- OOB Agent in local mode directly invokes the in-container `claude` / `codex` CLIs

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
│ Artifacts: OOB MCP / TraceLens / InferenceX / Skills / scripts  │
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
| Layer 3 | TraceLens + OOB Agent Python/Node deps | Medium |
| Layer 4 | InferenceX, Skills, mcp.json, entrypoint | High (lightweight COPY, minimal rebuild cost) |

### 3.3 Dual Framework Support

The same `hyperloom-src` artifacts feed into two inference framework base images (`sglang` / `vllm`). Both Dockerfiles share identical structure — only the `FROM` image and `FRAMEWORK` env var differ.

---

## 4. Service Topology & Lifecycle

### 4.1 In-Container Services

| Service | Port | Role |
|---------|------|------|
| sshd | 22 | Cursor Remote SSH entry point |
| TraceLens MCP | 8001 | Performance analysis (kernel breakdown, roofline) |
| GEAK REST API | 8000 | Kernel optimization task management backend |
| GEAK MCP | 8002 | GEAK MCP protocol adapter |
| OOB Agent MCP | 8003 | Claude Code / Codex code generation backend |

### 4.2 Lifecycle

**Docker mode**: `entrypoint.sh` runs as PID 1 — starts all services sequentially → waits for ports to be ready (30s timeout) → enters a supervisor loop (5s interval, restarts crashed services). On SIGTERM/SIGINT, gracefully kills all child processes.

**K8s mode**: When the Pod CMD is overridden, MCP services are instead started by `/etc/profile.d/hyperloom.sh` on first SSH login (idempotent — uses port probes to avoid duplicate starts).

---

## 5. Deployment Models

### 5.1 Docker vs Kubernetes

| Dimension | Docker | Kubernetes |
|-----------|--------|------------|
| GPU allocation | `--device=/dev/kfd --device=/dev/dri` | `amd.com/gpu` device plugin |
| Service startup | entrypoint.sh direct management | autostart.sh idempotent launch |
| Secret management | `docker run -e` | K8s Secret |
| Network exposure | `-p 20022:22` port mapping | Service NodePort |
| Best for | Single node, quick validation | Multi-user, production |

See [README.md](README.md) for deployment examples and full parameter reference.

---

## 6. Networking & Security

### 6.1 Network Model

```
External ──► SSH (:22) ──► Container        (only inbound point)
Container ──► LLM API (GEAK/OOB outbound)  (only outbound point)
Internal: all MCP services on localhost     (not externally exposed)
```

| Port | Direction | Externally Exposed |
|------|-----------|--------------------|
| 22 | Inbound | Yes (required) |
| 8000/8001/8002/8003 | Internal | No |

### 6.2 Security Boundary

- **Data stays in user's network**: Model files are volume-mounted; benchmark data and optimization results are stored on the container's local filesystem
- **Only outbound traffic**: LLM API calls (GEAK kernel optimization, OOB code generation) — can be restricted via network policies
- **MCP service isolation**: All MCP servers listen on localhost only, ports are not mapped externally
- **API Key consolidation**: Unified entry points (`LLM_API_KEY` / `OOB_API_KEY`), entrypoint auto-maps to service-specific variables

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
├── hyperloom-autostart.sh     # K8s SSH login auto-start script
└── mcp.json                   # Cursor MCP server config (localhost ports)
```
