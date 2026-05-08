# Hyperloom Local Mode Design Document

> Local node support for user-owned infrastructure.

## 1. Overview & Motivation

### 1.1 What is Local Mode

Local mode lets users run the full Hyperloom inference optimization loop on their **own GPU infrastructure**, without depending on AMD-hosted PrimusClaw sandboxes or Primus-SaFE Authoring Pods.

"Local" means **local Agent + local GPU** — the Agent (Cursor IDE) runs locally, benchmarks execute on local GPUs, and **all optimization tooling is invoked as in-container CLIs** (TraceLens; GEAK via `geak_ray_submit.py` through Ray; OOB via `oob_ray_submit.py run` through Ray). No persistent MCP/REST services are exposed.

Local mode supports **two deployment methods**:

| | Method A: Prebuilt image | Method B: BYOI (Bring Your Own Image) |
|--|--|--|
| Image source | Hyperloom official images (`Dockerfile.sglang` / `Dockerfile.vllm`) | Any user image (must meet minimum requirements) |
| Dependency install | Done at image build time | `bootstrap.sh` after Agent connects over SSH |
| Best for | Standard rollout, quick start | Existing images, special drivers, internal registries |
| Entrypoint | Hyperloom `entrypoint.sh` (starts all services) | User-defined (Agent runs bootstrap) |

### 1.2 Why Local

| Scenario | PrimusClaw (cloud-hosted) | Local (user-owned infra) |
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
│   PrimusClaw Mode        │      Local Mode                  │
│   ────────────────       │    ──────────────────           │
│   Web UI job submission  │    Cursor SSH into container    │
│   AMD cloud sandboxes    │    User-owned GPU nodes         │
│   Remote MCP connections │    CLI-only in-container        │
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
└──────────────────┘                    │   • oob          (Ray-scheduled)   │
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

**Pure CLI tooling, scheduled through Ray where needed**: TraceLens, GEAK, and OOB are all invoked as in-container CLIs by Skills, but the scheduling path differs.
- **TraceLens**: direct subprocess — `tracelens-*` for offline trace analysis, **not routed through Ray**
- **GEAK**: submitted to the local Ray cluster via `geak_ray_submit.py`, one GPU per task, parallel kernel optimization
- **OOB**: submitted via `oob_ray_submit.py run`; the `claude` / `codex` CLIs run as Ray worker subprocesses and block until completion
- **Why OOB uses Ray**: `claude` / `codex` workloads drive GPU kernel benchmark/tuning; concurrent tasks need Ray to enforce GPU isolation via `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`, avoiding contention when multiple tasks would otherwise share one GPU

---

### 2.3 BYOI (Method B) Architecture

Method B turns **build-time installation from the Dockerfile into on-demand bootstrap at Agent runtime**.

```
User Laptop / Workstation                User GPU Node (any base image)
┌──────────────────┐                    ┌────────────────────────────────────┐
│                  │                    │  User container (framework + ROCm) │
│  Cursor IDE      │◄── SSH (22) ──────►│                                    │
│  + Agent         │                    │  On first Agent SSH session:        │
│  + Skills        │                    │   1. Probe existing components     │
│                  │                    │   2. Install missing deps under     │
└──────────────────┘                    │      /opt/hyperloom                 │
                                        │   3. Start Ray + export env vars   │
                                        │                                    │
                                        │  After bootstrap, same as Method A: │
                                        │   • tracelens-* / geak / oob CLIs   │
                                        │   • Ray head :6379                  │
                                        │   • /opt/hyperloom/ workspace       │
                                        │                                    │
                                        │  GPU: /dev/kfd + /dev/dri          │
                                        └────────────────────────────────────┘
```

**Principles**:
- **Behavior matches Method A after bootstrap** — Skills do not branch on A vs B
- **Idempotent**: `bootstrap.sh` can be re-run; already-installed steps are skipped
- **Persistence**: everything installs under `/opt/hyperloom`; survives container `restart` but not Pod recreate or a fresh `docker run` — mount `/opt/hyperloom` on WekaFS or a local persistent volume

---

## 3. Container Build Design

> This section applies **only to Method A (prebuilt images)**. Method B skips it; see [Section 7](#7-byoi-design-method-b).

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

Persistent processes (Docker: `entrypoint.sh`; K8s: `/etc/profile.d/hyperloom.sh` on SSH login — deployed from repo `hyperloom-autostart.sh`):

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
| OOB | `oob_ray_submit.py run -a {claude,codex} -p ... -f ...` — Ray-scheduled GPU isolation; `oob` CLI runs as worker subprocess and blocks until done |

### 4.2 Lifecycle

**Docker mode**: `entrypoint.sh` runs as PID 1 — provisions agent CLI auth files, starts sshd / Ray / (optional) auth-proxy → waits for ports to be ready (30s timeout) → enters a supervisor loop (5s interval, restarts crashed Ray or auth-proxy). On SIGTERM/SIGINT, gracefully kills all child processes.

**K8s mode**: When the Pod CMD is overridden, `/etc/profile.d/hyperloom.sh` (from `hyperloom-autostart.sh`) runs on SSH login instead of PID 1. It renders the GEAK LiteLLM config, starts Ray if needed, and when `OOB_BASE_URL` is set, starts the local OOB auth-proxy and rewrites `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` to `http://127.0.0.1:4002/...`. The script avoids duplicate background services by checking `ray status` and whether `:4002` is already listening.

**GEAK template fallback**: When rendering GEAK config, `hyperloom-autostart.sh` prefers `/opt/hyperloom/geak-config/template.yaml` (copied into Method A images by the Dockerfile). In dev mode, where `/opt/hyperloom` is a host bind-mount of the repo, that path may be missing; the script then falls back to `/opt/hyperloom/deploy/local/geak-litellm.yaml` (the template in-repo) so GEAK config always renders regardless of mount layout.

---

## 5. Deployment Models

### 5.1 Docker vs Kubernetes

| Dimension | Docker | Kubernetes |
|-----------|--------|------------|
| GPU allocation | `--device=/dev/kfd --device=/dev/dri` | `amd.com/gpu` device plugin |
| Background process startup | `entrypoint.sh` direct management | `hyperloom-autostart.sh` idempotent launch on first SSH login (installed as `/etc/profile.d/hyperloom.sh`) |
| Secret management | `docker run -e` | K8s Secret |
| Network exposure | `-p 20022:22` port mapping | Service NodePort (only port 22) |
| Best for | Single node, quick validation | Centralized rollout, production |

### 5.2 Deployment method × orchestration matrix

Method A/B and Docker/K8s are orthogonal; all four combinations are supported:

| | Docker | Kubernetes |
|--|--------|------------|
| **Method A (prebuilt image)** | `entrypoint.sh` is PID 1 and starts all services; user runs e.g. `docker run hyperloom-sglang` | Default Pod CMD is `entrypoint.sh`; behavior matches Docker |
| **Method B (BYOI)** | User entrypoint starts sshd; first Cursor SSH session runs `bootstrap.sh` | Pod uses the user image + user CMD (must include sshd); first Cursor SSH session runs `bootstrap.sh` |

**Extra requirements for BYOI + K8s**:
- Pod spec must expose SSH (NodePort or Ingress)
- User image must run sshd in the foreground (or sshd alongside the user main process)
- WekaFS bundle is mounted in the Pod at `$HYPERLOOM_BUNDLE`
- Mount `/opt/hyperloom` with emptyDir or PVC so bootstrap output survives Pod recreation when possible

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
- **API Key consolidation**: Unified entry points (`LLM_API_KEY` / `OOB_API_KEY`), startup scripts auto-map to provider-specific variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `AMD_LLM_API_KEY`) and route OOB traffic through the local `:4002` proxy when configured

---

## 7. BYOI Design (Method B)

### 7.1 Minimum requirements

The user-supplied image must satisfy:

| Component | Requirement | Notes |
|-----------|-------------|-------|
| Python | ≥ 3.10 | Base runtime; not installed by bootstrap |
| GPU driver | ROCm (`/dev/kfd` + `/dev/dri` available) | Kernel-level; not installed by bootstrap |
| Inference framework | `sglang` or `vllm` installed | **Must be present**; bootstrap does not install it |
| Network | Reachable PyPI + GitHub + npm | Bootstrap downloads dependencies |
| WekaFS mount | `$HYPERLOOM_BUNDLE` readable | **Required** — source of OOB / TraceLens / InferenceX artifacts |

**Bootstrap can install** if missing: pip, git, curl, Node.js, Ray, GEAK, intellikit, TraceLens, OOB (including npm `claude` / `codex`).

**Provided by the Cursor workspace**: Skills (loaded when the user opens `/opt/hyperloom` in Cursor).

### 7.2 Bootstrap flow

**Preconditions**: The user started a container from their image, sshd is running, Cursor Remote SSH is connected, and the workspace is `/opt/hyperloom` with `.cursor/skills/inference-optimization/` present (**Skills only** at first).

**Artifact sources** (main difference vs Method A):

| Component | Source | Bootstrap behavior |
|-----------|--------|-------------------|
| Skills | Cursor workspace `/opt/hyperloom/.cursor/skills/` | Already present; skipped |
| GEAK / intellikit | GitHub | `git clone` + `pip install -e` |
| Ray / click | PyPI | `pip install` |
| Node CLIs (`claude` / `codex`) | npm | `npm install -g` |
| OOB / TraceLens | **WekaFS** (`$HYPERLOOM_BUNDLE/`) | `cp` into `/opt/hyperloom/` + `pip install -e` |
| InferenceX | **WekaFS** | `INFERENCEX_PATH` points into the bundle (large; not copied) |

**WekaFS bundle layout** (user places Hyperloom artifacts on shared WekaFS for all GPU nodes):

```
$HYPERLOOM_BUNDLE/                # default /wekafs/hyperloom/
├── OOB/                          # OOB sources (flat: cli.py, auth_proxy.py, pyproject.toml at root)
├── TraceLens-internal/           # TraceLens sources
└── inference_optimization/
    └── InferenceX/               # InferenceX data + scripts
```

> **GEAK LiteLLM template** is not shipped on WekaFS — `bootstrap.sh` Step 5 writes an embedded heredoc to `$HYPERLOOM_ROOT/geak-config/template.yaml`, matching the in-image `geak-litellm.yaml` from Method A.
>
> **OOB layout**: On WekaFS, `OOB/` is flat (no `oob_cli/` subdirectory). Bootstrap wraps it as `$HYPERLOOM_ROOT/OOB/oob_cli/` so paths match Method A (`/opt/OOB/oob_cli/`) — no branching for `auth_proxy.py` or `pip install -e` targets.

```
bootstrap.sh execution
──────────────────────────────────────────────────────────

Step 1: Hard prerequisites (exit on failure, no auto-fix)
  ├── Python ≥ 3.10?
  ├── GPU? (rocm-smi or amd-smi)
  ├── Inference framework? (import $FRAMEWORK — sglang or vllm)
  └── WekaFS bundle reachable? ($HYPERLOOM_BUNDLE/{OOB,TraceLens-internal,
      inference_optimization/InferenceX} all present)

Step 2: Soft system deps (install only what is missing)
  ├── apt: pip, git, curl, gnupg, ca-certificates
  ├── Node.js 20 (NodeSource if missing)
  ├── AMD CA certs (amd-root-ca.crt, amd-issuing-ca.crt if missing)
  └── mkdir -p /opt/hyperloom/geak-config /tmp/geak-data

Step 3: External deps (GitHub + PyPI)
  ├── git clone GEAK → /opt/hyperloom/geak
  ├── git clone intellikit → /opt/hyperloom/intellikit (pinned SHA)
  ├── pip install -e /opt/hyperloom/geak
  ├── pip install -e /opt/hyperloom/intellikit/metrix/
  └── pip install ray "click<8.3"

Step 4: Copy Hyperloom components from WekaFS and install
  (WekaFS is often read-only; copy then pip install -e under /opt/hyperloom)
  ├── cp -r $HYPERLOOM_BUNDLE/TraceLens-internal → /opt/hyperloom/TraceLens
  │   └── pip install -e /opt/hyperloom/TraceLens
  ├── cp -r $HYPERLOOM_BUNDLE/OOB → /opt/hyperloom/OOB/oob_cli  ← wrap for Method A parity
  │   ├── pip install -r /opt/hyperloom/OOB/oob_cli/requirements.txt
  │   └── pip install -e /opt/hyperloom/OOB/oob_cli
  ├── inject certifi CAs if AMD CA was installed
  └── npm install -g @anthropic-ai/claude-code @openai/codex@0.100.0

Step 5: Render config + agent CLI auth files
  ├── write /opt/hyperloom/geak-config/template.yaml (embedded heredoc, not from WekaFS)
  │   render → /opt/hyperloom/geak-config/local.yaml (inject model/key/url)
  ├── write /root/.claude/config.json if OOB_API_KEY is set
  └── write /root/.codex/auth.json if OOB_API_KEY is set

Step 6: Start background services, export env, completion marker
  ├── ray start --head --num-gpus=$GPU_COUNT if not already running
  ├── OOB auth-proxy on :4002 if OOB_BASE_URL is set (reuse /opt/hyperloom/OOB/oob_cli/auth_proxy.py)
  ├── write /etc/profile.d/hyperloom-env.sh
  │   ├── MODE=local
  │   ├── FRAMEWORK=sglang|vllm
  │   ├── GEAK_CONFIG=/opt/hyperloom/geak-config/local.yaml
  │   ├── INFERENCEX_PATH=$HYPERLOOM_BUNDLE/inference_optimization/InferenceX  ← points at WekaFS
  │   ├── SKILL_ROOT=/opt/hyperloom/.cursor/skills/inference-optimization
  │   ├── LLM_API_KEY → AMD_LLM_API_KEY mapping
  │   └── OOB_API_KEY → ANTHROPIC_API_KEY / OPENAI_API_KEY
  ├── touch /opt/hyperloom/.bootstrap_done  ← idempotency marker; skip on later SSH
  └── source that file so the current shell picks up env immediately
──────────────────────────────────────────────────────────
Done → toolchain matches Method A (geak / oob / tracelens usable, Ray up)
```

> **MODE convention**: Both Method A and B export `MODE=local` for this containerized deployment.

**Method A vs Method B responsibilities**:

| What Method A does | Method B | Why |
|--------------------|----------|-----|
| Install + configure sshd / default password | Skipped | User image already provides SSH |
| `COPY --from=hyperloom-src /OOB` at build | Runtime `cp` from WekaFS | No sources at build time |
| `COPY --from=hyperloom-src /TraceLens` at build | Runtime `cp` from WekaFS | Same |
| `COPY --from=hyperloom-src /InferenceX` at build | `INFERENCEX_PATH` → WekaFS | Large dataset; avoid copy |
| `COPY` Skills at build | Skipped | Skills live in Cursor workspace |
| `COPY` entrypoint.sh / hyperloom-autostart.sh | Skipped | No Hyperloom PID 1 in Method B |
| Supervisor loop (5s restart) | Skipped | User or re-bootstrap handles crashes |

### 7.3 Idempotency

Each step has a **pre-check**; completed work is skipped:

| Step | Skip when |
|------|-----------|
| Step 1: WekaFS bundle | All of `$HYPERLOOM_BUNDLE/{OOB,TraceLens-internal,inference_optimization/InferenceX}` exist |
| Step 2: pip / git / curl | `command -v` succeeds for each |
| Step 2: Node.js 20 | `command -v node` and version ≥ 20 |
| Step 2: AMD CA | `/usr/local/share/ca-certificates/amd-root-ca.crt` exists |
| Step 3: GEAK | `command -v geak` succeeds |
| Step 3: intellikit | `python -c "import metrix"` succeeds |
| Step 3: Ray | `command -v ray` succeeds |
| Step 4: TraceLens | `python -c "import TraceLens"` succeeds |
| Step 4: OOB CLI | `command -v oob` succeeds |
| Step 4: OOB Node CLIs | `command -v claude` and `command -v codex` |
| Step 5: GEAK config | `/opt/hyperloom/geak-config/local.yaml` exists |
| Step 5: Claude auth | `/root/.claude/config.json` exists |
| Step 5: Codex auth | `/root/.codex/auth.json` exists |
| Step 6: Ray head | `ray status` exits 0 |
| Step 6: OOB auth-proxy | something listens on `:4002` |

Implications: first run full install (~3–5 minutes, network-dependent); later runs skip quickly; partial failure + rerun only fills gaps.

### 7.4 Mode detection

During Setup the Agent uses:

```bash
if [ -f /opt/entrypoint.sh ]; then
    # Method A: prebuilt image; entrypoint finished init
    DEPLOY_METHOD="prebuilt"
elif [ -f /opt/hyperloom/.bootstrap_done ]; then
    # Method B: BYOI, bootstrap already completed
    DEPLOY_METHOD="byoi"
else
    # Method B: first entry; bootstrap (idempotent)
    DEPLOY_METHOD="byoi"
    bash /opt/hyperloom/.cursor/skills/inference-optimization/scripts/bootstrap.sh
fi
export MODE="${MODE:-local}"
```

> `/opt/hyperloom/.bootstrap_done` is written at the end of bootstrap Step 6 so long-lived containers do not re-scan on every SSH login. Delete the file to force a full re-run.

### 7.5 Method A vs Method B summary

| Dimension | Method A (prebuilt) | Method B (BYOI) |
|-----------|---------------------|-----------------|
| Image | Hyperloom official | Any user image |
| First boot | ~30s (services) | 3–5 min (download + install) |
| Later boots | ~30s | ~30s (idempotent skip) |
| Inference framework | In image | User-provided |
| Version pinning | Image tag | Bootstrap defaults to latest; env vars can pin |
| Offline | Supported (self-contained image) | Not supported (needs network) |
| Entrypoint | Hyperloom `entrypoint.sh` | User-defined; Agent runs `bootstrap.sh` |

### 7.6 BYOI environment variables

In addition to Method A variables, BYOI supports:

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_BUNDLE` | `/wekafs/hyperloom` | **Required** — shared storage root for Hyperloom artifacts (`OOB/`, `TraceLens-internal/`, `inference_optimization/InferenceX`) |
| `HYPERLOOM_ROOT` | `/opt/hyperloom` | Install root inside the container (OOB/TraceLens copy targets) |
| `GEAK_REPO` | `https://github.com/AMD-AGI/GEAK.git` | GEAK Git URL |
| `GEAK_BRANCH` | `main` | GEAK branch when `GEAK_SHA` is empty |
| `GEAK_SHA` | *(empty)* | Pinned GEAK commit; wins over `GEAK_BRANCH` when set |
| `INTELLIKIT_SHA` | `bcbfa0252df...` | Pinned intellikit commit |
| `SKIP_BOOTSTRAP` | — | Set to `1` to skip bootstrap (manual install) |

---

## Appendix: File Inventory

```
deploy/local/
├── DESIGN.md                  # This design document
├── README.md                  # User-facing quick start guide
├── Dockerfile.hyperloom-src   # Stage 0: repo artifact bundle image (Method A)
├── Dockerfile.sglang          # SGLang base image build (Method A)
├── Dockerfile.vllm            # vLLM base image build (Method A)
├── geak-litellm.yaml          # GEAK LiteLLM template (rendered at startup)
├── entrypoint.sh              # Container entry: service startup + supervisor (Method A)
└── hyperloom-autostart.sh     # K8s SSH login auto-start script

.cursor/skills/inference-optimization/
└── scripts/
    └── bootstrap.sh           # BYOI bootstrap script (Method B)
```
