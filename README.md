# ROCm Hyperloom

An agentic system that autonomously optimizes LLM inference on AMD GPUs. Hyperloom treats optimization as a **search problem**: given a workload, it builds a tree of candidate optimizations — backend swaps, server parameters, GEMM tuning, kernel rewrites, parallelism configs — scores each by expected gain and cost, then explores depth-first, always measuring against the real workload. Simply provide your workload and the agent delivers a fully optimized codebase — profiling against peak hardware potential, identifying bottlenecks, and iteratively rewriting code to maximize throughput on AMD GPUs, so the team gets production-ready optimized code.

<p align="center"><img width="600" alt="HyperLoom Architecture" src="slides/hyperloom_loop.png" /></p>

Block 1-3 - Workload understanding and profiling: Submit your workload as the starting point for the agent to understand your codebase, profile using [TraceLens Agentic Analysis](https://github.com/AMD-AGI/TraceLens-internal/) (relies on [Magpie](https://github.com/AMD-AGI/Magpie) for trace collection), capture bottlenecks and roofline targets.

Block 4 - Code Optimization Loop: The core of Hyperloom. The agent builds a scored tree of candidates — config overrides, code patches, backend switches, kernel rewrites — and explores depth-first, one change at a time: **Think → Implement → Benchmark → Decide**. Each result re-scores the remaining tree. 

In parallel, hot kernels are asynchronously optimized via external backends ([GEAK](https://github.com/AMD-AGI/GEAK/tree/main), and OOB kernel optimization via Claude Code and OpenAI Codex relying on kernel optimization flow of [Apex](https://github.com/AMD-AGI/Apex)). Kernel profiling and validation is powered by [Magpie](https://github.com/AMD-AGI/Magpie), which relies on [IntelliKit](https://github.com/AMDResearch/intellikit) for some of low-level GPU profiling tools.

Block 5-6 - Validated Delivery: The agent optimizes for throughput while maintaining accuracy — every change is correctness-gated before acceptance. Once the loop exits, the agent packages the optimized code, submits a PR to your repo, and merges into your codebase, completing the full loop.

### Learn More

| | |
|---|---|
| **[How the Optimization Loop Works](docs/HOW_THE_OPTIMIZATION_LOOP_WORKS.md)** | Scoring heuristics, stack mechanics, dynamic branching, and the self-evolving knowledge base |
| **[GLM-5 — Discovering Optimizations Hard to Spot Manually](docs/CASE_STUDY_GLM5.md)** | Hidden GEMM configs, cross-repo kernel patches, +193% throughput |
| **[DeepSeek-R1 — Fast Scale-Up on a New Workload](docs/CASE_STUDY_DEEPSEEK_R1.md)** | 7 configs to optimal in one session, MTP scheduling fix, +97% over B200 |

---

## Prerequisites

Bind your **[LLM Gateway](https://llm.amd.com/)** key to **[Hyperloom](https://core42.primus-safe.amd.com/hyperloom/)** to obtain your `AK_YOUR_API_KEY`. This key is required for both the Hyperloom UI and the local optimization workflow — it provides access to TraceLens, GEAK, and OOB services.

---

## Quickstart — Hyperloom UI (PrimusClaw)

The fastest way to start is through the hosted **AMD Hyperloom** web interface — powered by **[PrimusClaw](https://github.com/AMD-AGI/Primus-Claw)**, the hosted online mode designed for **large-scale reachability**. Any team member can launch an optimization through the browser without local GPU setup or environment configuration.

- **Easy to scale** — each job runs in isolated sandboxed containers (GPU or CPU). Single-node optimizations run in-sandbox; multi-node workloads fan out via RayJob for distributed benchmarking.
- **Data flywheel** — every optimization run feeds results back through Minio storage and Langfuse observability, creating a closed feedback loop that continuously improves the agent's knowledge base and scoring heuristics.
- **Full Skills support** — sandboxes load optimization Skills on demand, giving the agent the same profiling, kernel-rewrite, and domain-specific capabilities at cloud scale.

1. Go to **[core42.primus-safe.amd.com/hyperloom](https://core42.primus-safe.amd.com/hyperloom/)**
2. Select **Claw Agent** or **Get Started** from the landing page to enter PrimusClaw
   <p align="center"><img width="500" alt="Hyperloom Landing" src="slides/hyperloom_landing.png" /></p>
3. Hyperloom (tab): End-to-end Model Performance Optimization
   <p align="center"><img width="500" alt="Hyperloom PrimusClaw UI" src="slides/hyperloom_claw_v2.png" /></p>
4. TraceLens-only (tab): Model Performance/gap analysis and bridge planning
   <p align="center"><img width="500" alt="TraceLens Config" src="slides/tracelens_quickstart.png" /></p>
5. GEAK-only: Kernel optimization
   <p align="center"><img width="500" alt="GEAK Config" src="slides/geak_quickstart.png" /></p>

---

## Quickstart — Local Mode (Cursor)

### Inference Optimization

`inference_optimizer/` is the single, canonical skill in this repo. It uses a multi-agent architecture with a Python Coordinator orchestrating four agent roles: Orchestration, Kernel, Critic, and Robustness. All optimization tools (GEAK, OOB, TraceLens) are scheduled via a local Ray cluster — no MCP remote services required.

#### Step 1 — Prepare GPU Environment

An AMD GPU node is required, with MI300X and MI355X supported. Use a local machine or request an Authoring Pod on [Primus-SaFE](https://core42.primus-safe.amd.com/authoring).

The inference image can be an SGLang or vLLM ROCm image. The example below uses SGLang:

```bash
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video \
  lmsysorg/sglang:v0.5.11-rocm720-mi30x
```

Image examples:

- SGLang MI300X: `lmsysorg/sglang:v0.5.11-rocm720-mi30x`
- SGLang MI355X: `lmsysorg/sglang:v0.5.11-rocm720-mi35x`
- vLLM MI300X: `vllm/vllm-openai-rocm:v0.18.0`
- vLLM MI355X: `vllm/vllm-openai-rocm:v0.18.0`

#### Step 2 — Prepare Source Trees and Configure Environment Variables

Prepare Hyperloom and its dependency source trees on the GPU node, then set the path environment variables explicitly. Hyperloom does not pin these internal source repositories to fixed paths; runtime uses the repo paths you provide through the environment.

**Required:**

| Variable | Description | Example |
|----------|-------------|---------|
| `SAFE_API_KEY` | LLM gateway auth key | `ak-your-safe-apikey` |
| `OPENAI_BASE_URL` | LLM gateway endpoint | `https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1` |

**Optional (Cursor kernel-opt backend):**

| Variable | Description | Example |
|----------|-------------|---------|
| `CURSOR_API_KEY` | Cursor SDK key for the OOB cursor backend; independent issuer (Cursor account, prefix `crsr_...`). When unset, Hyperloom auto-skips cursor from default backend selection and only races claude/codex/geak. | `crsr_xxxxxxxxxxxx` |
| `CURSOR_DEFAULT_MODEL` | Override the default Cursor model id. | `claude-opus-4-7-thinking-xhigh` (default) |

**Path configuration:**

| Variable | Description | Example |
|----------|-------------|---------|
| `REPO_ROOT` | Hyperloom repo root | `/workspace/Hyperloom` |
| `OOB_SRC` | OOB source root | `/workspace/OOB` |
| `INFERENCEX_PATH` | InferenceX repo root | `/workspace/InferenceX` |
| `TRACELENS_ROOT` | TraceLens-internal repo root | `/workspace/TraceLens-internal` |

Prepare the source trees from the corresponding repositories:

- Hyperloom: this repository; clone it and point `REPO_ROOT` at the repo root.
- OOB: clone the [AMD-AGI/Primus-Claw](https://github.com/AMD-AGI/Primus-Claw) repository, then point `OOB_SRC` at its `OOB/` subdirectory.
- InferenceX: [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX); point `INFERENCEX_PATH` at the local repo root.
- TraceLens-internal: [AMD-AGI/TraceLens-internal](https://github.com/AMD-AGI/TraceLens-internal/); checkout `release/hyperloom_integration_v0.3` and point `TRACELENS_ROOT` at that checkout.

> `SAFE_API_KEY` is obtained from [LLM Gateway](https://core42.primus-safe.amd.com/litellm-gateway). GEAK and OOB (claude/codex) API Key / Base URL are automatically inherited from `SAFE_API_KEY` / `OPENAI_BASE_URL` — no separate configuration needed. The OOB **cursor** backend is the exception: it talks to Cursor's own gateway and requires a separate `CURSOR_API_KEY`. If `CURSOR_API_KEY` is unset, cursor is silently skipped from default kernel-opt selection.

#### Step 3 — Connect via Cursor Remote SSH

1. Connect to the GPU node via Remote SSH in Cursor
2. Open the Hyperloom directory pointed to by `$REPO_ROOT`
3. `cd "$REPO_ROOT"` to ensure skill files load correctly

#### Step 4 — Launch Inference Optimization in Cursor Chat

Reference `inference_optimizer/SKILL.md` and describe the task in natural language.

**Basic usage:**

```
@$REPO_ROOT/inference_optimizer/SKILL.md
Optimize /path/to/your/model inference on MI300X.
--framework sglang --max-hours 2
```

**Full parameter example (long-running):**

```
@$REPO_ROOT/inference_optimizer/SKILL.md
Optimize /path/to/your/model inference on MI300X.

Environment:
- FRAMEWORK=sglang
- GPU_TYPE=MI300X
- TP=8, CONC=64, ISL=1024, OSL=1024
- PRECISION=bf16
- --target-gain 10
- --max-hours 24
- Run in background: setsid nohup

REPO_ROOT: /workspace/Hyperloom
OOB_SRC: /workspace/OOB
INFERENCEX_PATH: /workspace/InferenceX
TRACELENS_ROOT: /workspace/TraceLens-internal

Requirements:
1. Report the session ID, log path, PID, and initial health check result.
2. Monitor the process every 300s until done.
```

The agent automatically:

1. Installs all dependencies (Ray, TraceLens CLI, GEAK v3.1.0 CLI, OOB CLI + auth-proxy)
2. Launches the `inference_optimizer optimize` CLI
3. Multi-agent system autonomously executes: baseline → profile → param tuning → kernel optimization → E2E validation
4. Reports progress periodically (cumulative gain, current phase, best config)

#### Parameter Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--framework` | Inference framework (sglang / vllm) | sglang |
| `--max-hours` | Maximum run time (hours) | 2 |
| `--target-gain` | Target throughput improvement % | 10 |
| `--no-kernel` | Param tuning only, skip kernel optimization | disabled |
| `--gpu-type` | GPU model (MI300X / MI355X) | auto-detect |

> Training and MLPerf-training skills have been retired from this repo. Only inference optimization is supported here.

---

## Key Results

### Inference Optimization — InferenceX Challenge

Hyperloom optimized 4 flagship models for the [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) benchmark on AMD Instinct MI355X, matching or beating NVIDIA B200 on 3 out of 4 models.

| Model | Best tok/s/GPU | vs MI355X Baseline | vs NVIDIA B200 |
|-------|---------------:|:------------------:|:--------------:|
| DeepSeek-R1-0528 (671B MoE) | **1,476** | — | **+97% ahead** |
| GLM-5-FP8 (756B MoE+NSA) | **509** | **+193%** | **+27% ahead** |
| Qwen3.5-397B (397B MoE) | **350** | **+40%** | **+2.5% ahead** |
| MiniMax-M2.5 (MoE 256E) | **2,276** | **+6.5%** | **+5.7% ahead** |
| gpt-oss-120b (120B MoE, mxfp4) | **11,643** | — | **+34% ahead** |

All benchmarks: ISL=1024, OSL=1024 on MI355X (gfx950). "vs B200" shows best concurrency point. Full concurrency/ISL/OSL sweeps, patches, configs, and reproduction scripts: **[Agentic-InferenceX](https://github.com/AMD-AGI/Agentic-InferenceX)**.

## Detailed Skill Documentation

The repo ships a single skill — `inference_optimizer/` — with the full optimization protocol, examples, and a knowledge base of lessons learned from prior runs:

| Domain | Skill | Description |
|--------|-------|-------------|
| **Inference** | [SKILL.md](inference_optimizer/SKILL.md) | Multi-agent system, CLI-driven, fully automated |

The skill file is the agent's instructions. It encodes the full optimization methodology — setup, profiling protocol, what to try, how to measure, when to stop, and how to report. The knowledge base sections are updated live during runs with new pitfalls and validated results.

---

## Repo Structure

```
Hyperloom/
├── inference_optimizer/                  # Inference optimization skill (sole entry point)
│   ├── SKILL.md                          # Skill spec (Cursor/Claw entry point)
│   ├── cli.py                            # CLI entry: inference_optimizer optimize
│   ├── actions/_meta/                    # Action metadata and scheduling policy
│   ├── baseline_comparison/              # InferenceX baseline comparison and target analysis
│   ├── orchestrator/                     # Coordinator + agent roles + action executors
│   │   ├── action_executors/             # Executors for baseline/profile/params/sweep, etc.
│   │   ├── backends/                     # Claude/Codex/Critic backend adapters
│   │   └── system_prompts/               # Orchestration prompt construction
│   ├── scripts/                          # Install scripts, baseline/profile configs
│   └── tests/                            # Inference optimizer unit and regression tests
├── kernel-agent/                         # Kernel Agent toolkit (TraceLens/GEAK/OOB tools)
│   ├── SKILL.md                          # Kernel Agent operation spec
│   ├── tools/                            # TraceLens analysis, kernel optimization, patch apply
│   │   └── backends/                     # GEAK/OOB submission (Ray-scheduled)
│   ├── scripts/                          # Runtime setup scripts: install.sh, auth proxy, etc.
│   └── tests/                            # Kernel Agent tool tests
├── critic-agent/                         # Critic-agent subprocess runtime (proposal review)
├── robustness-agent/                     # Robustness-agent subprocess runtime (health/RCA)
├── ci/                                   # CI orchestration (PR submitter, AB test)
├── docs/                                 # Architecture docs, case studies, and Mermaid diagrams
├── scripts/                              # Repo-level helper scripts
├── .env.template                         # Environment variables
└── README.md
```

