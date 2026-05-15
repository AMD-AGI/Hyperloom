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

Bind your **[LLM Gateway](https://llm.amd.com/)** key to **[Hyperloom](https://core42.example-internal-host.invalid/hyperloom/)** to obtain your `AK_YOUR_API_KEY`. This key is required for both the Hyperloom UI and the local optimization workflow — it provides access to TraceLens, GEAK, and OOB services.

---

## Quickstart — Hyperloom UI (PrimusClaw)

The fastest way to start is through the hosted **AMD Hyperloom** web interface — powered by **[PrimusClaw](https://github.com/AMD-AGI/Primus-Claw)**, the hosted online mode designed for **large-scale reachability**. Any team member can launch an optimization through the browser without local GPU setup or environment configuration.

- **Easy to scale** — each job runs in isolated sandboxed containers (GPU or CPU). Single-node optimizations run in-sandbox; multi-node workloads fan out via RayJob for distributed benchmarking.
- **Data flywheel** — every optimization run feeds results back through Minio storage and Langfuse observability, creating a closed feedback loop that continuously improves the agent's knowledge base and scoring heuristics.
- **Full Skills support** — sandboxes load optimization Skills on demand, giving the agent the same profiling, kernel-rewrite, and domain-specific capabilities at cloud scale.

1. Go to **[core42.example-internal-host.invalid/hyperloom](https://core42.example-internal-host.invalid/hyperloom/)**
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

An AMD GPU node is required, with MI300X and MI355X supported. Two recommended setups:

- **Recommended — SaFE Authoring Pod**: create an Authoring Pod on [Primus-SaFE](https://core42.example-internal-host.invalid/authoring) using an SGLang or vLLM inference image.
- **Bring your own GPU host**: if you run on your own AMD GPU server, you can use one of the image examples below.

Image examples:

- SGLang MI300X: `lmsysorg/sglang:v0.5.11-rocm720-mi30x`
- SGLang MI355X: `lmsysorg/sglang:v0.5.11-rocm720-mi35x`
- vLLM MI300X: `vllm/vllm-openai-rocm:v0.18.0`
- vLLM MI355X: `vllm/vllm-openai-rocm:v0.18.0`

#### Step 2 — Prepare Source Trees and Configure Environment Variables

Prepare Hyperloom and its dependency source trees on the GPU node, then set the path environment variables explicitly. Hyperloom does not pin these internal source repositories to fixed paths; runtime uses the repo paths you provide through the environment.

**Required credentials:**

Hyperloom can read credentials from `$REPO_ROOT/.env`. The recommended setup is to copy the template and fill in your key:

```bash
cd "$REPO_ROOT"
cp .env.template .env
```

Edit `.env`:

```env
SAFE_API_KEY=ak-your-safe-apikey
OPENAI_BASE_URL=https://core42.example-internal-host.invalid/api/v1/llm-proxy/v1
```

| Variable | Description | Example |
|----------|-------------|---------|
| `SAFE_API_KEY` | LLM gateway auth key | `ak-your-safe-apikey` |
| `OPENAI_BASE_URL` | LLM gateway endpoint | `https://core42.example-internal-host.invalid/api/v1/llm-proxy/v1` |

Shell environment variables take precedence over values in `.env`, so advanced users can still export these variables directly.

**Optional (Cursor kernel-opt backend):**

| Variable | Description | Example |
|----------|-------------|---------|
| `CURSOR_API_KEY` | Cursor SDK key for the OOB cursor backend; independent issuer (Cursor account, prefix `crsr_...`). When unset, Hyperloom auto-skips cursor from default backend selection and only races claude/codex/geak. | `crsr_xxxxxxxxxxxx` |
| `CURSOR_DEFAULT_MODEL` | Override the default Cursor model id. | `claude-opus-4-7` (default) |

**Path configuration:**

These paths are used by the agent and installer to wire together the local Hyperloom stack:

| Path | Why it is needed |
|------|------------------|
| `REPO_ROOT` | Locates this Hyperloom repo, including `inference_optimizer/`, `kernel-agent/`, skills, scripts, and runtime assets. |
| `OOB_SRC` | Provides the OOB CLI and auth-proxy used by kernel optimization backends. |
| `INFERENCEX_PATH` | Provides InferenceX benchmark/evaluation code and reference data used during baseline and target analysis. |
| `TRACELENS_ROOT` | Provides TraceLens profiling tooling for bottleneck analysis and kernel selection. |
| `USER_DATA_PATH` | Session directory root for logs, runs, source mirrors, and all per-session artefacts. Optional, defaults to `/workspace/hyperloom`. |

Prepare the source trees from the corresponding repositories:

- Hyperloom: this repository; clone it and point `REPO_ROOT` at the repo root.
- OOB: clone the [AMD-AGI/Primus-Claw](https://github.com/AMD-AGI/Primus-Claw) repository, then point `OOB_SRC` at its `OOB/` subdirectory.
- InferenceX: [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX); point `INFERENCEX_PATH` at the local repo root.
- TraceLens-internal: [AMD-AGI/TraceLens-internal](https://github.com/AMD-AGI/TraceLens-internal/); checkout `release/hyperloom_integration_v0.3` and point `TRACELENS_ROOT` at that checkout.

> `SAFE_API_KEY` is obtained from [LLM Gateway](https://core42.example-internal-host.invalid/litellm-gateway). GEAK and OOB (claude/codex) API Key / Base URL are automatically inherited from `SAFE_API_KEY` / `OPENAI_BASE_URL`. You can place these values in `$REPO_ROOT/.env`; no separate GEAK or OOB configuration is needed. The OOB **cursor** backend is the exception: it talks to Cursor's own gateway and requires a separate `CURSOR_API_KEY`. If `CURSOR_API_KEY` is unset, cursor is silently skipped from default kernel-opt selection.

#### Step 3 — Connect via Cursor Remote SSH

1. Connect to the GPU node via Remote SSH in Cursor
2. Open the Hyperloom directory pointed to by `$REPO_ROOT`
3. `cd "$REPO_ROOT"` to ensure skill files load correctly

#### Step 4 — Launch Inference Optimization in Cursor Chat

Reference `inference_optimizer/SKILL.md` in Cursor chat and describe the workload. The agent reads the skill, installs dependencies automatically, translates your workload into the appropriate CLI/environment settings, launches `inference_optimizer optimize`, and reports progress.

**Minimal launch:**

Example prompt:

```text
@$REPO_ROOT/inference_optimizer/SKILL.md

Optimize this model:
- Model: /path/to/your/model
- GPU: MI300X
- Framework: sglang
- Budget: 2 hours
```

**Typical long-running launch:**

Example prompt:

```text
@$REPO_ROOT/inference_optimizer/SKILL.md

Optimize inference for this workload:
- Model: /path/to/your/model
- Framework: sglang
- GPU: MI300X
- TP: 8
- CONC: 64
- ISL: 1024
- OSL: 1024
- Precision: bf16
- Goal: improve throughput by at least 10%
- Budget: 24 hours

Paths:
- Hyperloom repo: $REPO_ROOT
- OOB source: $OOB_SRC
- InferenceX repo: $INFERENCEX_PATH
- TraceLens repo: $TRACELENS_ROOT

Requirements:
1. Report the session ID, log path, PID, and initial health check result.
2. Monitor the process every 300s until the optimization is complete or failed.
```

**Resume an existing session:**

Example prompt:

```text
@$REPO_ROOT/inference_optimizer/SKILL.md

Resume the existing Hyperloom optimization session.

Requirements:
1. Launch `inference_optimizer optimize --resume`; do not start a new session.
2. Do not pass `--model`; read the model and workload from the saved manifest/state.
3. Before launching, verify `manifest.json` and `state.json` exist.
4. Report the log path, PID, initial health check result, current phase, cumulative gain, and best config.
5. Monitor the process every 300s until the optimization is complete or failed.
```

The agent automatically:

1. Installs all dependencies (Ray, TraceLens CLI, GEAK v3.1.0 CLI, OOB CLI + auth-proxy)
2. Launches `inference_optimizer optimize` with the resolved workload settings
3. Executes baseline → profile → param tuning → kernel optimization → E2E validation
4. Reports the session directory, log path, PID, initial health check, current phase, cumulative gain, and best config until the run completes or fails

For first-launch errors, see `inference_optimizer/SKILL.md` §"Failure Handling".

#### Workload Fields and Internal Mapping

The chat examples use user-facing field names. The agent maps them to the optimizer's CLI flags and environment variables:

| User-facing field | Maps to | Description | Default |
|-------------------|---------|-------------|---------|
| Model | `--model`, `MODEL_PATH` | Model path. Required for a new run; ignored when resuming. | required |
| Framework | `--framework`, `FRAMEWORK` | Inference framework: `sglang` or `vllm`. A session cannot mix frameworks. | `sglang` |
| GPU | `--gpu-type`, `GPU_TYPE` | Target GPU type: `mi300x`, `mi325x`, or `mi355x`; can also be auto-detected. | auto-detect |
| Model class | `--model-class` | Model architecture family used by the orchestrator's scoring and action selection. Supply explicitly; the agent no longer derives this from a `classify` action. | unset |
| Compare against GPU | `--compare-against-gpu` | Opt into InferenceX reference fetching for the named GPU (e.g. `B200`). When omitted, `target_analysis` records a `no_target_gpu_configured` marker and the run proceeds without an external reference. | unset |
| TP | `TP` | Tensor parallel size used by Magpie benchmark configs. | `1` |
| CONC | `CONC` | Benchmark concurrency. | YAML default, commonly `8` |
| ISL | `--isl`, `ISL` | Input sequence length. | `256` |
| OSL | `--osl`, `OSL` | Output sequence length. | `256` |
| Precision | `--precision`, `PRECISION` | Model precision, for example `bf16`. | `bf16` |
| Goal | `--target-gain`, `--target-tput`, `--target-baseline-dir` | Optional stop condition. If omitted, the run continues until the time budget or no-more-leverage stop reason. | unset |
| Budget | `--max-hours` | Wall-clock optimization budget in hours. | `2.0` |
| Kernel optimization | `--no-kernel` | By default kernel optimization is enabled. Ask to skip kernel optimization for parameter/backend search only. | enabled |
| Hyperloom repo | `REPO_ROOT` | Hyperloom repository root. | set in Step 2 |
| OOB source | `OOB_SRC` | OOB source root. | set in Step 2 |
| InferenceX repo | `INFERENCEX_PATH` | InferenceX repository root. | set in Step 2 |
| TraceLens repo | `TRACELENS_ROOT` | TraceLens-internal repository root. | set in Step 2 |
| Session directory | `USER_DATA_PATH` | Session directory for state, logs, runs, reports, and resume. To override, set `USER_DATA_PATH` in the shell or specify the path in your prompt. | `/workspace/hyperloom` |
| Resume | `--resume` | Resume the existing session from the session directory; requires `manifest.json` and `state.json`. | disabled |

> Training and MLPerf-training skills have been retired from this repo. Only inference optimization is supported here.

#### Migration Notes (upgrading from earlier Hyperloom releases)

1. **Session directory env renamed.** Set `USER_DATA_PATH` instead of `INFERENCE_OPTIMIZER_SESSION_DIR`. The legacy variable is no longer read.
2. **`setup` and `classify` actions removed.** If your launcher relied on them being in the action graph, supply the equivalents on the CLI:
   - `--model-class <…>` for what `classify` used to derive.
   - `--compare-against-gpu <…>` to opt into InferenceX reference fetching (otherwise `target_analysis` writes a `no_target_gpu_configured` marker and the run proceeds).
3. **Magpie benchmark script is now generic-pinned by default.** When `--gpu-type` is set, the YAML renderer pins `benchmark_script=<framework>_<gpu_type>.sh` to stop InferenceX-native scripts from silently leaking `result.json` outside the session dir. If you intentionally use a model-specific script (e.g. `dsr1_fp8_mi300x.sh`), keep passing `benchmark_script=` explicitly — operator overrides still win against the generic-script pin. You may additionally want to set `$INFERENCE_OPTIMIZER_RESCUE_PATHS` so the harvest step can recover leaked `result.json` files written to hardcoded `--result-dir` locations.

---

## Key Results

### Inference Optimization — InferenceX Challenge

Hyperloom optimized 4 flagship models for the [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) benchmark on AMD Instinct MI355X, matching target performance on 3 out of 4 models.

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

