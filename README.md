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

### Environment Setup

Local Mode runs Hyperloom in a remote AMD GPU environment, then uses Cursor to connect to that environment and launch optimization. Complete these three steps in order:

1. Prepare the GPU environment.
2. Connect Cursor to that environment.
3. Clone Hyperloom in the remote environment and run the bootstrap script.

#### 1. Prepare the GPU Environment

You need an AMD GPU machine that supports MI300X or MI355X, using an SGLang or vLLM ROCm inference image. Example images:

- SGLang MI300X: `lmsysorg/sglang:v0.5.11-rocm720-mi30x`
- SGLang MI355X: `lmsysorg/sglang:v0.5.11-rocm720-mi35x`
- vLLM MI300X: `vllm/vllm-openai-rocm:v0.19.0`
- vLLM MI355X: `vllm/vllm-openai-rocm:v0.19.0`

Choose one environment:

- **SaFE Authoring Pod**: create an Authoring Pod on [Primus-SaFE Authoring](https://core42.primus-safe.amd.com/authoring), select one of the SGLang or vLLM images above, and wait for the Pod to become ready.
- **Your own GPU machine**: start a long-running ROCm inference container that can access the GPU. The container name, workspace path, model path, and image version in the example below can all be changed for your environment.

Minimal Docker example for your own GPU machine:

```bash
docker run -d \
  --name hyperloom-local \
  --network host \
  --shm-size 64g \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -v /path/to/workspace:/workspace \
  -v /path/to/models:/models \
  lmsysorg/sglang:v0.5.11-rocm720-mi30x \
  tail -f /dev/null
```

If Hyperloom is already cloned on the host, you can mount that checkout directly into the container, for example by replacing `-v /path/to/workspace:/workspace` with `-v /path/on/host/Hyperloom:/workspace/Hyperloom`. Then open `/workspace/Hyperloom` after attaching Cursor to the container; you do not need to clone Hyperloom again inside the container.

> If HTTPS requests to `core42.primus-safe.amd.com` or the AMD LLM Gateway fail with a certificate verification error inside the container, install the AMD certificate bundle manually. This is most common when running on your own GPU server or a custom container image:
>
> ```bash
> curl -fsSL https://raw.githubusercontent.com/AMD-AGI/Primus-SaFE/main/Scripts/setup-certs/setup.sh | bash
> ```

#### 2. Connect Cursor to the Runtime Environment

- **SaFE Authoring Pod**: when the Pod is ready, check the connection instructions in the SaFE Authoring page and follow them to connect with Cursor Remote SSH.
- **Your own GPU machine + Docker**: first connect Cursor Remote SSH to the server running Docker, then use Dev Containers / Attach to Running Container to select `hyperloom-local` and open `/workspace` inside the container.

> Tip: to attach Cursor to a Docker container on a remote server:
>
> 1. Open the command palette in Cursor: `Ctrl+Shift+P`.
> 2. Search for `Remote-SSH: Connect to Host...` and connect to the server running Docker.
> 3. In that SSH remote window, open Extensions: `Ctrl+Shift+X`.
> 4. Search for and install `Dev Containers`, making sure it is installed in the current remote environment.
> 5. Open the command palette again: `Ctrl+Shift+P`.
> 6. Search for `Dev Containers: Attach to Running Container...`.
> 7. Select `hyperloom-local` (or your container name) and open `/workspace` inside the container.

#### 3. Clone Hyperloom and Bootstrap Local Mode

In the remote environment, first make sure GitHub authentication and AMD-AGI repository access are available. `local_setup.sh` will use the same access to clone dependency repositories.

If the remote environment does not already have a Hyperloom checkout, clone this repository:

```bash
git clone https://github.com/AMD-AGI/Hyperloom.git
cd Hyperloom
```

If Hyperloom was mounted through a Docker volume or a SaFE Authoring Pod, enter the mounted checkout directly:

```bash
cd /workspace/Hyperloom
```

Prepare Hyperloom credentials:

```bash
cp .env.template .env
```

Edit `.env`:

```env
SAFE_API_KEY=ak-your-safe-apikey
OPENAI_BASE_URL=https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1
```

| Variable | Description | Example |
|----------|-------------|---------|
| `SAFE_API_KEY` | LLM gateway auth key | `ak-your-safe-apikey` |
| `OPENAI_BASE_URL` | LLM gateway endpoint | `https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1` |

Then run the Local Mode bootstrap:

```bash
export USER_DATA_PATH=/path/to/hyperloom-run
bash inference_optimizer/scripts/local_setup.sh
```

`USER_DATA_PATH` is Hyperloom's runtime directory for dependency checkouts, logs, state, and optimization results. It is not the Hyperloom source checkout, and you can point it at any location with enough space. `local_setup.sh` derives `REPO_ROOT` from its own location, clones and wires OOB, InferenceX, and TraceLens, and writes a local env file. When it finishes, it prints:

- the Hyperloom workspace path to open in Cursor;
- the prompt template to paste into Cursor Chat;
- the env file that the agent should source before launch.

Example output snippet:

````text
Open this folder in Cursor as the workspace:
  /path/to/Hyperloom

Before launching Hyperloom from Cursor Chat, ask the agent to run exactly:

```bash
source '/path/to/hyperloom-run/runtime/local-setup.env.sh'
export USER_DATA_PATH='/path/to/hyperloom-run'
```

Paste this into Cursor Chat and fill in your workload:

@inference_optimizer/SKILL.md

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

Before launch, run exactly:
```bash
source '/path/to/hyperloom-run/runtime/local-setup.env.sh'
export USER_DATA_PATH='/path/to/hyperloom-run'
```

Requirements:
1. Report the session ID, log path, PID, and initial health check result.
2. Monitor the process every 300s until the optimization is complete or failed.
````

Follow the script output. In the default flow, users do not need to manually configure GEAK, OOB, InferenceX, or TraceLens.

**Optional (Cursor kernel-opt backend):**

| Variable | Description | Example |
|----------|-------------|---------|
| `CURSOR_API_KEY` | Cursor SDK key for the OOB cursor backend; independent issuer (Cursor account, prefix `crsr_...`). When unset, Hyperloom auto-skips cursor from default backend selection and only races claude/codex/geak. | `crsr_xxxxxxxxxxxx` |
| `CURSOR_DEFAULT_MODEL` | Override the default Cursor model id. | `claude-opus-4-7` (default) |

> `SAFE_API_KEY` is obtained from [LLM Gateway](https://core42.primus-safe.amd.com/litellm-gateway). GEAK and OOB (claude/codex) API Key / Base URL are automatically inherited from `SAFE_API_KEY` / `OPENAI_BASE_URL`. You can place these values in `$REPO_ROOT/.env`; no separate GEAK, OOB, InferenceX, or TraceLens configuration is needed. The OOB **cursor** backend is the exception: it talks to Cursor's own gateway and requires a separate `CURSOR_API_KEY`. If `CURSOR_API_KEY` is unset, cursor is silently skipped from default kernel-opt selection.

### Launch Inference Optimization

After setup, open the Hyperloom workspace printed by `local_setup.sh` in Cursor, then paste the generated prompt template into Cursor Chat. Replace the model path, framework, GPU type, budget, and any other workload parameters before sending.

**Resume an existing session:**

Example prompt:

```text
@inference_optimizer/SKILL.md

Resume the existing Hyperloom optimization session.

Requirements:
1. Launch `inference_optimizer optimize --resume`; do not start a new session.
2. Do not pass `--model`; read the model and workload from the saved manifest/state.
3. Before launching, verify `manifest.json` and `state.json` exist.
4. Report the log path, PID, initial health check result, current phase, cumulative gain, and best config.
5. Monitor the process every 300s until the optimization is complete or failed.
```

Prompt field reference:

| Field | Maps to | Description | Default |
|---|---|---|---|
| `Model` | `--model`, `MODEL_PATH` | Model path. Required for a new run; ignored when resuming. | required |
| `Framework` | `--framework`, `FRAMEWORK` | Inference framework: `sglang` or `vllm`. Do not mix frameworks within one session. | `sglang` |
| `GPU` | `--gpu-type`, `GPU_TYPE` | Target GPU type, such as `MI300X`, `MI325X`, or `MI355X`; can also be auto-detected. | auto-detect |
| `Model class` | `--model-class` | Model architecture type used for action selection and scoring. | unset |
| `Compare against GPU` | `--compare-against-gpu` | Optional external reference GPU, such as `B200`. When unset, optimization continues without fetching an external baseline. | unset |
| `TP` | `TP` | Tensor parallel size. | `1` |
| `CONC` | `CONC` | Benchmark concurrency. | YAML default, commonly `8` |
| `ISL` | `--isl`, `ISL` | Input sequence length. | `256` |
| `OSL` | `--osl`, `OSL` | Output sequence length. | `256` |
| `Precision` | `--precision`, `PRECISION` | Model precision, for example `bf16`. | `bf16` |
| `Goal` | `--target-gain`, `--target-tput`, `--target-baseline-dir` | Optional stop condition, such as target throughput gain. | unset |
| `Budget` | `--max-hours` | Maximum optimization time. | `2.0` hours |
| `Kernel optimization` | `--no-kernel` | Kernel optimization is enabled by default; ask to skip it if you only want parameter/backend search. | enabled |
| `Resume` | `--resume` | Resume an existing session; requires `manifest.json` and `state.json`. | disabled |

For first-launch errors, see `inference_optimizer/SKILL.md` §"Failure Handling".

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

