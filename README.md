# ROCm Hyperloom

An agentic system that autonomously optimizes LLM inference and training on AMD GPUs. Hyperloom treats optimization as a **search problem**: given a workload, it builds a tree of candidate optimizations — backend swaps, server parameters, GEMM tuning, kernel rewrites, parallelism configs — scores each by expected gain and cost, then explores depth-first, always measuring against the real workload. Simply provide your workload and the agent delivers a fully optimized codebase — profiling against peak hardware potential, identifying bottlenecks, and iteratively rewriting code to maximize throughput on AMD GPUs, so the team gets production-ready optimized code.

<p align="center"><img width="500" height="400" alt="HyperLoom Architecture" src="slides/hyperloom_loop.png" /></p>

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

Bind your **[LLM Gateway](https://llm.amd.com/)** key to **[Hyperloom](https://oci-slc.primus-safe.amd.com/hyperloom/)** to obtain your `AK_YOUR_API_KEY`. This key is required for both the Hyperloom UI and the local optimization workflow — it provides access to TraceLens, GEAK, and OOB services.

---

## Quickstart — Hyperloom UI (PrimusClaw)

The fastest way to start is through the hosted **AMD Hyperloom** web interface — powered by **[PrimusClaw](https://github.com/AMD-AGI/Primus-Claw)**, the hosted online mode designed for **large-scale reachability**. Any team member can launch an optimization through the browser without local GPU setup or environment configuration.

- **Easy to scale** — each job runs in isolated sandboxed containers (GPU or CPU). Single-node optimizations run in-sandbox; multi-node workloads fan out via RayJob for distributed benchmarking.
- **Data flywheel** — every optimization run feeds results back through Minio storage and Langfuse observability, creating a closed feedback loop that continuously improves the agent's knowledge base and scoring heuristics.
- **Full MCP + Skills support** — sandboxes connect to BenchMark/RayJob, TraceLens Jarvis, GEAK, OOB, and InferenceX via MCP (local and remote), and load optimization Skills on demand, giving the agent the same profiling, kernel-rewrite, and domain-specific capabilities at cloud scale.

1. Go to **[oci-slc.primus-safe.amd.com/hyperloom](https://oci-slc.primus-safe.amd.com/hyperloom/)**
2. Select **Claw Agent** or **Get Started** from the landing page to enter PrimusClaw
   <p align="center"><img width="500" alt="Hyperloom Landing" src="slides/hyperloom_landing.png" /></p>
3. Hyperloom (tab): End-to-end Model Performance Optimization
   <p align="center"><img width="500" alt="Hyperloom PrimusClaw UI" src="slides/hyperloom_claw_quickstart.png" /></p>
4. TraceLens-only (tab): Model Performance/gap analysis and bridge planning
   <p align="center"><img width="500" alt="TraceLens Config" src="slides/tracelens_quickstart.png" /></p>
5. GEAK-only: Kernel optimization
   <p align="center"><img width="500" alt="GEAK Config" src="slides/geak_quickstart.png" /></p>

---

## Quickstart — Local Optimization (Cursor / Claude / VS Code)

### 1. Configure MCP Servers

Hyperloom uses two external tools as MCP servers, configured in `.cursor/mcp.json`:

- **TraceLens Agentic Analysis** — for profiling analysis (kernel breakdown, roofline modeling). Used during the profile phase.
- **GEAK** — for kernel-level optimization (rewrites Triton/HIP source). Used when the agent identifies hot custom kernels worth optimizing.

Update the GEAK authorization key in `.cursor/mcp.json`:

```json
{
  "geak-agent": {
    "headers": {
      "Authorization": "$AK_YOUR_API_KEY"
    }
  }
}
```

### 2. Environment

A GPU node is required to run benchmarks. You can either use a local GPU machine or request an Authoring Pod on **[Primus-SaFE](https://oci-slc.primus-safe.amd.com/authoring)**. For example, an inference optimization workload typically runs on an image like:

```bash
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video \
  rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260324
```

Then configure your API key:

```bash
cp .env.template .env
# Edit .env — set AK_YOUR_API_KEY for TraceLens, GEAK, and OOB
```

### 3. Run an Optimization

Reference a skill file in your Cursor chat with `@` and describe the workload:

**Inference:**
```
@.cursor/skills/inference-optimization/SKILL.md
Optimize Qwen3-30B-A3B inference on MI355X.
Target: (Specify throughput targets)
Model: /path/to/Qwen3-30B-A3B
InferenceX: /path/to/InferenceX
```

**Training:**
```
@.cursor/skills/training-optimization/SKILL.md
Optimize GPT-OSS 20B training on 8x MI355X.
Config: examples/megatron/configs/MI355X/gpt_oss_20B-BF16-pretrain.yaml
```

**MLPerf Training (GPT-OSS-20B):**

> **Image:** `harbor.oci-slc.primus-safe.amd.com/custom/tasimage/primus:202604070309`
>
> Before running, configure two API keys:
> 1. `training_optimization/mlperf/config_MI355X_1x8x1_fp8.sh` line 29 — set `HF_TOKEN` to your HuggingFace token
> 2. `.cursor/mcp.json` — replace all `<SAFE_API_KEY>ak-xxxxx` with your SAFE API key
>
> See the [MLPerf skill Readme](.cursor/skills/mlperf-optimization/Readme.MD) for full setup instructions.

```
@.cursor/skills/mlperf-optimization/SKILL.md
Use the mlperf-optimization skill to optimize GPT-OSS-20B MLPerf training performance.
Benchmark: gpt-oss-20b (MLPerf Training 5.1.0)
Quality target: validation log perplexity = 3.34
MLPERF_DIR: /root/Hyperloom-plus-mlperf/training_optimization/mlperf
Config script: config_MI355X_1x8x1_fp8.sh
GPU: 8x MI355X, 1 node
```

The agent takes it from there — baseline, profile, loop, report.

---

## Quickstart — Fully Local Mode (Docker / K8s)

Run Hyperloom on your own GPU infrastructure — a single container bundles all MCP services (TraceLens, GEAK, OOB Agent), InferenceX, and Skills. No manual MCP or environment setup required.

```bash
docker run -d --shm-size=16g \
  --device=/dev/kfd --device=/dev/dri \
  -v /path/to/models:/models \
  -p 20022:22 \
  -e LLM_API_KEY=<your-key> \
  -e LLM_API_BASE=https://api.openai.com/v1 \
  hyperloom-local:sglang-latest
```

Connect via Cursor Remote SSH → `localhost:20022` → open `/opt/hyperloom`, then:

```
@inference-optimization Optimize /models/Qwen3-30B-A3B
```

Full setup guide: **[deploy/fully-local/README.md](deploy/fully-local/README.md)** | Design doc: **[deploy/fully-local/DESIGN.md](deploy/fully-local/DESIGN.md)**

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

### Training Optimization

| Workload | GPUs | Baseline | Optimized | Speedup | Attempts |
|----------|------|----------|-----------|---------|----------|
| GPT-OSS 20B (MoE, BF16) | 8x MI355X | 13,708 ms/iter | 13,060 ms/iter | **+4.7%** | 57 (4hr) |
| GPT-OSS 20B (MoE, BF16) | 8x MI355X | 13,265 ms/iter | 13,042 ms/iter | **+1.7%** | 9 (1hr) |
| Qwen3 8B (full finetune) | 8x MI355X | 863 tok/s/gpu | 18,860 tok/s/gpu | **+2,085%** | 19 |
| Qwen3 32B (LoRA) | 8x MI355X | 467 tok/s/gpu | 4,984 tok/s/gpu | **+967%** | 19 |
| Llama 4 Scout 17B-16E (MoE) | 8x MI355X | 20 tok/s/gpu | 170 tok/s/gpu | **+750%** | 18 |

## Detailed Skill Documentation

Each domain has a comprehensive skill file with the full optimization protocol, examples, and a knowledge base of lessons learned from prior runs:

| Domain | Skill | Detailed README |
|--------|-------|-----------------|
| **Training** | [SKILL.md](.cursor/skills/training-optimization/SKILL.md) | [README](.cursor/skills/training-optimization/README.md) |
| **Inference** | [SKILL.md](.cursor/skills/inference-optimization/SKILL.md) | [README](.cursor/skills/inference-optimization/README.md) |
| **MLPerf Training** | [SKILL.md](.cursor/skills/mlperf-optimization/SKILL.md) | [Readme](.cursor/skills/mlperf-optimization/Readme.MD) |

The skill files are the agent's instructions. They encode the full optimization methodology — setup, profiling protocol, what to try, how to measure, when to stop, and how to report. The knowledge base sections are updated live during runs with new pitfalls and validated results.

---

## Repo Structure

```
Hyperloom/
├── .cursor/
│   ├── mcp.json                          # MCP server config (TraceLens + GEAK)
│   └── skills/
│       ├── training-optimization/        # Training optimization skill + knowledge base
│       ├── inference-optimization/       # Inference optimization skill + scripts
│       └── mlperf-optimization/          # MLPerf training optimization skill
├── inference_optimization/
│   └── InferenceX/                       # Inference benchmarking framework
├── training_optimization/
│   ├── turboquant/                       # Quantization evaluation library
│   └── mlperf/                           # MLPerf GPT-OSS-20B benchmark code
├── deploy/
│   └── fully-local/                      # Fully Local mode: containerized deployment for user-owned infra
├── dashboards/                            # Interactive optimization dashboard (HTML)
├── slides/                               # Architecture diagrams
├── .env.template                         # Environment variables
└── README.md
```

