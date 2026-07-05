# ROCm Hyperloom

An agentic system that autonomously optimizes LLM inference on AMD GPUs. Hyperloom treats optimization as a **search problem**: given a workload, it explores candidate optimizations — backend swaps, server parameters, GEMM tuning, kernel rewrites, parallelism configs — one change at a time, always measuring against the real workload and prioritizing the next move from prior results and KB-driven priors. Simply provide your workload and the agent delivers a fully optimized codebase — profiling against peak hardware potential, identifying bottlenecks, and iteratively rewriting code to maximize throughput on AMD GPUs, so the team gets production-ready optimized code.

<p align="center"><img width="600" alt="HyperLoom Architecture" src="slides/hyperloom_loop.png" /></p>

Block 1-3 - Workload understanding and profiling: Submit your workload as the starting point for the agent to understand your codebase, profile using [TraceLens Agentic Analysis](https://github.com/AMD-AGI/TraceLens/) (relies on [Magpie](https://github.com/AMD-AGI/Magpie) for trace collection), capture bottlenecks and roofline targets. Hyperloom uses the public TraceLens package (`TRACELENS_ROOT`) by default (open-source-only report). An optional internal TraceLens extension — roofline numbers, gains estimates, and MI355/MI455 MAF data — can be enabled by internal users who set `TRACELENS_INTERNAL_ROOT` to point at their own internal checkout (path self-provided); leave it unset to stay on the open-source-only report. There is no separate on/off toggle.

Block 4 - Code Optimization Loop: The core of Hyperloom. The agent explores candidates — config overrides, code patches, backend switches, kernel rewrites — one change at a time: **Think → Implement → Benchmark → Decide**. Each result informs which candidate to try next. 

In parallel, hot kernels are asynchronously optimized via external backends ([GEAK](https://github.com/AMD-AGI/GEAK/tree/main), and OOB kernel optimization via Claude Code and OpenAI Codex relying on kernel optimization flow of [Apex](https://github.com/AMD-AGI/Apex)). Kernel profiling and validation is powered by [Magpie](https://github.com/AMD-AGI/Magpie), which relies on [IntelliKit](https://github.com/AMDResearch/intellikit) for some of low-level GPU profiling tools.

Block 5-6 - Validated Delivery: The agent optimizes for throughput while maintaining accuracy — every change is correctness-gated before acceptance. Once the loop exits, the runtime writes the final report, reproducible session artifacts, and `session_breakdown.json` so downstream delivery workflows can package or review the optimized stack.

### Learn More

| | |
|---|---|
| **[Local Mode Quickstart (Cursor)](docs/QUICKSTART_LOCAL_MODE.md)** | Run Hyperloom in Docker on your own AMD GPU machine and drive it from Cursor |
| **[Quantization (AMD Quark)](docs/QUANTIZATION_QUARK.md)** | Optional `--quantize` prelude: Quark checkout requirement and `QUARK_ROOT` resolution |
| **[How the Optimization Loop Works](docs/HOW_THE_OPTIMIZATION_LOOP_WORKS.md)** | Conversational orchestration, phase sequencing, action gates, and KB-driven priors |
| **[GLM-5 — Discovering Optimizations Hard to Spot Manually](docs/CASE_STUDY_GLM5.md)** | Hidden GEMM configs, cross-repo kernel patches, +193% throughput |
| **[DeepSeek-R1 — Fast Scale-Up on a New Workload](docs/CASE_STUDY_DEEPSEEK_R1.md)** | 7 configs to optimal in one session, MTP scheduling fix, +97% over B200 |
| **[Auth & Environment Guide](docs/ENV_AND_AUTH.md)** | Single authoritative auth/env reference; the inline tables in this README are a convenience excerpt |
| **[Configuration Reference](docs/CONFIGURATION_REFERENCE.md)** | Every environment variable read by the runtime |
| **[Knowledge-Base Guide](docs/KB_GUIDE.md)** | Local recipe KB and optional Cortex KB setup |
| **[`session_breakdown.json` Integration](docs/INTEGRATION_SESSION_BREAKDOWN.md)** | Stable contract for downstream consumers (`claw-stats-service`, dashboards) |
| **[Operations & Self-Host Runbook](docs/OPERATIONS.md)** | k8s sizing, `USER_DATA_PATH` backup, disaster recovery |
| **[Troubleshooting](docs/TROUBLESHOOTING.md)** | Gateway 401, Ray `--num-gpus`, VRAM IR-1, and other recurring failures |
| **[Upgrading](docs/UPGRADING.md)** | Per-version migration steps (companion to [`CHANGELOG.md`](CHANGELOG.md)) |

---

## Quickstart — Hyperloom UI (PrimusClaw)

The fastest way to start is through the hosted **AMD Hyperloom** web interface — powered by **[PrimusClaw](https://github.com/AMD-AGI/Primus-Claw)**, the hosted online mode designed for **large-scale reachability**. Any team member can launch an optimization through the browser without local GPU setup or environment configuration.

- **Easy to scale** — each job runs in isolated sandboxed containers (GPU or CPU). Single-node optimizations run in-sandbox; multi-node workloads fan out via RayJob for distributed benchmarking.
- **Data flywheel** — every optimization run feeds results back through Minio storage and Langfuse observability, creating a closed feedback loop that continuously improves the agent's knowledge base and scoring heuristics.
- **Full Skills support** — sandboxes load optimization Skills on demand, giving the agent the same profiling, kernel-rewrite, and domain-specific capabilities at cloud scale.

1. Go to **[crusoe.example-internal-host.invalid/hyperloom](https://crusoe.example-internal-host.invalid/hyperloom/)**
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

Local Mode runs Hyperloom in a Docker container on your AMD GPU machine. Cursor attaches to the container and launches optimization. See **[docs/QUICKSTART_LOCAL_MODE.md](docs/QUICKSTART_LOCAL_MODE.md)** for the full setup guide (own-GPU quickstart, optional Primus-SaFE path, and launch instructions).

---

## Key Results

### Inference Optimization — InferenceX Challenge

Hyperloom optimized 4 flagship models for the [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) benchmark on AMD Instinct MI355X, matching or comparing with reference GPU on 3 out of 4 models.

| Model | Best tok/s/GPU | vs MI355X Baseline | vs NVIDIA B200 |
|-------|---------------:|:------------------:|:--------------:|
| DeepSeek-R1-0528 (671B MoE) | **1,476** | — | **+97% ahead** |
| GLM-5-FP8 (756B MoE+NSA) | **509** | **+193%** | **+27% ahead** |
| Qwen3.5-397B (397B MoE) | **350** | **+40%** | **+2.5% ahead** |
| MiniMax-M2.5 (MoE 256E) | **2,276** | **+6.5%** | **+5.7% ahead** |
| gpt-oss-120b (120B MoE, mxfp4) | **11,643** | — | **+34% ahead** |

All benchmarks: ISL=1024, OSL=1024 on MI355X (gfx950). "vs B200" shows best concurrency point. Full concurrency/ISL/OSL sweeps, patches, configs, and reproduction scripts: **[Agentic-InferenceX](https://github.com/AMD-AGI/Agentic-InferenceX)**.

## Hosted Tier — Limits & Pricing

The hosted [Hyperloom UI / PrimusClaw](https://crusoe.example-internal-host.invalid/hyperloom/)
is operated by AMD on shared infrastructure. Defaults for the public
PrimusClaw tier:

| Resource                          | Hosted default                                                                 |
|-----------------------------------|---------------------------------------------------------------------------------|
| Per-session GPU budget            | 1–8 × MI300X / MI325X / MI355X for single-node runs (matches TP); 16+ GPUs via RayJob for multi-node (nodes ≥ 2) |
| Concurrent sessions per account   | 2                                                                               |
| Session wall-clock                | 24 hours (extensible on request)                                                |
| `USER_DATA_PATH` quota            | 200 GB per session, with daily snapshots                                        |
| LLM-gateway request rate          | Bound to your `SAFE_API_KEY` quota (see [LLM Gateway](https://llm.amd.com/))     |
| Outbound network                  | Allowlisted (model registries, HuggingFace, GitHub)                             |

Pricing for the hosted tier is currently **free for AMD-internal users
and approved AMD partners** via Primus-SaFE. Public / enterprise
pricing is under active definition by the BRAIN team; reach out via
the [Hyperloom support form](https://crusoe.example-internal-host.invalid/hyperloom/)
or open an issue if your organization needs a quote.

For higher limits, dedicated capacity, or air-gapped deployment, self-host
Hyperloom in your own cluster following [`docs/OPERATIONS.md`](docs/OPERATIONS.md).
Self-hosted Hyperloom is MIT licensed (see
[Licensing](#licensing)); there is no per-seat or per-session fee.

---

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
├── kernel-agent/                         # Kernel-agent toolkit (TraceLens/GEAK/OOB tools)
│   ├── SKILL.md                          # Kernel-agent operation spec
│   ├── tools/                            # TraceLens analysis, kernel optimization, patch apply
│   │   └── backends/                     # GEAK/OOB submission (Ray-scheduled)
│   ├── scripts/                          # Runtime setup scripts: install.sh, etc.
│   └── tests/                            # Kernel-agent tool tests
├── critic-agent/                         # Critic-agent subprocess runtime (proposal review)
├── robustness-agent/                     # Robustness-agent subprocess runtime (health/RCA)
├── ci/                                   # CI orchestration (PR submitter, AB test)
├── docs/                                 # Architecture docs, case studies, and Mermaid diagrams
├── scripts/                              # Repo-level helper scripts
├── .env.template                         # Environment variables
├── CHANGELOG.md                          # Per-release notes
├── CONTRIBUTING.md                       # Contribution guidelines
├── LICENSE                               # MIT
├── SECURITY.md                           # Vulnerability disclosure policy
└── README.md
```

---

## Licensing

Hyperloom is released under the **MIT License**. The full license text
is in [`LICENSE`](LICENSE).

You may use Hyperloom commercially, modify it, and distribute it under
the terms of the MIT license, provided the copyright notice and the
permission notice are retained in all copies or substantial portions of
the software.

Third-party tools and agents (Cursor, Visual Studio, and Claude Code)
that Hyperloom invokes are governed by their own separate license terms
and are NOT covered by the MIT license above — see the "Third-Party
Tools and Agents" section in [`LICENSE`](LICENSE). You are responsible
for reviewing and complying with each tool's individual license.

For security-relevant issues, see [`SECURITY.md`](SECURITY.md). For
contribution conventions, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

