---
myst:
    html_meta:
        "description": "Learn about AgentKernelArena, a standardized AMD evaluation arena for measuring AI coding agent performance on GPU kernel optimization tasks on ROCm."
        "keywords": "AgentKernelArena, GPU kernel, optimization, AI agents, ROCm, HIP, Triton, AMD GPU, evaluation, benchmark, leaderboard, Hyperloom"
---
# AgentKernelArena

AgentKernelArena is a standardized evaluation arena, built by AMD, that measures
how well AI coding agents perform on real GPU kernel optimization tasks. It runs
LLM-powered agents side-by-side on the same tasks in isolated workspaces and
scores them with objective, reproducible metrics for compilation, correctness,
and GPU performance.

Within Hyperloom, AgentKernelArena provides the benchmarking harness used to
compare Hyperloom's kernel-optimization agents against other AI coding agents
under identical, reproducible conditions. It is also used to validate that
kernel tasks are correct and self-contained before they enter the leaderboard.

- **Source**: <https://github.com/AMD-AGI/AgentKernelArena>
- **License**: MIT

## Capabilities and task categories

AgentKernelArena runs each agent in a Docker-first runtime with pinned ROCm
SGLang images, isolates every task in its own timestamped workspace, and scores
results through a common evaluation pipeline:

- **Multi-agent arena** — Cursor Agent, Claude Code, Codex, and custom agent
  integrations.
- **Multi-model support** — OpenAI, Anthropic, and other models through
  OpenRouter or a local vLLM server.
- **Task categories** — HIP (`hip2hip`), CUDA-to-HIP (`cuda2hip`), Triton
  (`triton2triton`, `instruction2triton`), Torch-to-HIP (`torch2hip`), and
  FlyDSL (`flydsl2flydsl`), plus repository-level tasks.
- **Objective metrics** — automated compilation, correctness, and real GPU
  performance speedups recorded per task in `task_result.yaml`.
- **Workspace isolation** — each task runs in its own timestamped workspace for
  reproducibility.
- **Multi-GPU parallel runs** — one isolated Docker worker per GPU, sharing a
  common task queue.
- **A/B testing** — run the same task set with and without a new Model Context
  Protocol (MCP) server, skill, prompt, or tool and compare scores directly.
- **Task validator** — an agent that runs automated checks on task quality
  before tasks enter the leaderboard.
- **Visualization dashboard** — a static dashboard for comparing run reports
  across agents and configurations.

## Installation

Clone the repository and verify the Docker runtime and GPU:

```bash
git clone https://github.com/AMD-AGI/AgentKernelArena.git
cd AgentKernelArena

# Verify the container runtime and GPU.
make docker-smoke

# Verify agent CLI login reuse (Codex, Claude Code, Cursor Agent).
make docker-check-agents
```

Install whichever agent CLIs you plan to evaluate:

```bash
# Cursor Agent CLI
make install-cursor-agent

# Claude Code
npm install -g @anthropic-ai/claude-code
```

Export API keys for the providers you will use:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export OPENROUTER_API_KEY="your_openrouter_key"
```

## Usage

Evaluations are driven through `config.yaml` and the Docker-first Makefile
targets. Each entry in `tasks` is a path relative to the `tasks/` directory:

```yaml
agent:
  template: cursor          # one agent template per run

tasks:
  - hip2hip/gpumode/GELU
  - triton2triton/vllm/triton_rms_norm

target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace
```

| Entry | Selects |
|-------|---------|
| `all` | Every task in `tasks/` |
| `hip2hip` | All tasks under `tasks/hip2hip/` |
| `triton2triton/vllm` | All tasks under that subdirectory |
| `hip2hip/gpumode/GELU` | A single task |

Run an evaluation:

```bash
make docker-run CONFIG=config.yaml

# Add a suffix to label a run (useful for A/B testing)
make docker-run CONFIG=config.yaml RUN_ARGS="--run-suffix cursor_with_mcp"

# Distribute across multiple GPUs
make docker-parallel-run CONFIG=config.yaml GPU_IDS=0,1,2,3
```

Each task writes a `task_result.yaml` with the scored outcome:

```yaml
task_name: hip2hip/gpumode/GELU
pass_compilation: true
pass_correctness: true
base_execution_time: 1.82
best_optimized_execution_time: 1.15
speedup_ratio: 1.58
score: 278.0
```

Interrupted runs can be resumed without repeating completed tasks:

```bash
make docker-run CONFIG=config.yaml RUN_ARGS="--resume-latest"
```

## Role in Hyperloom

AgentKernelArena is the evaluation harness used to compare Hyperloom's
kernel-optimization agents against other AI coding agents on standardized GPU
kernel tasks. Its task categories (`hip2hip`, `triton2triton`, `flydsl2flydsl`,
and others) overlap directly with the kernel types that Hyperloom's kernel agent
handles through GEAK and Kernel-Forge.

The task validator is also used to verify that kernels produced by Hyperloom
optimization runs are correct and self-contained before they are published as
benchmark tasks.

## API reference

AgentKernelArena ships its own documentation in-repo; see the
[docs directory](https://github.com/AMD-AGI/AgentKernelArena/tree/main/docs)
covering installation, how-to guides, and the configuration and API reference.
