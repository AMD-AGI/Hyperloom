---
myst:
    html_meta:
        "description": "Learn about GEAK, Hyperloom's agent-driven GPU kernel optimization framework. Covers Triton, HIP, and FlyDSL kernel rewriting, parallel optimization, and patch validation."
        "keywords": "GEAK, Hyperloom, GPU kernel optimization, Triton, HIP, FlyDSL, AMD GPU, ROCm, kernel rewriting, benchmarking, parallel optimization, LLM inference, agent, Ray"
---
# GEAK

GEAK (Generating Efficient AI-Centric Kernels) is an agent-driven framework for end-to-end GPU kernel optimization in
real codebases. It runs a closed loop of profiling, optimization, and
validation, and produces reviewable patches backed by reproducible benchmarks.
GEAK supports Triton, HIP (and CUDA / Composable Kernel (CK) / HSA Code Object (HSACO)), and FlyDSL
kernels, and extends [mini-SWE-agent](https://github.com/SWE-agent/mini-SWE-agent)
for its agent loop and environment tooling.

Within Hyperloom, GEAK is one of the kernel-rewrite backends: when a hot kernel
is identified, it is optimized asynchronously through GEAK. The kernel agent
dispatches GEAK runs with placement precedence SSH (Dynamo multi-node) > Ray
(when available) > direct CLI, so multiple candidates can be explored in
parallel on the cluster's GPUs.

- **Source**: <https://github.com/AMD-AGI/GEAK>
- **License**: MIT

## Optimization workflow and features

GEAK turns a natural-language task plus a target repository into a verified,
patch-based kernel optimization:

- **End-to-end optimization** — discovers or generates tests and a harness,
  then runs a closed loop of profiling → optimization → validation, emitting
  reproducible patches and benchmarks.
- **Patch-driven iteration** — every step produces a reproducible diff, guided
  by a task-defined or custom metric, with strategy tracking across the run.
- **Parallel exploration** — runs multiple optimization agents, each in an
  isolated git worktree, with optional GPU pinning (`--gpu-ids`).
- **Best-patch selection** — verifies candidates against the run's benchmark
  contract, writes `round_N_evaluation.json` per round, and records the winning
  result in `final_report.json`.
- **Skills & subagents** — domain skills (`triton`, `hip`, `flydsl`,
  `pytorch2flydsl-translation`, `fp8-gemm-tuning-sglang-aiter`) and specialist
  subagents (for example, `general-kernel-optimization`, `harness-generator`,
  `codebase-explore`, `gemm-tuning`, `speedup-verify`) handle different kernel
  types and tasks.
- **Tooling layer** — kernel profiling for bottleneck analysis, optional retrieval-augmented generation (RAG)
  for GPU knowledge retrieval, and within-/cross-session memory of past
  optimization insights.

The package is published as `minisweagent` and ships its agent loop, config
templates, skills, and subagents inside the wheel.

## Installation

Clone the GEAK repository and install using make or pip:

```bash
git clone https://github.com/AMD-AGI/GEAK
cd GEAK
make install          # core + MCP tools
# or, pip-only:
pip install -e .      # editable
pip install .         # non-editable (recommended for embedding consumers)
```

Configure a model and provider key before running, for example:

```bash
export MSWEA_MODEL_NAME="openai/gpt-5"
export OPENAI_API_KEY="YOUR_KEY"
```

```{note}
Hyperloom installs GEAK for you. `src/hyperloom/inference_optimizer/assets/install.sh`
chains into `src/hyperloom/agents/kernel/scripts/install.sh`, whose `ensure_geak()` step clones
GEAK under the pod-local open-source checkout root by default
(`${HYPERLOOM_OPEN_SOURCE_ROOT:-/opt/hyperloom/open-source-repos}/GEAK`)
and pip-installs it. Runtime config is written under
`$USER_DATA_PATH/runtime/kernel-agent.env.sh`. For multi-node Dynamo runs
(`--mn-backend dynamo`), `python -m hyperloom.inference_optimizer.multi_node install-geak`
can pip-install a supplied GEAK checkout into GPU pods so the `geak` CLI lands on `PATH`; pass `--geak-src` or
`HYPERLOOM_GEAK_SRC` when the checkout is not in a shared runtime path. GEAKv4
uses the Claude Code workflow and reads its model from `GEAK_CLAUDE_MODEL`.
```

## Usage

The primary entry point is the `geak` CLI. `pyproject.toml` registers three
console scripts:

| CLI entry point | Purpose |
|-----------------|---------|
| `geak` | Optimize a kernel / repository (`minisweagent.run.mini:app`) |
| `geak-preprocess` | Pre-process a target and set up the harness contract |
| `geak-gemm-tuning` | General matrix multiplication (GEMM) selection and configuration tuning for SGLang + AITer |

### Optimize a kernel

Run the `geak` CLI with a target repository and optimization task:

```bash
geak --repo /path/to/kernel/repo \
  --task "Optimize the block_reduce kernel. Metric: bandwidth in GB/s (higher is better)."
```

### Parallel optimization

Use `--num-parallel` and `--gpu-ids` to run multiple optimization agents simultaneously:

```bash
geak --repo /path/to/kernel/repo \
  --task "Optimize the block_reduce kernel." \
  --num-parallel 4 \
  --gpu-ids 0,1,2,3
```

Key flags:

- `--repo` (required target repo)
- `--kernel-path` / `--kernel-url` (target kernel file)
- `--test-command` (the correctness/benchmark contract; when omitted GEAK discovers or generates a harness)
- `--num-parallel`, `--gpu-ids`, `--cost-limit`, `--config`, and `--mode quick|full` (60 / 120 min wall-clock caps).

Outputs land under `optimization_logs/<kernel>_<timestamp>/`, keeping `final_report.json`, the winning `.diff`, and `geak_agent.log`.

For the full CLI reference and examples, see the
[GEAK Quick start](https://github.com/AMD-AGI/GEAK/blob/main/docs/quick_start.md).

## Role in Hyperloom

Hyperloom uses GEAK as the **whole-pipeline e2e delegate** when
`KERNEL_OPT_BACKEND_ORDER=geak` (the bare-metal default). In this mode the
orchestrator hands the optimization workload to
`src/hyperloom/agents/kernel/tools/backends/geak_runner.py`, which resolves the
GEAK checkout and launches GEAK's e2e runner (`interface/run_e2e.py`) with the
generated session context.

GEAK is distinct from the per-kernel `forge` backend:

- `geak` runs the whole e2e optimization loop through GEAK.
- `forge` targets individual kernel/GEMM opportunities through KernelForge and
  related forge tools.

This keeps the backend split explicit: use `geak` for whole-pipeline delegation
and `forge` for per-kernel optimization. See
[Hyperloom optimization loop](../conceptual/optimization-loop.md).

## API reference

GEAK ships its own documentation in-repo; see the
[docs index](https://github.com/AMD-AGI/GEAK/tree/main/docs) covering
[Quick start](https://github.com/AMD-AGI/GEAK/blob/main/docs/quick_start.md).
