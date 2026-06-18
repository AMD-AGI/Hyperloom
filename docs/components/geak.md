# GEAK

GEAK is an agent-driven framework for end-to-end GPU kernel optimization in
real codebases. It runs a closed loop of profiling, optimization, and
validation, and produces reviewable patches backed by reproducible benchmarks.
GEAK supports **Triton**, **HIP** (and CUDA / CK / HSACO), and **FlyDSL**
kernels, and extends [mini-SWE-agent](https://github.com/SWE-agent/mini-SWE-agent)
for its agent loop and environment tooling.

Within Hyperloom, GEAK is one of the kernel-rewrite backends: when a hot kernel
is identified, it is optimized asynchronously through GEAK (alongside the OOB
kernel-optimization path that uses Claude Code / OpenAI Codex). The kernel agent
dispatches GEAK runs through Ray so multiple candidates can be explored in
parallel on the cluster's GPUs.

- **Source:** <https://github.com/AMD-AGI/GEAK>
- **License:** MIT

## Overview

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
  subagents (e.g. `general-kernel-optimization`, `harness-generator`,
  `codebase-explore`, `gemm-tuning`, `speedup-verify`) handle different kernel
  types and tasks.
- **Tooling layer** — kernel profiling for bottleneck analysis, optional RAG
  for GPU knowledge retrieval, and within-/cross-session memory of past
  optimization insights.

The package is published as `minisweagent` and ships its agent loop, config
templates, skills, and subagents inside the wheel.

## Installation

```bash
git clone https://github.com/AMD-AGI/GEAK
cd GEAK
make install          # core + MCP tools
# or, pip-only:
pip install -e .      # editable
pip install .         # non-editable (recommended for embedding consumers)
```

Configure a model and provider key before running, e.g.:

```bash
export MSWEA_MODEL_NAME="openai/gpt-5"
export OPENAI_API_KEY="YOUR_KEY"
```

```{note}
Hyperloom installs GEAK for you. `inference_optimizer/scripts/install.sh`
chains into `kernel-agent/scripts/install.sh`, whose `ensure_geak()` step clones
GEAK under the pod-local open-source checkout root by default
(`${HYPERLOOM_OPEN_SOURCE_ROOT:-${TMPDIR:-/tmp}/hyperloom/open-source-repos}/GEAK`)
and pip-installs it. Runtime config is written under
`$USER_DATA_PATH/runtime/geak-config/local.yaml`. For multi-node runs,
`inference_optimizer.multi_node install-geak` can pip-install a supplied GEAK
checkout into GPU pods so the `geak` CLI lands on `PATH`; pass `--geak-src` or
`HYPERLOOM_GEAK_SRC` when the checkout is not in a shared runtime path. The GEAK
run config is resolved from `$GEAK_CONFIG` and must set
`model.model_class: litellm`.
```

## Usage

The primary entry point is the `geak` CLI. `pyproject.toml` registers three
console scripts:

| CLI entry point | Purpose |
|-----------------|---------|
| `geak` | Optimize a kernel / repository (`minisweagent.run.mini:app`) |
| `geak-preprocess` | Preprocess a target and set up the harness contract |
| `geak-gemm-tuning` | GEMM selection / configuration tuning for SGLang + AITer |

### Optimize a kernel

```bash
geak --repo /path/to/kernel/repo \
  --task "Optimize the block_reduce kernel. Metric: bandwidth in GB/s (higher is better)."
```

### Parallel optimization

```bash
geak --repo /path/to/kernel/repo \
  --task "Optimize the block_reduce kernel." \
  --num-parallel 4 \
  --gpu-ids 0,1,2,3
```

Key flags: `--repo` (required target repo), `--kernel-path` / `--kernel-url`
(target kernel file), `--test-command` (the correctness/benchmark contract;
when omitted GEAK discovers or generates a harness), `--num-parallel`,
`--gpu-ids`, `--cost-limit`, `--config`, and `--mode quick|full` (60 / 120 min
wall-clock caps). Outputs land under `optimization_logs/<kernel>_<timestamp>/`,
keeping `final_report.json`, the winning `.diff`, and `geak_agent.log`.

For the full CLI reference and examples, see the
[GEAK Quick start](https://github.com/AMD-AGI/GEAK/blob/main/docs/quick_start.md).

## Role in Hyperloom

GEAK is wired in as a kernel-rewrite backend of the kernel agent:

- `kernel-agent/tools/kernel_optimization.py` selects the backend ladder
  (defaulting to `forge,geak`; OOB backends such as `claude`, `codex`, and
  `cursor` require an explicit backend override) and builds the GEAK task
  prompt, mapping the candidate's `source_type` to GEAK's `kernel_type`
  vocabulary and rendering the `--test-command` and budget.
- `kernel-agent/tools/backends/geak_submit.py` is the GEAK submission backend.
  It locates the `geak` CLI on `PATH`, resolves `$GEAK_CONFIG`, assembles the
  argument vector (`geak -t <prompt> --yolo --output <dir> --gpu-ids <ids>
  --config <cfg> ...`), and dispatches it inside a `num_gpus`-pinned **Ray**
  remote task (`run_via_ray`), remapping visible GPUs to logical ids and
  isolating per-attempt compile caches so a co-running OOB ladder cannot clobber
  artifacts.
- `kernel-agent/tools/geak_prompt_patcher.py` adapts the task prompt that GEAK
  receives.

This lets Hyperloom optimize hot kernels asynchronously and in parallel on the
cluster. See [How the optimization loop works](../HOW_THE_OPTIMIZATION_LOOP_WORKS.md).

```{note}
GEAK is distinct from the Kernel-Forge backend
(`kernel-agent/tools/backends/forge_submit.py`), which is a separate
self-contained rewrite backend and does not depend on GEAK.
```

## API reference

GEAK ships its own documentation in-repo; see the
[docs index](https://github.com/AMD-AGI/GEAK/tree/main/docs) covering
[Quick start](https://github.com/AMD-AGI/GEAK/blob/main/docs/quick_start.md),
[Configuration](https://github.com/AMD-AGI/GEAK/blob/main/docs/configuration.md),
and the
[Subagent guide](https://github.com/AMD-AGI/GEAK/blob/main/docs/subagent_guide.md).
