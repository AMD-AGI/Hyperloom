---
myst:
    html_meta:
        "description": "Learn about Magpie, Hyperloom's benchmark engine for GPU kernel correctness and performance evaluation. Covers Analyze, Compare, and Benchmark modes for AMD and NVIDIA GPUs."
        "keywords": "Magpie, Hyperloom, GPU benchmarking, kernel evaluation, AMD GPU, ROCm, HIP, CUDA, vLLM, SGLang, TraceLens, MCP, benchmark engine, throughput, LLM inference"
---
# Magpie

Magpie is a lightweight, general-purpose framework for evaluating GPU kernel
correctness and performance on AMD (HIP) and NVIDIA (CUDA) GPUs. It exposes
three evaluation modes — Analyze, Compare, and Benchmark — plus framework-level
(vLLM / SGLang / Atom) benchmarking with built-in TraceLens trace analysis.

Within Hyperloom, Magpie is the benchmark engine. The kernel agent and the
optimization loop drive Magpie to spin up a serving framework, run the workload,
collect traces, and emit a structured `benchmark_report.json`; those traces are
the input that [TraceLens](tracelens.md) then analyzes. Magpie relies on
[IntelliKit](intellikit.md) for some low-level GPU profiling tools.

- **Documentation**:
- **Source**: <https://github.com/AMD-AGI/Magpie>
- **License**: MIT

## Evaluation modes and capabilities

Magpie provides a hardware-aware kernel and workload evaluation framework:

- **Three evaluation modes** — Analyze (single kernel + testcase), Compare
  (multi-kernel ranking), and Benchmark (framework-level vLLM/SGLang/Atom runs).
- **Heterogeneous hardware** — AMD (HIP) and NVIDIA (CUDA) GPUs.
- **Execution environments** — Local, sandboxed container, and remote Ray
  cluster scheduling.
- **Hardware control** — Kernel evaluation under controlled power/frequency
  settings, with automatic idle-GPU selection in Benchmark mode.
- **Trace analysis** — TraceLens integration plus gap analysis on torch
  profiler traces.
- **MCP server** — Model Context Protocol integration for AI agents.
- **Structured reports** — JSON output (`benchmark_report.json`) for pipeline
  integration.

The package is organized into composable modules: `config`, `core` (executor,
scheduler, Ray executor, job store), `eval` (compiling, correctness,
performance), `modes` (`analyze_eval`, `compare_eval`, `benchmark`), `mcp`, and
`utils`.

## Installation

Install Magpie directly from the GitHub repository using pip:

```bash
pip install git+https://github.com/AMD-AGI/Magpie.git
```

For development (editable install from source):

```bash
git clone https://github.com/AMD-AGI/Magpie.git && cd Magpie
pip install -e .
```

```{note}
Hyperloom installs Magpie for you during install/preflight rather than using
`local_setup.sh`. `inference_optimizer/scripts/install.sh` clones it (pinned to
`MAGPIE_REF`, default under `$MAGPIE_DIR`) and editable-installs it, while also
applying the Hyperloom atomic benchmark-script patch when needed.
`inference_optimizer/cli.py` can also clone + `pip install -e` Magpie on
preflight if it is not importable; that fallback clone uses the repository
default branch rather than the install-time `MAGPIE_REF` pin. Magpie subprocesses
are launched with the interpreter
resolved from `$MAGPIE_PYTHON` (validated to be able to `import Magpie`, else
auto-detected; see
`inference_optimizer/orchestrator/action_executors/_grid_runner.py`).
```

## Usage

The CLI entry point is `magpie` (console script `Magpie.main:main`); the
equivalent `python -m Magpie` form accepts the same subcommands.

```bash
# Analyze a single kernel from a config file
magpie analyze --kernel-config Magpie/kernel_config.yaml.example

# Compare multiple kernel implementations
magpie compare --kernel-config examples/ck_grouped_gemm_compare.yaml

# Framework-level benchmark (vLLM/SGLang/Atom) with optional trace analysis
magpie benchmark --benchmark-config examples/benchmarks/benchmark_vllm_dsr1.yaml

# GPU / toolchain summary
magpie --gpu-info

# MCP server
python -m Magpie.mcp
```

| Mode | Description | CLI |
|------|-------------|-----|
| Analyze | Single kernel evaluation with a testcase | `magpie analyze --kernel-config <yaml>` |
| Compare | Multi-kernel comparison and ranking | `magpie compare --kernel-config <yaml>` |
| Benchmark | Framework-level benchmarking (vLLM/SGLang/Atom) with trace analysis | `magpie benchmark --benchmark-config <yaml>` |

## Role in Hyperloom

The optimization loop drives Magpie's benchmark mode as a subprocess. The
grid runner builds the command line and launches one run per variant in
`inference_optimizer/orchestrator/action_executors/_grid_runner.py`:

```python
cmd = [
    magpie_python, "-m", "Magpie", "-v", "benchmark",
    "--benchmark-config", str(config_path),
    "--output-dir", str(output_dir),
    "--run-mode", "local",
]
```

Each run produces a `benchmark_report.json` that Hyperloom parses to extract
throughput/measurements and pick winners. To make concurrent benchmark runs
robust, `inference_optimizer/orchestrator/action_executors/_magpie_patcher.py`
applies an idempotent, atomic-write patch to Magpie's cloned `benchmarker.py`
(`_prepare_benchmark_scripts`) so a concurrent reader never sees a half-copied
script. See [Hyperloom optimization loop](../conceptual/optimization-loop.md) for more information.

## API reference

Magpie ships its own documentation in-repo (ROCm docs-site source under
`docs/`); see the
[reference docs](https://github.com/AMD-AGI/Magpie/tree/main/docs/reference)
and the [how-to guides](https://github.com/AMD-AGI/Magpie/tree/main/docs/how-to).
