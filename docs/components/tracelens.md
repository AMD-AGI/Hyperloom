---
myst:
    html_meta:
        "description": "Learn about TraceLens, Hyperloom's trace analysis library. Covers hierarchical GPU performance breakdowns, roofline modeling, multi-GPU analysis, and trace comparison."
        "keywords": "TraceLens, Hyperloom, GPU performance, trace analysis, roofline modeling, PyTorch profiler, JAX, rocprofv3, AMD GPU, ROCm, benchmarking, LLM inference, vLLM, SGLang"
---
# TraceLens

TraceLens is a Python library focused on automating analysis from trace
files. It turns raw PyTorch / JAX / rocprofv3 traces into hierarchical
performance breakdowns, roofline metrics, and multi-GPU communication analysis.

Within Hyperloom, TraceLens is the profiling brain of the workload
understanding stage: it consumes traces collected by [Magpie](magpie.md),
captures bottlenecks, and derives the roofline targets that seed the
optimization search tree.

- **Source**: <https://github.com/AMD-AGI/TraceLens>
- **License**: MIT

## Capabilities and modules

TraceLens provides a top-down view of GPU performance and a hackable SDK:

- **Hierarchical performance breakdowns** — drill from the overall GPU timeline
  (idle/busy) down to operator categories, individual operators, and unique
  argument shapes.
- **Compute & roofline modeling** — translate raw timings into TFLOP/s and TB/s,
  and classify ops as compute- or memory-bound.
- **Multi-GPU communication analysis** — separate pure communication time from
  synchronization skew and compute effective collective bandwidth.
- **Trace comparison** — diff two traces at the CPU-dispatch level to quantify
  the impact of a change across hardware or software versions.
- **Event replay** — generate minimal, self-contained, IP-safe replay scripts
  for a single operation.

The package is organized into composable modules: `Trace2Tree`, `TreePerf`,
`NcclAnalyser`, `TraceDiff`, `EventReplay`, `TraceFusion`, `PerfModel`,
`Reporting`, and an `Agent` layer for agentic analysis.

## Installation

Install TraceLens directly from the GitHub repository using pip.

```bash
pip install git+https://github.com/AMD-AGI/TraceLens.git
```

For development (editable install with test extras):

```bash
git clone https://github.com/AMD-AGI/TraceLens.git && cd TraceLens
pip install -e .[dev]
python -m pytest tests/ -v
```

```{note}
Hyperloom resolves the public checkout via `TRACELENS_ROOT`. Runtime
installation is performed by `src/hyperloom/inference_optimizer/assets/install.sh`,
which chains into `src/hyperloom/agents/kernel/scripts/install.sh`; that
installer clones, pins, and editable-installs the checkout. When
`TRACELENS_ROOT` is unset, the installer clones the public repo under the
pod-local open-source checkout root
(`${HYPERLOOM_OPEN_SOURCE_ROOT:-/opt/hyperloom/open-source-repos}`).
The optional internal extension (roofline gap / MI355+ MAF data) is enabled by
setting `TRACELENS_INTERNAL_ROOT`; leave it unset for the open-source-only
report.
```

## Usage

TraceLens provides CLI entry points for generating performance reports, comparing traces, and running agentic analysis.

### Generate a report from a PyTorch trace

```bash
TraceLens_generate_perf_report_pytorch --profile_json_path path/to/trace.json
```

This produces an Excel workbook with the GPU timeline breakdown, ops summary,
and roofline metrics.

### Supported profile formats

TraceLens supports the following trace formats.

| Format | Producer | CLI entry point |
|--------|----------|-----------------|
| PyTorch | `torch.profiler` | `TraceLens_generate_perf_report_pytorch` |
| PyTorch inference | `torch.profiler` for vLLM/SGLang traces | `TraceLens_generate_perf_report_pytorch_inference` |
| JAX | XPlane protobuf | `TraceLens_generate_perf_report_jax` |
| rocprofv3 JSON | rocprofiler-sdk | `TraceLens_generate_perf_report_rocprof` |
| rocprofv3 pftrace | Perfetto-style | `TraceLens_generate_perf_report_pftrace_hip_api` |

### Compare two reports

Compare two TraceLens Excel reports to quantify the impact of a change.

```bash
TraceLens_compare_perf_reports_pytorch baseline.xlsx candidate.xlsx \
    --names baseline candidate --sheets all -o comparison.xlsx
```

For the full CLI reference and module deep-dives, see the
[TraceLens documentation](https://github.com/AMD-AGI/TraceLens/tree/main/docs).

## Role in Hyperloom

The orchestration runtime enters TraceLens through two paths:

- The kernel request path:
  `src/hyperloom/orchestrator/kernel/request_handlers.py` dispatches
  `trace_analyze` requests as subprocesses that run
  `src/hyperloom/agents/kernel/tools/tracelens_analysis.py`. That script itself
  imports and calls `run_tracelens_skill` (the skill runner) internally under
  the agent route — the runner is not a separate subprocess dispatched by the
  orchestrator.
- The composite roofline action: `RooflineExecutor`
  (`src/hyperloom/orchestrator/actions/executors/roofline.py`) is an atomic
  `profile` + `trace_analyze` pipeline that first profiles the workload with
  Magpie, then calls `trace_analyze_handler()` directly on the trace.

Before profiling, Hyperloom can patch the active vLLM/SGLang server tree with
TraceLens-specific runtime flags through
`src/hyperloom/orchestrator/actions/executors/_server_patcher.py` and the
workload environment helpers. The generated report feeds the roofline ceilings
and bottleneck list used to score candidate optimizations. See
[Hyperloom optimization loop](../conceptual/optimization-loop.md).

## API reference

TraceLens ships its own SDK documentation in-repo; see the
[module docs](https://github.com/AMD-AGI/TraceLens/tree/main/docs) and the
[example notebooks](https://github.com/AMD-AGI/TraceLens/tree/main/examples).
