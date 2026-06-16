# TraceLens

TraceLens is a Python library focused on **automating analysis from trace
files**. It turns raw PyTorch / JAX / rocprofv3 traces into hierarchical
performance breakdowns, roofline metrics, and multi-GPU communication analysis.

Within Hyperloom, TraceLens is the profiling brain of the **workload
understanding** stage: it consumes traces collected by [Magpie](magpie.md),
captures bottlenecks, and derives the roofline targets that seed the
optimization search tree.

- **Source:** <https://github.com/AMD-AGI/TraceLens>
- **License:** MIT

## Overview

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
Hyperloom resolves the public checkout via `TRACELENS_ROOT`. `local_setup.sh`
can clone or update the checkout and write it into `local-setup.env.sh`; the
runtime installation is performed by `inference_optimizer/scripts/install.sh`,
which chains into `kernel-agent/scripts/install.sh` and editable-installs the
checkout. When `TRACELENS_ROOT` is unset, the installer clones the public repo
under the pod-local open-source checkout root
(`${HYPERLOOM_OPEN_SOURCE_ROOT:-${TMPDIR:-/tmp}/hyperloom/open-source-repos}`).
The optional internal extension (roofline gap / MI355+ MAF data) is enabled by
setting `TRACELENS_INTERNAL_ROOT`; leave it unset for the open-source-only
report.
```

## Usage

### Generate a report from a PyTorch trace

```bash
TraceLens_generate_perf_report_pytorch --profile_json_path path/to/trace.json
```

This produces an Excel workbook with the GPU timeline breakdown, ops summary,
and roofline metrics.

### Supported profile formats

| Format | Producer | CLI entry point |
|--------|----------|-----------------|
| PyTorch | `torch.profiler` | `TraceLens_generate_perf_report_pytorch` |
| PyTorch inference | `torch.profiler` for vLLM/SGLang traces | `TraceLens_generate_perf_report_pytorch_inference` |
| JAX | XPlane protobuf | `TraceLens_generate_perf_report_jax` |
| rocprofv3 JSON | rocprofiler-sdk | `TraceLens_generate_perf_report_rocprof` |
| rocprofv3 pftrace | Perfetto-style | `TraceLens_generate_perf_report_pftrace_hip_api` |

### Compare two reports

```bash
TraceLens_compare_perf_reports_pytorch baseline.xlsx candidate.xlsx \
    --names baseline candidate --sheets all -o comparison.xlsx
```

For the full CLI reference and module deep-dives, see the
[TraceLens documentation](https://github.com/AMD-AGI/TraceLens/tree/main/docs).

## Role in Hyperloom

The orchestration runtime enters TraceLens through the kernel request path:
`inference_optimizer/orchestrator/kernel_request_handlers.py` dispatches
`trace_analyze` requests as subprocesses that run
`kernel-agent/tools/tracelens_analysis.py` (and the skill runner when needed).
The composite roofline executor first profiles the workload with Magpie, then
hands the trace to that `trace_analyze` path.

Before profiling, Hyperloom can patch the active vLLM/SGLang server tree with
TraceLens-specific runtime flags through
`inference_optimizer/orchestrator/action_executors/_server_patcher.py` and the
workload environment helpers. The generated report feeds the roofline ceilings
and bottleneck list used to score candidate optimizations. See
[How the optimization loop works](../HOW_THE_OPTIMIZATION_LOOP_WORKS.md).

## API reference

TraceLens ships its own SDK documentation in-repo; see the
[module docs](https://github.com/AMD-AGI/TraceLens/tree/main/docs) and the
[example notebooks](https://github.com/AMD-AGI/TraceLens/tree/main/examples).
