---
myst:
    html_meta:
        "description": "Learn about Magpie, Hyperloom's benchmark engine for GPU kernel correctness and performance evaluation. Covers Analyze, Compare, and Benchmark modes for AMD and NVIDIA GPUs."
        "keywords": "Magpie, Hyperloom, GPU benchmarking, kernel evaluation, AMD GPU, ROCm, HIP, CUDA, vLLM, SGLang, TraceLens, MCP, benchmark engine, throughput, LLM inference"
---
# Magpie

Magpie is a lightweight, general-purpose framework for evaluating GPU kernel
correctness and performance on AMD (HIP) GPUs. It exposes
three evaluation modes — Analyze, Compare, and Benchmark — plus framework-level
(vLLM or SGLang) benchmarking with built-in TraceLens trace analysis.

Within Hyperloom, Magpie is the benchmark engine. The kernel agent and the
optimization loop drive Magpie to spin up a serving framework, run the workload,
collect traces, and emit a structured `benchmark_report.json` file; those traces are
the input that [TraceLens](tracelens.md) then analyzes. Magpie relies on
[IntelliKit](intellikit.md) for some low-level GPU profiling tools.

- **Source**: <https://github.com/AMD-AGI/Magpie>
- **License**: MIT

## Role in Hyperloom

The optimization loop drives Magpie's benchmark mode as a subprocess. The
command line is built by `build_benchmark_command()` /
`MagpieBackend.build_command()` in
`src/hyperloom/orchestrator/actions/executors/benchmark_backend.py`; callers
such as `_grid_runner.py` and `baseline.py` invoke it to launch one run per
variant:

```python
cmd = [
    magpie_python, "-m", "Magpie", "-v", "benchmark",
    "--benchmark-config", str(config_path),
    "--output-dir", str(output_dir),
    "--run-mode", "local",
]
```

Each run produces a `benchmark_report.json` that Hyperloom parses to extract
throughput/measurements and pick winners. Hyperloom pins the Magpie
[v0.3.0 release](https://github.com/AMD-AGI/Magpie/releases/tag/v0.3.0) plus native launch overrides, custom-model replay, and the
generic eval source-path fix at immutable commit
`c5c80698fef1b89cc6882264b80d5b306d4e9328`. The hotfix keeps evaluation scripts
bound to their intended InferenceX source after directory changes. The launch extensions are pinned development commits rather than a new Magpie
release. Earlier GPU validation used the v0.3.0 release commit and does not
validate these extensions; their source and wheel share an audited execution tree.
With `benchmark.agentx: enable`, Magpie resolves the serving specification,
owns server startup and cleanup, and runs the maintained InferenceX/AIPerf client.
The integration validates the package, recipe/client sources, and effective
server launch evidence. Both the repository root and the `inferencex-e2e/`
project layout are supported; no Slurm launcher extension is required.

Fresh `HYPERLOOM_AGENTX=1` sessions use Magpie native AgentX and retain
Hyperloom's optimization loop. Magpie resolves the recipe and launcher from
the model, framework, GPU, precision, and concurrency; ambiguous inputs fail
with a selector error. New sessions record backend `native` and epoch 4.
With no registered recipe match, an explicit image, TP, EP, concurrency, and
canonical model identify a custom SGLang/vLLM workload; Magpie validates its
model context and resolves a formal generic launcher. Registered ambiguities
remain errors. Candidate KEEPs retain the exact native launch snapshot, so
resume and subsequent candidates do not apply the same launch arguments twice.
Persisted epoch-1 sessions resume the legacy client, and epoch-2 native sessions
retain their measurement-only contract. Saved epoch-3 sessions retain their
upstream-launcher contract. No baseline or KEEP record is migrated.

Native profiling derives a diagnostic run from the accepted candidate and uses
Magpie's phase-gated, step-bounded torch profiler. Magpie owns the first-capture
delay, repetition interval, count limit, and framework-specific controls.
Hyperloom selects a complete capture from the manifest for kernel/roofline
analysis. Diagnostic success never makes the run a publishable KEEP result.
Detailed annotations require a compatible instrumented serving runtime; an
ordinary GPU trace alone does not establish complete kernel shape metadata. See
[Run an InferenceX AgentX workload](../how-to/optimize.md#run-an-inferencex-agentx-workload)
for the example and current optimization limitations, and
[Hyperloom optimization loop](../conceptual/optimization-loop.md) for more information.

## Magpie documentation

For detailed documentation on Magpie, see [Magpie on ROCm Docs](https://rocm.docs.amd.com/projects/magpie/en/latest/).
