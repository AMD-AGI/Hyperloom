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

`HYPERLOOM_AGENTX=1 hyperloom optimize --model ... --tp ... --conc ...` converts
Hyperloom workload flags into native AgentX YAML. `--benchmark-config` accepts an
explicit YAML for advanced settings. Both paths archive `benchmark.yaml` in the
session and use the same Magpie execution contract.

Each run produces a `benchmark_report.json` that Hyperloom parses to extract
throughput/measurements and pick winners. Native AgentX uses Magpie’s managed
server/replay/profiling contract and public validity verdict; Hyperloom does not
patch its replay scripts or parse raw AIPerf exports. Hyperloom pins Magpie to a ref that
already copies benchmark scripts atomically upstream; `_magpie_patcher.py` applies
install-time compatibility patches (SGLang trust, eval concurrency) to the
pinned scripts. See
[Hyperloom optimization loop](../conceptual/optimization-loop.md) for more information.

## Magpie documentation

For detailed documentation on Magpie, see [Magpie on ROCm Docs](https://rocm.docs.amd.com/projects/magpie/en/latest/).

Native AgentX currently pins Magpie to
`6a1ee26f9053363d3750b43e3d5afb5d1c4ce84e` and InferenceX to
`408c015be4b22d14c69518643609669405507077`. Magpie owns the native InferenceX
checkout and resolves its `inferencex-e2e/` project layout. To supply a checkout,
set `benchmark.inferencex_path` in the workload YAML or `MAGPIE_INFERENCEX_PATH`.
Native runs do not inherit the synthetic checkout from `INFERENCEX_PATH`.

The general installer and synthetic/MLPerf preflight retain InferenceX
`3d5581562f643f9bdeb8410cd924e2c70906c966`, which the GSM8K and profiling patches
support. Those paths continue to honor `INFERENCEX_PATH` and `INFERENCEX_REF`;
native AgentX skips these compatibility patches.
See [the AgentX workflow](../how-to/optimize.md#optimize-an-agentx-workload).
