"""Benchmark resolution and execution.

Provides resolve_benchmark_plugin() and run_benchmark() — the two functions
imported by cli.py to run benchmarks during optimization sessions.
"""

from __future__ import annotations

import logging
import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SessionConfig
    from .capabilities import Capabilities

from .plugins.base import BenchmarkPlugin, BenchResult

log = logging.getLogger(__name__)


def resolve_benchmark_plugin(config: "SessionConfig", caps: "Capabilities") -> BenchmarkPlugin:
    """Select the appropriate benchmark plugin based on config.

    Priority:
      1. Explicit benchmark script → CustomBenchmarkPlugin
      2. Framework = "vllm" → VLLMPlugin (direct benchmark_serving.py)
      3. Framework in (sglang, atom, inferencex) → InferenceXPlugin (Magpie)
      4. Fallback → InferenceXPlugin (uses benchmark_serving.py directly)
    """
    if config.benchmark.script:
        from .plugins.custom import CustomBenchmarkPlugin
        return CustomBenchmarkPlugin(
            script=config.benchmark.script,
            output_format=config.benchmark.output_format,
            throughput_key=config.benchmark.throughput_key,
            latency_key=config.benchmark.latency_key,
            env=config.benchmark.env,
            timeout=config.benchmark.timeout,
        )

    framework = config.benchmark.framework.lower() if config.benchmark.framework else ""

    if framework == "vllm":
        from .plugins.vllm import VLLMPlugin
        plugin = VLLMPlugin(config)
        if plugin._find_bench_script():
            return plugin
        from .plugins.inferencex import InferenceXPlugin
        return InferenceXPlugin(config)

    if framework in ("sglang", "atom", "inferencex"):
        from .plugins.inferencex import InferenceXPlugin
        return InferenceXPlugin(config)

    from .plugins.inferencex import InferenceXPlugin
    return InferenceXPlugin(config)


def run_benchmark(plugin: BenchmarkPlugin, config: "SessionConfig") -> BenchResult:
    """Execute a benchmark using the given plugin and session config.

    Builds a config dict from SessionConfig and environment variables,
    then delegates to the plugin's run() method.
    """
    bench_config: dict[str, Any] = {
        "model_path": config.model_path,
        "port": config.port,
        "isl": int(os.environ.get("ISL", "1024")),
        "osl": int(os.environ.get("OSL", "1024")),
        "concurrency": int(os.environ.get("CONCURRENCY", os.environ.get("CONC", "64"))),
        "framework": config.benchmark.framework,
        "gpu_type": config.gpu_type,
        "gpus": config.gpus,
        "timeout": config.benchmark.timeout,
        "inferencex_path": os.environ.get("INFERENCEX_PATH", ""),
    }

    num_prompts = bench_config["concurrency"] * 10
    bench_config["num_prompts"] = num_prompts

    log.info(
        "Running benchmark: plugin=%s model=%s port=%d ISL=%d OSL=%d CONC=%d",
        plugin.name, config.model_path, config.port,
        bench_config["isl"], bench_config["osl"], bench_config["concurrency"],
    )

    result = plugin.run(bench_config)

    if result.success:
        log.info("Benchmark result: %.1f %s", result.throughput, result.throughput_unit)
    else:
        log.warning("Benchmark failed or returned 0 throughput. Output: %s", result.raw_output[:500])

    return result
