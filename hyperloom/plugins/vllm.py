"""vLLM benchmark plugin.

Wraps vLLM's benchmark_serving.py for direct vLLM benchmarking
without Magpie/InferenceX.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .base import BenchmarkPlugin, BenchResult

log = logging.getLogger(__name__)


class VLLMPlugin(BenchmarkPlugin):
    """Benchmark via vLLM's benchmark_serving.py."""

    def __init__(self, config: Any = None):
        self._config = config

    @property
    def name(self) -> str:
        return "vllm"

    def run(self, config: dict[str, Any]) -> BenchResult:
        bench_script = self._find_bench_script()
        if not bench_script:
            return BenchResult(raw_output="vLLM benchmark_serving.py not found")

        model_path = config.get("model_path", "")
        port = config.get("port", 8000)
        isl = config.get("isl", 1024)
        osl = config.get("osl", 256)
        num_prompts = config.get("num_prompts", 100)
        concurrency = config.get("concurrency", 16)

        cmd = [
            "python3", bench_script,
            "--backend", "vllm",
            "--base-url", f"http://localhost:{port}",
            "--model", model_path,
            "--dataset-name", "random",
            "--random-input-len", str(isl),
            "--random-output-len", str(osl),
            "--random-range-ratio", "1.0",
            "--num-prompts", str(num_prompts),
            "--max-concurrency", str(concurrency),
            "--request-rate", "inf",
            "--ignore-eos",
        ]

        timeout = config.get("timeout", 3600)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return BenchResult(raw_output="vLLM benchmark timed out")

        output = result.stdout + "\n" + result.stderr
        return self._parse_output(output)

    def _parse_output(self, output: str) -> BenchResult:
        """Parse vLLM benchmark_serving.py text output."""
        throughput = _extract(r"Output token throughput.*?:\s*([\d.]+)", output)
        latency = _extract(r"Mean TPOT.*?:\s*([\d.]+)", output)
        return BenchResult(
            throughput=throughput,
            latency_mean_ms=latency,
            raw_output=output,
        )

    def _find_bench_script(self) -> str:
        """Find vLLM's benchmark_serving.py via environment or python path."""
        explicit = os.environ.get("VLLM_BENCH_SCRIPT", "")
        if explicit and Path(explicit).exists():
            return explicit

        try:
            result = subprocess.run(
                ["python3", "-c", "import vllm; import os; print(os.path.dirname(vllm.__file__))"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                vllm_dir = Path(result.stdout.strip())
                bench = vllm_dir.parent / "benchmarks" / "benchmark_serving.py"
                if bench.exists():
                    return str(bench)
        except (subprocess.TimeoutExpired, OSError):
            pass

        return ""


def _extract(pattern: str, text: str) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else 0.0
