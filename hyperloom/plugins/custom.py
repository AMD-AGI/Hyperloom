"""Custom plugin: wraps user-provided benchmark/accuracy scripts.

The user provides a script path and output format specification.
This plugin runs the script, captures output, and parses it into
standardized BenchResult/AccuracyResult.

Supported output formats:
  - "json": Script prints JSON with configurable keys
  - "regex": Script prints text; we extract numbers via regex patterns
  - "last_line": Last line of stdout is the metric value
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .base import AccuracyPlugin, AccuracyResult, BenchmarkPlugin, BenchResult


class CustomBenchmarkPlugin(BenchmarkPlugin):
    """Wraps any user-provided benchmark script."""

    def __init__(
        self,
        script: str,
        output_format: str = "json",
        throughput_key: str = "throughput",
        latency_key: str = "latency_mean_ms",
        env: dict[str, str] | None = None,
        timeout: int = 7200,
        cwd: str | None = None,
    ):
        self._script = script
        self._output_format = output_format
        self._throughput_key = throughput_key
        self._latency_key = latency_key
        self._env = env or {}
        self._timeout = timeout
        self._cwd = cwd

    @property
    def name(self) -> str:
        return "custom"

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        errors = []
        if not Path(self._script).exists():
            errors.append(f"Benchmark script not found: {self._script}")
        if not os.access(self._script, os.X_OK):
            errors.append(f"Benchmark script not executable: {self._script}")
        return errors

    def run(self, config: dict[str, Any]) -> BenchResult:
        env = {**os.environ, **self._env}
        for k, v in config.items():
            if isinstance(v, (str, int, float)):
                env[f"HYPERLOOM_{k.upper()}"] = str(v)

        start = time.time()
        result = subprocess.run(
            [self._script],
            capture_output=True,
            text=True,
            timeout=self._timeout,
            env=env,
            cwd=self._cwd,
        )
        elapsed = time.time() - start

        output = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            return BenchResult(raw_output=output, extra={"elapsed_s": elapsed, "returncode": result.returncode})

        return self._parse_output(output, elapsed)

    def _parse_output(self, output: str, elapsed: float) -> BenchResult:
        if self._output_format == "json":
            return self._parse_json(output, elapsed)
        elif self._output_format == "regex":
            return self._parse_regex(output, elapsed)
        elif self._output_format == "last_line":
            return self._parse_last_line(output, elapsed)
        return BenchResult(raw_output=output, extra={"elapsed_s": elapsed})

    def _parse_json(self, output: str, elapsed: float) -> BenchResult:
        for line in reversed(output.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    return BenchResult(
                        throughput=float(data.get(self._throughput_key, 0)),
                        latency_mean_ms=float(data.get(self._latency_key, 0)),
                        raw_output=output,
                        extra={"parsed": data, "elapsed_s": elapsed},
                    )
                except (json.JSONDecodeError, ValueError):
                    continue
        return BenchResult(raw_output=output, extra={"elapsed_s": elapsed, "parse_error": "no JSON found"})

    def _parse_regex(self, output: str, elapsed: float) -> BenchResult:
        throughput = 0.0
        m = re.search(rf"{self._throughput_key}\s*[:=]?\s*([\d.]+)", output)
        if m:
            throughput = float(m.group(1))
        latency = 0.0
        m = re.search(rf"{self._latency_key}\s*[:=]?\s*([\d.]+)", output)
        if m:
            latency = float(m.group(1))
        return BenchResult(
            throughput=throughput,
            latency_mean_ms=latency,
            raw_output=output,
            extra={"elapsed_s": elapsed},
        )

    def _parse_last_line(self, output: str, elapsed: float) -> BenchResult:
        lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
        if not lines:
            return BenchResult(raw_output=output, extra={"elapsed_s": elapsed})
        try:
            throughput = float(lines[-1])
        except ValueError:
            throughput = 0.0
        return BenchResult(throughput=throughput, raw_output=output, extra={"elapsed_s": elapsed})


class CustomAccuracyPlugin(AccuracyPlugin):
    """Wraps any user-provided accuracy eval script."""

    def __init__(
        self,
        script: str,
        output_format: str = "json",
        score_key: str = "score",
        threshold: float = 0.0,
        env: dict[str, str] | None = None,
        timeout: int = 7200,
        cwd: str | None = None,
    ):
        self._script = script
        self._output_format = output_format
        self._score_key = score_key
        self._threshold = threshold
        self._env = env or {}
        self._timeout = timeout
        self._cwd = cwd

    @property
    def name(self) -> str:
        return "custom"

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        errors = []
        if not Path(self._script).exists():
            errors.append(f"Accuracy script not found: {self._script}")
        if not os.access(self._script, os.X_OK):
            errors.append(f"Accuracy script not executable: {self._script}")
        return errors

    def run(self, config: dict[str, Any]) -> AccuracyResult:
        env = {**os.environ, **self._env}
        for k, v in config.items():
            if isinstance(v, (str, int, float)):
                env[f"HYPERLOOM_{k.upper()}"] = str(v)

        result = subprocess.run(
            [self._script],
            capture_output=True,
            text=True,
            timeout=self._timeout,
            env=env,
            cwd=self._cwd,
        )

        output = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            return AccuracyResult(
                passed=False,
                raw_output=output,
                extra={"returncode": result.returncode},
            )

        return self._parse_output(output)

    def _parse_output(self, output: str) -> AccuracyResult:
        if self._output_format == "json":
            for line in reversed(output.strip().splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        data = json.loads(line)
                        score = float(data.get(self._score_key, 0))
                        passed = score >= self._threshold if self._threshold > 0 else True
                        return AccuracyResult(
                            score=score,
                            passed=passed,
                            threshold=self._threshold,
                            raw_output=output,
                            extra={"parsed": data},
                        )
                    except (json.JSONDecodeError, ValueError):
                        continue

        elif self._output_format == "last_line":
            lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
            if lines:
                try:
                    score = float(lines[-1])
                    passed = score >= self._threshold if self._threshold > 0 else True
                    return AccuracyResult(score=score, passed=passed, threshold=self._threshold, raw_output=output)
                except ValueError:
                    pass

        return AccuracyResult(passed=True, raw_output=output, extra={"parse_error": "could not extract score"})
