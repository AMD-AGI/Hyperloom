# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark backend seam.

Central place that builds the benchmark subprocess command line so the
optimizer can run against different benchmark engines without every executor
knowing which engine is active. The default backend is Magpie and its command
is byte-for-byte identical to the previously hardcoded Magpie benchmark
invocation (python -m Magpie -v benchmark --benchmark-config CFG
--output-dir OUT --run-mode local).

A future bypass backend can implement the same contract (same input YAML,
same workspace/report artifacts) and be selected via
HYPERLOOM_BENCHMARK_BACKEND=bypass without touching the executors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

# Backend selection env var. Absent/empty/unknown resolves to the Magpie
# backend so existing deployments keep their current behavior.
BENCHMARK_BACKEND_ENV = "HYPERLOOM_BENCHMARK_BACKEND"
DEFAULT_BENCHMARK_BACKEND = "magpie"


class BenchmarkBackend(Protocol):
    """Builds the benchmark subprocess command for one benchmark run."""

    name: str

    def build_command(
        self,
        *,
        python_exe: str,
        config_path: Path,
        output_dir: Path,
    ) -> list[str]:
        """Return the argv list for one local benchmark run.

        Args:
            python_exe: Interpreter used to launch the benchmark engine.
            config_path: Materialized benchmark config YAML.
            output_dir: Per-task output/workspace directory.

        Returns:
            The argv list to hand to the subprocess runner.
        """
        ...


class MagpieBackend:
    """Default backend: launches Magpie's local benchmark subprocess.

    The produced command matches the historically hardcoded invocation so this
    seam is a behavior-preserving refactor.
    """

    name = "magpie"

    def build_command(
        self,
        *,
        python_exe: str,
        config_path: Path,
        output_dir: Path,
    ) -> list[str]:
        """Return the canonical python -m Magpie ... --run-mode local argv."""
        return [
            python_exe,
            "-m",
            "Magpie",
            "-v",
            "benchmark",
            "--benchmark-config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--run-mode",
            "local",
        ]


def resolve_backend_name() -> str:
    """Resolve the active backend name from the environment.

    Returns:
        The lowercased backend name; magpie when unset/blank.
    """
    raw = (os.environ.get(BENCHMARK_BACKEND_ENV) or "").strip().lower()
    return raw or DEFAULT_BENCHMARK_BACKEND


def resolve_backend() -> BenchmarkBackend:
    """Resolve the active benchmark backend instance.

    Only the Magpie backend exists today; any unknown value falls back to it so
    a typo cannot silently disable benchmarking.

    Returns:
        The selected BenchmarkBackend implementation.
    """
    name = resolve_backend_name()
    if name == "magpie":
        return MagpieBackend()
    # Unknown backend: fall back to Magpie (defensive). Later stages register
    # additional backends (e.g. bypass) here.
    return MagpieBackend()


def build_benchmark_command(
    *,
    python_exe: str,
    config_path: Path,
    output_dir: Path,
) -> list[str]:
    """Build the benchmark command using the active backend.

    Args:
        python_exe: Interpreter used to launch the benchmark engine.
        config_path: Materialized benchmark config YAML.
        output_dir: Per-task output/workspace directory.

    Returns:
        The argv list for one local benchmark run.
    """
    return resolve_backend().build_command(
        python_exe=python_exe,
        config_path=config_path,
        output_dir=output_dir,
    )