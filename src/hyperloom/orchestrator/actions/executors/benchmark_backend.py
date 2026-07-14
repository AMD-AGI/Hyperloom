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

    def resolve_interpreter(self) -> str:
        """Return the Magpie-importable interpreter for the Magpie backend."""
        from ._grid_runner import _resolve_magpie_python

        return _resolve_magpie_python()

    def lifecycle_eligibility(self, bench: dict) -> dict | None:
        """Return None to use the default (Magpie script-based) eligibility.

        Magpie keeps the historical behavior: server_lifecycle is decided
        by the built-in benchmark_script name in :func:`resolve_lifecycle_params`.
        Returning None signals the caller to run that default path.
        """
        return None

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


class BypassBackend:
    """Bypass backend: launches Hyperloom's own benchmark runner.

    Accepts the same CLI flags as Magpie and reuses the InferenceX scripts,
    so the input YAML and the workspace/report contract are unchanged. See
    :mod:`bypass_runner`.
    """

    name = "bypass"

    def resolve_interpreter(self) -> str:
        """Return a plain python3 for bypass (no Magpie import needed).

        Prefers the current interpreter, then a PATH ``python3``. bypass
        drives InferenceX directly, so it must NOT fall back to Magpie's
        ``/opt/venv/bin/python`` canonical path.
        """
        import shutil
        import sys

        return sys.executable or shutil.which("python3") or "python3"

    # Serving frameworks whose OpenAI server bypass can persist for reuse.
    _LIFECYCLE_FRAMEWORKS = frozenset({"vllm", "atom", "sglang"})

    def lifecycle_eligibility(self, bench: dict) -> dict | None:
        """Decide bypass server_lifecycle eligibility.

        Eligible for a single-node serving framework with profiling off:
        bypass honors the YAML ``server_lifecycle`` block (persist server on
        the first round, reuse on the next, teardown on cleanup), so
        run_grid's warmup+measure reuse works. Scriptable/diffusion and
        torch_profiler runs stay ineligible (no persistent server to reuse).
        """
        framework = str(bench.get("framework") or "").lower()
        envs = bench.get("envs") or {}
        try:
            port = int(envs.get("PORT", 8888))
        except (TypeError, ValueError):
            port = 8888
        verdict = {"eligible": False, "framework": framework, "port": port, "reason": ""}
        # The reuse protocol boots a local server and re-attaches a client
        # round to it; that only holds single-node. Mirror the Magpie path's
        # multi-node gate (see _server_lifecycle.resolve_lifecycle_params),
        # which our non-None verdict would otherwise short-circuit past.
        from ._multi_node_env import is_multi_node

        if is_multi_node():
            verdict["reason"] = "multi-node (server_lifecycle is local-only)"
            return verdict
        if framework not in self._LIFECYCLE_FRAMEWORKS:
            verdict["reason"] = f"framework {framework!r} is not a serving framework"
            return verdict
        profiler_on = bool((bench.get("profiler") or {}).get("torch_profiler", {}).get("enabled"))
        if profiler_on:
            verdict["reason"] = "torch_profiler enabled (incompatible with reuse)"
            return verdict
        verdict["eligible"] = True
        return verdict

    def build_command(
        self,
        *,
        python_exe: str,
        config_path: Path,
        output_dir: Path,
    ) -> list[str]:
        """Return the bypass runner argv mirroring Magpie's flags."""
        return [
            python_exe,
            "-m",
            "hyperloom.orchestrator.actions.executors.bypass_runner",
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
    if name == "bypass":
        return BypassBackend()
    if name == "magpie":
        return MagpieBackend()
    # Unknown backend: fall back to Magpie (defensive) so a typo cannot
    # silently disable benchmarking.
    return MagpieBackend()


def resolve_benchmark_interpreter() -> str:
    """Resolve the interpreter for the active benchmark backend."""
    return resolve_backend().resolve_interpreter()


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