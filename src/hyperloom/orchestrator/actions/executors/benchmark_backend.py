# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Benchmark backend seam."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

# Backend selection env var.
BENCHMARK_BACKEND_ENV = "HYPERLOOM_BENCHMARK_BACKEND"
DEFAULT_BENCHMARK_BACKEND = "magpie"
KNOWN_BENCHMARK_BACKENDS = frozenset({"magpie", "bypass"})


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
        """Return the argv list for one local benchmark run."""
        ...


class MagpieBackend:
    """Default backend: launches Magpie's local benchmark subprocess."""

    name = "magpie"

    def resolve_interpreter(self) -> str:
        """Return the Magpie-importable interpreter for the Magpie backend."""
        from ._benchmark_interpreter import _resolve_magpie_python

        return _resolve_magpie_python()

    def lifecycle_eligibility(self, bench: dict) -> dict | None:
        """Return None to use the default (Magpie script-based) eligibility."""
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
    """Bypass backend: launches Hyperloom's own benchmark runner."""

    name = "bypass"

    def resolve_interpreter(self) -> str:
        """Return a plain python3 for bypass (no Magpie import needed)."""
        import shutil
        import sys

        return sys.executable or shutil.which("python3") or "python3"

    # Serving frameworks whose OpenAI server bypass can persist for reuse.
    _LIFECYCLE_FRAMEWORKS = frozenset({"vllm", "atom", "sglang"})

    def lifecycle_eligibility(self, bench: dict) -> dict | None:
        """Decide bypass server_lifecycle eligibility."""
        framework = str(bench.get("framework") or "").lower()
        envs = bench.get("envs") or {}
        try:
            port = int(envs.get("PORT", 8888))
        except (TypeError, ValueError):
            port = 8888
        verdict = {"eligible": False, "framework": framework, "port": port, "reason": ""}
        # The reuse protocol boots a local server and re-attaches a client round to it; that only holds single-node.
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
    """Resolve the active backend name from the environment."""
    raw = (os.environ.get(BENCHMARK_BACKEND_ENV) or "").strip().lower()
    if not raw or raw not in KNOWN_BENCHMARK_BACKENDS:
        return DEFAULT_BENCHMARK_BACKEND
    return raw


def resolve_backend() -> BenchmarkBackend:
    """Resolve the active benchmark backend instance."""
    name = resolve_backend_name()
    if name == "bypass":
        return BypassBackend()
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
    """Build the benchmark command using the active backend."""
    return resolve_backend().build_command(
        python_exe=python_exe,
        config_path=config_path,
        output_dir=output_dir,
    )
