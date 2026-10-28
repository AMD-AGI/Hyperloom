# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Benchmark backend seam."""

from __future__ import annotations


import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

# Backend selection env var.
BENCHMARK_BACKEND_ENV = "HYPERLOOM_BENCHMARK_BACKEND"
DEFAULT_BENCHMARK_BACKEND = "magpie"
KNOWN_BENCHMARK_BACKENDS = frozenset({"magpie", "bypass"})

# Magpie built-in benchmark scripts that support the server_lifecycle reuse protocol.
MAGPIE_BUILTIN_SCRIPTS = frozenset(
    {
        "vllm_mi300x.sh",
        "vllm_mi355x.sh",
        "vllm_gfx12.sh",
        "sglang_mi300x.sh",
        "sglang_mi355x.sh",
        "atom_mi300x.sh",
        "atom_mi355x.sh",
    }
)


class BenchmarkBackend(Protocol):
    """Builds the benchmark subprocess command for one benchmark run."""

    name: str
    #: What a round that only boots the persistent server (and leaves it up for the rounds that measure) must carry in
    #: its config's ``benchmark.envs`` to stop after the server phase; empty when the backend needs nothing said.
    boot_phase_envs: Mapping[str, str]

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
    #: Nothing. Magpie sets ``MAGPIE_RUN_PHASE`` itself on the processes it starts, and it keeps every config env that is
    #: not one of its client-only keys as the persistent server's reuse identity: a boot round that carried the key
    #: would make every round that measures on that server fail its reuse check ("Reuse metadata mismatch").
    boot_phase_envs: Mapping[str, str] = MappingProxyType({})

    def resolve_interpreter(self) -> str:
        """Return the Magpie-importable interpreter for the Magpie backend."""
        from ._benchmark_interpreter import _resolve_magpie_python

        return _resolve_magpie_python()

    def lifecycle_refusal(self, bench: dict, framework: str) -> str | None:
        """Why Magpie cannot run ``bench`` under server_lifecycle, or ``None`` when it can."""
        from hyperloom.inference_optimizer import framework_registry

        # Server-less (scriptable) frameworks such as xDiT never boot a persistent server.
        workload_kind = str(bench.get("workload_kind") or "").strip().lower()
        if workload_kind == framework_registry.SCRIPTABLE or framework_registry.is_scriptable(framework):
            return "scriptable framework (server-less; no server_lifecycle)"
        script_name = Path(str(bench.get("benchmark_script") or "")).name
        if script_name not in MAGPIE_BUILTIN_SCRIPTS:
            return f"benchmark_script={script_name!r} is not a Magpie built-in ({sorted(MAGPIE_BUILTIN_SCRIPTS)})"
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
    #: The runner reads the phase from ``benchmark.envs`` (it has no reuse check on envs, and nothing else says it).
    boot_phase_envs: Mapping[str, str] = MappingProxyType({"MAGPIE_RUN_PHASE": "server"})

    def resolve_interpreter(self) -> str:
        """Return a plain python3 for bypass (no Magpie import needed)."""
        import shutil
        import sys

        return sys.executable or shutil.which("python3") or "python3"

    def lifecycle_refusal(self, bench: dict, framework: str) -> str | None:
        """Why bypass cannot persist ``framework``'s server for reuse, or ``None`` when it can."""
        from .bypass_engine import SERVER_FRAMEWORKS

        if framework not in SERVER_FRAMEWORKS:
            return f"framework {framework!r} is not a serving framework"
        return None

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
