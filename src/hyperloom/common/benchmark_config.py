# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``benchmark`` block of a point YAML: the operator's configuration and every materialized point read through it."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

BENCHMARK_CONFIG_ENV = "HYPERLOOM_BENCHMARK_CONFIG"

_BLOCK = "benchmark"
_AGENTX = "agentx"


class BenchmarkPointError(ValueError):
    """A point YAML whose ``benchmark`` block is missing or not shaped as Magpie reads it."""


@dataclass(frozen=True)
class AgentxLaunch:
    """``agentx.launch_overrides`` of a point: what Magpie launches an AgentX server with, beside the recipe's argv.

    ``unset_env`` names the inherited variables Magpie drops before applying ``env``. ``composed`` is whether the
    overrides carry ``append_args``, which every launch Hyperloom composes writes and a workload YAML never
    materialized does not.
    """

    env: Mapping[str, Any]
    executable: str
    append_args: tuple[str, ...]
    remove_args: tuple[str, ...]
    replace_args: bool
    unset_env: tuple[str, ...]
    source_files: Mapping[str, Any]
    composed: bool


@dataclass(frozen=True)
class BenchmarkPoint:
    """The ``benchmark`` block of a point YAML, read once.

    Text fields absent from the block are ``""``; ``envs`` keeps each value as the YAML gave it. ``agentx_launch`` is
    read whenever the block's ``agentx`` is a mapping, whether or not it is ``enabled``. ``torch_profiler`` is whether
    ``profiler.torch_profiler.enabled`` is ``true``; ``server_ready_timeout_s`` is the number
    ``server_lifecycle.server_ready_timeout_s`` carries, ``None`` when it carries none. ``workload_spec`` is the replay
    shape only AgentX materializations carry.
    """

    framework: str
    model: str
    precision: str
    runner_type: str
    benchmark_script: str
    workload_kind: str
    workload_mode: str
    quant_scheme: str
    inferencex_path: str
    envs: Mapping[str, Any]
    workload_spec: Mapping[str, Any]
    agentx: bool
    agentx_launch: AgentxLaunch
    torch_profiler: bool
    server_ready_timeout_s: float | None

    @classmethod
    def from_document(cls, document: object, *, source: str = "the point") -> BenchmarkPoint:
        """Read the ``benchmark`` block of a loaded point YAML; ``source`` names it in a :class:`BenchmarkPointError`."""
        return cls.from_block(_benchmark_block(document, source), source=source)

    @classmethod
    def from_block(cls, block: Mapping[str, Any], *, source: str = "the point") -> BenchmarkPoint:
        """Read a ``benchmark`` block already taken out of its point YAML."""
        agentx = block.get(_AGENTX)
        launch = _mapping(agentx.get("launch_overrides") if isinstance(agentx, Mapping) else None, source)
        torch_profiler = _mapping(_mapping(block.get("profiler"), source).get("torch_profiler"), source)
        ready = _mapping(block.get("server_lifecycle"), source).get("server_ready_timeout_s")
        return cls(
            framework=_text(block, "framework"),
            model=_text(block, "model"),
            precision=_text(block, "precision"),
            runner_type=_text(block, "runner_type"),
            benchmark_script=_text(block, "benchmark_script"),
            workload_kind=_text(block, "workload_kind"),
            workload_mode=_text(block, "workload_mode"),
            quant_scheme=_text(block, "quant_scheme"),
            inferencex_path=_text(block, "inferencex_path").strip(),
            envs=_mapping(block.get("envs"), source),
            workload_spec=_mapping(block.get("workload_spec"), source),
            agentx=is_agentx(block),
            agentx_launch=AgentxLaunch(
                env=_mapping(launch.get("env"), source),
                executable=_text(launch, "executable"),
                append_args=_strings(launch.get("append_args"), source),
                remove_args=_strings(launch.get("remove_args"), source),
                replace_args=bool(launch.get("replace_args")),
                unset_env=_strings(launch.get("unset_env"), source),
                source_files=_mapping(launch.get("source_files"), source),
                composed="append_args" in launch,
            ),
            torch_profiler=torch_profiler.get("enabled") is True,
            server_ready_timeout_s=_number(ready),
        )


def read_benchmark_point(path: Path) -> BenchmarkPoint:
    """Read the point YAML at ``path``.

    Raises:
        OSError: ``path`` cannot be read.
        yaml.YAMLError: It is not YAML.
        BenchmarkPointError: It holds no ``benchmark`` mapping, or a part of it is not shaped as Magpie reads it.
    """
    return BenchmarkPoint.from_document(yaml.safe_load(path.read_text(encoding="utf-8")), source=str(path))


def _benchmark_block(document: object, source: str) -> Mapping[str, Any]:
    block = document.get(_BLOCK) if isinstance(document, Mapping) else None
    if not isinstance(block, Mapping):
        raise BenchmarkPointError(f"{source} must contain a {_BLOCK} mapping")
    return block


def _mapping(value: object, source: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise BenchmarkPointError(f"{source} holds {value!r} where Magpie reads a mapping")
    return MappingProxyType(dict(value))


def _strings(value: object, source: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise BenchmarkPointError(f"{source} holds {value!r} where Magpie reads a list")
    return tuple(str(item) for item in value)


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _text(block: Mapping[str, Any], key: str) -> str:
    return str(block.get(key) or "")


def benchmark_config_path(env: Mapping[str, str] | None = None) -> Path | None:
    runtime = os.environ if env is None else env
    value = runtime.get(BENCHMARK_CONFIG_ENV, "").strip()
    return Path(value) if value else None


@lru_cache(maxsize=16)
def _read_config(path: Path, _modified_ns: int) -> dict[str, Any]:
    return dict(_benchmark_block(yaml.safe_load(path.read_text(encoding="utf-8")), str(path)))


def configured_benchmark(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    path = benchmark_config_path(env)
    if path is None:
        return {}
    return _read_config(path, path.stat().st_mtime_ns).copy()


def is_agentx(benchmark: Mapping[str, Any]) -> bool:
    """Whether a ``benchmark`` block turns AgentX on: ``agentx`` is true, an enabling word, or a mapping not disabled."""
    value = benchmark.get(_AGENTX)
    if isinstance(value, dict):
        value = value.get("enabled", True)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "enabled", "enable", "yes", "1"}
    return value is True


is_magpie_agentx = is_agentx
