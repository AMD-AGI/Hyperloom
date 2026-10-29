# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One workspace's ceiling estimated on its own, outside a forge-loop campaign, from its task configuration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from kernelforge.agent_backends.base import AgentRuntimeSettings, with_writable_sandbox
from kernelforge.agent_backends.registry import (
    create_registered_backend,
    get_agent_provider,
    resolve_agent_runtime,
    select_default_agent_provider,
)
from kernelforge.roofline_ceiling.analyst import AnalystSession, CeilingTask
from kernelforge.roofline_ceiling.estimate import (
    ESTIMATE_ERRORS,
    CeilingOutcome,
    estimate_ceiling,
)
from kernelforge.roofline_ceiling.evidence import PerformanceRun
from kernelforge.config import Config, resolve_agent_model, resolve_agent_reasoning_effort

CONFIG_FILENAME = "config.yaml"


class CeilingError(Exception):
    """The estimate cannot be taken; the message says why, for the operator who asked for it."""


def _load_config(path: Path) -> dict[str, Any]:
    """Read the task configuration, or an empty mapping when it is absent."""
    if not path.is_file():
        return {}
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CeilingError(f"{path} could not be read: {exc}") from exc
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise CeilingError(f"{path} does not parse to a mapping of task settings")
    return document


def _resolve_performance_command(explicit: str, document: dict[str, Any], config_path: Path) -> list[str]:
    """Resolve the command that runs the timed benchmark.

    ``config.yaml`` declares commands as a list of shell command strings -- the
    same contract ``evaluation.correctness.canonical`` reads -- so they are run through
    a shell rather than exec'd as argv.
    """
    if explicit.strip():
        return ["bash", "-c", explicit.strip()]

    declared = document.get("performance_command")
    if not declared:
        raise CeilingError(f"no --performance-command given and {config_path} declares no 'performance_command'")
    if isinstance(declared, str):
        commands = [declared]
    elif isinstance(declared, (list, tuple)) and all(isinstance(entry, str) for entry in declared):
        commands = [str(entry) for entry in declared]
    else:
        raise CeilingError(
            f"{config_path} declares 'performance_command' as {declared!r}; "
            "it must be a shell command string or a list of them"
        )
    return ["bash", "-c", " && ".join(command.strip() for command in commands if command.strip())]


def _resolve_kernel_files(explicit: Sequence[str], document: dict[str, Any], workspace: Path) -> list[str]:
    """Resolve the sources the analyst reads to build its work model."""
    if explicit:
        return [str(Path(entry)) for entry in explicit]
    declared = document.get("source_file_path") or document.get("target_kernel_file") or []
    if isinstance(declared, str):
        declared = [declared]
    resolved = [str((workspace / str(entry)).resolve()) for entry in declared if str(entry).strip()]
    if not resolved:
        raise CeilingError(
            "no --kernel given and the task configuration declares no 'source_file_path'; "
            "the analyst needs at least one source file to model"
        )
    return resolved


@dataclass(frozen=True)
class AnalystAgent:
    """The provider, model, session budget and sandbox the ceiling analyst runs with."""

    provider: str
    model: str
    timeout_sec: int
    sandbox_mode: str


def resolve_analyst_backend(agent: AnalystAgent):
    """Build the backend the ceiling analyst session runs on.

    Follows the same provider/model ladder forge-loop reads, so a campaign that
    estimates its own ceiling reaches the same agent the standalone command
    would have. ``sandbox_mode`` is the deployment's, so the analyst runs in the
    sandbox every other session runs in. A ``read-only`` one is widened to
    ``workspace-write`` and no further, as for every session that must write:
    the analyst writes its answer, so read-only would leave it unable to. A
    sandbox that hides the device nodes leaves the profiler without a GPU, and
    the derivation then has to say its roofs were recalled rather than measured.
    """
    name = get_agent_provider(agent.provider).name if agent.provider.strip() else select_default_agent_provider().name
    registration = get_agent_provider(name)
    selected = agent.model.strip() or resolve_agent_model(name) or registration.default_model
    runtime = resolve_agent_runtime(
        name,
        model=selected,
        settings=AgentRuntimeSettings(
            timeout_sec=agent.timeout_sec,
            reasoning_effort=resolve_agent_reasoning_effort(),
            sandbox_mode=agent.sandbox_mode,
            fallback_provider="",
        ),
    )
    return create_registered_backend(with_writable_sandbox(runtime))


@dataclass(frozen=True)
class CeilingInputs:
    """The workspace and task settings the estimate is taken from."""

    workspace_dir: str
    kernel_files: tuple[str, ...]
    driver_script: str
    config_path: str
    performance_command: str


@dataclass(frozen=True)
class AnalystOptions:
    """The analyst session's provider, model, budget and sandbox."""

    agent_provider: str
    agent_model: str
    agent_timeout_sec: int
    agent_sandbox_mode: str | None


@dataclass(frozen=True)
class CeilingRequest:
    """Every option of one standalone ceiling estimate, grouped by concern."""

    inputs: CeilingInputs
    output_dir: str
    arch: str
    analyst: AnalystOptions
    run_timeout_sec: int


@dataclass(frozen=True)
class PreparedCeiling:
    """A ceiling request with its workspace inputs resolved: everything the estimate runs, before it starts."""

    request: CeilingRequest
    workspace: Path
    document: dict[str, Any]
    command: list[str]
    sources: list[str]
    sandbox_mode: str


def prepare_workspace_ceiling(request: CeilingRequest) -> PreparedCeiling:
    """Resolve the workspace, its config, the benchmark command and the kernel sources an estimate needs.

    Raises :class:`CeilingError` when one of them cannot be resolved, before anything is run.
    """
    inputs = request.inputs
    workspace = Path(inputs.workspace_dir).expanduser().resolve()
    if not workspace.is_dir():
        raise CeilingError(f"workspace is not a directory: {workspace}")

    config_file = Path(inputs.config_path).expanduser() if inputs.config_path.strip() else workspace / CONFIG_FILENAME
    document = _load_config(config_file)
    sandbox = request.analyst.agent_sandbox_mode
    return PreparedCeiling(
        request=request,
        workspace=workspace,
        document=document,
        command=_resolve_performance_command(inputs.performance_command, document, config_file),
        sources=_resolve_kernel_files(inputs.kernel_files, document, workspace),
        sandbox_mode=Config.from_env(
            **({"agent_sandbox_mode": sandbox} if sandbox is not None else {})
        ).agent_sandbox_mode,
    )


def estimate_workspace_ceiling(prepared: PreparedCeiling) -> CeilingOutcome:
    """Estimate the theoretical achievable latency of one workspace's kernel, per scored shape.

    The answer is an optimistic lower bound under hardware limits and legal
    algorithm constraints. It reports no attainment ratio: the only latency this
    estimate has to divide by is a single run of the benchmark, and a campaign
    measures attainment against its own per-case medians instead. Raises
    :class:`CeilingError` when the estimate cannot be taken.
    """
    request, analyst = prepared.request, prepared.request.analyst
    try:
        return asyncio.run(
            estimate_ceiling(
                resolve_analyst_backend(
                    AnalystAgent(
                        analyst.agent_provider, analyst.agent_model, analyst.agent_timeout_sec, prepared.sandbox_mode
                    )
                ),
                CeilingTask(
                    run=PerformanceRun(prepared.command, prepared.workspace, float(request.run_timeout_sec)),
                    kernel_files=prepared.sources,
                    driver_script=request.inputs.driver_script,
                    case_params=prepared.document.get("shapes") or prepared.document.get("cases") or {},
                    arch=request.arch,
                ),
                AnalystSession(model=analyst.agent_model, timeout_sec=analyst.agent_timeout_sec),
                Path(request.output_dir).expanduser() if request.output_dir.strip() else None,
            )
        )
    except ESTIMATE_ERRORS as exc:
        raise CeilingError(str(exc) or type(exc).__name__) from exc


__all__ = [
    "AnalystAgent",
    "AnalystOptions",
    "CONFIG_FILENAME",
    "CeilingError",
    "CeilingInputs",
    "CeilingRequest",
    "PreparedCeiling",
    "estimate_workspace_ceiling",
    "prepare_workspace_ceiling",
    "resolve_analyst_backend",
]
