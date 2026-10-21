# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry point for ``kernelforge ceiling``."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import click
import yaml

from kernelforge.agent_backends.base import AgentRuntimeSettings, with_writable_sandbox
from kernelforge.agent_backends.registry import (
    create_registered_backend,
    get_agent_provider,
    resolve_agent_runtime,
    select_default_agent_provider,
)
from kernelforge.option_groups import option_group
from kernelforge.roofline_ceiling.analyst import AnalystSession, CeilingTask
from kernelforge.roofline_ceiling.contract import CeilingReport
from kernelforge.roofline_ceiling.estimate import (
    ESTIMATE_ERRORS,
    estimate_ceiling,
)
from kernelforge.roofline_ceiling.evidence import PerformanceRun
from kernelforge.roofline_ceiling.report import WORKSPACE_SUBDIR
from kernelforge.config import Config, resolve_agent_model, resolve_agent_reasoning_effort

CONFIG_FILENAME = "config.yaml"

#: Printed around the machine-readable result so a caller driving this command
#: from a script can find it in the log, the way forge-loop's is found.
RESULT_SENTINEL = "__FORGE_ROOFLINE_CEILING_RESULT__"


def _load_config(path: Path) -> dict[str, Any]:
    """Read the task configuration, or an empty mapping when it is absent."""
    if not path.is_file():
        return {}
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise click.ClickException(f"{path} could not be read: {exc}") from exc
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise click.ClickException(f"{path} does not parse to a mapping of task settings")
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
        raise click.ClickException(
            f"no --performance-command given and {config_path} declares no 'performance_command'"
        )
    if isinstance(declared, str):
        commands = [declared]
    elif isinstance(declared, (list, tuple)) and all(isinstance(entry, str) for entry in declared):
        commands = [str(entry) for entry in declared]
    else:
        raise click.ClickException(
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
        raise click.ClickException(
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


def _emit(report: CeilingReport, *, source: str, report_path: Path | None) -> None:
    """Print the machine-readable result between its sentinels."""
    payload = {
        "source": source,
        "report_path": str(report_path) if report_path else "",
        "ideal_ms": report.ideal_ms(),
        "mean_ideal_ms": report.mean_ideal_ms(),
    }
    click.echo(RESULT_SENTINEL)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))
    click.echo(RESULT_SENTINEL)


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
    """The analyst session's provider, model, budget and sandbox as typed on the command line."""

    agent_provider: str
    agent_model: str
    agent_timeout_sec: int
    agent_sandbox_mode: str | None


@dataclass(frozen=True)
class CeilingRequest:
    """Every option of one ``roofline-ceiling`` invocation, grouped by concern."""

    inputs: CeilingInputs
    output_dir: str
    arch: str
    analyst: AnalystOptions
    run_timeout_sec: int


@click.command("roofline-ceiling")
@option_group(
    "request",
    CeilingRequest,
    option_group(
        "inputs",
        CeilingInputs,
        click.option("--workspace", "workspace_dir", required=True, help="Kernel workspace to analyse"),
        click.option(
            "--kernel",
            "kernel_files",
            multiple=True,
            help="Source file the analyst models; repeatable. Defaults to config.yaml 'source_file_path'.",
        ),
        click.option("--driver", "driver_script", default="", help="Measurement driver, for the analyst's context"),
        click.option(
            "--config",
            "config_path",
            default="",
            help=f"Task configuration. Defaults to <workspace>/{CONFIG_FILENAME}.",
        ),
        click.option(
            "--performance-command",
            default="",
            help="Shell command that runs the timed benchmark. Overrides config.yaml 'performance_command'.",
        ),
    ),
    click.option("--output-dir", default="", help=f"Where to publish. Defaults to <workspace>/{WORKSPACE_SUBDIR}."),
    click.option("--arch", default="", help="Target arch (gfx950, gfx942). Detected via rocminfo when omitted."),
    option_group(
        "analyst",
        AnalystOptions,
        click.option(
            "--agent-provider", default="", help="Agent provider (claude, codex). Auto-selected when omitted."
        ),
        click.option("--agent-model", default="", help="Agent model. Falls back to the provider default."),
        click.option("--agent-timeout-sec", default=3600, type=int, help="Wall-clock budget for the analyst session"),
        click.option(
            "--agent-sandbox-mode",
            default=None,
            help="Provider sandbox mode. Defaults to FORGE_AGENT_SANDBOX_MODE, as for every other command.",
        ),
    ),
    click.option("--run-timeout-sec", default=1800, type=int, help="Wall-clock budget for each measurement subprocess"),
)
def roofline_ceiling_command(request: CeilingRequest) -> None:
    """Estimate the theoretical achievable latency of one kernel, per scored shape.

    The answer is an optimistic lower bound under hardware limits and legal
    algorithm constraints. It reports no attainment ratio: the only latency this
    command has to divide by is a single run of the benchmark, and a campaign
    measures attainment against its own per-case medians instead.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    inputs = request.inputs
    analyst = request.analyst
    workspace = Path(inputs.workspace_dir).expanduser().resolve()
    if not workspace.is_dir():
        raise click.ClickException(f"workspace is not a directory: {workspace}")

    config_file = Path(inputs.config_path).expanduser() if inputs.config_path.strip() else workspace / CONFIG_FILENAME
    document = _load_config(config_file)
    command = _resolve_performance_command(inputs.performance_command, document, config_file)
    sources = _resolve_kernel_files(inputs.kernel_files, document, workspace)
    sandbox_mode = Config.from_env(
        **({"agent_sandbox_mode": analyst.agent_sandbox_mode} if analyst.agent_sandbox_mode is not None else {})
    ).agent_sandbox_mode

    click.echo("[ceiling] collecting evidence (driver run, device profile, trace)...")
    try:
        outcome = asyncio.run(
            estimate_ceiling(
                resolve_analyst_backend(
                    AnalystAgent(analyst.agent_provider, analyst.agent_model, analyst.agent_timeout_sec, sandbox_mode)
                ),
                CeilingTask(
                    run=PerformanceRun(command, workspace, float(request.run_timeout_sec)),
                    kernel_files=sources,
                    driver_script=inputs.driver_script,
                    case_params=document.get("shapes") or document.get("cases") or {},
                    arch=request.arch,
                ),
                AnalystSession(model=analyst.agent_model, timeout_sec=analyst.agent_timeout_sec),
                Path(request.output_dir).expanduser() if request.output_dir.strip() else None,
            )
        )
    except ESTIMATE_ERRORS as exc:
        raise click.ClickException(str(exc) or type(exc).__name__) from exc

    for note in outcome.notes:
        click.echo(f"[ceiling] note: {note}")
    click.echo(f"[ceiling] {len(outcome.scored_case_ids)} scored case(s)")
    _emit(outcome.report, source=outcome.source, report_path=outcome.report_path)


__all__ = ["RESULT_SENTINEL", "AnalystAgent", "resolve_analyst_backend", "roofline_ceiling_command"]
