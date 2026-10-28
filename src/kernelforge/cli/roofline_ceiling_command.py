# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``roofline-ceiling`` command: estimate one workspace's ceiling and print it for the caller that asked."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from kernelforge.cli.option_groups import option_group
from kernelforge.roofline_ceiling.contract import CeilingReport
from kernelforge.roofline_ceiling.report import WORKSPACE_SUBDIR
from kernelforge.roofline_ceiling.standalone import (
    CONFIG_FILENAME,
    AnalystOptions,
    CeilingError,
    CeilingInputs,
    CeilingRequest,
    estimate_workspace_ceiling,
    prepare_workspace_ceiling,
)

#: Printed around the machine-readable result so a caller driving this command
#: from a script can find it in the log, the way forge-loop's is found.
RESULT_SENTINEL = "__FORGE_ROOFLINE_CEILING_RESULT__"


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
    try:
        prepared = prepare_workspace_ceiling(request)
        click.echo("[ceiling] collecting evidence (driver run, device profile, trace)...")
        outcome = estimate_workspace_ceiling(prepared)
    except CeilingError as exc:
        raise click.ClickException(str(exc)) from exc

    for note in outcome.notes:
        click.echo(f"[ceiling] note: {note}")
    click.echo(f"[ceiling] {len(outcome.scored_case_ids)} scored case(s)")
    _emit(outcome.report, source=outcome.source, report_path=outcome.report_path)


__all__ = ["RESULT_SENTINEL", "roofline_ceiling_command"]
