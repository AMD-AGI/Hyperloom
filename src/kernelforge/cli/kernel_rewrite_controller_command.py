# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Click command for the kernel rewrite controller."""

from __future__ import annotations

from pathlib import Path

import click

from kernelforge.kernel_rewrite_controller.controller import ControllerRunError, run_controller


@click.command("kernel-rewrite-controller")
@click.option(
    "--handoff-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
    help="Directory containing workload.md, serving-context.md, and trace-evidence.md.",
)
@click.option(
    "--budget-minutes",
    required=True,
    type=click.FloatRange(min=0.0, min_open=True),
    help="Total controller wall-clock budget in minutes.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    help="Fresh macro-cycle output directory owned by the controller.",
)
def kernel_rewrite_controller_command(
    handoff_dir: Path,
    budget_minutes: float,
    output_dir: Path,
) -> None:
    """Analyze and run autonomous single-operator rewrite campaigns."""
    try:
        state = run_controller(
            handoff_dir=handoff_dir,
            budget_minutes=budget_minutes,
            output_dir=output_dir,
        )
    except ControllerRunError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"kernel-rewrite-controller: {state.status}")


__all__ = ["kernel_rewrite_controller_command"]
