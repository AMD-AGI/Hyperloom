# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry point for kernelforge."""

from __future__ import annotations

import click

from kernelforge.cli.export_runs_command import export_runs
from kernelforge.cli.forge_fuse_command import forge_fuse
from kernelforge.cli.forge_loop_command import forge_loop
from kernelforge.cli.forge_rewrite_command import forge_rewrite
from kernelforge.cli.gemm_tune_command import gemm_tune
from kernelforge.cli.kernel_rewrite_controller_command import kernel_rewrite_controller_command
from kernelforge.cli.roofline_ceiling_command import roofline_ceiling_command


@click.group()
@click.version_option(package_name="hyperloom-inference_optimizer")
def main():
    """Kernel Agents — Agentic GPU kernel development system."""


main.add_command(forge_loop)
main.add_command(forge_rewrite)
main.add_command(kernel_rewrite_controller_command, name="kernel-rewrite-controller")
main.add_command(forge_fuse, name="forge-fuse")
main.add_command(gemm_tune, name="gemm-tune")
main.add_command(roofline_ceiling_command, name="roofline-ceiling")
main.add_command(export_runs, name="export-runs")
