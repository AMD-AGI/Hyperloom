# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry point for kernelforge."""

from __future__ import annotations

import click

from kernelforge.cli.forge_ablate_command import forge_ablate
from kernelforge.cli.forge_dashboard_command import forge_dashboard
from kernelforge.cli.forge_finalize_command import forge_finalize
from kernelforge.cli.forge_loop_command import forge_loop
from kernelforge.cli.forge_rewrite_command import forge_rewrite
from kernelforge.cli.forge_say_command import forge_say


@click.group()
@click.version_option(package_name="hyperloom-inference_optimizer")
def main():
    """Kernel Agents — Agentic GPU kernel development system."""


main.add_command(forge_loop)
main.add_command(forge_rewrite)
main.add_command(forge_say)
main.add_command(forge_ablate)
main.add_command(forge_dashboard)
main.add_command(forge_finalize)


def _register_kernel_rewrite_controller() -> None:
    """Attach the autonomous kernel rewrite controller command."""
    from kernelforge.kernel_rewrite_controller.command import (
        kernel_rewrite_controller_command,
    )

    main.add_command(
        kernel_rewrite_controller_command,
        name="kernel-rewrite-controller",
    )


_register_kernel_rewrite_controller()


def _register_forge_fuse() -> None:
    """Attach the fusion pipeline's own Click command under `forge-fuse`."""
    from kernelforge.fusion.command import run as forge_fuse

    main.add_command(forge_fuse, name="forge-fuse")


_register_forge_fuse()


def _register_gemm_tune() -> None:
    """Attach the deterministic GEMM tuner under `gemm-tune`."""
    from kernelforge.gemm_tune.cli import gemm_tune

    main.add_command(gemm_tune, name="gemm-tune")


_register_gemm_tune()


def _register_roofline_ceiling() -> None:
    """Attach the per-shape theoretical-ceiling estimator under `roofline-ceiling`."""
    from kernelforge.roofline_ceiling.command import roofline_ceiling_command

    main.add_command(roofline_ceiling_command, name="roofline-ceiling")


_register_roofline_ceiling()
