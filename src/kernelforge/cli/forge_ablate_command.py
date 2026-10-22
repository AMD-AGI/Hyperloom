# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``kernelforge forge-ablate``: price each change a finished forge-loop campaign kept."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click


@click.command("forge-ablate")
@click.option(
    "--workspace",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Workspace of a forge-loop campaign that is not running",
)
@click.option("--pairs", default=3, show_default=True, type=click.IntRange(min=2), help="Pairs timed per change")
@click.option(
    "--null-control-pairs",
    default=5,
    show_default=True,
    type=click.IntRange(min=2),
    help="Pairs of the best kernel beside itself that set the noise floor",
)
@click.option(
    "--max-hours",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
    help="Budget; changes not reached are marked skipped_budget and a rerun prices them",
)
@click.option(
    "--bench-timeout-sec",
    default=900,
    show_default=True,
    type=click.IntRange(min=1),
    help="Timeout of one benchmark or correctness run",
)
def forge_ablate(workspace: str, pairs: int, null_control_pairs: int, max_hours: float, bench_timeout_sec: int) -> None:
    """Time the best kernel beside itself with each kept change reverted, and with all of them reverted.

    Writes forge_experiments/ablation.json and ablation.md and refreshes the dashboard. A rerun keeps every
    change already priced against the same best commit.
    """
    from kernelforge.loop.ablation import ABLATION_MD, AblationError, ablate
    from kernelforge.loop.dashboard import publish_dashboard
    from kernelforge.loop.run_state import WorkspaceLockError

    try:
        asyncio.run(
            ablate(
                workspace,
                pairs=pairs,
                null_pairs=null_control_pairs,
                max_hours=max_hours,
                timeout_sec=bench_timeout_sec,
            )
        )
    except WorkspaceLockError as error:
        raise click.ClickException(f"{error}: stop forge-loop before ablating") from error
    except AblationError as error:
        raise click.ClickException(str(error)) from error
    root = Path(workspace) / "forge_experiments"
    publish_dashboard(root)
    click.echo(f"ablation: {root / ABLATION_MD}")
