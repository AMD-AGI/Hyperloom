# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``kernelforge forge-finalize``: deliver a finished forge-loop campaign as one reviewed commit."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from kernelforge.cli.options import AgentRuntimeOptions, _agent_runtime_options, _agent_runtime_overrides


@click.command("forge-finalize")
@click.option(
    "--workspace",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Workspace of a forge-loop campaign that is not running",
)
@click.option(
    "--author",
    default="",
    help="'Name <email>' that authors and commits the delivered commit (default: the workspace's git identity)",
)
@click.option(
    "--lint",
    "lint_commands",
    multiple=True,
    help="A check the delivered sources must pass, run with their paths appended, e.g. 'ruff check'. Repeatable; "
    "one that already failed on the starting kernel is reported, not held against the review",
)
@click.option(
    "--review-guide",
    "guides",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="A file of the kernel repository's own rules for its language, read by the review first. Repeatable",
)
@click.option(
    "--review-rounds",
    default=2,
    show_default=True,
    type=click.IntRange(min=1),
    help="Review sessions at most; another runs only while style or lint findings remain",
)
@click.option(
    "--pairs",
    default=3,
    show_default=True,
    type=click.IntRange(min=2),
    help="Pairs that time the reviewed kernel beside the best, extended to five while a slowdown is not ruled out",
)
@click.option(
    "--bench-timeout-sec",
    default=900,
    show_default=True,
    type=click.IntRange(min=1),
    help="Timeout of one benchmark or correctness run",
)
@_agent_runtime_options
def forge_finalize(
    workspace: str,
    author: str,
    lint_commands: tuple[str, ...],
    guides: tuple[str, ...],
    review_rounds: int,
    pairs: int,
    bench_timeout_sec: int,
    agent: AgentRuntimeOptions,
) -> None:
    """Review the campaign's kept change as a whole, check and time it, and squash its rounds into one commit.

    Run after forge-ablate, whose ladder tells the review which rounds pay. Writes forge_experiments/finalize.json
    and finalize.md; the per-round commits stay on the forge-history branch.
    """
    from kernelforge.config import Config
    from kernelforge.finalize import FINALIZE_MD, DeliveryOptions, FinalizeError, finalize_campaign
    from kernelforge.finalize.review import make_review_fn
    from kernelforge.loop.dashboard import publish_dashboard
    from kernelforge.loop.run_state import WorkspaceLockError

    config = Config.from_env(workspace=workspace, **_agent_runtime_overrides(agent))
    try:
        report = asyncio.run(
            finalize_campaign(
                workspace,
                review_fn=make_review_fn(config),
                options=DeliveryOptions(
                    author=author,
                    lint_commands=lint_commands,
                    guides=guides,
                    review_rounds=review_rounds,
                    pairs=pairs,
                    timeout_sec=bench_timeout_sec,
                ),
            )
        )
    except WorkspaceLockError as error:
        raise click.ClickException(f"{error}: stop forge-loop before finalizing") from error
    except FinalizeError as error:
        raise click.ClickException(str(error)) from error
    root = Path(workspace) / "forge_experiments"
    publish_dashboard(root)
    gates = report["gates"]
    click.echo(
        f"delivered {report['final_commit'][:12]}; review {'kept' if gates['kept'] else 'not kept: ' + gates['reason']}"
    )
    click.echo(f"finalize: {root / FINALIZE_MD}")
