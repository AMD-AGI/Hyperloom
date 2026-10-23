# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``kernelforge forge-distill``: propose a patch to an optimization skill from what a finished campaign learned."""

from __future__ import annotations

import asyncio

import click

from kernelforge.cli.options import AgentRuntimeOptions, _agent_runtime_options, _agent_runtime_overrides


@click.command("forge-distill")
@click.option(
    "--workspace",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Workspace of a finished forge-loop campaign, best after forge-ablate and forge-finalize",
)
@click.option(
    "--skill-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="The skill to propose edits to (a folder with SKILL.md); it is copied, never edited",
)
@click.option(
    "--out",
    default=None,
    type=click.Path(file_okay=False, resolve_path=True),
    help="Where the patch, its report and the edited copy go (default: <workspace>/forge_experiments/knowledge_patch)",
)
@click.option(
    "--reference",
    "references",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="A source tree the curator may read to check a claim, e.g. the installed FlyDSL. Repeatable",
)
@_agent_runtime_options
def forge_distill(
    workspace: str,
    skill_dir: str,
    out: str | None,
    references: tuple[str, ...],
    agent: AgentRuntimeOptions,
) -> None:
    """Audit what the campaign learned against the skill and write the edits worth making as a patch to review.

    Writes knowledge_patch.diff (applies with `git apply` from the skill's repository root), distill.md (each edit's
    evidence and the candidates rejected) and distill.json. The skill itself is left as it is.
    """
    from pathlib import Path

    from kernelforge.config import Config
    from kernelforge.finalize.distill import DistillError, distill_campaign, make_distill_fn

    config = Config.from_env(workspace=workspace, **_agent_runtime_overrides(agent))
    target = Path(out) if out else Path(workspace) / "forge_experiments" / "knowledge_patch"
    try:
        result = asyncio.run(
            distill_campaign(
                workspace,
                skill_dir=skill_dir,
                out=target,
                distill_fn=make_distill_fn(config),
                references=references,
            )
        )
    except DistillError as error:
        raise click.ClickException(str(error)) from error
    if result["failed"]:
        raise click.ClickException(f"the curator failed; nothing to review in {target}")
    click.echo(
        f"proposed edits to {len(result['files'])} file(s), {result['added_lines']} added lines, "
        f"not applied: {result['patch']}"
    )
    click.echo(f"evidence and rejected candidates: {result['report']}")
