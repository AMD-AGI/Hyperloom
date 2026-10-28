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
@click.option(
    "--guide",
    "guides",
    multiple=True,
    help="A file of the skill, by its path inside it (e.g. CONTRIBUTING.md), that holds the skill's own rules for "
    "adding to it; the curator follows it where it differs from the built-in rules. Repeatable",
)
@click.option(
    "--lint",
    "lint_commands",
    multiple=True,
    help="A check the files the patch adds or changes must pass, run in the edited copy with their paths appended; "
    "{repo} stands for the skill's repository root, e.g. 'python3 tools/kb_lint.py --repo-root {repo}'. Repeatable; "
    "one that already fails on those files in the skill is reported as such",
)
@_agent_runtime_options
def forge_distill(
    workspace: str,
    skill_dir: str,
    out: str | None,
    references: tuple[str, ...],
    guides: tuple[str, ...],
    lint_commands: tuple[str, ...],
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
                guides=guides,
                lint_commands=lint_commands,
            )
        )
    except DistillError as error:
        raise click.ClickException(str(error)) from error
    if result["failed"]:
        raise click.ClickException(f"the curator failed; nothing to review in {target}")
    click.echo(Path(result["patch"]).read_text() or "No edit proposed.")
    click.echo(f"proposed edits to {len(result['files'])} file(s), not applied: {result['patch']}")
    click.echo(f"evidence and rejected candidates: {result['report']}")
    if result["lint"]:
        click.echo(f"lint fails on the edited files: {', '.join(result['lint'])}; see the report")
