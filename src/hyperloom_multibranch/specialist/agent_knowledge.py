# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an autonomous branch's agent is told of the session beyond its branch: the integration head.

The agent works from its branch's start commit and hears nothing of the gate, so what other branches merged reaches it
only through :class:`IntegrationNotes`: a markdown file rewritten whenever the integration head moves, with the head's
server arguments and environment, its patches and kernel files, written out beside the notes, and the changes merged
into it, each with the intent its branch claimed. The agent's onboarding has it read the notes every iteration.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from hyperloom.contracts.recipe import Recipe
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.specialist.agent_session import SessionNotebook

NOTES_FILE = "mb-integration.md"
#: Where the head's patch and kernel files are written, beside the notes, at the paths its recipe names.
FILES_DIR = "mb-integration"
ENCODING = "utf-8"
NONE = "- none"


@dataclass(frozen=True)
class MergedChange:
    """A branch commit the integration head holds: its branch, its gain over its parent and what its branch said it
    changes, empty for an import of an agent, which claims nothing."""

    commit_id: str
    branch: str
    gain_pct: float
    intent: str


def merged_changes(snapshot: NotebookSnapshot, head: str) -> tuple[MergedChange, ...]:
    """The branch commits merged into ``head`` or a commit it builds on."""

    held = set(snapshot.lineage.ancestry(head))
    records = {record.request_id: record for record in snapshot.records}
    changes = []
    for commit in snapshot.committed:
        if commit.merged_into not in held:
            continue
        record = records.get(commit.request_id)
        intent = None if record is None else snapshot.intents.get(record.request.intent_fingerprint or "")
        said = "" if intent is None else f"{intent.target}: {intent.mechanism} ({intent.layer.value})"
        changes.append(MergedChange(commit.commit_id, commit.branch, commit.gain_pct, said))
    return tuple(changes)


def render_notes(head: str, recipe: Recipe, merged: Iterable[MergedChange], files: Path) -> str:
    """The notes of the integration ``head`` whose recipe is ``recipe``, its files written under ``files``."""

    envs = [f"- `{name}={value}`" for name, value in sorted(recipe.launch.envs.items())] or [NONE]
    patches = [
        f"- {component.component}: {files / patch.path}"
        for component in recipe.components
        for patch in component.patches
    ]
    kernels = [f"- {kernel.kind.value} {kernel.target}: {files / kernel.ref}" for kernel in recipe.kernels]
    changes = [
        f"- {change.branch} {change.commit_id[:12]} {change.gain_pct:+.2f}%: {change.intent or 'no intent recorded'}"
        for change in merged
    ]
    return "\n".join(
        [
            "# The session's integration head",
            "",
            f"Commit {head}. Everything below is already measured and merged: do not redo it, and do not take it into",
            "your workspace either, since the gate measures your changes on top of your branch's own head.",
            "",
            "## Launch",
            "",
            f"Server arguments: `{recipe.launch.server_args or 'none'}`",
            "",
            "Environment:",
            *envs,
            "",
            "## Patches",
            "",
            *(patches or [NONE]),
            "",
            "## Kernel files",
            "",
            *(kernels or [NONE]),
            "",
            "## Merged changes",
            "",
            *(changes or ["- none yet"]),
            "",
        ]
    )


class IntegrationNotes:
    """The notes at ``path`` of the integration head of ``repo``, with what ``notebook`` says was merged into it."""

    def __init__(self, repo: ExperimentRepo, notebook: SessionNotebook, path: Path) -> None:
        self._repo, self._notebook, self._path = repo, notebook, path
        self._written = ""

    @property
    def path(self) -> Path:
        return self._path

    async def refresh(self) -> bool:
        """Rewrite the notes when the integration head moved since they were last written; whether it did."""

        head = await self._repo.head(INTEGRATION_BRANCH)
        if head == self._written:
            return False
        files = self._path.parent / FILES_DIR
        await asyncio.to_thread(shutil.rmtree, files, ignore_errors=True)
        files.mkdir(parents=True)
        await self._repo.checkout(head, files)
        merged = merged_changes(await self._notebook.snapshot(), head)
        text = render_notes(head, await self._repo.read_recipe(head), merged, files)
        self._path.write_text(text, encoding=ENCODING)
        self._written = head
        return True


__all__ = ["FILES_DIR", "NOTES_FILE", "IntegrationNotes", "MergedChange", "merged_changes", "render_notes"]
