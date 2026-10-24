# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The CLOSE exports: the Recipe KB write-back of the final choice, the session's knowledge, the session breakdown
and the dashboard.

They read a notebook that no longer changes. The write-back goes first, reading the patch and kernel files of the
chosen commit's recipe from the session's blob store, then the notebook's knowledge steps (the search tree to the archive;
the Experiences and the specialists' learnings go nowhere while MB1 runs on no Experience KB, see
:func:`~hyperloom_multibranch.scheduling.mb_fakes.mb_fake_close_knowledge`), because the breakdown records both
outcomes next to the chosen commit's recipe and how FINAL chose it; once the search tree is archived, the scheduler's
LLM writes a generation of frontier scorers into the pool beside the archive (:func:`author_scorers`) and curates the
RSI lesson pool the knowledge steps published the session's learnings to (:func:`curate_lessons`), steps that never
stop CLOSE; ``session_breakdown.json``, the knowledge steps' receipts and :data:`IMAGE_CHECK_FILE`, the whole
image check the session ran before its first phase, land in ``MB-reports/``, and the dashboard is rebuilt a last time
with the queue the stop left.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.outside_facts.image_check import ImageCheckReport
from hyperloom.contracts.recipe import RecipeError
from hyperloom.measure.compile import BlobError, BlobStore
from hyperloom.orchestrator.knowledge.remote_recipe import KBStoreError
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom.orchestrator.knowledge.remote_recipe.models import RemoteRecipeValidationError, RemoteWriteResult
from hyperloom_multibranch.contracts.notebook.interfaces import Notebook
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import QueueState
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.dashboard import DashboardFiles
from hyperloom_multibranch.notebook.knowledge.breakdown import (
    CloseOutcome,
    SessionPlace,
    build_breakdown,
    write_breakdown,
)
from hyperloom_multibranch.notebook.knowledge.recipe_writeback import SKIPPED, FinalRecipe, publish_final_recipe
from hyperloom_multibranch.notebook.rsi.authoring import author_scorers
from hyperloom_multibranch.notebook.rsi.lesson_curation import curate_lessons
from hyperloom_multibranch.notebook.rsi.session import CloseStep, KnowledgeClose
from hyperloom_multibranch.scheduling.dashboard import SessionDashboard
from hyperloom_multibranch.scheduling.final import FinalChoice
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.contracts.llm import Completion
from hyperloom_multibranch.scheduling.manifest import read_manifest
from hyperloom_multibranch.scheduling.mb_fakes import mb_fake_close_knowledge

#: The single-lineage CLOSE's status when no Recipe KB is configured.
KB_DISABLED = "disabled"
KB_FAILED = "failed"
#: The whole image check's report, under ``MB-reports/``.
IMAGE_CHECK_FILE = "image_check.json"


def knowledge_steps(knowledge: KnowledgeClose, scorers: CloseStep, curation: CloseStep) -> dict[str, CloseStep]:
    """The knowledge steps, the scorers' step and the lesson pool's curation by their names in ``close.knowledge``:
    the steps of the breakdown and of the CLOSE report alike."""

    return {**knowledge.steps(), "scorers": scorers, "lesson_curation": curation}


def knowledge_document(knowledge: KnowledgeClose, scorers: CloseStep, curation: CloseStep) -> dict[str, object]:
    """The :func:`knowledge_steps` as JSON-ready data: ``close.knowledge`` of the breakdown and of the CLOSE report
    alike."""

    return {name: encode(step) for name, step in knowledge_steps(knowledge, scorers, curation).items()}


def step_lines(steps: Mapping[str, CloseStep]) -> list[str]:
    """The CLOSE report's line for each of ``steps``, labelled by its name in ``close.knowledge``, so the report lists
    every step the breakdown records."""

    return [
        f"{name.replace('_', ' ').capitalize()}: {step.status} {step.reason}".rstrip() for name, step in steps.items()
    ]


@dataclass(frozen=True)
class CloseExports:
    """What CLOSE handed on: the Recipe KB write-back, the knowledge steps, the generation of frontier scorers, the
    breakdown file, the dashboard files, and the curation of the RSI lesson pool."""

    kb_write_back: RemoteWriteResult
    knowledge: KnowledgeClose
    scorers: CloseStep
    breakdown: Path
    dashboard: DashboardFiles
    curation: CloseStep

    def knowledge_steps(self) -> dict[str, CloseStep]:
        """The knowledge steps, the scorers' step and the curation: the steps the CLOSE report lists."""

        return knowledge_steps(self.knowledge, self.scorers, self.curation)

    def knowledge_document(self) -> dict[str, object]:
        """The knowledge steps, the scorers' step and the curation as JSON-ready data: the breakdown's
        ``close.knowledge``."""

        return knowledge_document(self.knowledge, self.scorers, self.curation)


@dataclass(frozen=True)
class ScorerAuthoring:
    """The completion CLOSE asks for a generation of frontier scorers, and ``model``, the model it runs on, which
    each written scorer's provenance names."""

    completion: Completion
    model: str


@dataclass(frozen=True)
class ExportSources:
    """What CLOSE exports from: the scheduler's notebook handle, the repository and the blob store that hold the
    chosen commit's recipe and its files, the session's dashboard and the image check it ran before its first phase,
    ``None`` for a FINAL and CLOSE over a session that ended before FINAL, which run none."""

    notebook: Notebook
    repo: ExperimentRepo
    blobs: BlobStore
    dashboard: SessionDashboard
    image_check: ImageCheckReport | None


@dataclass(frozen=True)
class ExportServices:
    """What CLOSE calls out to: the Recipe KB the final recipe is written back to, ``None`` when none is configured,
    the LLM that writes a generation of frontier scorers, and the one that curates the RSI lesson pool, in the same
    shape."""

    kb_client: RemoteRecipeClient | None
    scorers: ScorerAuthoring
    curator: ScorerAuthoring


class SessionExports:
    """Exports one stopped session into ``layout``, through ``services``.

    ``metric`` is the recorded-value metric and ``budget_seconds`` the session's budget. The archive is the
    environment's; the Experience KB is :data:`~hyperloom_multibranch.scheduling.mb_fakes.MB_FAKE_EXPERIENCE_KB`.
    """

    def __init__(
        self,
        layout: SessionLayout,
        sources: ExportSources,
        metric: str,
        budget_seconds: float,
        services: ExportServices,
    ) -> None:
        self._layout = layout
        self._scorers = services.scorers
        self._curator = services.curator
        self._kb_client = services.kb_client
        self._notebook = sources.notebook
        self._repo = sources.repo
        self._blobs = sources.blobs
        self._dashboard = sources.dashboard
        self._image_check = sources.image_check
        self._metric = metric
        self._budget_seconds = budget_seconds

    async def export(self, final: FinalChoice | None, queue: QueueState | None) -> CloseExports:
        """Write the exports; ``queue`` is what the toolbox reported after the stop."""

        if self._image_check is not None:
            await asyncio.to_thread(self._image_check.write, self._layout.reports / IMAGE_CHECK_FILE)
        snapshot = await self._notebook.snapshot()
        chosen = await self._final_recipe(final)
        kb_write_back = await self._publish(snapshot, chosen)
        knowledge = await asyncio.to_thread(
            mb_fake_close_knowledge,
            snapshot,
            self._metric,
            budget_seconds=self._budget_seconds,
            receipts=self._layout.reports,
        )
        scorers = await author_scorers(snapshot, self._ask_scorers, model=self._scorers.model)
        curation = await self._curate(snapshot)
        selection = encode(final) if final is not None else None
        stepped = knowledge_document(knowledge, scorers, curation)
        document = build_breakdown(
            snapshot,
            CloseOutcome(final=chosen, selection=selection, kb_write_back=kb_write_back, knowledge=stepped),
            SessionPlace(
                session_dir=self._layout.root,
                runner_dir=self._layout.logs,
                code_revision=read_manifest(self._layout.manifest).code_version.git_sha or None,
            ),
        )
        breakdown = write_breakdown(self._layout.reports, document)
        dashboard = await self._dashboard.write(queue)
        return CloseExports(kb_write_back, knowledge, scorers, breakdown, dashboard, curation)

    async def _ask_scorers(self, system: str, user: str) -> str:
        return (await self._scorers.completion(system, user)).text

    async def _curate(self, snapshot: NotebookSnapshot) -> CloseStep:
        """The lesson pool curated by the session's curator, after the knowledge steps published to it."""

        async def ask(system: str, user: str) -> str:
            return (await self._curator.completion(system, user)).text

        return await curate_lessons(snapshot, ask, model=self._curator.model)

    async def _final_recipe(self, final: FinalChoice | None) -> FinalRecipe | None:
        """The chosen commit's recipe, whose files the write-back reads from the blob store; ``files_root`` is not
        read."""

        chosen = final.chosen if final is not None else None
        if chosen is None:
            return None
        recipe = await self._repo.read_recipe(chosen.commit)
        return FinalRecipe(chosen.commit, recipe, self._layout.final, self._metric, self._blobs)

    async def _publish(self, snapshot: NotebookSnapshot, final_recipe: FinalRecipe | None) -> RemoteWriteResult:
        if self._kb_client is None:
            return RemoteWriteResult(KB_DISABLED, "no Recipe KB client is configured")
        if final_recipe is None:
            return RemoteWriteResult(SKIPPED, "no final choice")
        rule_pack = read_manifest(self._layout.manifest).rule_pack
        try:
            return await asyncio.to_thread(publish_final_recipe, self._kb_client, snapshot, final_recipe, rule_pack)
        except (RecipeError, RemoteRecipeValidationError, KBStoreError, BlobError, OSError) as exc:
            return RemoteWriteResult(KB_FAILED, f"{type(exc).__name__}: {exc}")


__all__ = [
    "KB_DISABLED",
    "IMAGE_CHECK_FILE",
    "KB_FAILED",
    "CloseExports",
    "ExportServices",
    "ExportSources",
    "ScorerAuthoring",
    "SessionExports",
    "knowledge_document",
    "knowledge_steps",
    "step_lines",
]
