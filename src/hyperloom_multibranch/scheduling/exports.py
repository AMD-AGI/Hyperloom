# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The CLOSE exports: the Recipe KB write-back of the final choice, the session's knowledge, the session breakdown
and the dashboard.

They read a notebook that no longer changes. The write-back goes first, reading the patch and kernel files of the
chosen commit's recipe from the session's blob store, then the notebook's knowledge steps (the search tree to the archive;
the Experiences and the specialists' learnings go nowhere while MB1 runs on no Experience KB, see
:func:`~hyperloom_multibranch.scheduling.mb_fakes.mb_fake_close_knowledge`), because the breakdown records both
outcomes next to the chosen commit's recipe and how FINAL chose it; once the search tree is archived, the scheduler's
LLM writes a generation of frontier scorers into the pool beside the archive (:func:`author_scorers`), a step that
never stops CLOSE; ``session_breakdown.json`` and the knowledge steps' receipts land in ``MB-reports/``, and the
dashboard is rebuilt a last time with the queue the stop left.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

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
from hyperloom_multibranch.notebook.knowledge.breakdown import build_breakdown, write_breakdown
from hyperloom_multibranch.notebook.knowledge.recipe_writeback import SKIPPED, FinalRecipe, publish_final_recipe
from hyperloom_multibranch.notebook.rsi.authoring import author_scorers
from hyperloom_multibranch.notebook.rsi.session import CloseStep, KnowledgeClose
from hyperloom_multibranch.scheduling.dashboard import SessionDashboard
from hyperloom_multibranch.scheduling.final import FinalChoice
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.llm import Completion
from hyperloom_multibranch.scheduling.mb_fakes import mb_fake_close_knowledge

#: The single-lineage CLOSE's status when no Recipe KB is configured.
KB_DISABLED = "disabled"
KB_FAILED = "failed"


@dataclass(frozen=True)
class CloseExports:
    """What CLOSE handed on: the Recipe KB write-back, the knowledge steps, the generation of frontier scorers, the
    breakdown file and the dashboard files."""

    kb_write_back: RemoteWriteResult
    knowledge: KnowledgeClose
    scorers: CloseStep
    breakdown: Path
    dashboard: DashboardFiles

    def knowledge_document(self) -> dict[str, object]:
        """The knowledge steps and the scorers' step as JSON-ready data: the breakdown's ``close.knowledge``."""

        return {**self.knowledge.document(), "scorers": encode(self.scorers)}


@dataclass(frozen=True)
class ScorerAuthoring:
    """The completion CLOSE asks for a generation of frontier scorers, and ``model``, the model it runs on, which
    each written scorer's provenance names."""

    completion: Completion
    model: str


@dataclass(frozen=True)
class ExportSources:
    """What CLOSE exports from: the scheduler's notebook handle, the repository and the blob store that hold the
    chosen commit's recipe and its files, and the session's dashboard."""

    notebook: Notebook
    repo: ExperimentRepo
    blobs: BlobStore
    dashboard: SessionDashboard


@dataclass(frozen=True)
class ExportServices:
    """What CLOSE calls out to: the Recipe KB the final recipe is written back to, ``None`` when none is configured,
    and the LLM that writes a generation of frontier scorers."""

    kb_client: RemoteRecipeClient | None
    scorers: ScorerAuthoring


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
        self._kb_client = services.kb_client
        self._notebook = sources.notebook
        self._repo = sources.repo
        self._blobs = sources.blobs
        self._dashboard = sources.dashboard
        self._metric = metric
        self._budget_seconds = budget_seconds

    async def export(self, final: FinalChoice | None, queue: QueueState | None) -> CloseExports:
        """Write the exports; ``queue`` is what the toolbox reported after the stop."""

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
        selection = encode(final) if final is not None else None
        stepped = {**knowledge.document(), "scorers": encode(scorers)}
        document = build_breakdown(
            snapshot, final=chosen, selection=selection, kb_write_back=kb_write_back, knowledge=stepped
        )
        breakdown = write_breakdown(self._layout.reports, document)
        return CloseExports(kb_write_back, knowledge, scorers, breakdown, await self._dashboard.write(queue))

    async def _ask_scorers(self, system: str, user: str) -> str:
        return (await self._scorers.completion(system, user)).text

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
        try:
            return await asyncio.to_thread(publish_final_recipe, self._kb_client, snapshot, final_recipe)
        except (RecipeError, RemoteRecipeValidationError, KBStoreError, BlobError, OSError) as exc:
            return RemoteWriteResult(KB_FAILED, f"{type(exc).__name__}: {exc}")


__all__ = [
    "KB_DISABLED",
    "KB_FAILED",
    "CloseExports",
    "ExportServices",
    "ExportSources",
    "ScorerAuthoring",
    "SessionExports",
]
