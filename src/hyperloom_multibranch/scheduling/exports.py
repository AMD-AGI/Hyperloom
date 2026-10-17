# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The CLOSE exports: the Recipe KB write-back of the final choice, the session's knowledge, the session breakdown
and the dashboard.

They read a notebook that no longer changes. The write-back goes first, reading the patch and kernel files of the
chosen commit's recipe from the session's blob store, then the notebook's knowledge steps (the search tree to the archive;
the Experiences and the specialists' learnings go nowhere while MB1 runs on no Experience KB, see
:func:`~hyperloom_multibranch.scheduling.mb_fakes.mb_fake_close_knowledge`), because the breakdown records both
outcomes next to the chosen commit's recipe and how FINAL chose it; ``session_breakdown.json`` and the knowledge
steps' receipts land in ``MB-reports/``, and the dashboard is rebuilt a last time with the queue the stop left.
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
from hyperloom_multibranch.notebook.rsi.session import KnowledgeClose
from hyperloom_multibranch.scheduling.dashboard import SessionDashboard
from hyperloom_multibranch.scheduling.final import FinalChoice
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.mb_fakes import mb_fake_close_knowledge

#: The single-lineage CLOSE's status when no Recipe KB is configured.
KB_DISABLED = "disabled"
KB_FAILED = "failed"


@dataclass(frozen=True)
class CloseExports:
    """What CLOSE handed on: the Recipe KB write-back, the knowledge steps, the breakdown file and the dashboard files."""

    kb_write_back: RemoteWriteResult
    knowledge: KnowledgeClose
    breakdown: Path
    dashboard: DashboardFiles


@dataclass(frozen=True)
class ExportSources:
    """What CLOSE exports from: the scheduler's notebook handle, the repository and the blob store that hold the
    chosen commit's recipe and its files, and the session's dashboard."""

    notebook: Notebook
    repo: ExperimentRepo
    blobs: BlobStore
    dashboard: SessionDashboard


class SessionExports:
    """Exports one stopped session into ``layout``; ``kb_client`` is ``None`` when no Recipe KB is configured.

    ``metric`` is the recorded-value metric and ``budget_seconds`` the session's budget. The archive is the
    environment's; the Experience KB is :data:`~hyperloom_multibranch.scheduling.mb_fakes.MB_FAKE_EXPERIENCE_KB`.
    """

    def __init__(
        self,
        layout: SessionLayout,
        sources: ExportSources,
        metric: str,
        budget_seconds: float,
        *,
        kb_client: RemoteRecipeClient | None,
    ) -> None:
        self._layout = layout
        self._kb_client = kb_client
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
        selection = encode(final) if final is not None else None
        document = build_breakdown(
            snapshot,
            final=chosen,
            selection=selection,
            kb_write_back=kb_write_back,
            knowledge=knowledge.document(),
        )
        breakdown = write_breakdown(self._layout.reports, document)
        return CloseExports(kb_write_back, knowledge, breakdown, await self._dashboard.write(queue))

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


__all__ = ["KB_DISABLED", "KB_FAILED", "CloseExports", "ExportSources", "SessionExports"]
