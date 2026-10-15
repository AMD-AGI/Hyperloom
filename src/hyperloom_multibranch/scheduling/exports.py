# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The CLOSE exports: the Recipe KB write-back of the final choice, the session's knowledge, the session breakdown
and the dashboard.

They read a notebook that no longer changes. The write-back goes first, then the notebook's knowledge steps (the
Experiences and the specialists' learnings to the Experience KB, the search tree to the archive), because the
breakdown records both outcomes next to the chosen commit's recipe and how FINAL chose it; ``session_breakdown.json``
and the knowledge steps' receipts land in ``MB-reports/`` and the dashboard, with frontier-scored progress, in
``MB-dashboard/``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from hyperloom.contracts.recipe import RecipeError
from hyperloom.measure.compile import BlobError
from hyperloom.orchestrator.knowledge.remote_recipe import KBStoreError
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom.orchestrator.knowledge.remote_recipe.models import RemoteRecipeValidationError, RemoteWriteResult
from hyperloom_multibranch.contracts.notebook.interfaces import Notebook, ProgressQuery
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import QueueState
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.dashboard import DashboardFiles, build_dashboard, write_dashboard
from hyperloom_multibranch.notebook.knowledge.breakdown import build_breakdown, write_breakdown
from hyperloom_multibranch.notebook.knowledge.recipe_writeback import SKIPPED, FinalRecipe, publish_final_recipe
from hyperloom_multibranch.notebook.rsi.session import KnowledgeClose, close_knowledge
from hyperloom_multibranch.scheduling.final import FinalChoice
from hyperloom_multibranch.scheduling.layout import SessionLayout

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


class SessionExports:
    """Exports one stopped session into ``layout``; ``kb_client`` is ``None`` when no Recipe KB is configured.

    ``notebook`` is the scheduler's handle, ``progress`` the frontier-scored progress, ``metric`` the recorded-value
    metric and ``budget_seconds`` the session's budget. The Experience KB and the archive
    are the environment's.
    """

    def __init__(
        self,
        layout: SessionLayout,
        notebook: Notebook,
        repo: ExperimentRepo,
        progress: ProgressQuery,
        metric: str,
        budget_seconds: float,
        *,
        kb_client: RemoteRecipeClient | None,
    ) -> None:
        self._layout = layout
        self._kb_client = kb_client
        self._notebook = notebook
        self._repo = repo
        self._progress = progress
        self._metric = metric
        self._budget_seconds = budget_seconds

    async def export(self, final: FinalChoice | None, queue: QueueState | None) -> CloseExports:
        """Write the exports; ``queue`` is what the toolbox reported after the stop."""

        snapshot = await self._notebook.snapshot()
        chosen = await self._final_recipe(final)
        kb_write_back = await self._publish(snapshot, chosen)
        knowledge = await asyncio.to_thread(
            close_knowledge,
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
        dashboard = build_dashboard(snapshot, await self._progress.progress(), metric=self._metric, queue=queue)
        return CloseExports(kb_write_back, knowledge, breakdown, write_dashboard(self._layout.dashboard, dashboard))

    async def _final_recipe(self, final: FinalChoice | None) -> FinalRecipe | None:
        """The chosen commit's recipe, whose files the write-back checks out into the layout's final checkout."""

        chosen = final.chosen if final is not None else None
        if chosen is None:
            return None
        recipe = await self._repo.read_recipe(chosen.commit)
        return FinalRecipe(chosen.commit, recipe, self._layout.final_checkout, self._metric)

    async def _publish(self, snapshot: NotebookSnapshot, final_recipe: FinalRecipe | None) -> RemoteWriteResult:
        if self._kb_client is None:
            return RemoteWriteResult(KB_DISABLED, "no Recipe KB client is configured")
        if final_recipe is None:
            return RemoteWriteResult(SKIPPED, "no final choice")
        try:
            await self._repo.checkout(final_recipe.commit_id, final_recipe.files_root)
            return await asyncio.to_thread(publish_final_recipe, self._kb_client, snapshot, final_recipe)
        except (RecipeError, RemoteRecipeValidationError, KBStoreError, BlobError, OSError) as exc:
            return RemoteWriteResult(KB_FAILED, f"{type(exc).__name__}: {exc}")


#: Builds a session's exports from the session's layout, scheduler notebook handle, repository, progress, metric and
#: budget.
ExportsFactory = Callable[[SessionLayout, Notebook, ExperimentRepo, ProgressQuery, str, float], SessionExports]


def exports_from_env() -> ExportsFactory:
    """Session exports with the Recipe KB client the environment configures, resolved now; none when it has none."""

    return partial(SessionExports, kb_client=RemoteRecipeClient.from_env_optional())


__all__ = ["KB_DISABLED", "KB_FAILED", "CloseExports", "ExportsFactory", "SessionExports", "exports_from_env"]
