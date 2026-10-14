# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The CLOSE exports: the Recipe KB write-back of the final choice, the session breakdown and the dashboard.

They read a notebook that no longer changes. The write-back goes first because the breakdown records its outcome;
``session_breakdown.json`` lands in ``MB-reports/`` and the dashboard, with frontier-scored progress, in
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
from hyperloom_multibranch.scheduling.final import FinalChoice
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.patches import PatchStore

#: The single-lineage CLOSE's status when no Recipe KB is configured.
KB_DISABLED = "disabled"
KB_FAILED = "failed"


@dataclass(frozen=True)
class CloseExports:
    """What CLOSE handed on: the Recipe KB write-back, the breakdown file and the dashboard files."""

    kb_write_back: RemoteWriteResult
    breakdown: Path
    dashboard: DashboardFiles


class SessionExports:
    """Exports one stopped session into ``layout``; ``kb_client`` is ``None`` when no Recipe KB is configured.

    ``notebook`` is the scheduler's handle, ``patches`` holds the files the chosen recipe names, ``progress`` is the
    frontier-scored progress and ``metric`` the recorded-value metric.
    """

    def __init__(
        self,
        layout: SessionLayout,
        notebook: Notebook,
        repo: ExperimentRepo,
        patches: PatchStore,
        progress: ProgressQuery,
        metric: str,
        *,
        kb_client: RemoteRecipeClient | None,
    ) -> None:
        self._layout = layout
        self._kb_client = kb_client
        self._notebook = notebook
        self._repo = repo
        self._patches = patches
        self._progress = progress
        self._metric = metric

    async def export(self, final: FinalChoice | None, queue: QueueState | None) -> CloseExports:
        """Write the exports; ``queue`` is what the toolbox reported after the stop."""

        snapshot = await self._notebook.snapshot()
        kb_write_back = await self._publish(snapshot, final)
        breakdown = write_breakdown(self._layout.reports, build_breakdown(snapshot, kb_write_back=kb_write_back))
        dashboard = build_dashboard(snapshot, await self._progress.progress(), metric=self._metric, queue=encode(queue))
        return CloseExports(kb_write_back, breakdown, write_dashboard(self._layout.dashboard, dashboard))

    async def _publish(self, snapshot: NotebookSnapshot, final: FinalChoice | None) -> RemoteWriteResult:
        if self._kb_client is None:
            return RemoteWriteResult(KB_DISABLED, "no Recipe KB client is configured")
        chosen = final.chosen if final is not None else None
        if chosen is None:
            return RemoteWriteResult(SKIPPED, "no final choice")
        destination = self._layout.final_checkout
        recipe = await self._repo.read_recipe(chosen.commit)
        final_recipe = FinalRecipe(chosen.commit, recipe, destination, self._metric)
        try:
            await asyncio.to_thread(self._patches.checkout, recipe, destination)
            return await asyncio.to_thread(publish_final_recipe, self._kb_client, snapshot, final_recipe)
        except (RecipeError, RemoteRecipeValidationError, KBStoreError, BlobError, OSError) as exc:
            return RemoteWriteResult(KB_FAILED, f"{type(exc).__name__}: {exc}")


#: Builds a session's exports from the session's layout, scheduler notebook handle, repository, patch store, progress
#: and metric.
ExportsFactory = Callable[[SessionLayout, Notebook, ExperimentRepo, PatchStore, ProgressQuery, str], SessionExports]


def exports_from_env() -> ExportsFactory:
    """Session exports with the Recipe KB client the environment configures, resolved now; none when it has none."""

    return partial(SessionExports, kb_client=RemoteRecipeClient.from_env_optional())


__all__ = ["KB_DISABLED", "KB_FAILED", "CloseExports", "ExportsFactory", "SessionExports", "exports_from_env"]
