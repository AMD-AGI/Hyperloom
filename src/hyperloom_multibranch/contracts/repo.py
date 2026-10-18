# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's experiment repository: one git repository, a recipe per commit, one writer per branch.

A commit holds only its recipe; the patch and kernel files the recipe names live in the session's blob store, where
the toolbox reads them to build a test's environment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hyperloom.contracts.recipe import PatchRef, Recipe, RecipeDelta, ValueChange


class BranchMoved(RuntimeError):
    """A branch's head is no longer the commit a candidate was judged against."""


@dataclass(frozen=True)
class PatchConflict:
    """A patch of ``component`` that does not apply where the merge puts it; ``detail`` is what applying it said."""

    component: str
    path: str
    detail: str


@dataclass(frozen=True)
class RepoMerge:
    """A candidate commit's own change, relative to its parent, carried onto ``onto``.

    ``delta`` turns ``onto``'s recipe into the merged one. Where both changed a server argument or an environment
    variable, the candidate's value is taken and the setting listed in ``config_conflicts`` as ``onto``'s value
    before the candidate's. ``patch_conflicts`` are the patches that do not apply to the installed package where the
    merge puts them; ``delta`` leaves them out.
    """

    onto: str
    delta: RecipeDelta
    config_conflicts: tuple[ValueChange, ...] = ()
    patch_conflicts: tuple[PatchConflict, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.patch_conflicts


class ExperimentRepo(Protocol):
    """A specialist branch takes commits only from the rule gate and ``INTEGRATION_BRANCH`` only from the integrator;
    every other caller reads.

    ``commit`` lands only while ``branch`` is still at ``parent``, the commit the candidate was measured against, and
    raises ``BranchMoved`` otherwise: a gain judged against one parent must not land on another. ``rewind`` takes
    ``branch`` from ``commit`` back to ``parent`` while it is still at ``commit``, for a commit whose judgement could
    not be recorded. Unknown branches and commits raise ``KeyError``, an existing branch ``ValueError``. A commit's
    recorded value lives with its judgement in the notebook, and two commits are compared with
    ``hyperloom.contracts.recipe.diff`` on the recipes read here.
    """

    async def init_root(self, recipe: Recipe) -> str:
        """The baseline as the root commit, which starts ``INTEGRATION_BRANCH``; a repository takes one root."""

    async def read_recipe(self, commit: str) -> Recipe: ...

    async def create_branch(self, branch: str, at_commit: str) -> None: ...

    async def head(self, branch: str) -> str: ...

    async def commit(self, branch: str, parent: str, recipe: Recipe, message: str) -> str: ...

    async def merge(self, candidate: str, onto: str) -> RepoMerge:
        """Carry ``candidate``'s change onto ``onto``, writing nothing; ``RecipeError`` when the two cannot combine."""

    async def commit_integration(self, onto: str, delta: RecipeDelta, merged_from: Sequence[str]) -> str:
        """``delta`` on ``onto`` as a merge of ``merged_from``, landing on ``INTEGRATION_BRANCH`` as ``commit`` does.

        The merged recipe names as unexpressed whatever ``onto`` or a merged commit names.
        """

    async def rewind(self, branch: str, commit: str, parent: str) -> None: ...

    async def store_patch(self, component: str, text: str) -> PatchRef:
        """Keep a patch file of ``component`` for a later delta to add; the same text gets the same reference."""

    async def patch_text(self, commit: str, path: str) -> str:
        """The patch file ``path`` of ``commit``'s recipe; ``KeyError`` when the recipe names no such patch."""

    async def checkout(self, commit: str, destination: Path) -> None:
        """Write every patch and kernel file ``commit``'s recipe names under ``destination``, at the path it names."""


__all__ = ["BranchMoved", "ExperimentRepo", "PatchConflict", "RepoMerge"]
