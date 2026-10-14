# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's experiment repository: one git repository, a recipe per commit, one writer per branch."""

from __future__ import annotations

from typing import Protocol

from hyperloom.contracts.recipe import Recipe


class BranchMoved(RuntimeError):
    """A branch's head is no longer the commit a candidate was judged against."""


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

    async def read_recipe(self, commit: str) -> Recipe: ...

    async def create_branch(self, branch: str, at_commit: str) -> None: ...

    async def head(self, branch: str) -> str: ...

    async def commit(self, branch: str, parent: str, recipe: Recipe, message: str) -> str: ...

    async def rewind(self, branch: str, commit: str, parent: str) -> None: ...


__all__ = ["BranchMoved", "ExperimentRepo"]
