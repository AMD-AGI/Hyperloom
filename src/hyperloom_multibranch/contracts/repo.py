# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's experiment repository: one git repository, a recipe per commit, one writer per branch."""

from __future__ import annotations

from typing import Protocol

from hyperloom.contracts.recipe import Recipe


class ExperimentRepo(Protocol):
    """A specialist branch takes commits only from the rule gate and ``INTEGRATION_BRANCH`` only from the integrator;
    every other caller reads.

    A commit's recorded value lives with its judgement in the notebook, and two commits are compared with
    ``hyperloom.contracts.recipe.diff`` on the recipes read here.
    """

    async def read_recipe(self, commit: str) -> Recipe: ...

    async def create_branch(self, branch: str, at_commit: str) -> None: ...

    async def head(self, branch: str) -> str: ...

    async def commit(self, branch: str, recipe: Recipe, message: str) -> str: ...


__all__ = ["ExperimentRepo"]
