# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""In-memory implementations of the multi-branch contracts, for tests that need a repository or a lifecycle without
git or processes."""

from __future__ import annotations

from hyperloom.contracts.recipe import Recipe
from hyperloom_multibranch.contracts.lifecycle import SpecialistLaunch
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH


class InMemoryRepo:
    """Starts with the baseline recipe as the root commit on ``INTEGRATION_BRANCH``."""

    def __init__(self, baseline: Recipe) -> None:
        self._recipes: dict[str, Recipe] = {}
        self._heads: dict[str, str] = {}
        self.messages: dict[str, str] = {}
        self._heads[INTEGRATION_BRANCH] = self._store(baseline, "baseline")

    def _store(self, recipe: Recipe, message: str) -> str:
        commit = f"c{len(self._recipes)}"
        self._recipes[commit] = recipe
        self.messages[commit] = message
        return commit

    async def read_recipe(self, commit: str) -> Recipe:
        return self._recipes[commit]

    async def create_branch(self, branch: str, at_commit: str) -> None:
        if branch in self._heads:
            raise ValueError(f"branch {branch!r} already exists")
        if at_commit not in self._recipes:
            raise KeyError(at_commit)
        self._heads[branch] = at_commit

    async def head(self, branch: str) -> str:
        return self._heads[branch]

    async def commit(self, branch: str, recipe: Recipe, message: str) -> str:
        if branch not in self._heads:
            raise KeyError(branch)
        self._heads[branch] = self._store(recipe, message)
        return self._heads[branch]


class RecordingLifecycle:
    """Keeps the launch and the current K of every running branch."""

    def __init__(self) -> None:
        self.running: dict[str, SpecialistLaunch] = {}
        self.k: dict[str, int] = {}

    async def start(self, launch: SpecialistLaunch) -> None:
        if launch.branch in self.running:
            raise ValueError(f"branch {launch.branch!r} already has a specialist")
        self.running[launch.branch] = launch
        self.k[launch.branch] = launch.k

    async def stop(self, branch: str) -> None:
        del self.running[branch]
        del self.k[branch]

    async def set_k(self, branch: str, k: int) -> None:
        if branch not in self.running:
            raise KeyError(branch)
        if k < 1:
            raise ValueError("stop the branch instead of giving it no room")
        self.k[branch] = k
