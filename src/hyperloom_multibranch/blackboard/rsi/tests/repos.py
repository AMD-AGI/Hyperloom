# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The experiment repository as the briefing reads it, in memory: each branch's head, and the recipe of each commit."""

from __future__ import annotations

import dataclasses

from hyperloom.contracts.recipe import Recipe


@dataclasses.dataclass
class FakeRepo:
    """Each branch's head and each commit's recipe; ``reads`` lists the commits whose recipe was read, in order."""

    heads: dict[str, str]
    recipes: dict[str, Recipe]
    reads: list[str] = dataclasses.field(default_factory=list)

    async def head(self, branch: str) -> str:
        return self.heads[branch]

    async def read_recipe(self, commit: str) -> Recipe:
        self.reads.append(commit)
        return self.recipes[commit]
