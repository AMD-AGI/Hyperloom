# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""In-memory implementations of the multi-branch contracts, for tests that need a repository or a lifecycle without
git or processes."""

from __future__ import annotations

from collections.abc import Iterable

from hyperloom.contracts.recipe import Recipe
from hyperloom_multibranch.contracts.fork import AdmissionStatus, ForkAdmission, ForkRequest, RepairRequest
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.repo import BranchMoved


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

    async def commit(self, branch: str, parent: str, recipe: Recipe, message: str) -> str:
        if self._heads[branch] != parent:
            raise BranchMoved(f"{branch} is at {self._heads[branch]}, not {parent}")
        self._heads[branch] = self._store(recipe, message)
        return self._heads[branch]

    async def rewind(self, branch: str, commit: str, parent: str) -> None:
        if self._heads[branch] != commit:
            raise BranchMoved(f"{branch} is at {self._heads[branch]}, not {commit}")
        self._heads[branch] = parent


class RecordingScheduler:
    """Answers every fork and repair request with ``status``, naming branch ``fork-<n>`` on acceptance, and keeps what
    it was asked."""

    def __init__(self, status: AdmissionStatus) -> None:
        self.status = status
        self.forks: list[ForkRequest] = []
        self.repairs: list[RepairRequest] = []

    async def admit(self, request: ForkRequest) -> ForkAdmission:
        self.forks.append(request)
        return self._answer()

    async def admit_repair(self, request: RepairRequest) -> ForkAdmission:
        self.repairs.append(request)
        return self._answer()

    def _answer(self) -> ForkAdmission:
        if self.status is not AdmissionStatus.ACCEPTED:
            return ForkAdmission(self.status)
        return ForkAdmission(self.status, branch=f"fork-{len(self.forks) + len(self.repairs)}")


class RecordingVerifier:
    """Re-verifies every result but those of ``rejects``, and keeps the request ids it was asked about."""

    def __init__(self, rejects: Iterable[str] = ()) -> None:
        self.rejects = frozenset(rejects)
        self.asked: list[str] = []

    async def reverify(self, request_id: str) -> None:
        self.asked.append(request_id)
        if request_id in self.rejects:
            raise ResultUnverified(request_id, "the manifest hash is not the recorded one")


class RecordingLifecycle:
    """Keeps the start commit, the brief, the repair and the current K of every running branch."""

    def __init__(self) -> None:
        self.running: dict[str, str] = {}
        self.briefs: dict[str, str] = {}
        self.repairs: dict[str, RepairRequest | None] = {}
        self.k: dict[str, int] = {}

    async def start(self, branch: str, start_commit: str, brief: str = "", repair: RepairRequest | None = None) -> None:
        if branch in self.running:
            raise ValueError(f"branch {branch!r} already has a specialist")
        self.running[branch] = start_commit
        self.briefs[branch] = brief
        self.repairs[branch] = repair
        self.k[branch] = 1

    async def stop(self, branch: str) -> None:
        del self.running[branch]
        del self.briefs[branch]
        del self.repairs[branch]
        del self.k[branch]

    async def set_k(self, branch: str, k: int) -> None:
        if branch not in self.running:
            raise KeyError(branch)
        if k < 1:
            raise ValueError("stop the branch instead of giving it no room")
        self.k[branch] = k
