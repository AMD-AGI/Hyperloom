# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist that proposes from a script, for running the session's skeleton without a model.

It submits each change of its script as an explore test on its branch's head, with at most K of them running at once,
then idles until it is stopped. What it submitted is read back from the notebook, so a restarted run goes on where
the last one stopped instead of measuring a change twice.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.notebook.interfaces import SnapshotQuery
from hyperloom_multibranch.contracts.notebook.records import TestRecord
from hyperloom_multibranch.contracts.toolbox import Toolbox


class BranchHeads(Protocol):
    async def head(self, branch: str) -> str: ...


class ScriptedSession:
    """One run of a scripted specialist on ``branch``."""

    def __init__(
        self,
        branch: str,
        script: Sequence[RecipeDelta],
        k: int,
        toolbox: Toolbox,
        repo: BranchHeads,
        notebook: SnapshotQuery,
        *,
        poll_seconds: float = 5.0,
    ) -> None:
        self._branch = branch
        self._script = tuple(script)
        self._k = k
        self._toolbox = toolbox
        self._repo = repo
        self._notebook = notebook
        self._poll_seconds = poll_seconds
        self._stopped = asyncio.Event()

    async def run(self) -> None:
        submitted = {record.request.delta.fingerprint() for record in await self._explored()}
        for change in self._script:
            if change.fingerprint() in submitted:
                continue
            while not self._stopped.is_set() and sum(r.in_flight for r in await self._explored()) >= self._k:
                await asyncio.sleep(self._poll_seconds)
            if self._stopped.is_set():
                return
            parent = await self._repo.head(self._branch)
            request = TestRequest(TestKind.TEST, Purpose.EXPLORE, self._branch, parent, change)
            await self._toolbox.submit_test(request)
        await self._stopped.wait()

    async def set_k(self, k: int) -> None:
        self._k = k

    async def stop(self) -> None:
        self._stopped.set()

    async def _explored(self) -> list[TestRecord]:
        snapshot = await self._notebook.snapshot()
        return [
            record
            for record in snapshot.records
            if record.request.branch_id == self._branch and record.request.purpose is Purpose.EXPLORE
        ]


__all__ = ["ScriptedSession"]
