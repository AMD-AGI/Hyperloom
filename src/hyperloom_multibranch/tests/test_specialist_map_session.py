# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A mapping branch's run: the pass the scheduler starts, then its outcome handed back for the split, and the run
finished; no pass when the scheduler gives none; a stop kills a pass still running."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.specialist.arbor_adapter import MAPPED
from hyperloom_multibranch.specialist.dossier_pass import DossierOutcome, DossierStart
from hyperloom_multibranch.specialist.map_session import MapCalls, MapSession
from hyperloom_multibranch.specialist.runtime import SessionFinished, SessionStart

ARBORS = ("/agents/arbor-a", "/agents/arbor-b")
START = SessionStart("b01", "c0", 1, BranchTask(agent=ARBORS[0], deadline=5000.0, map_for=ARBORS))
OUTCOME = DossierOutcome(Path("/pass/directions.json"), None, MAPPED)


class Scheduler:
    def __init__(self, *, runs_pass: bool = True) -> None:
        self.runs_pass = runs_pass
        self.asked: list[tuple[str, str, float]] = []
        self.split: list[tuple[str, str, tuple[str, ...], DossierOutcome | None]] = []

    async def start(self, source: str, start_commit: str, deadline: float) -> DossierStart | None:
        self.asked.append((source, start_commit, deadline))
        return DossierStart(source, start_commit, deadline) if self.runs_pass else None

    async def handed(
        self, branch: str, start_commit: str, map_for: tuple[str, ...], outcome: DossierOutcome | None
    ) -> str:
        self.split.append((branch, start_commit, map_for, outcome))
        return "2 share branches asked"

    def calls(self) -> MapCalls:
        return MapCalls(self.start, self.handed)


async def test_the_pass_of_the_first_arbor_runs_and_its_map_is_handed_out_before_the_run_finishes() -> None:
    scheduler, ran = Scheduler(), []

    async def run_pass(start: DossierStart) -> DossierOutcome:
        ran.append(start)
        return OUTCOME

    with pytest.raises(SessionFinished, match="2 share branches asked"):
        await MapSession(START, scheduler.calls(), run_pass).run()

    assert scheduler.asked == [(ARBORS[0], "c0", 5000.0)]
    assert ran == [DossierStart(ARBORS[0], "c0", 5000.0)]
    assert scheduler.split == [("b01", "c0", ARBORS, OUTCOME)]


async def test_without_a_pass_the_map_is_handed_out_with_no_outcome() -> None:
    scheduler = Scheduler(runs_pass=False)

    async def run_pass(_start: DossierStart) -> DossierOutcome:
        raise AssertionError("no pass runs")

    with pytest.raises(SessionFinished):
        await MapSession(START, scheduler.calls(), run_pass).run()

    assert scheduler.split == [("b01", "c0", ARBORS, None)]


async def test_a_stop_cancels_the_pass_still_running_and_nothing_is_handed_out() -> None:
    scheduler, cancelled = Scheduler(), asyncio.Event()

    async def run_pass(_start: DossierStart) -> DossierOutcome:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("unreachable")

    session = MapSession(START, scheduler.calls(), run_pass)
    run = asyncio.ensure_future(session.run())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    await session.stop()

    assert cancelled.is_set()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert scheduler.split == []
