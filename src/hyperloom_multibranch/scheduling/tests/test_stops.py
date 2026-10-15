# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session stops: the roster closes, the right branches stop for the session's reason, and no process is left."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from pathlib import Path

import pytest

from hyperloom.common.proctree import descendants, running
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import Recipe
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.scheduling.branches import BranchControl
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.mb_fakes import MB_FAKE_BASE_IMAGE_DIGEST
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import (
    BranchOrigin,
    BranchStop,
    Roster,
    RosterCaps,
    RosterFull,
    StopCause,
)
from hyperloom_multibranch.scheduling.stops import STOP_OVERRAN, SessionStopper, SessionStopReason, StopSignals
from hyperloom_multibranch.scheduling.tests.peer_fakes import TOOLBOX_DOWN, DownToolbox, MbFakeLifecycle
from hyperloom_multibranch.toolbox.queue import BRANCH_STOPPED

NOW = 1_700_000_000.0
HARD_STOP = NOW + 60.0
#: Real seconds the clock in these tests lets a stop run before the hard stop passes.
SHORT_GRACE = 0.3
JUDGE_STOP = BranchStop(StopCause.JUDGE, "no headroom", NOW - 1)


class KeepingToolbox(DownToolbox):
    """A toolbox that withdraws nothing it was asked to; a failure double, not a placeholder."""

    async def cancel_branch(self, branch_id: str, reason: str = "branch stopped") -> int:
        self.cancelled.append(branch_id)
        return 0

    async def withdraw_all(self, reason: str = "session stopped") -> tuple[str, ...]:
        self.withdrawals.append(reason)
        return ()


class Stage:
    """A roster with the integration branch, ``b01`` running, ``b02`` stopped by the judge and ``b03`` an admitted fork,
    stopped through the real :class:`BranchControl` over the peer stand-ins."""

    def __init__(
        self,
        tmp_path: Path,
        lifecycle: MbFakeLifecycle | None = None,
        toolbox_type: type[DownToolbox] = DownToolbox,
    ) -> None:
        clock = Clock(now=lambda: NOW)
        self.roster = Roster(RosterCaps(4, 2, 8), tmp_path / "MB-roster.json", clock)
        self.processes = ProcessRegistry(tmp_path / "MB-pids.jsonl", clock)
        self.decisions_file = tmp_path / "MB-decisions.jsonl"
        decisions = DecisionLog(self.decisions_file, clock)
        self.notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.SCHEDULER, "scheduler"), clock=clock.now)
        self.toolbox = toolbox_type(
            LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"), clock=clock.now)
        )
        self.lifecycle = lifecycle or MbFakeLifecycle()
        self.repo_path = tmp_path / "repo"
        self.root = ""
        self.repo = GitExperimentRepo(self.repo_path, LocalBlobStore(tmp_path / "blobs"))
        self.branches = BranchControl(
            self.roster, self.notebook, self.lifecycle, self.toolbox, self.repo, self.processes, decisions, clock
        )
        self.stopper = SessionStopper(self.roster, self.branches, self.toolbox, self.processes, decisions, clock)

    async def opened(self) -> Stage:
        baseline = Recipe(MB_FAKE_BASE_IMAGE_DIGEST)
        self.root = await self.repo.init_root(baseline)
        await self.notebook.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=self.root))
        self.roster.add_integration(self.root, k=1)
        await self.branches.open_branches(2, self.root, 1, BranchOrigin.FIRST_BATCH)
        await self.branches.stop_branch("b02", JUDGE_STOP)
        self.roster.open("b03", origin=BranchOrigin.FORK, start_commit="", k=1, from_branch="b01")
        self.lifecycle.stopped.clear()
        self.toolbox.cancelled.clear()
        return self

    def stops(self) -> dict[str, BranchStop | None]:
        return {entry.branch: entry.stop for entry in self.roster.entries()}

    async def spawn_trees(self, tmp_path: Path, branches: tuple[str, ...]) -> list[int]:
        pids: list[int] = []
        for branch in branches:
            proc = await self.processes.spawn(
                ("sh", "-c", "sleep 60 & sleep 60"), branch=branch, log=tmp_path / "p.log"
            )
            pids.append(proc.record.pid)
        await asyncio.sleep(0.3)
        return pids + [child for pid in pids for child, _ in descendants(pid)]


def _ended(reason: SessionStopReason) -> BranchStop:
    return BranchStop(StopCause.SESSION_END, reason.value, NOW)


def _ended_with_b01(reason: SessionStopReason) -> BranchStop:
    """How the admitted fork ``b03`` ends: its parent ``b01`` stops it, as the first of the branches stopped."""

    return BranchStop(StopCause.SESSION_END, f"parent b01 stopped: {reason.value}", NOW)


async def test_end_exploration_stops_open_specialists_and_keeps_integration(tmp_path: Path) -> None:
    session = await Stage(tmp_path).opened()

    await session.stopper.end_exploration(SessionStopReason.TIME_EXHAUSTED)

    ended = _ended(SessionStopReason.TIME_EXHAUSTED)
    fork = _ended_with_b01(SessionStopReason.TIME_EXHAUSTED)
    assert session.stops() == {INTEGRATION_BRANCH: None, "b01": ended, "b02": JUDGE_STOP, "b03": fork}
    assert session.lifecycle.stopped == ["b01"]
    assert dict(session.toolbox.policies[-1].branch_k) == {INTEGRATION_BRANCH: 1}
    assert session.roster.free_slots() == 0
    with pytest.raises(RosterFull):
        session.roster.open("b04", origin=BranchOrigin.FORK, start_commit="", k=1)


async def test_a_failed_branch_stop_does_not_hold_up_the_end_of_exploration(tmp_path: Path) -> None:
    session = await Stage(tmp_path, MbFakeLifecycle(refusing=frozenset({"b01"}))).opened()

    await session.stopper.end_exploration(SessionStopReason.GLOBAL_CONVERGED)

    ended = _ended(SessionStopReason.GLOBAL_CONVERGED)
    fork = _ended_with_b01(SessionStopReason.GLOBAL_CONVERGED)
    assert (session.stops()["b01"], session.stops()["b03"]) == (ended, fork)
    assert sorted(session.toolbox.cancelled) == ["b01", "b03"]


async def test_stop_everything_withdraws_every_branch_closes_the_toolbox_and_leaves_no_process(tmp_path: Path) -> None:
    session = await Stage(tmp_path).opened()
    pids = await session.spawn_trees(tmp_path, ("b01", INTEGRATION_BRANCH, "prelude"))

    report = await session.stopper.stop_everything(SessionStopReason.SIGNAL, HARD_STOP)

    assert session.toolbox.withdrawals == [SessionStopReason.SIGNAL.value]
    assert set(session.toolbox.cancelled) == {"b01", "b03", INTEGRATION_BRANCH}
    assert session.toolbox.closed == 1
    ended = _ended(SessionStopReason.SIGNAL)
    fork = _ended_with_b01(SessionStopReason.SIGNAL)
    assert session.stops() == {INTEGRATION_BRANCH: ended, "b01": ended, "b02": JUDGE_STOP, "b03": fork}
    assert not any(running(pid) for pid in pids)
    assert report.clean
    decision = json.loads(session.decisions_file.read_text(encoding="utf-8").splitlines()[-1])
    assert decision["kind"] == "session_stop"
    assert decision["reason"] == "signal"


async def test_a_failed_branch_stop_is_reported_and_every_process_still_killed(tmp_path: Path) -> None:
    session = await Stage(tmp_path, MbFakeLifecycle(refusing=frozenset({"b01"}))).opened()
    pids = await session.spawn_trees(tmp_path, ("b01",))

    report = await session.stopper.stop_everything(SessionStopReason.PRELUDE_FAILED, HARD_STOP)

    assert not report.clean
    assert report.failures == ("RuntimeError: the runtime refused to stop b01",)
    assert not any(running(pid) for pid in pids)
    assert await session.processes.leftovers() == ()


async def test_a_stop_that_overruns_the_hard_stop_is_cut_and_every_process_killed(tmp_path: Path) -> None:
    session = await Stage(tmp_path, MbFakeLifecycle(hanging=frozenset({"b01"}))).opened()
    pids = await session.spawn_trees(tmp_path, ("b01", "b03"))

    report = await session.stopper.stop_everything(SessionStopReason.TIME_EXHAUSTED, NOW + SHORT_GRACE)

    assert STOP_OVERRAN in report.failures and not report.clean
    assert session.lifecycle.stopped == ["b01"]
    assert session.toolbox.withdrawals == [SessionStopReason.TIME_EXHAUSTED.value]
    assert not any(running(pid) for pid in pids)
    assert session.roster.closed


async def test_a_failing_toolbox_is_reported_and_every_branch_still_stopped(tmp_path: Path) -> None:
    session = await Stage(tmp_path).opened()
    session.toolbox.down = True

    report = await session.stopper.stop_everything(SessionStopReason.SIGNAL, HARD_STOP)

    assert report.failures and set(report.failures) == {f"ConnectionError: {TOOLBOX_DOWN}"}
    assert report.queue is None and not report.clean
    assert all(stop is not None for stop in session.stops().values())
    assert sorted(session.lifecycle.stopped) == ["b01"]


async def test_a_test_of_a_stopped_branch_is_refused(tmp_path: Path) -> None:
    session = await Stage(tmp_path).opened()

    with pytest.raises(RequestRejected) as refused:
        await session.toolbox.submit_test(TestRequest(TestKind.TEST, Purpose.EXPLORE, "b02", session.root))

    assert refused.value.code == BRANCH_STOPPED


async def test_every_test_is_withdrawn_whatever_branch_it_is_of(tmp_path: Path) -> None:
    session = await Stage(tmp_path).opened()
    session.toolbox.executor.gate.clear()
    held = await session.toolbox.submit_test(TestRequest(TestKind.TEST, Purpose.EXPLORE, "b01", session.root))
    session.lifecycle.hanging = frozenset({"b01"})

    report = await session.stopper.stop_everything(SessionStopReason.GLOBAL_CONVERGED, NOW + SHORT_GRACE)

    assert STOP_OVERRAN in report.failures
    assert held in session.toolbox.withdrawn and "b01" not in session.toolbox.cancelled


async def test_a_test_the_toolbox_still_holds_makes_the_report_unclean(tmp_path: Path) -> None:
    session = await Stage(tmp_path, toolbox_type=KeepingToolbox).opened()
    session.toolbox.executor.gate.clear()
    await session.toolbox.submit_test(TestRequest(TestKind.TEST, Purpose.EXPLORE, "b01", session.root))

    report = await session.stopper.stop_everything(SessionStopReason.GLOBAL_CONVERGED, HARD_STOP)

    assert report.queue is not None and report.queue.running_by_branch == {"b01": 1}
    assert not report.clean
    assert report.leftovers == ()


async def test_stop_signals_report_the_first_signal() -> None:
    signals = StopSignals()
    signals.install()
    try:
        assert signals.received is None
        os.kill(os.getpid(), signal.SIGHUP)
        assert await asyncio.wait_for(signals.wait(), 5.0) == signal.SIGHUP
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.sleep(0.05)
        assert signals.received == signal.SIGHUP
    finally:
        signals.uninstall()


async def test_a_second_signal_is_remembered_before_anyone_waits_for_it() -> None:
    signals = StopSignals()
    signals.install()
    try:
        second = asyncio.ensure_future(signals.wait_second())
        os.kill(os.getpid(), signal.SIGTERM)
        assert await asyncio.wait_for(signals.wait(), 5.0) == signal.SIGTERM
        await asyncio.sleep(0.05)
        assert not second.done()
        second.cancel()
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.sleep(0.05)
        assert await asyncio.wait_for(signals.wait_second(), 5.0) == signal.SIGINT
        assert signals.received == signal.SIGTERM
    finally:
        signals.uninstall()


async def test_waiting_before_install_is_an_error() -> None:
    with pytest.raises(RuntimeError):
        await StopSignals().wait()
