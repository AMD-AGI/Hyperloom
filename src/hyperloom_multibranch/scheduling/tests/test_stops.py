# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session stops: the roster closes, the right branches stop for the session's reason, and no process is left."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.common.proctree import descendants, running
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import ConfigDelta, Recipe, RecipeDelta
from hyperloom.hosted.mcp_server import MIN_TOKEN_LENGTH, McpServer, McpServerConfig
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.fork import AdmissionStatus, ForkRequest
from hyperloom_multibranch.contracts.lifecycle import SpecialistStatus
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import LineageEvent
from hyperloom_multibranch.contracts.blackboard.vocab import LineageKind
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.rsi.briefing import SessionBriefs
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.scheduling import branches
from hyperloom_multibranch.scheduling.branches import BatchStart, BranchControl
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.explore import gave_up_reason
from hyperloom_multibranch.scheduling.forks import ForkDesk
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
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    BASE_IMAGE_DIGEST,
    SPECIALIST_POLL_SECONDS,
    TOOLBOX_DOWN,
    DownToolbox,
    RecordedReservations,
    RecordedSessions,
    specialist_runtime,
)
from hyperloom_multibranch.toolbox.queue import BRANCH_STOPPED

NOW = 1_700_000_000.0
HARD_STOP = NOW + 60.0
#: Real seconds the clock in these tests lets a stop run before the hard stop passes.
SHORT_GRACE = 0.3
JUDGE_STOP = BranchStop(StopCause.JUDGE, "no headroom", NOW - 1)

_CHANGE = RecipeDelta(config=ConfigDelta(server_args="--max-num-seqs 256"))  # an explore must change something


def test_the_stop_of_a_gave_up_branch_names_the_counts_the_runtime_kept() -> None:
    crashed = SpecialistStatus(running=False, crashes=4, lost=0, gave_up=True, judge_failed=False)

    assert gave_up_reason(crashed) == "its specialist gave up after 4 crashes"
    assert gave_up_reason(replace(crashed, crashes=3, lost=3)).endswith("3 crashes, 3 of them with a lost heartbeat")
    assert gave_up_reason(replace(crashed, crashes=0)).endswith("its sessions in a row found nothing to test")
    finished = replace(crashed, crashes=0, finished="the agent finished its run: deadline")
    assert gave_up_reason(finished) == "the agent finished its run: deadline"
    unready = replace(crashed, crashes=0, unready="attempt 1: the onboarding model's call failed: API Error: 400")
    assert gave_up_reason(unready) == (
        "its specialist could not be started: attempt 1: the onboarding model's call failed: API Error: 400"
    )


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
    stopped through the real :class:`BranchControl` and specialist runtime over the peer stand-ins; the runtime refuses
    to stop a branch in ``refusing``, and the run of a branch in ``hanging`` never confirms its stop."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        refusing: frozenset[str] = frozenset(),
        hanging: frozenset[str] = frozenset(),
        toolbox_type: type[DownToolbox] = DownToolbox,
    ) -> None:
        clock = Clock(now=lambda: NOW)
        self.roster = Roster(RosterCaps(4, 2, 8), tmp_path / "MB-roster.json", clock)
        self.processes = ProcessRegistry(tmp_path / "MB-pids.jsonl", clock)
        self.decisions_file = tmp_path / "MB-decisions.jsonl"
        decisions = DecisionLog(self.decisions_file, clock)
        self.blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.SCHEDULER, "scheduler"), clock=clock.now)
        self.repo_path = tmp_path / "repo"
        self.root = ""
        self.repo = GitExperimentRepo(self.repo_path, LocalBlobStore(tmp_path / "blobs"))
        self.toolbox = toolbox_type(
            LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"), clock=clock.now), repo=self.repo
        )
        self.sessions = RecordedSessions(self.toolbox, self.repo, self.blackboard, hanging=hanging)
        self.runtime = specialist_runtime(
            tmp_path / "blackboard", self.toolbox, self.repo, self.sessions, now=clock.now
        )
        self.runtime.refusing = refusing
        self.desk = ForkDesk(self.roster, decisions, clock)
        self.reserved = RecordedReservations()
        context = SchedulerContext(
            SessionLayout(tmp_path),
            self.roster,
            self.blackboard,
            decisions,
            self.processes,
            clock,
            self.desk,
            self.reserved,
        )
        self.branches = BranchControl(context, self.runtime, self.toolbox, self.repo, SessionBriefs(self.blackboard))
        self.mcp = McpServer(McpServerConfig(host_token="t" * MIN_TOKEN_LENGTH), campaign="stops")
        self.stopper = SessionStopper(context, self.branches, self.toolbox, self.mcp)

    async def opened(self) -> Stage:
        baseline = Recipe(BASE_IMAGE_DIGEST)
        self.root = await self.repo.init_root(baseline)
        await self.blackboard.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=self.root))
        self.roster.add_integration(self.root, k=1)
        await self.branches.open_branches(2, 1, BatchStart(BranchOrigin.FIRST_BATCH, self.root, HARD_STOP))
        await asyncio.sleep(SPECIALIST_POLL_SECONDS)
        await self.branches.stop_branch("b02", JUDGE_STOP)
        fork = self.roster.next_branch_id()
        self.roster.open(fork, origin=BranchOrigin.FORK, start_commit="", k=1, from_branch="b01")
        self.sessions.stopped.clear()
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
    assert session.reserved.voided == [("b03", fork.reason)]
    assert session.sessions.stopped == ["b01"]
    assert dict(session.toolbox.policies[-1].branch_k) == {INTEGRATION_BRANCH: 1}
    assert session.roster.free_slots() == 0
    with pytest.raises(RosterFull):
        session.roster.open("b04", origin=BranchOrigin.FORK, start_commit="", k=1)


async def test_the_end_of_explore_withdraws_the_tests_a_stopped_branch_was_left_to_finish(tmp_path: Path) -> None:
    session = await Stage(tmp_path).opened()
    session.toolbox.executor.gate.clear()
    left = await session.toolbox.submit_test(
        TestRequest(TestKind.TEST, Purpose.EXPLORE, "b01", session.root, delta=_CHANGE)
    )
    await session.branches.stop_branch("b01", JUDGE_STOP)
    assert session.toolbox.withdrawn == {} and "b01" in session.toolbox.policies[-1].branch_k

    await session.stopper.end_exploration(SessionStopReason.TIME_EXHAUSTED)

    assert session.toolbox.withdrawn == {left: SessionStopReason.TIME_EXHAUSTED.value}
    assert dict(session.toolbox.policies[-1].branch_k) == {INTEGRATION_BRANCH: 1}


@pytest.mark.parametrize("reason", [SessionStopReason.TIME_EXHAUSTED, SessionStopReason.SIGNAL])
async def test_a_session_stop_voids_every_fork_queued_for_a_slot(tmp_path: Path, reason: SessionStopReason) -> None:
    session = await Stage(tmp_path).opened()
    for _ in range(2):
        branch = session.roster.next_branch_id()
        session.roster.open(branch, origin=BranchOrigin.FIRST_BATCH, start_commit=session.root, k=1)
    queued = await session.desk.admit(ForkRequest("b01", session.root, "fix f1 regresses"))

    if reason is SessionStopReason.TIME_EXHAUSTED:
        await session.stopper.end_exploration(reason)
    else:
        await session.stopper.stop_everything(reason, HARD_STOP)

    assert (queued.status, queued.branch) == (AdmissionStatus.PENDING, "b06") and session.desk.pending() == ()
    logged = [json.loads(line) for line in session.decisions_file.read_text(encoding="utf-8").splitlines()]
    assert [row["reason"] for row in logged if row["kind"] == "fork" and row["detail"].get("status") == "voided"] == [
        f"voided: {reason.value}"
    ]
    assert session.reserved.voided == [("b06", reason.value), ("b03", _ended_with_b01(reason).reason)]


async def test_a_failed_branch_stop_does_not_hold_up_the_end_of_exploration(tmp_path: Path) -> None:
    session = await Stage(tmp_path, refusing=frozenset({"b01"})).opened()

    await session.stopper.end_exploration(SessionStopReason.JUDGE_STOPPED)

    ended = _ended(SessionStopReason.JUDGE_STOPPED)
    fork = _ended_with_b01(SessionStopReason.JUDGE_STOPPED)
    assert (session.stops()["b01"], session.stops()["b03"]) == (ended, fork)
    assert sorted(session.toolbox.cancelled) == ["b01", "b02", "b03"]


async def test_stop_everything_withdraws_every_branch_closes_the_toolbox_and_leaves_no_process(tmp_path: Path) -> None:
    session = await Stage(tmp_path).opened()
    pids = await session.spawn_trees(tmp_path, ("b01", INTEGRATION_BRANCH, "prelude"))
    session.mcp.start()

    report = await session.stopper.stop_everything(SessionStopReason.SIGNAL, HARD_STOP)

    with pytest.raises(RuntimeError, match="not running"):
        session.mcp.url

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
    session = await Stage(tmp_path, refusing=frozenset({"b01"})).opened()
    pids = await session.spawn_trees(tmp_path, ("b01",))

    report = await session.stopper.stop_everything(SessionStopReason.PRELUDE_FAILED, HARD_STOP)

    assert not report.clean
    assert report.failures == (
        "RuntimeError: the runtime refused to stop b01",
        "the specialist of b01 did not confirm its stop",
    )
    assert session.sessions.stopped == [] and session.sessions.running == {"b01"}
    assert not any(running(pid) for pid in pids)
    assert await session.processes.leftovers() == ()


async def test_a_specialist_that_did_not_stop_at_the_end_of_explore_is_asked_again_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(branches, "SPECIALIST_CALL_SECONDS", 0.1)
    session = await Stage(tmp_path, hanging=frozenset({"b01"})).opened()
    await session.stopper.end_exploration(SessionStopReason.JUDGE_STOPPED)

    report = await session.stopper.stop_everything(SessionStopReason.JUDGE_STOPPED, HARD_STOP)

    assert session.sessions.stopped == ["b01"] and session.sessions.running == {"b01"}
    assert report.failures == ("the specialist of b01 did not confirm its stop",) and not report.clean


async def test_a_stop_that_overruns_the_hard_stop_is_cut_and_every_process_killed(tmp_path: Path) -> None:
    session = await Stage(tmp_path, hanging=frozenset({"b01"})).opened()
    pids = await session.spawn_trees(tmp_path, ("b01", "b03"))

    report = await session.stopper.stop_everything(SessionStopReason.TIME_EXHAUSTED, NOW + SHORT_GRACE)

    assert STOP_OVERRAN in report.failures and not report.clean
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
    assert session.sessions.stopped == ["b01"]


async def test_a_test_of_a_stopped_branch_is_refused(tmp_path: Path) -> None:
    session = await Stage(tmp_path).opened()

    with pytest.raises(RequestRejected) as refused:
        await session.toolbox.submit_test(
            TestRequest(TestKind.TEST, Purpose.EXPLORE, "b02", session.root, delta=_CHANGE)
        )

    assert refused.value.code == BRANCH_STOPPED


async def test_every_test_is_withdrawn_whatever_branch_it_is_of(tmp_path: Path) -> None:
    session = await Stage(tmp_path).opened()
    session.toolbox.executor.gate.clear()
    held = await session.toolbox.submit_test(
        TestRequest(TestKind.TEST, Purpose.EXPLORE, "b01", session.root, delta=_CHANGE)
    )
    session.sessions.hanging = frozenset({"b01"})

    report = await session.stopper.stop_everything(SessionStopReason.JUDGE_STOPPED, NOW + SHORT_GRACE)

    assert STOP_OVERRAN in report.failures
    assert held in session.toolbox.withdrawn and "b01" not in session.toolbox.cancelled


async def test_a_test_the_toolbox_still_holds_makes_the_report_unclean(tmp_path: Path) -> None:
    session = await Stage(tmp_path, toolbox_type=KeepingToolbox).opened()
    session.toolbox.executor.gate.clear()
    await session.toolbox.submit_test(TestRequest(TestKind.TEST, Purpose.EXPLORE, "b01", session.root, delta=_CHANGE))

    report = await session.stopper.stop_everything(SessionStopReason.JUDGE_STOPPED, HARD_STOP)

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
