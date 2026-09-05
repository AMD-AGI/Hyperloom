# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a budget does when it runs out, played on a clock nobody has to wait for.

The failure these pin is not that a timeout was too long. It is that a timeout
of exactly zero was read one layer down as no timeout at all, so the sessions
that most needed stopping were the only ones that never did. The shape of the
check is therefore always comparative: whatever a session with time left gets,
a session with none must get *less* of, at every layer the number crosses.

The clock is virtual, so an hours-long ceiling is asserted in milliseconds and
the ordering between deadlines -- which of them fires first -- stays the thing
under test rather than something shortened until it fits.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hyperloom.common.deadline import Deadline, seconds_until
from hyperloom.orchestrator.rehearsal import VirtualClock, installed_clock
from hyperloom.orchestrator.specialists import subprocess_ as subprocess_module
from hyperloom.orchestrator.specialists.subprocess_ import (
    UNBOUNDED_REAP_CAP_SEC,
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
)
from hyperloom.orchestrator.state.shared_state import SharedState

#: Turn budget the old ceiling multiplied into a duration. Kept here because
#: the number it produced -- days -- is the whole reason these tests exist.
_PRODUCTION_MAX_TURNS = 1000

#: The per-turn ceiling it multiplied by, in seconds.
_LEGACY_PER_TURN_SEC = 600.0


def _at(unix: float) -> datetime:
    """Return ``unix`` as an aware UTC datetime, for budget math on fixed times."""
    return datetime.fromtimestamp(unix, tz=timezone.utc)


class _NeverExits:
    """A process that has to be killed, because it will not stop on its own."""

    def __init__(self) -> None:
        self.pid = -1
        self.returncode: int | None = None
        self.killed = False

    def poll(self) -> int | None:
        """Return ``None`` until something kills this process."""
        return self.returncode


def _dispatcher() -> SpecialistSubprocessDispatcher:
    """A dispatcher whose only live stop is the deadline.

    The staleness window is pushed past every deadline under test so that the
    reap the assertions read is the one the deadline caused, not one the
    heartbeat check got to first.
    """
    return SpecialistSubprocessDispatcher(
        config=SpecialistSubprocessConfig(heartbeat_stale_seconds=10 * UNBOUNDED_REAP_CAP_SEC),
    )


async def _reap(deadline: Deadline | None, clock: VirtualClock, workspace: Path) -> dict:
    """Run the reap loop against a process that never exits.

    Args:
        deadline: The bound handed to the reaper.
        clock: The virtual clock the loop advances.
        workspace: A directory for the loop's process log lookups.

    Returns:
        dict: The reaper's outcome.
    """
    dispatcher = _dispatcher()
    proc = _NeverExits()

    def _kill(target: object) -> None:
        target.returncode = -9  # type: ignore[attr-defined]
        target.killed = True  # type: ignore[attr-defined]

    dispatcher._kill = _kill  # type: ignore[assignment,method-assign]
    return await dispatcher._reap_loop(
        proc=proc,
        workspace=workspace,
        done_files=(),
        heartbeat_file=workspace / "heartbeat.json",
        deadline=deadline,
        started=clock.monotonic(),
        task_id="t",
    )


class TestASpentBudgetTightens:
    """An exhausted budget must shorten the ceiling below, never remove it."""

    def test_a_spent_session_yields_an_expired_deadline_not_an_absent_one(self):
        spent = SharedState(session_id="spent", max_minutes=60)
        spent.begin_leg()
        spent.elapsed_charged_sec = 120 * 60.0

        deadline = spent.session_deadline()

        assert deadline is not None, "an exhausted budget must still bound the work"
        assert deadline.expired()

    def test_a_spent_session_gives_a_shorter_ceiling_than_a_fresh_one(self):
        fresh = SharedState(session_id="fresh", max_minutes=240)
        fresh.begin_leg()
        spent = SharedState(session_id="spent", max_minutes=240)
        spent.begin_leg()
        spent.elapsed_charged_sec = 240 * 60.0

        # The dispatcher combines the task-shape budget with the session bound
        # by taking whichever stops sooner.
        shape = 60 * 60.0
        fresh_at = Deadline.after(shape).tightened_to(fresh.session_deadline())
        spent_at = Deadline.after(shape).tightened_to(spent.session_deadline())

        assert spent_at.remaining() < fresh_at.remaining()
        assert spent_at.expired()

    @pytest.mark.asyncio
    async def test_the_reaper_kills_a_spent_budget_at_once(self, tmp_path):
        clock = VirtualClock()
        with installed_clock(clock, subprocess_module):
            outcome = await _reap(Deadline.after(0.0), clock, tmp_path)

        assert outcome["timed_out"] is True
        assert clock.elapsed < 60.0, "a spent budget must not buy another window"

    @pytest.mark.asyncio
    async def test_the_reaper_kills_a_live_budget_at_its_instant(self, tmp_path):
        clock = VirtualClock()
        with installed_clock(clock, subprocess_module):
            outcome = await _reap(Deadline.after(1_800.0), clock, tmp_path)

        assert outcome["timed_out"] is True
        assert 1_800.0 <= clock.elapsed <= 1_800.0 + 30.0

    @pytest.mark.asyncio
    async def test_an_unbounded_dispatch_still_stops_at_a_finite_cap(self, tmp_path):
        clock = VirtualClock()
        with installed_clock(clock, subprocess_module):
            outcome = await _reap(None, clock, tmp_path)

        assert outcome["timed_out"] is True
        assert clock.elapsed <= UNBOUNDED_REAP_CAP_SEC + 30.0
        # The ceiling that used to stand in for a missing budget was turns
        # multiplied by a per-turn allowance: over a hundred and sixty hours.
        assert clock.elapsed < _PRODUCTION_MAX_TURNS * _LEGACY_PER_TURN_SEC / 10.0


class TestUnboundedIsNotUnlimited:
    """``None`` means nobody set a bound, and still buys a finite wait."""

    def test_an_unbounded_wait_is_capped_at_what_the_caller_named(self):
        assert seconds_until(None, unbounded_cap=30.0) == 30.0

    def test_an_expired_deadline_waits_no_time_at_all(self):
        assert seconds_until(Deadline.after(-600.0), unbounded_cap=30.0) == 0.0

    def test_an_expired_deadline_is_not_confused_with_an_absent_one(self):
        expired = seconds_until(Deadline.after(-600.0), unbounded_cap=30.0)
        absent = seconds_until(None, unbounded_cap=30.0)
        assert expired < absent

    def test_an_infinite_cap_is_refused(self):
        with pytest.raises(ValueError):
            seconds_until(None, unbounded_cap=float("inf"))

    def test_remaining_keeps_its_sign_so_overrun_stays_visible(self):
        assert Deadline.after(-600.0).remaining() == pytest.approx(-600.0, abs=1.0)


class TestCombiningBoundsOnlyEverTightens:
    """No spelling here grants time; that is what survives a layer crossing."""

    def test_the_earlier_bound_wins(self):
        soon, later = Deadline.after(10.0), Deadline.after(1_000.0)
        assert soon.tightened_to(later) is soon
        assert later.tightened_to(soon) is soon

    def test_an_unbounded_partner_leaves_the_bound_standing(self):
        bound = Deadline.after(10.0)
        assert bound.tightened_to(None) is bound

    def test_an_expired_bound_beats_a_live_one(self):
        expired = Deadline.after(-1.0)
        assert Deadline.after(3_600.0).tightened_to(expired) is expired


class TestTheBudgetSurvivesALegBoundary:
    """Elapsed is summed forward, so a new leg cannot restart the clock."""

    def test_a_second_leg_inherits_what_the_first_spent(self):
        state = SharedState(session_id="s", max_minutes=120)
        state.begin_leg(now_unix=1_000.0)
        state.charge_elapsed(now_unix=4_600.0)  # one hour of leg one

        state.begin_leg(now_unix=50_000.0)  # a long gap, then leg two

        assert state.elapsed_charged_sec == pytest.approx(3_600.0)
        deadline = state.session_deadline(now=_at(50_000.0))
        assert deadline is not None
        assert deadline.remaining() == pytest.approx(3_600.0, abs=2.0)

    def test_a_killed_leg_is_charged_exactly_like_a_stopped_one(self):
        # Neither state records how its leg ended, and no code path may ask:
        # the branch that used to ask read a kill as a clean stop and handed
        # back a full budget.
        killed = SharedState(session_id="killed", max_minutes=120)
        stopped = SharedState(session_id="stopped", max_minutes=120)
        for state in (killed, stopped):
            state.begin_leg(now_unix=1_000.0)
            state.charge_elapsed(now_unix=4_600.0)
        stopped.stop_reason = "time_exhausted"
        stopped.stop_ts = "2026-01-01T00:00:00+00:00"

        for state in (killed, stopped):
            state.begin_leg(now_unix=50_000.0)

        assert killed.elapsed_charged_sec == stopped.elapsed_charged_sec
        killed_left = killed.session_deadline(now=_at(50_000.0))
        stopped_left = stopped.session_deadline(now=_at(50_000.0))
        assert killed_left is not None and stopped_left is not None
        assert killed_left.remaining() == pytest.approx(stopped_left.remaining(), abs=2.0)

    def test_many_legs_cannot_outrun_the_budget(self):
        state = SharedState(session_id="s", max_minutes=60)
        now = 1_000.0
        for _ in range(10):
            state.begin_leg(now_unix=now)
            now += 600.0  # each leg runs ten minutes
            state.charge_elapsed(now_unix=now)
            now += 5_000.0  # and a long gap before the next

        assert state.remaining_minutes(now=_at(now)) == 0.0
        deadline = state.session_deadline(now=_at(now))
        assert deadline is not None and deadline.expired()

    def test_only_an_operator_extension_lengthens_a_spent_session(self):
        state = SharedState(session_id="s", max_minutes=60)
        state.begin_leg(now_unix=1_000.0)
        state.charge_elapsed(now_unix=4_600.0)
        assert state.session_deadline(now=_at(4_600.0)).expired()  # type: ignore[union-attr]

        state.extend_budget_minutes(30.0, reason="operator asked for more")

        deadline = state.session_deadline(now=_at(4_600.0))
        assert deadline is not None and not deadline.expired()
        assert state.budget_extensions[-1]["minutes"] == 30.0
        assert state.budget_extensions[-1]["reason"] == "operator asked for more"

    def test_an_extension_does_not_refund_elapsed_time(self):
        state = SharedState(session_id="s", max_minutes=60)
        state.begin_leg(now_unix=1_000.0)
        state.charge_elapsed(now_unix=4_600.0)
        before = state.elapsed_charged_sec

        state.extend_budget_minutes(30.0)

        assert state.elapsed_charged_sec == before


class TestOffloadedWorkCannotHoldTheTick:
    """Blocking work returns the loop at a known instant, whatever it is doing."""

    @pytest.mark.asyncio
    async def test_work_that_outlasts_its_deadline_returns_the_default(self):
        from hyperloom.orchestrator.loop.offload import offload

        released = threading.Event()

        def _blocks() -> str:
            # Bounded on its own, the way offloaded work is required to be: the
            # awaiter is released at its deadline either way, and the thread
            # finishing on its own is what stops them accumulating. It outlasts
            # the deadline because nothing releases it until the wait is over.
            released.wait(timeout=30.0)
            return "finished"

        result = await offload(
            _blocks,
            deadline=Deadline.after(0.05),
            label="a call that will not return",
            default="moved on",
        )
        released.set()

        assert result == "moved on"

    @pytest.mark.asyncio
    async def test_work_with_a_spent_deadline_is_never_started(self):
        from hyperloom.orchestrator.loop.offload import offload

        started: list[bool] = []

        def _work() -> str:
            started.append(True)
            return "ran"

        result = await offload(
            _work,
            deadline=Deadline.after(-1.0),
            label="already too late",
            default="skipped",
        )

        assert result == "skipped"
        assert started == []

    @pytest.mark.asyncio
    async def test_work_that_raises_does_not_end_the_tick(self):
        from hyperloom.orchestrator.loop.offload import offload

        def _boom() -> str:
            raise RuntimeError("the network went away")

        result = await offload(_boom, deadline=None, label="boom", default="degraded")

        assert result == "degraded"


class TestAStopIsRecordedBeforeItIsDispatched:
    """A signal must land while the loop is busy, not once it is free."""

    def test_the_drain_publishes_to_both_audiences(self):
        import os
        import signal as signal_module

        from hyperloom.orchestrator.loop.signals import SignalDrain

        async def _exercise() -> tuple[bool, bool]:
            stop = asyncio.Event()
            drain = SignalDrain(loop=asyncio.get_running_loop(), stop_event=stop)
            if not drain.armed:  # pragma: no cover — no handlers off the main thread
                pytest.skip("signal handlers are unavailable here")
            try:
                os.kill(os.getpid(), signal_module.SIGTERM)
                # The threading event is set by the reading thread; the asyncio
                # one only once the loop runs, which is the ordering under test.
                await asyncio.wait_for(stop.wait(), timeout=5.0)
                return drain.requested.is_set(), stop.is_set()
            finally:
                drain.close()

        requested, stopped = asyncio.run(_exercise())

        assert requested is True
        assert stopped is True

    def test_closing_restores_the_previous_handler(self):
        import signal as signal_module

        from hyperloom.orchestrator.loop.signals import SignalDrain

        async def _exercise() -> None:
            before = signal_module.getsignal(signal_module.SIGTERM)
            drain = SignalDrain(loop=asyncio.get_running_loop(), stop_event=asyncio.Event())
            drain.close()
            assert signal_module.getsignal(signal_module.SIGTERM) is before
            # Teardown of a stop mechanism must never be the thing that raises.
            drain.close()

        asyncio.run(_exercise())


class TestTheChargeAnchorIsNotPersisted:
    """A state on disk has no leg running, so it charges nothing by sitting."""

    def test_a_reload_does_not_charge_the_time_the_state_sat_on_disk(self, tmp_path):
        state = SharedState(session_id="s", max_minutes=180)
        state.begin_leg(now_unix=1_000.0)
        state.charge_elapsed(now_unix=4_600.0)
        state.save(tmp_path)
        spent_at_save = state.elapsed_charged_sec

        reloaded = SharedState.load_or_init(tmp_path)
        # A tool that loads and saves without running a leg must not bill one.
        reloaded.save(tmp_path)

        assert reloaded.leg_anchor_unix == 0.0
        assert SharedState.load_or_init(tmp_path).elapsed_charged_sec == pytest.approx(spent_at_save, abs=2.0)

    def test_a_save_during_a_leg_records_what_the_leg_has_spent(self, tmp_path):
        state = SharedState(session_id="s", max_minutes=180)
        state.begin_leg(now_unix=time.time() - 600.0)

        state.save(tmp_path)

        assert SharedState.load_or_init(tmp_path).elapsed_charged_sec == pytest.approx(600.0, abs=5.0)

    def test_a_state_written_before_the_elapsed_sum_carries_its_spend_over(self):
        # The old shape recorded spend only as a stamped deadline; what it had
        # used was the time since start_ts, and that must survive the upgrade.
        started = time.time() - 2 * 3600.0
        state = SharedState.from_dict(
            {
                "schema_version": 6,
                "session_id": "s",
                "max_minutes": 180,
                "start_ts": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
                "deadline_unix": started + 180 * 60.0,
            }
        )

        assert state.elapsed_charged_sec == pytest.approx(2 * 3600.0, abs=5.0)
        assert state.remaining_minutes() == pytest.approx(60.0, abs=1.0)
