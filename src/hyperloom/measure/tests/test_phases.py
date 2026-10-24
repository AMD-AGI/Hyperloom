# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The phase clock: phases tile the span they are stamped on, the host puts the runner's between its own, and what is
left over is said, never dropped."""

from __future__ import annotations

import pytest

from hyperloom.contracts.measurement import PhaseTiming
from hyperloom.measure.phases import (
    ACQUIRED,
    LANDED,
    RUNNER_DONE,
    HostTimeline,
    PhaseRecorder,
    account,
    manifest_document,
)


class Clock:
    """A clock the test moves."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


def _named(phases: tuple[PhaseTiming, ...]) -> list[tuple[str, float]]:
    return [(p.name, round(p.seconds, 6)) for p in phases]


# -- the recorder -------------------------------------------------------------------------------------------------


def test_each_phase_ends_where_the_next_begins_so_they_tile_the_span() -> None:
    clock = Clock()
    recorder = PhaseRecorder(clock)
    recorder.begin("preflight")
    clock.advance(2.0)
    recorder.begin("env_rebuild")
    clock.advance(5.5)
    recorder.begin("server_boot", point=0)
    clock.advance(80.0)

    phases = recorder.finish()

    assert _named(phases) == [("preflight", 2.0), ("env_rebuild", 5.5), ("server_boot", 80.0)]
    assert [p.started_at for p in phases] == [1000.0, 1002.0, 1007.5]
    assert sum(p.seconds for p in phases) == clock.now - 1000.0
    assert [p.point for p in phases] == [None, None, 0]


def test_a_phase_that_took_no_time_is_not_recorded() -> None:
    clock = Clock()
    recorder = PhaseRecorder(clock)
    recorder.begin("a")
    recorder.begin("b")
    clock.advance(1.0)

    assert _named(recorder.finish()) == [("b", 1.0)]


def test_a_mark_that_lies_before_the_open_phase_is_moved_up_so_no_phase_is_negative_and_none_loses_time() -> None:
    recorder = PhaseRecorder(Clock())
    recorder.begin("a", at=100.0)
    recorder.begin("b", at=90.0)  # a file time ahead of the phase it follows
    recorder.begin("c", at=110.0)

    phases = recorder.finish(at=130.0)

    assert _named(phases) == [("b", 10.0), ("c", 20.0)]
    assert [p.started_at for p in phases] == [100.0, 110.0]
    assert sum(p.seconds for p in phases) == 130.0 - 100.0


def test_the_open_phase_is_not_a_result_until_it_is_finished() -> None:
    clock = Clock()
    recorder = PhaseRecorder(clock)
    recorder.begin("a")
    clock.advance(1.0)
    recorder.begin("b")
    clock.advance(1.0)

    assert _named(recorder.phases) == [("a", 1.0)]
    assert _named(recorder.finish()) == [("a", 1.0), ("b", 1.0)]


def test_the_manifest_states_lengths_on_the_runners_clock_and_no_card_time() -> None:
    clock = Clock(50.0)
    recorder = PhaseRecorder(clock)
    recorder.begin("warmup", point=1)
    clock.advance(3.0)

    assert manifest_document(recorder.finish()) == [
        {"name": "warmup", "started_unix": 50.0, "seconds": 3.0, "point": 1}
    ]


# -- the host's side ----------------------------------------------------------------------------------------------


def _timeline(clock: Clock, **marks: float) -> HostTimeline:
    """A timeline whose marks fall ``marks`` seconds after it began."""
    timeline = HostTimeline(clock)
    for name, at in marks.items():
        clock.now = timeline.began + at
        timeline.mark(name)
    return timeline


RUNNER = (
    PhaseTiming("bootstrap", 9000.0, 30.0),
    PhaseTiming("sheet_wait", 9030.0, 4.0),
    PhaseTiming("server_boot", 9034.0, 82.0, point=0),
    PhaseTiming("measured", 9116.0, 20.0, point=0),
)


def test_the_hosts_phases_surround_the_runners_which_are_placed_by_their_lengths_not_their_clock() -> None:
    clock = Clock()
    timeline = _timeline(clock, **{ACQUIRED: 10.0, RUNNER_DONE: 160.0, LANDED: 163.0})
    clock.now = timeline.began + 164.0

    phases = timeline.compose(RUNNER)

    assert _named(phases) == [
        ("sandbox_start", 10.0),
        ("bootstrap", 30.0),
        ("sheet_wait", 4.0),
        ("server_boot", 82.0),
        ("measured", 20.0),
        ("runner_handoff", 14.0),  # the 150 s the host waited, less the runner's 136
        ("result_landing", 3.0),
        ("sandbox_release", 1.0),
    ]
    assert [p.started_at - 1000.0 for p in phases] == [0.0, 10.0, 40.0, 44.0, 126.0, 146.0, 160.0, 163.0]
    assert sum(p.seconds for p in phases) == 164.0, "the host's phases cover its whole span"
    assert [p.point for p in phases if p.name == "server_boot"] == [0]


@pytest.mark.parametrize(
    "started", [0.0, 5e10, 99_999_999_990.0], ids=["epoch", "year 3500", "the contract's last unix times"]
)
def test_the_runners_phases_are_placed_back_to_back_whatever_its_own_clock_says(started: float) -> None:
    clock = Clock()
    timeline = _timeline(clock, **{ACQUIRED: 10.0, RUNNER_DONE: 160.0, LANDED: 163.0})
    clock.now = timeline.began + 164.0
    runner = (
        PhaseTiming("sheet_wait", started, 4.0),
        PhaseTiming("server_boot", started + 4.0, 82.0),
        PhaseTiming("measured", 99_999_999_999.0, 20.0),
    )

    phases = timeline.compose(runner)

    assert [p.started_at - 1000.0 for p in phases][:5] == [0.0, 10.0, 14.0, 96.0, 116.0]
    assert sum(p.seconds for p in phases) == 164.0


def test_a_test_whose_runner_stated_no_phases_is_one_wait_for_the_runner() -> None:
    clock = Clock()
    timeline = _timeline(clock, **{ACQUIRED: 5.0, RUNNER_DONE: 65.0, LANDED: 66.0})
    clock.now = timeline.began + 66.0

    assert _named(timeline.compose(())) == [
        ("sandbox_start", 5.0),
        ("runner_wait", 60.0),
        ("result_landing", 1.0),
    ]


def test_a_test_that_never_got_its_sandbox_is_all_sandbox_start() -> None:
    clock = Clock()
    timeline = HostTimeline(clock)
    clock.advance(42.0)

    assert _named(timeline.compose(RUNNER)) == [("sandbox_start", 42.0)], "a runner never ran: nothing of it is claimed"


def test_a_test_that_ended_in_the_wait_has_the_wait_it_saw_and_no_landing() -> None:
    clock = Clock()
    timeline = _timeline(clock, **{ACQUIRED: 2.0})
    clock.now = timeline.began + 3600.0

    assert _named(timeline.compose(())) == [("sandbox_start", 2.0), ("runner_wait", 3598.0)]


def test_a_runner_that_claims_more_than_the_host_waited_is_kept_whole_and_adds_no_handoff() -> None:
    clock = Clock()
    timeline = _timeline(clock, **{ACQUIRED: 1.0, RUNNER_DONE: 11.0, LANDED: 11.0})
    clock.now = timeline.began + 11.0

    phases = timeline.compose(RUNNER)

    assert [p.name for p in phases] == ["sandbox_start", "bootstrap", "sheet_wait", "server_boot", "measured"]
    assert sum(p.seconds for p in phases) > 11.0, "the excess shows as a negative unaccounted time, not as lost time"


# -- the accounting -----------------------------------------------------------------------------------------------


def test_the_cards_are_held_by_every_phase_and_the_uncovered_seconds_make_the_phases_add_up_to_the_wall_time() -> None:
    phases = (PhaseTiming("a", 0.0, 60.0), PhaseTiming("b", 60.0, 30.0))

    accounted = account(phases, 100.0, 4)

    assert [p.gpu_seconds for p in accounted.phases] == [240.0, 120.0]
    assert accounted.unaccounted_seconds == 10.0
    assert sum(p.seconds for p in accounted.phases) + accounted.unaccounted_seconds == 100.0


def test_phases_that_outrun_the_wall_time_leave_a_negative_remainder() -> None:
    accounted = account((PhaseTiming("a", 0.0, 120.0),), 100.0, 1)

    assert accounted.unaccounted_seconds == -20.0


def test_the_time_before_the_runner_and_the_cards_it_never_held_are_the_hosts_to_state() -> None:
    accounted = account((), 5.0, 8)

    assert accounted.phases == () and accounted.unaccounted_seconds == 5.0


@pytest.mark.parametrize("gpus", [0, 1, 8])
def test_a_phase_holds_its_seconds_times_the_cards_of_the_test(gpus: int) -> None:
    (phase,) = account((PhaseTiming("a", 0.0, 2.5),), 3.0, gpus).phases
    assert phase.gpu_seconds == 2.5 * gpus
