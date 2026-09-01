# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A device-contention finding, and what it takes to stop it blocking."""

from __future__ import annotations

from pathlib import Path

from kernelforge.llm import process_reaping
from kernelforge.loop.device_hazard import (
    MAX_BLOCKED_ITERATIONS,
    DeviceHazard,
    DeviceHazardLog,
)


class _FakeDevice:
    """The device state the re-check reads, without a real process on it.

    Holds pid -> start time for the processes that currently have a device node
    open. Nothing here is a process: the reaper's two readers are replaced, so a
    test decides what ``/proc`` would have said.
    """

    def __init__(self, monkeypatch, holders: dict[int, int]) -> None:
        self.holders = dict(holders)
        monkeypatch.setattr(process_reaping, "_read_proc", self._proc)
        monkeypatch.setattr(process_reaping, "_holds_device", self._holds)

    def _proc(self, pid: int):
        if pid not in self.holders:
            return None
        return process_reaping._Proc(
            pid=pid,
            state="R",
            ppid=1,
            pgid=pid,
            starttime=self.holders[pid],
        )

    def _holds(self, pid: int) -> bool:
        return pid in self.holders

    def release(self, pid: int) -> None:
        self.holders.pop(pid, None)


def test_a_hazard_blocks_until_the_device_is_free(tmp_path, monkeypatch):
    """Both directions, and neither of them is the clock.

    The hazard keeps refusing while the process it recorded still has the
    device, and stops the moment it does not -- not after a fixed number of
    iterations, and not because an iteration ended.
    """
    device = _FakeDevice(monkeypatch, {4321: 99})
    log = DeviceHazardLog(tmp_path)

    log.record(iteration=1, detail="pid(s) [4321] hold a device node", pids=[4321])

    assert log.recheck(2) is not None
    assert log.recheck(3) is not None

    device.release(4321)

    assert log.recheck(4) is None
    assert log.live is None


def test_a_recycled_pid_does_not_keep_a_hazard_alive(tmp_path, monkeypatch):
    """The holder is an identity, not a number.

    Pid ranges wrap in under an hour on a busy host. A hazard that re-checked on
    the bare pid would go on refusing measurements because something unrelated
    landed on the same number and happens to touch the GPU.
    """
    device = _FakeDevice(monkeypatch, {4321: 99})
    log = DeviceHazardLog(tmp_path)

    log.record(iteration=1, detail="pid(s) [4321] hold a device node", pids=[4321])
    device.holders[4321] = 4242

    assert log.recheck(2) is None


def test_a_hazard_with_nothing_on_the_device_clears_at_the_next_check(tmp_path, monkeypatch):
    """The reaper's "could not clear" and "is on the device" are not the same.

    A directory the reaper could not empty is reason enough for the iteration
    that found it to refuse. It is not reason for the next one to refuse, unless
    something it named is actually holding the device -- otherwise there is
    nothing for the re-check to wait on and the campaign would stall on a
    process that demonstrably is not in the way.
    """
    _FakeDevice(monkeypatch, {})
    log = DeviceHazardLog(tmp_path)

    hazard = log.record(iteration=1, detail="pid(s) [4321] survived", pids=[4321])

    assert hazard.holders == {}
    assert log.recheck(2) is None


def test_re_checking_twice_in_one_iteration_counts_as_one_refusal(tmp_path, monkeypatch):
    """The loop consults the hazard before and after its fan-out round.

    Counting the second look as a second refusal would end the campaign in half
    the iterations the cap names, which is the difference between "waited as
    long as we said" and "gave up early".
    """
    _FakeDevice(monkeypatch, {4321: 99})
    log = DeviceHazardLog(tmp_path)

    log.record(iteration=1, detail="held", pids=[4321])
    first = log.recheck(2)
    second = log.recheck(2)

    assert first is not None and second is not None
    assert first.blocked_iterations == second.blocked_iterations == 2


def test_a_hazard_that_never_clears_reaches_the_cap(tmp_path, monkeypatch):
    """A hazard nothing can clear must end somewhere, not spin.

    Nothing about a foreign process guarantees it ever exits, so the count is
    what makes the refusal terminal rather than permanent.
    """
    _FakeDevice(monkeypatch, {4321: 99})
    log = DeviceHazardLog(tmp_path)

    hazard = log.record(iteration=1, detail="held", pids=[4321])

    assert hazard.exhausted is False
    for iteration in range(2, MAX_BLOCKED_ITERATIONS + 1):
        hazard = log.recheck(iteration)
    assert hazard is not None
    assert hazard.exhausted is True


def test_a_hazard_survives_the_process_that_recorded_it(tmp_path, monkeypatch):
    """A campaign ending between iterations is the ordinary case.

    The reaper's finding was in memory and the resumed process would measure on
    the held device knowing nothing, which is exactly the number the refusal
    exists to prevent.
    """
    _FakeDevice(monkeypatch, {4321: 99})
    DeviceHazardLog(tmp_path).record(iteration=7, detail="pid(s) [4321] hold a device node", pids=[4321])

    resumed = DeviceHazardLog(tmp_path)

    assert resumed.live is not None
    assert resumed.live.found_iteration == 7
    assert resumed.live.holders == {4321: 99}
    assert resumed.recheck(8) is not None


def test_an_unreadable_record_is_not_a_hazard(tmp_path):
    """A corrupt file must not refuse every measurement for the rest of the run.

    There is nothing to wait on in it -- no pid, no identity -- so it can only
    block forever. Measuring is the recoverable mistake here.
    """
    path = tmp_path / "forge_experiments" / "device_hazard.json"
    path.parent.mkdir(parents=True)
    path.write_text("{ truncated", encoding="utf-8")

    assert DeviceHazardLog(tmp_path).live is None


def test_a_refusal_describes_what_it_is_waiting_on_without_repeating_itself():
    """The line is read once per refused iteration, so it cannot accumulate."""
    hazard = DeviceHazard(detail="lane 2: pid(s) [4321] survived SIGKILL")

    assert hazard.describe() == "lane 2: pid(s) [4321] survived SIGKILL"

    blocked = DeviceHazard(detail="lane 2: pid(s) [4321] survived SIGKILL", still_held_by=(4321,))

    assert blocked.describe().startswith("lane 2: pid(s) [4321] survived SIGKILL")
    assert "still hold a device node" in blocked.describe()


def test_a_cleared_hazard_leaves_nothing_behind(tmp_path, monkeypatch):
    """Otherwise the next process to start inherits a hazard that is over."""
    device = _FakeDevice(monkeypatch, {4321: 99})
    log = DeviceHazardLog(tmp_path)
    log.record(iteration=1, detail="held", pids=[4321])
    device.release(4321)

    log.recheck(2)

    assert not Path(log.path).exists()
    assert DeviceHazardLog(tmp_path).live is None
