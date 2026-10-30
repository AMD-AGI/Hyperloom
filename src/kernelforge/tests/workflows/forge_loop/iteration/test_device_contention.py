# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A device another process holds: the measurement it costs, the round it costs, and when it stops the run."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from kernelforge.utils import process_reaping
from kernelforge.utils.process_reaping import ReapReport
from kernelforge.workflows.forge_loop.iteration import fanout
from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.iteration.device_hazard import MAX_BLOCKED_ITERATIONS
from kernelforge.workflows.forge_loop.campaign.run_state import LoopStateStore
from kernelforge.workflows.forge_loop.roles.contracts import OrchestrationRunResult
from kernelforge.workflows.forge_loop.iteration import iteration_phases
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _counting_plan,
    _lane_agent_factory,
    _make_loop,
    _no_change_agent,
    _unused_supervisor,
)


def test_a_contended_workspace_skips_the_canonical_measurement(
    tmp_path,
    monkeypatch,
):
    """A leftover process the reaper could not clear is still holding the GPU.

    The session ended, the candidate is on disk, and the benchmark that decides
    KEEP is about to run against a device something else is using. That number
    would be this candidate's plus whatever it is sharing the device with, and
    the loop would act on it -- so the measurement is skipped and the candidate
    reverted, the same way a protected-integrity violation is handled.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    kernel = workspace / "kernel.py"
    validation_calls: list[int] = []

    async def contended_agent(kernel_path, _history, session_sink):
        Path(kernel_path).write_text("def kernel():\n    return 99\n")
        session_sink["plan"] = "a candidate nothing can measure"
        session_sink["end_reason"] = "candidate_submitted"
        session_sink["workspace_contention"] = "pid(s) [4321] survived SIGKILL; pid(s) [4321] hold a device node"
        return "contended candidate"

    async def unexpected_validation(*_args, **_kwargs):
        validation_calls.append(1)
        raise AssertionError("canonical validation must be skipped")

    monkeypatch.setattr(loop, "run_one_iteration", unexpected_validation)

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=contended_agent, supervisor_fn=_unused_supervisor)))

    assert validation_calls == []
    # Unmeasured means unkept: HEAD stays at the last state a benchmark ever
    # backed, rather than carrying a candidate nobody verified.
    assert kernel.read_text() == "def kernel():\n    return 1\n"
    assert "4321" in loop.results[0].workspace_contention
    assert loop.results[0].kept is False
    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]
    assert decisions == ["REVERT_CONTENDED"]


def _fake_device(monkeypatch, holders: dict[int, int]) -> dict[int, int]:
    """The device state the hazard re-check reads, without a process on it.

    ``holders`` maps pid to start time for whatever currently has a device node
    open; mutating it afterwards is how a test frees the device. Both of the
    reaper's readers are replaced, so nothing here depends on what is really
    running on the machine the suite is on.
    """

    def _read_proc(pid: int):
        if pid not in holders:
            return None
        return process_reaping._Proc(pid=pid, state="R", ppid=1, pgid=pid, starttime=holders[pid])

    monkeypatch.setattr(process_reaping, "_read_proc", _read_proc)
    monkeypatch.setattr(process_reaping, "_holds_device", lambda pid: pid in holders)
    return holders


async def _contended_agent(kernel_path, _history, session_sink):
    """A session that finished and whose workspace could not be cleared."""
    Path(kernel_path).write_text("def kernel():\n    return 99\n")
    session_sink["plan"] = "a candidate nothing can measure"
    session_sink["end_reason"] = "candidate_submitted"
    session_sink["workspace_contention"] = "pid(s) [4321] hold a device node"
    return "contended candidate"


def test_one_contended_lane_costs_the_whole_round_its_measurement(
    tmp_path,
    monkeypatch,
):
    """The device is not per-lane, so neither is a lane that could not clear it.

    Dropping the contended lane and measuring its healthy sibling is the wrong
    half of the response: what the lane left running is on the same GPU the
    canonical benchmark is about to use, so the number the round would take is
    the sibling's plus whatever is still benching. The round takes no
    measurement, and the sibling's candidate -- already paid for -- stays queued
    for an iteration that can measure it.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, lanes=2)
    rounds: list[int] = []
    monkeypatch.setattr(
        loop,
        "_run_orchestration",
        _counting_plan(loop, workspace, rounds, plans=("tune prefill", "tune decode")),
    )

    async def one_lane_left_something_running(lane_dir: Path) -> ReapReport:
        if lane_dir.name == "2":
            return ReapReport(
                directory=str(lane_dir),
                unkillable=(4321,),
                holding_device=(4321,),
            )
        return ReapReport(directory=str(lane_dir))

    monkeypatch.setattr(fanout, "_reap_lane_processes", one_lane_left_something_running)

    async def unexpected_validation(*_args, **_kwargs):
        raise AssertionError("a contended round must measure nothing")

    monkeypatch.setattr(loop, "run_one_iteration", unexpected_validation)

    asyncio.run(
        loop.run(
            runner_module.LoopAgents(
                agent_fn=_no_change_agent,
                agent_factory=_lane_agent_factory({"tune prefill": ("return 1", "return 2")}),
                orchestration_service=object(),
                supervisor_fn=_unused_supervisor,
            )
        )
    )

    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]

    assert decisions == ["REVERT_CONTENDED"]
    assert loop.results[0].kept is False
    assert loop.results[0].validation_passed is False
    assert "4321" in loop.results[0].workspace_contention
    # Refused, not discarded: the healthy lane's session was the expensive part
    # and its candidate is still worth measuring once the device is free.
    assert [item.plan for item in loop._lane_queue] == ["tune prefill"]


def test_a_leaked_probe_costs_the_round_its_measurement(tmp_path, monkeypatch):
    """A probe is a benchmark, and the device it holds is not the round's alone.

    A specialist killed by its session timeout mid-probe leaves a process on the
    same GPU the canonical measurement is about to use. The lanes queue behind
    it on the campaign sentinel; the canonical measurement takes no lock and
    would have run straight into it. So the analysis phase's teardown finding
    becomes a hazard exactly as a contended lane's does, and the round it
    belongs to measures nothing.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, lanes=2)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )

    class _ProbeLeakingService:
        """Planning that succeeded and left a probe running behind it."""

        async def run(self, _context, *, usage=None, lanes=1):
            return OrchestrationRunResult(
                dispatch_plan=None,
                optimization_plans=("tune prefill", "tune decode"),
                structured_output_diagnostics={
                    "probe_device_hazard": {
                        "describe": ("pid(s) [4321] survived SIGKILL; pid(s) [4321] hold a device node"),
                        "pids": [4321],
                    }
                },
            )

    async def unexpected_validation(*_args, **_kwargs):
        raise AssertionError("a round with a leaked probe must measure nothing")

    monkeypatch.setattr(loop, "run_one_iteration", unexpected_validation)

    asyncio.run(
        loop.run(
            runner_module.LoopAgents(
                agent_fn=_no_change_agent,
                agent_factory=_lane_agent_factory({"tune prefill": ("return 1", "return 2")}),
                orchestration_service=_ProbeLeakingService(),
                supervisor_fn=_unused_supervisor,
            )
        )
    )

    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]

    assert decisions == ["REVERT_CONTENDED"]
    assert loop.results[0].kept is False
    assert loop.results[0].validation_passed is False
    assert "4321" in loop.results[0].workspace_contention
    assert "probe round" in loop.results[0].workspace_contention
    # Bought and kept: the lane sessions were the expensive part, and their
    # candidates are still worth measuring once the device is free.
    assert [item.plan for item in loop._lane_queue] == ["tune prefill"]


def test_a_hazard_outlives_the_iteration_that_found_it(tmp_path, monkeypatch):
    """Nothing about an iteration ending makes a foreign process let go.

    The first iteration refuses because the reaper said so. The second has no
    reaper finding of its own -- it never ran a session -- and must still refuse
    while the device is held, then run as usual once it is free. Both
    directions, decided by the device rather than by how many iterations have
    passed.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=3)
    holders = _fake_device(monkeypatch, {4321: 99})
    monkeypatch.setattr(iteration_phases, "processes_under", lambda _dir: {4321})
    sessions: list[int] = []

    async def agent(kernel_path, history, session_sink):
        sessions.append(len(loop.results) + 1)
        if len(loop.results) == 0:
            return await _contended_agent(kernel_path, history, session_sink)
        session_sink["plan"] = "inspect only"
        return "No source change was needed."

    def free_the_device_after_the_second_refusal(_result):
        if len(loop.results) == 2:
            holders.clear()

    asyncio.run(
        loop.run(
            runner_module.LoopAgents(agent_fn=agent, supervisor_fn=_unused_supervisor),
            callbacks=runner_module.LoopCallbacks(on_iteration=free_the_device_after_the_second_refusal),
        )
    )

    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]

    assert decisions == ["REVERT_CONTENDED", "REVERT_CONTENDED", "NO_CHANGES"]
    # The refused iteration bought nothing: no session, so no plan and no diff.
    assert sessions == [1, 3]
    assert loop.results[1].workspace_contention
    assert loop.results[2].workspace_contention == ""
    assert loop.termination_reason != "device_contended"


def test_a_hazard_nothing_clears_stops_the_run_rather_than_spinning(
    tmp_path,
    monkeypatch,
):
    """A foreign process may hold the device for the rest of the campaign.

    Retrying until the budget runs out spends a whole run producing nothing
    while reporting nothing wrong, which is no better than the bad measurement
    the refusal exists to prevent. The run ends under a reason of its own.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=20)
    _fake_device(monkeypatch, {4321: 99})
    monkeypatch.setattr(iteration_phases, "processes_under", lambda _dir: {4321})

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_contended_agent, supervisor_fn=_unused_supervisor)))

    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]

    assert loop.termination_reason == "device_contended"
    assert set(decisions) == {"REVERT_CONTENDED"}
    # The iteration that found it plus the ones it refused after: the re-check
    # that exhausts the hazard runs before that iteration spends anything, so
    # the run stops there rather than filing one more unmeasured result.
    assert len(decisions) == MAX_BLOCKED_ITERATIONS - 1
