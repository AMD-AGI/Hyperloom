# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The phase event, recorded by the real phase machine and dispatcher.

:mod:`test_sbd_v6_phase_timeline` drives the recorder directly; these tests drive the production call
sites -- ``record_phase_transition``, ``append_phase_history_event`` and the dispatcher's
``run_task_registered`` -- so a call site that stops recording fails here even while the recorder stays
correct. Coverage rests on those three being the only producers of their facts, which is pinned below:
``run_task_registered`` holds the tree's sole call to ``sub.run_task``, and that is what lets one hook
cover every action kind instead of a hand-grown whitelist.
"""

from __future__ import annotations

import asyncio
import re
import types
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder import phase_event
from hyperloom.inference_optimizer.breakdown.recorder.assembler import phase_event_parts
from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator
from hyperloom.orchestrator.phases import machine_state
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so the machine records into it."""
    with session_scope(tmp_path):
        yield tmp_path


def _ext(phase: str, macro_cycle: int = 0) -> dict[str, Any]:
    ext, _status = phase_event.assemble_phase_ext(
        phase_event_parts(),
        event=phase_event.phase_event_id(phase, macro_cycle),
    )
    return ext


def _events(session_dir: Path) -> list[dict[str, Any]]:
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "phase"]


def _state(tmp_path: Path) -> SharedState:
    """A SharedState the real phase machine can transition."""
    state = SharedState()
    state._session_dir = tmp_path
    state.phase = ""
    state.phase_history = []
    state.macro_cycle = 0
    state.tick = 0
    return state


def test_the_real_transition_opens_the_phase_it_entered(tmp_path):
    state = _state(tmp_path)
    machine_state.record_phase_transition(state, to_phase="PRELUDE", reason="session_start")

    ext = _ext("PRELUDE")
    assert ext["phase"] == "PRELUDE"
    assert ext["entries"] == 1
    assert ext["open"] is True
    assert ext["segments"][0]["entered_reason"] == "session_start"


def test_the_real_transition_closes_the_phase_it_left(tmp_path):
    state = _state(tmp_path)
    machine_state.record_phase_transition(state, to_phase="PRELUDE", reason="session_start")
    machine_state.record_phase_transition(state, to_phase="FRAMEWORK_AGENT", reason="baseline_ready")

    prelude = _ext("PRELUDE")
    assert prelude["open"] is False
    assert prelude["exit_reason"] == "baseline_ready"
    assert prelude["segments"][0]["to_phase"] == "FRAMEWORK_AGENT"
    assert prelude["duration_sec"] is not None
    assert _ext("FRAMEWORK_AGENT")["open"] is True


def test_the_real_transition_carries_its_evidence_both_ways(tmp_path):
    state = _state(tmp_path)
    machine_state.record_phase_transition(state, to_phase="FRAMEWORK_AGENT", reason="start")
    machine_state.record_phase_transition(
        state,
        to_phase="KERNEL_AGENT",
        reason="plateau_no_gain",
        evidence={"no_gain_streak": 3},
    )

    leaving = _ext("FRAMEWORK_AGENT")["segments"][0]
    assert leaving["exit_evidence"] == {"no_gain_streak": 3}
    entering = _ext("KERNEL_AGENT")["segments"][0]
    assert entering["entered_evidence"] == {"no_gain_streak": 3}
    assert entering["from_phase"] == "FRAMEWORK_AGENT"


def test_a_real_re_entry_is_a_second_segment_on_one_event(tmp_path):
    state = _state(tmp_path)
    machine_state.record_phase_transition(state, to_phase="FRAMEWORK_AGENT", reason="start")
    machine_state.record_phase_transition(state, to_phase="KERNEL_AGENT", reason="plateau_no_gain")
    machine_state.record_phase_transition(state, to_phase="FRAMEWORK_AGENT", reason="kernel_done")
    machine_state.record_phase_transition(state, to_phase="SWEEP", reason="plateau_no_gain")

    ext = _ext("FRAMEWORK_AGENT")
    assert ext["entries"] == 2
    assert [row["exit_reason"] for row in ext["segments"]] == ["plateau_no_gain", "plateau_no_gain"]
    assert [row["entered_reason"] for row in ext["segments"]] == ["start", "kernel_done"]
    # The re-entry's close reuses the sequence the first open took, so it overwrites rather than appends.
    published = [event["id"] for event in _events(tmp_path)]
    assert published.count("framework_agent:0:phase") == 1


def test_a_real_loopback_closes_the_cycle_it_ran_in(tmp_path):
    state = _state(tmp_path)
    machine_state.record_phase_transition(state, to_phase="EXPLORE", reason="start")
    state.macro_cycle = 1
    machine_state.record_phase_transition(state, to_phase="FRAMEWORK_AGENT", reason="cycle_reloop")

    ran = _ext("EXPLORE", 0)
    assert ran["open"] is False
    assert ran["exit_reason"] == "cycle_reloop"
    assert _ext("EXPLORE", 1)["segments"] == []
    assert _ext("FRAMEWORK_AGENT", 1)["open"] is True


def test_the_real_marker_lands_in_the_phase_it_was_raised_in(tmp_path):
    state = _state(tmp_path)
    machine_state.record_phase_transition(state, to_phase="FRAMEWORK_AGENT", reason="start")
    machine_state.append_phase_history_event(
        state,
        reason="plateau_proxy_provisional",
        evidence={"r09_provisional": True},
    )

    markers = _ext("FRAMEWORK_AGENT")["markers"]
    assert markers["count"] == 1
    assert markers["rows"][0]["reason"] == "plateau_proxy_provisional"
    assert markers["rows"][0]["evidence"] == {"r09_provisional": True}
    assert _ext("FRAMEWORK_AGENT")["entries"] == 1


def test_a_transition_still_happens_when_recording_cannot(tmp_path, monkeypatch):
    state = _state(tmp_path)

    def _boom(**_kwargs):
        raise RuntimeError("spool is gone")

    monkeypatch.setattr(phase_event, "record_entry", _boom)
    machine_state.record_phase_transition(state, to_phase="PRELUDE", reason="session_start")

    assert state.phase == "PRELUDE"
    assert len(state.phase_history) == 1


class _Sub:
    """Stands in for the runner, recording what it was asked to run."""

    def __init__(self, result: Any = None) -> None:
        self.result = result
        self.ran: list[str] = []

    async def run_task(self, task, *, prebound_lease=None, extra_context=None):
        self.ran.append(str(task.kind))
        return self.result


def _dispatcher(tmp_path: Path, state: SharedState, sub: _Sub) -> Any:
    """A DispatcherCollaborator with only what ``run_task_registered`` touches."""
    fake = types.SimpleNamespace(
        shared_state=state,
        sub=sub,
        session_dir=tmp_path,
        locks=None,
        gpu_specialist_pool=None,
        _inflight_actions={},
    )
    fake.run_task_registered = types.MethodType(DispatcherCollaborator.run_task_registered, fake)
    return fake


def _task(kind: str, task_id: str) -> Any:
    return types.SimpleNamespace(
        kind=kind,
        task_id=task_id,
        params={},
        requires_lanes=(),
        lease_ttl_sec=60,
    )


def test_the_real_runner_records_the_dispatch(tmp_path):
    state = _state(tmp_path)
    machine_state.record_phase_transition(state, to_phase="FRAMEWORK_AGENT", reason="start")
    state.tick = 11
    sub = _Sub(result="done")
    dispatcher = _dispatcher(tmp_path, state, sub)

    asyncio.run(dispatcher.run_task_registered(_task("baseline", "t-1")))

    assert sub.ran == ["baseline"]
    row = _ext("FRAMEWORK_AGENT")["actions"]["rows"][0]
    assert row["action"] == "baseline"
    assert row["task_id"] == "t-1"
    assert row["phase"] == "FRAMEWORK_AGENT"
    assert row["tick"] == 11
    # Still in flight as far as this event knows: the verdict is the reap's to record.
    assert row.get("status", "") == ""


def test_the_real_runner_records_every_kind_it_is_given(tmp_path):
    state = _state(tmp_path)
    machine_state.record_phase_transition(state, to_phase="CLOSE", reason="start")
    sub = _Sub(result="done")
    dispatcher = _dispatcher(tmp_path, state, sub)

    unaudited = ("report", "recover", "session_breakdown", "target_analysis")
    for index, kind in enumerate(unaudited):
        asyncio.run(dispatcher.run_task_registered(_task(kind, f"t-{index}")))

    assert _ext("CLOSE")["actions"]["kinds"] == sorted(unaudited)


def test_the_runner_is_the_only_path_an_action_takes(tmp_path):
    pattern = re.compile(r"\bsub\.run_task\(")
    root = Path(__file__).resolve().parents[3] / "hyperloom"
    found = [
        f"{path}:{lineno}"
        for path in sorted(root.rglob("*.py"))
        if "tests" not in path.parts
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert len(found) == 1, f"sub.run_task is called from more than one place: {found}"
    assert "loop/dispatcher.py" in found[0]


def test_a_dispatch_still_runs_when_recording_cannot(tmp_path, monkeypatch):
    state = _state(tmp_path)
    machine_state.record_phase_transition(state, to_phase="PRELUDE", reason="start")

    def _boom(**_kwargs):
        raise RuntimeError("spool is gone")

    monkeypatch.setattr(phase_event, "record_dispatch", _boom)
    sub = _Sub(result="done")
    dispatcher = _dispatcher(tmp_path, state, sub)

    assert asyncio.run(dispatcher.run_task_registered(_task("baseline", "t-1"))) == "done"
    assert sub.ran == ["baseline"]


def test_a_dispatch_that_raised_is_still_on_the_timeline(tmp_path):
    state = _state(tmp_path)
    machine_state.record_phase_transition(state, to_phase="KERNEL_AGENT", reason="start")

    class _Boom(_Sub):
        async def run_task(self, task, *, prebound_lease=None, extra_context=None):
            raise RuntimeError("executor died")

    dispatcher = _dispatcher(tmp_path, state, _Boom())
    with pytest.raises(RuntimeError):
        asyncio.run(dispatcher.run_task_registered(_task("kernel_opt", "t-5")))

    rows = _ext("KERNEL_AGENT")["actions"]["rows"]
    assert [row["task_id"] for row in rows] == ["t-5"]
    assert rows[0].get("status", "") == ""


def test_the_recorder_needs_no_knowledge_of_the_catalogue(tmp_path):
    state = _state(tmp_path)
    machine_state.record_phase_transition(state, to_phase="FRAMEWORK_AGENT", reason="start")
    kinds = sorted(ACTION_CATALOGUE)
    for index, kind in enumerate(kinds):
        phase_event.record_dispatch(
            action=kind,
            task_id=f"t-{index}",
            phase="FRAMEWORK_AGENT",
            macro_cycle=0,
        )

    assert _ext("FRAMEWORK_AGENT")["actions"]["kinds"] == kinds
