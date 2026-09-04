# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the SBD V6 ``enablement`` event.

The breakdown used to publish only the patches of rounds that were kept,
flattened and detached from the round that applied them. Most of these tests pin
the part that was missing: the rounds that failed, in order, each with the gap
its boot reached and the archived files that prove it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.assembler import enablement_event_parts
from hyperloom.inference_optimizer.breakdown.recorder.enablement_event import (
    PRODUCER,
    STATUS_ADVANCED,
    STATUS_KEPT,
    STATUS_REVERTED,
    assemble_enablement_ext,
    enablement_event_id,
    make_enablement_recorder,
)
from hyperloom.inference_optimizer.breakdown.recorder.event_finalize import finalize_events
from hyperloom.inference_optimizer.breakdown.recorder.event_sink import make_sink
from hyperloom.inference_optimizer.breakdown.recorder.event_timeline import EVENT_STATUS_INTERRUPTED
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope

EVENT_ID = "prelude:0:enablement"


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so no call below takes a path."""
    with session_scope(tmp_path):
        yield tmp_path


def _events(session_dir: Path) -> list[dict[str, Any]]:
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "enablement"]


def _ext(session_dir: Path) -> dict[str, Any]:
    return _events(session_dir)[0]["ext"]


def _recorder(*, phase: str = "prelude", macro_cycle: int = 0):
    """Build a recorder the way the lane gets one on every tick."""
    recorder = make_enablement_recorder(
        make_sink(enablement_event_id(phase, macro_cycle), producer=PRODUCER),
    )
    assert recorder is not None
    return recorder


def _result(status: str, **overrides: Any) -> dict[str, Any]:
    """One ``integrate_patch`` verdict for an enablement round."""
    result: dict[str, Any] = {
        "enablement": True,
        "status": status,
        "patches_applied": ["/s/runs/specialist/t-1/patches/001_fix.patch"],
        "artifacts_applied": [
            {"target": "/sgl-workspace/sglang/python/sglang/srt/models/glm5.py", "source": "/s/runs/a", "backup": "/b"}
        ],
        "setup_commands_applied": ["pip install aiter==0.2"],
        "after_signature": {
            "kind": "missing_kernel",
            "offending_file": "fused_moe.py",
            "raw_excerpt": "RuntimeError: no matching kernel",
        },
        "enablement_effective_config": {
            "extra_server_args": "--disable-cuda-graph",
            "extra_envs": {"HL_X": "1"},
            "args_mode": "append",
        },
    }
    result.update(overrides)
    return result


def _files() -> list[dict[str, str]]:
    """What ``snapshot_round`` reports for a round that reached a bench."""
    return [
        {"path": "reports/enablement/t-1/patches/001_fix.patch", "role": "patch"},
        {"path": "reports/enablement/t-1/server.log", "role": "server_log"},
    ]


def test_the_event_is_on_the_timeline_before_the_effort_finishes(tmp_path: Path) -> None:
    """A stuck baseline is exactly the run an operator watches live."""
    _recorder().begin(mode="all", origin="")

    events = _events(tmp_path)
    assert len(events) == 1
    assert events[0]["status"] == "running"
    assert events[0]["id"] == EVENT_ID
    assert events[0]["start_time"]


def test_opening_twice_keeps_one_timeline_entry(tmp_path: Path) -> None:
    """The lane opens on the tick of every round rather than tracking the first."""
    _recorder().begin(mode="all")
    _recorder().begin(mode="all")

    assert len(_events(tmp_path)) == 1


def test_a_second_recorder_writes_into_the_same_round(tmp_path: Path) -> None:
    """Four recorders, one round: the dispatch and settle writes straddle a resume.

    The window has to come from the two writes rather than from the verdict --
    a round runs for minutes, and a start taken at the end is an instant.
    """
    _recorder().begin(mode="all")
    _recorder().round_dispatched(task_id="t-1", failure_kind="missing_kernel")
    _recorder().round_settled(task_id="t-1", result=_result(STATUS_KEPT), files=_files())
    _recorder().finish(succeeded=True)

    rounds = _ext(tmp_path)["rounds"]
    assert len(rounds) == 1
    assert rounds[0]["started_at"] <= rounds[0]["ended_at"]


def test_a_round_in_flight_is_already_readable(tmp_path: Path) -> None:
    """Held back until it settles, a stuck round would be the invisible one."""
    recorder = _recorder()
    recorder.begin(mode="all")
    recorder.round_dispatched(task_id="t-1", failure_kind="missing_kernel")

    ext, status = assemble_enablement_ext(enablement_event_parts(), event=EVENT_ID)
    assert status == "running"
    assert ext["rounds"][0]["failure_kind"] == "missing_kernel"
    assert "status" not in ext["rounds"][0]


def test_reverted_rounds_are_kept(tmp_path: Path) -> None:
    """The axis exists for these: a kept-only view drops the whole failure trail."""
    recorder = _recorder()
    recorder.begin(mode="all")
    for index, status in enumerate((STATUS_REVERTED, STATUS_ADVANCED, STATUS_KEPT), start=1):
        recorder.round_dispatched(task_id=f"t-{index}")
        recorder.round_settled(task_id=f"t-{index}", result=_result(status))
    recorder.finish(succeeded=True)

    rounds = _ext(tmp_path)["rounds"]
    assert [row["status"] for row in rounds] == [STATUS_REVERTED, STATUS_ADVANCED, STATUS_KEPT]
    assert [row["specialist_task_id"] for row in rounds] == ["t-1", "t-2", "t-3"]


def test_a_round_names_its_archived_files_and_not_its_workspace_paths(tmp_path: Path) -> None:
    """A runs/ path is what the archive drops, so publishing one 404s downstream."""
    recorder = _recorder()
    recorder.begin(mode="all")
    recorder.round_dispatched(task_id="t-1")
    recorder.round_settled(task_id="t-1", result=_result(STATUS_KEPT), files=_files())
    recorder.finish(succeeded=True)

    row = _ext(tmp_path)["rounds"][0]
    assert row["patches"] == ["001_fix.patch"]
    assert [entry["path"] for entry in row["files"]] == [
        "reports/enablement/t-1/patches/001_fix.patch",
        "reports/enablement/t-1/server.log",
    ]
    assert not any(str(entry["path"]).startswith("runs/") for entry in row["files"])


def test_an_artifact_keeps_its_target_and_drops_its_workspace_source(tmp_path: Path) -> None:
    """The target is where a replay installs it."""
    recorder = _recorder()
    recorder.begin(mode="all")
    recorder.round_dispatched(task_id="t-1")
    recorder.round_settled(task_id="t-1", result=_result(STATUS_KEPT))
    recorder.finish(succeeded=True)

    assert _ext(tmp_path)["rounds"][0]["artifacts"] == [
        {"target": "/sgl-workspace/sglang/python/sglang/srt/models/glm5.py"}
    ]


def test_the_config_delta_is_recorded(tmp_path: Path) -> None:
    """Base plus delta is the persisted contract."""
    recorder = _recorder()
    recorder.begin(mode="all")
    recorder.round_dispatched(task_id="t-1")
    recorder.round_settled(task_id="t-1", result=_result(STATUS_KEPT))
    recorder.finish(succeeded=True)

    assert _ext(tmp_path)["rounds"][0]["effective_config"] == {
        "extra_server_args": "--disable-cuda-graph",
        "extra_envs": {"HL_X": "1"},
        "remove_args": [],
        "args_mode": "append",
    }


def test_a_runnable_combo_closes_succeeded_with_no_failure(tmp_path: Path) -> None:
    recorder = _recorder()
    recorder.begin(mode="all", origin="eval")
    recorder.round_dispatched(task_id="t-1")
    recorder.round_settled(task_id="t-1", result=_result(STATUS_KEPT))
    recorder.finish(succeeded=True)

    event = _events(tmp_path)[0]
    assert event["status"] == "succeeded"
    assert event["ext"]["origin"] == "eval"
    assert event["ext"]["failure"] is None
    assert event["end_time"]
    # The envelope already carries the verdict, so ext does not repeat it.
    assert "succeeded" not in event["ext"]


def test_a_stalled_effort_closes_failed_on_the_last_gap_it_reached(tmp_path: Path) -> None:
    """A serial effort that advanced twice still failed; the progress is in the rounds."""
    recorder = _recorder()
    recorder.begin(mode="all")
    recorder.round_dispatched(task_id="t-1", failure_kind="unsupported_arch")
    recorder.round_settled(task_id="t-1", result=_result(STATUS_ADVANCED))
    recorder.round_dispatched(task_id="t-2", failure_kind="undefined_symbol")
    recorder.round_settled(task_id="t-2", result=_result(STATUS_REVERTED))
    recorder.finish(succeeded=False, stop_reason="enablement_stalled")

    event = _events(tmp_path)[0]
    assert event["status"] == "failed"
    # The gap the last round was aimed at, and the deeper one its boot reached.
    assert event["ext"]["failure_kind"] == "undefined_symbol"
    assert event["ext"]["failure"]["error_class"] == "missing_kernel"
    assert event["ext"]["failure"]["offending_file"] == "fused_moe.py"
    assert event["ext"]["failure"]["stop_reason"] == "enablement_stalled"
    assert event["ext"]["attempts"] == 2


def test_an_admitted_lane_that_never_dispatched_closes_skipped(tmp_path: Path) -> None:
    recorder = _recorder()
    recorder.begin(mode="launch")
    recorder.finish(succeeded=False)

    event = _events(tmp_path)[0]
    assert event["status"] == "skipped"
    assert event["ext"]["attempts"] == 0
    assert event["ext"]["mode"] == "launch"


def test_a_killed_effort_is_closed_as_interrupted_not_guessed(tmp_path: Path) -> None:
    """Its rounds look complete; nothing judged them, so nothing may call it success."""
    recorder = _recorder()
    recorder.begin(mode="all")
    recorder.round_dispatched(task_id="t-1")
    recorder.round_settled(task_id="t-1", result=_result(STATUS_KEPT), files=_files())

    assert finalize_events(tmp_path) == [EVENT_ID]

    event = _events(tmp_path)[0]
    assert event["status"] == EVENT_STATUS_INTERRUPTED
    assert len(event["ext"]["rounds"]) == 1
    assert event["ext"]["rounds"][0]["status"] == STATUS_KEPT


def _lane(tmp_path: Path, **enablement_kw: Any):
    """A lane over a state object minimal enough to resolve an event id."""
    from types import SimpleNamespace

    from hyperloom.orchestrator.enablement.lane import EnablementLane
    from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound

    state = SimpleNamespace(
        phase="PRELUDE",
        macro_cycle=0,
        stop_reason="",
        enablement=EnablementRound(**enablement_kw),
    )
    return EnablementLane(SimpleNamespace(shared_state=state, session_dir=str(tmp_path)))


def test_the_lane_mints_the_event_id_once_and_reuses_it(tmp_path: Path) -> None:
    """A round settles on a later tick than it was dispatched on, and the phase moves."""
    lane = _lane(tmp_path)
    dispatched = lane._enablement_recorder(create=True)
    assert dispatched is not None
    assert dispatched.event_id == EVENT_ID
    assert lane.shared_state.enablement.timeline_event_id == EVENT_ID

    lane.shared_state.phase = "CLOSE"
    lane.shared_state.macro_cycle = 3
    settled = lane._enablement_recorder()
    assert settled is not None
    assert settled.event_id == EVENT_ID


def test_the_lane_declines_rather_than_starting_a_second_event(tmp_path: Path) -> None:
    assert _lane(tmp_path)._enablement_recorder() is None


def test_a_resumed_process_does_not_reclose_the_event(tmp_path: Path, monkeypatch) -> None:
    """``succeeded`` is a persisted terminal guard, so every later tick retries the close.

    The in-memory memo is gone with the old process, so what has to stop the
    second close is the spool read -- and a second close would move the end of
    an effort that finished before the resume even started.
    """
    recorder = _recorder()
    recorder.begin(mode="all")
    recorder.round_dispatched(task_id="t-1")
    recorder.round_settled(task_id="t-1", result=_result(STATUS_KEPT))
    recorder.finish(succeeded=True)
    closed_at = _events(tmp_path)[0]["end_time"]

    # A resume is hours later, not inside the same ISO second.
    from hyperloom.inference_optimizer.breakdown.recorder import enablement_event

    monkeypatch.setattr(enablement_event, "_now_iso", lambda: "2099-01-01T00:00:00+00:00")
    lane = _lane(tmp_path, succeeded=True, timeline_event_id=EVENT_ID)
    lane.shared_state.phase = "CLOSE"
    lane.shared_state.macro_cycle = 3
    lane._maybe_close_enablement_event()

    events = _events(tmp_path)
    assert [event["id"] for event in events] == [EVENT_ID]
    assert events[0]["end_time"] == closed_at


def test_the_lane_closes_a_stalled_effort_and_leaves_other_stops_open(tmp_path: Path) -> None:
    """`interrupted` is for an effort nothing judged; a stall cap is a judgement."""
    _recorder().begin(mode="all")
    _recorder().round_dispatched(task_id="t-1")

    other = _lane(tmp_path, timeline_event_id=EVENT_ID)
    other.shared_state.stop_reason = "time_exhausted_during_prelude"
    other._maybe_close_enablement_event()
    assert _events(tmp_path)[0]["status"] == "running"

    stalled = _lane(tmp_path, timeline_event_id=EVENT_ID)
    stalled.shared_state.stop_reason = "enablement_stalled"
    stalled._maybe_close_enablement_event()
    assert _events(tmp_path)[0]["status"] == "failed"


def test_the_rows_stay_out_of_the_breakdown_envelope(tmp_path: Path) -> None:
    """They are the timeline's input; publishing them beside the event doubles them."""
    from hyperloom.inference_optimizer.breakdown.recorder.assembler import assemble_parts

    recorder = _recorder()
    recorder.begin(mode="all")
    recorder.round_dispatched(task_id="t-1")
    recorder.finish(succeeded=True)

    assembled = assemble_parts(tmp_path)
    assert "enablement_event" not in assembled
    assert "enablement_round" not in assembled
