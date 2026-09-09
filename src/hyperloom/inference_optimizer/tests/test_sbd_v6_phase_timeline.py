# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the SBD V6 ``phase`` event.

This event replaces two derived top-level keys, and most of what these tests
pin is something one of them got wrong.

``phase_segments`` paired ``phase_history`` rows off two at a time to synthesize
each segment's exit and duration, so it could not describe the segment a session
ended in -- there was no successor row to close it with. ``phase_timeline``
attributed actions to phases by testing each action's timestamp against the phase
windows, because the writer was handed the phase and dropped it; an action that
outlived the phase which ordered it was charged to whichever phase inherited it.

Both facts are now recorded where they are produced, and both readings are
pinned below against the cases that used to be wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder import phase_event
from hyperloom.inference_optimizer.breakdown.recorder.assembler import phase_event_parts
from hyperloom.inference_optimizer.breakdown.recorder.event_finalize import finalize_events
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so no call below takes a path."""
    with session_scope(tmp_path):
        yield tmp_path


def _ext(phase: str, macro_cycle: int = 0) -> dict[str, Any]:
    """One phase event's assembled ``ext``, whether or not it has closed.

    An open event carries only its shell on disk -- the facts live in fragments
    until something closes it -- so a test reading a phase the run is still in
    assembles the same way finalize would.
    """
    ext, _status = phase_event.assemble_phase_ext(
        phase_event_parts(),
        event=phase_event.phase_event_id(phase, macro_cycle),
    )
    return ext


def _status(phase: str, macro_cycle: int = 0) -> str:
    _ext_, status = phase_event.assemble_phase_ext(
        phase_event_parts(),
        event=phase_event.phase_event_id(phase, macro_cycle),
    )
    return status


def _events(session_dir: Path) -> list[dict[str, Any]]:
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "phase"]


def _enter(phase: str, *, sequence: int, cycle: int = 0, at: float, from_phase: str = "", reason: str = "") -> None:
    phase_event.record_entry(
        phase=phase,
        macro_cycle=cycle,
        sequence=sequence,
        from_phase=from_phase,
        reason=reason,
        entered_at=f"2026-01-01T00:00:{int(at):02d}+00:00",
        entered_unix=at,
    )


def _exit(phase: str, *, cycle: int = 0, at: float, to_phase: str = "", reason: str = "") -> None:
    phase_event.record_exit(
        phase=phase,
        macro_cycle=cycle,
        to_phase=to_phase,
        reason=reason,
        exited_at=f"2026-01-01T00:00:{int(at):02d}+00:00",
        exited_unix=at,
    )


# --- the phase lands on the timeline ---------------------------------------


def test_entering_a_phase_puts_it_on_the_timeline(tmp_path):
    """A phase the run entered is an event, before anything closes it."""
    _enter("PRELUDE", sequence=1, at=10.0, reason="session_start")

    finalize_events(tmp_path)
    events = _events(tmp_path)
    assert [event["type"] for event in events] == ["phase"]
    assert events[0]["id"] == "prelude:0:phase"


def test_the_phase_the_run_stopped_in_has_no_exit(tmp_path):
    """The open segment is recorded as open, not as a zero-length one.

    ``phase_segments`` could not express this: it closed each segment with the
    next row's timestamp, so the last segment had no successor and was published
    with an empty exit and a null duration that read the same as a phase the run
    passed through instantly.
    """
    _enter("PRELUDE", sequence=1, at=10.0)
    _exit("PRELUDE", at=40.0, to_phase="FRAMEWORK_AGENT")
    _enter("FRAMEWORK_AGENT", sequence=2, at=40.0, from_phase="PRELUDE")

    prelude = _ext("PRELUDE")
    assert prelude["open"] is False
    assert prelude["duration_sec"] == 30.0

    live = _ext("FRAMEWORK_AGENT")
    assert live["open"] is True
    assert live["duration_sec"] is None
    assert live["exited_at"] == ""
    assert _status("FRAMEWORK_AGENT") == phase_event.STATUS_INTERRUPTED


def test_the_exit_carries_the_reason_the_run_left(tmp_path):
    """The exit reason is the leaving transition's, not the entering one's."""
    _enter("FRAMEWORK_AGENT", sequence=1, at=10.0, reason="baseline_ready")
    _exit("FRAMEWORK_AGENT", at=70.0, to_phase="KERNEL_AGENT", reason="plateau_no_gain")

    ext = _ext("FRAMEWORK_AGENT")
    assert ext["exit_reason"] == "plateau_no_gain"
    assert ext["segments"][0]["entered_reason"] == "baseline_ready"
    assert ext["segments"][0]["to_phase"] == "KERNEL_AGENT"


# --- a phase re-entered inside one cycle -----------------------------------


def test_a_re_entered_phase_sums_its_own_time_only(tmp_path):
    """Two entries, one event, and the gap between them belongs to neither.

    Measuring the event from its first entry to its last exit would charge this
    phase the 100 seconds the run spent in KERNEL_AGENT in between, which is how
    a phase budget guard comes to believe a phase overran.
    """
    _enter("FRAMEWORK_AGENT", sequence=1, at=10.0)
    _exit("FRAMEWORK_AGENT", at=30.0, to_phase="KERNEL_AGENT")
    _enter("KERNEL_AGENT", sequence=2, at=30.0, from_phase="FRAMEWORK_AGENT")
    _exit("KERNEL_AGENT", at=130.0, to_phase="FRAMEWORK_AGENT")
    _enter("FRAMEWORK_AGENT", sequence=3, at=130.0, from_phase="KERNEL_AGENT")
    _exit("FRAMEWORK_AGENT", at=145.0, to_phase="SWEEP")

    ext = _ext("FRAMEWORK_AGENT")
    assert ext["entries"] == 2
    assert ext["duration_sec"] == 35.0
    assert [row["sequence"] for row in ext["segments"]] == [1, 3]


def test_the_second_entry_settles_its_own_segment(tmp_path):
    """An exit closes the open segment, not the first one it finds."""
    _enter("SWEEP", sequence=1, at=10.0)
    _exit("SWEEP", at=20.0, to_phase="FRAMEWORK_AGENT", reason="sweep_done")
    _enter("SWEEP", sequence=2, at=60.0, from_phase="FRAMEWORK_AGENT")
    _exit("SWEEP", at=95.0, to_phase="CLOSE", reason="budget_spent")

    segments = _ext("SWEEP")["segments"]
    assert [row["exit_reason"] for row in segments] == ["sweep_done", "budget_spent"]
    assert [row["duration_sec"] for row in segments] == [10.0, 35.0]


def test_the_same_phase_in_a_later_cycle_is_a_separate_event(tmp_path):
    """The macro cycle is part of the id, so cycle 1's time is its own event."""
    _enter("FRAMEWORK_AGENT", sequence=1, cycle=0, at=10.0)
    _exit("FRAMEWORK_AGENT", cycle=0, at=30.0, to_phase="EXPLORE")
    _enter("FRAMEWORK_AGENT", sequence=2, cycle=1, at=200.0, from_phase="EXPLORE")
    _exit("FRAMEWORK_AGENT", cycle=1, at=260.0, to_phase="CLOSE")

    assert _ext("FRAMEWORK_AGENT", 0)["duration_sec"] == 20.0
    assert _ext("FRAMEWORK_AGENT", 1)["duration_sec"] == 60.0


def test_a_loopback_that_bumped_the_cycle_still_closes_the_right_event(tmp_path):
    """The outgoing phase is closed by lookup, not by the caller's cycle.

    The loopback increments ``macro_cycle`` on its way out of a phase, so the
    cycle in scope at the transition can already be the next one. Computing the
    outgoing event id from it would close an event nothing ever opened and leave
    the real one hanging open forever.
    """
    _enter("EXPLORE", sequence=1, cycle=0, at=10.0)
    # The transition out of EXPLORE reports cycle 1: the bump already happened.
    _exit("EXPLORE", cycle=1, at=50.0, to_phase="FRAMEWORK_AGENT", reason="cycle_reloop")

    ext = _ext("EXPLORE", 0)
    assert ext["open"] is False
    assert ext["duration_sec"] == 40.0
    assert _ext("EXPLORE", 1)["segments"] == []


# --- actions are charged to the phase that ordered them --------------------


def test_a_dispatch_is_charged_to_the_phase_that_ordered_it(tmp_path):
    """The dispatching phase owns the action, and owns it before it settles."""
    _enter("FRAMEWORK_AGENT", sequence=1, at=10.0)
    phase_event.record_dispatch(
        action="baseline",
        task_id="t-1",
        phase="FRAMEWORK_AGENT",
        macro_cycle=0,
        tick=7,
        dispatched_unix=12.0,
    )

    actions = _ext("FRAMEWORK_AGENT")["actions"]
    assert actions["count"] == 1
    assert actions["settled"] == 0
    assert actions["kinds"] == ["baseline"]
    row = actions["rows"][0]
    assert row["action"] == "baseline"
    assert row["task_id"] == "t-1"
    assert row["phase"] == "FRAMEWORK_AGENT"
    assert row["tick"] == 7
    assert row["duration_sec"] is None


def test_an_action_that_outlived_its_phase_stays_with_its_phase(tmp_path):
    """The settle does not move the row to whichever phase inherited it.

    This is the case the export-time timestamp-window attribution got wrong. A
    baseline ordered in FRAMEWORK_AGENT that settles after a plateau exit has a
    settle timestamp inside KERNEL_AGENT's window, so the window test charged it
    to KERNEL_AGENT -- a phase that never asked for it.
    """
    _enter("FRAMEWORK_AGENT", sequence=1, at=10.0)
    phase_event.record_dispatch(
        action="baseline",
        task_id="t-1",
        phase="FRAMEWORK_AGENT",
        macro_cycle=0,
        dispatched_unix=12.0,
    )
    _exit("FRAMEWORK_AGENT", at=30.0, to_phase="KERNEL_AGENT")
    _enter("KERNEL_AGENT", sequence=2, at=30.0, from_phase="FRAMEWORK_AGENT")
    # The result lands well after the phase that ordered it is gone.
    phase_event.record_settle(
        task_id="t-1",
        status="succeeded",
        decision="promoted",
        settled_unix=44.0,
        phase="KERNEL_AGENT",
        macro_cycle=0,
        action="baseline",
    )

    ordered = _ext("FRAMEWORK_AGENT")["actions"]
    assert ordered["count"] == 1
    assert ordered["settled"] == 1
    assert ordered["rows"][0]["status"] == "succeeded"
    assert ordered["rows"][0]["decision"] == "promoted"
    # Measured across the phase boundary, because the action really did run
    # that long -- and charged to nobody but the phase that ordered it.
    assert ordered["rows"][0]["duration_sec"] == 32.0
    assert _ext("KERNEL_AGENT")["actions"]["count"] == 0


def test_a_dispatch_killed_mid_flight_reads_as_dispatched(tmp_path):
    """No verdict is a fact of its own, and not the same as never running.

    ``phase_timeline`` was written from the settled-attempt audit, so a
    dispatch cancelled at shutdown or over budget left no row at all and read
    exactly like one the phase never ordered.
    """
    _enter("SWEEP", sequence=1, at=10.0)
    phase_event.record_dispatch(action="conc_sweep", task_id="t-9", phase="SWEEP", macro_cycle=0, dispatched_unix=11.0)
    _exit("SWEEP", at=20.0, to_phase="CLOSE")

    actions = _ext("SWEEP")["actions"]
    assert actions["count"] == 1
    assert actions["settled"] == 0
    assert actions["rows"][0].get("status", "") == ""
    assert actions["rows"][0]["duration_sec"] is None


def test_a_failed_dispatch_keeps_its_error_class(tmp_path):
    """The verdict merges onto the dispatch row rather than opening a second."""
    _enter("PRELUDE", sequence=1, at=10.0)
    phase_event.record_dispatch(action="baseline", task_id="t-2", phase="PRELUDE", macro_cycle=0, dispatched_unix=11.0)
    phase_event.record_settle(
        task_id="t-2",
        status="failed",
        decision="no_promote",
        error_class="server_init_dead",
        workspace="/s/ws/t-2",
        settled_unix=25.0,
    )

    rows = _ext("PRELUDE")["actions"]["rows"]
    assert len(rows) == 1
    assert rows[0]["error_class"] == "server_init_dead"
    assert rows[0]["workspace"] == "/s/ws/t-2"
    assert rows[0]["duration_sec"] == 14.0


def test_every_kind_is_recorded_not_just_the_audited_four(tmp_path):
    """Coverage comes from the dispatch chokepoint, not from a kind whitelist.

    ``_AUDIT_ACTIONS`` covers four of the catalogue's fifteen kinds, so
    ``report``, ``recover``, ``session_breakdown`` and ``target_analysis`` were
    invisible on the timeline entirely. Nothing here consults that set.
    """
    _enter("CLOSE", sequence=1, at=10.0)
    for index, kind in enumerate(("report", "recover", "session_breakdown", "target_analysis")):
        phase_event.record_dispatch(
            action=kind,
            task_id=f"t-{index}",
            phase="CLOSE",
            macro_cycle=0,
            dispatched_unix=11.0 + index,
        )

    actions = _ext("CLOSE")["actions"]
    assert actions["count"] == 4
    assert actions["kinds"] == ["recover", "report", "session_breakdown", "target_analysis"]


def test_a_settle_with_no_dispatch_row_is_still_recorded(tmp_path):
    """A verdict from a path that never went through the runner is not dropped."""
    _enter("KERNEL_AGENT", sequence=1, at=10.0)
    phase_event.record_settle(
        task_id="t-orphan",
        status="succeeded",
        decision="promoted",
        settled_unix=20.0,
        phase="KERNEL_AGENT",
        macro_cycle=0,
        action="kernel_opt",
    )

    rows = _ext("KERNEL_AGENT")["actions"]["rows"]
    assert [row["task_id"] for row in rows] == ["t-orphan"]
    assert rows[0]["action"] == "kernel_opt"
    assert rows[0]["status"] == "succeeded"


def test_an_unidentified_settle_is_not_recorded(tmp_path):
    """Nothing is invented for a task with no id and no phase to place it in."""
    _enter("PRELUDE", sequence=1, at=10.0)
    phase_event.record_settle(task_id="", status="succeeded")
    phase_event.record_settle(task_id="t-nowhere", status="succeeded")

    assert _ext("PRELUDE")["actions"]["count"] == 0


# --- markers ----------------------------------------------------------------


def test_a_marker_lands_in_the_phase_that_raised_it(tmp_path):
    """Non-transition history rows are rows on the phase, not phases."""
    _enter("FRAMEWORK_AGENT", sequence=1, at=10.0)
    phase_event.record_marker(
        phase="FRAMEWORK_AGENT",
        macro_cycle=0,
        sequence=2,
        reason="plateau_proxy_provisional",
        evidence={"r09_provisional": True},
        ts="2026-01-01T00:00:15+00:00",
    )

    markers = _ext("FRAMEWORK_AGENT")["markers"]
    assert markers["count"] == 1
    assert markers["rows"][0]["reason"] == "plateau_proxy_provisional"
    assert markers["rows"][0]["evidence"] == {"r09_provisional": True}
    assert _ext("FRAMEWORK_AGENT")["entries"] == 1


def test_a_marker_alone_still_opens_the_phase(tmp_path):
    """Every entry point opens the event idempotently; none of them owns it."""
    phase_event.record_marker(phase="PRELUDE", macro_cycle=0, sequence=1, reason="install_degraded")

    finalize_events(tmp_path)
    assert [event["id"] for event in _events(tmp_path)] == ["prelude:0:phase"]


# --- status -----------------------------------------------------------------


def test_a_phase_that_dispatched_a_failure_still_succeeded(tmp_path):
    """A phase is not failed by having ordered an action that failed.

    The action's own verdict is on its row and in its stage event. A phase that
    tried something, was told no, and moved on did exactly what it is for.
    """
    _enter("FRAMEWORK_AGENT", sequence=1, at=10.0)
    phase_event.record_dispatch(action="explore", task_id="t-1", phase="FRAMEWORK_AGENT", macro_cycle=0)
    phase_event.record_settle(task_id="t-1", status="failed", decision="no_promote")
    phase_event.record_dispatch(action="baseline", task_id="t-2", phase="FRAMEWORK_AGENT", macro_cycle=0)
    phase_event.record_settle(task_id="t-2", status="succeeded", decision="promoted")
    _exit("FRAMEWORK_AGENT", at=60.0, to_phase="KERNEL_AGENT")

    assert _status("FRAMEWORK_AGENT") == phase_event.STATUS_SUCCEEDED


def test_a_phase_where_nothing_settled_well_is_degraded(tmp_path):
    """Every dispatch failing is the phase not doing what it was entered for."""
    _enter("KERNEL_AGENT", sequence=1, at=10.0)
    phase_event.record_dispatch(action="kernel_opt", task_id="t-1", phase="KERNEL_AGENT", macro_cycle=0)
    phase_event.record_settle(task_id="t-1", status="failed", decision="no_promote")
    _exit("KERNEL_AGENT", at=60.0, to_phase="SWEEP")

    assert _status("KERNEL_AGENT") == phase_event.STATUS_DEGRADED


def test_a_phase_that_dispatched_nothing_still_succeeded(tmp_path):
    """A phase can legitimately be a pass-through, and that is not degraded."""
    _enter("PRELUDE", sequence=1, at=10.0)
    _exit("PRELUDE", at=12.0, to_phase="FRAMEWORK_AGENT")

    assert _status("PRELUDE") == phase_event.STATUS_SUCCEEDED


# --- finalize ---------------------------------------------------------------


def test_finalize_publishes_every_phase_it_finds(tmp_path):
    """Closed and open phases alike reach the timeline."""
    _enter("PRELUDE", sequence=1, at=10.0)
    _exit("PRELUDE", at=30.0, to_phase="FRAMEWORK_AGENT")
    _enter("FRAMEWORK_AGENT", sequence=2, at=30.0, from_phase="PRELUDE")
    phase_event.record_dispatch(
        action="baseline",
        task_id="t-1",
        phase="FRAMEWORK_AGENT",
        macro_cycle=0,
        dispatched_unix=31.0,
    )

    finalize_events(tmp_path)
    events = {event["id"]: event for event in _events(tmp_path)}
    assert set(events) == {"prelude:0:phase", "framework_agent:0:phase"}
    assert events["prelude:0:phase"]["ext"]["duration_sec"] == 20.0
    live = events["framework_agent:0:phase"]
    assert live["ext"]["open"] is True
    assert live["ext"]["actions"]["rows"][0]["task_id"] == "t-1"


def test_the_phase_event_holds_no_copy_of_the_stage_detail(tmp_path):
    """The action row is thin on purpose; the join key is ``task_id``.

    The original plan for this section was a flat action stream carrying each
    attempt's key metric and per-action extras. That row is a strict subset of
    what the stage events record and could not express any of what makes them
    worth reading, so recording it would have put one semantic in two places.
    """
    _enter("FRAMEWORK_AGENT", sequence=1, at=10.0)
    phase_event.record_dispatch(action="baseline", task_id="t-1", phase="FRAMEWORK_AGENT", macro_cycle=0)
    phase_event.record_settle(task_id="t-1", status="succeeded", decision="promoted")

    row = _ext("FRAMEWORK_AGENT")["actions"]["rows"][0]
    assert set(row) == {
        "action",
        "task_id",
        "phase",
        "macro_cycle",
        "tick",
        "dispatched_at",
        "dispatched_unix",
        "status",
        "decision",
        "error_class",
        "workspace",
        "settled_at",
        "settled_unix",
        "duration_sec",
    }


def _propose(msg_id: str, *, phase: str = "KERNEL_AGENT", action: str = "kernel_opt", cycle: int = 0, **kw) -> None:
    phase_event.record_proposal(
        proposal_msg_id=msg_id,
        action=action,
        phase=phase,
        macro_cycle=cycle,
        from_agent="orchestration",
        **kw,
    )


def test_a_proposal_is_on_record_before_anything_acts_on_it(tmp_path):
    """Most proposals are never dispatched, so no other row would ever hold them."""
    _enter("KERNEL_AGENT", sequence=1, at=10.0)
    _propose("m-1", predicted_gain_pct=4.5)

    proposals = _ext("KERNEL_AGENT")["proposals"]
    assert proposals["count"] == 1
    row = proposals["rows"][0]
    assert row["proposal_msg_id"] == "m-1"
    assert row["action"] == "kernel_opt"
    assert row["from_agent"] == "orchestration"
    assert row["predicted_gain_pct"] == 4.5


def test_a_kernel_phase_ruling_is_filed_on_the_thing_it_ruled_on(tmp_path):
    """No framework event exists here, which is why this ruling used to vanish."""
    _enter("KERNEL_AGENT", sequence=1, at=10.0)
    _propose("m-1")
    phase_event.record_proposal_review(
        proposal_msg_id="m-1",
        verdict="reject",
        reasoning="the kernel is already fused",
        confidence=0.8,
        failure_reason_code="no_headroom",
    )

    review = _ext("KERNEL_AGENT")["proposals"]["rows"][0]["critic_review"]
    assert review["verdict"] == "reject"
    assert review["effective_verdict"] == "reject"
    assert review["held_to_rule"] is False
    assert review["confidence"] == 0.8
    assert review["failure_reason_code"] == "no_headroom"


def test_a_ruling_the_envelope_overrode_says_it_was_held_to_a_rule(tmp_path):
    """Two verdicts say they differ; only a third says the difference was imposed."""
    _enter("KERNEL_AGENT", sequence=1, at=10.0)
    _propose("m-1")
    phase_event.record_proposal_review(proposal_msg_id="m-1", verdict="approve", effective_verdict="needs_review")

    review = _ext("KERNEL_AGENT")["proposals"]["rows"][0]["critic_review"]
    assert review["verdict"] == "approve"
    assert review["effective_verdict"] == "needs_review"
    assert review["held_to_rule"] is True


def test_a_ruling_reaching_a_proposal_after_its_phase_exited_still_lands_on_it(tmp_path):
    """The Critic runs on its own tick, so the phase in scope is not the one that asked."""
    _enter("KERNEL_AGENT", sequence=1, at=10.0)
    _propose("m-1")
    _exit("KERNEL_AGENT", at=20.0, to_phase="FRAMEWORK_AGENT")
    _enter("FRAMEWORK_AGENT", sequence=2, at=20.0)
    phase_event.record_proposal_review(proposal_msg_id="m-1", verdict="approve")

    assert _ext("KERNEL_AGENT")["proposals"]["rows"][0]["critic_review"]["verdict"] == "approve"
    assert _ext("FRAMEWORK_AGENT")["proposals"]["count"] == 0


def test_a_ruling_for_a_proposal_that_was_never_recorded_mints_nothing(tmp_path):
    """A ruling cannot bring into existence the thing it claims to be about."""
    _enter("KERNEL_AGENT", sequence=1, at=10.0)
    phase_event.record_proposal_review(proposal_msg_id="ghost", verdict="approve")

    assert _ext("KERNEL_AGENT")["proposals"]["count"] == 0


def test_the_task_a_proposal_became_joins_it_to_its_dispatch(tmp_path):
    """Otherwise what was asked for and what was run sit on one event unconnected."""
    _enter("KERNEL_AGENT", sequence=1, at=10.0)
    _propose("m-1")
    phase_event.record_proposal_review(proposal_msg_id="m-1", verdict="approve")
    phase_event.record_proposal_outcome(proposal_msg_id="m-1", materialized=True, task_id="t-9")
    phase_event.record_dispatch(action="kernel_opt", task_id="t-9", phase="KERNEL_AGENT", macro_cycle=0)

    ext = _ext("KERNEL_AGENT")
    outcome = ext["proposals"]["rows"][0]["outcome"]
    assert outcome["materialized"] is True
    assert outcome["task_id"] == "t-9"
    assert [row["task_id"] for row in ext["actions"]["rows"]] == ["t-9"]


def test_a_proposal_the_critic_never_reached_is_not_a_refused_one(tmp_path):
    """The gap between count and reviewed is the difference, and it is reportable."""
    _enter("KERNEL_AGENT", sequence=1, at=10.0)
    _propose("m-1")
    _propose("m-2")
    phase_event.record_proposal_review(proposal_msg_id="m-1", verdict="reject")

    proposals = _ext("KERNEL_AGENT")["proposals"]
    assert proposals["count"] == 2
    assert proposals["reviewed"] == 1
    assert proposals["materialized"] == 0
    assert "critic_review" not in proposals["rows"][1]
