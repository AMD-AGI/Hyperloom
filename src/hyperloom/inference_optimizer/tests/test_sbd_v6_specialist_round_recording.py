# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist round came back with is recorded on the row that dispatched it.

The section this replaces was one row per round, keyed by a counter, sitting
beside the timeline with no link to the phase that ordered the dispatch or to
the framework run that consumed the proposals. Rounds are dispatched from three
places -- PRELUDE scouts, the periodic dispatch inside FRAMEWORK_AGENT, and the
plateau trajectory reviewer -- and only the middle one has a framework event
open, so a single landing site could never hold all three.

The product now merges onto whichever row already carries the dispatch: the
framework run for FRAMEWORK_AGENT rounds, the phase action row otherwise. These
tests pin both routes, and that the merge lands on the dispatch rather than
appending a second row.
"""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder import phase_event
from hyperloom.inference_optimizer.breakdown.recorder.assembler import phase_event_parts
from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
    ARM_SOURCE,
    ROLE_DISCOVERY,
    make_framework_recorder,
)
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.loop.writeback import WritebackCollaborator


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so no call below takes a path."""
    with session_scope(tmp_path):
        yield tmp_path


def _actions(phase: str, macro_cycle: int = 0) -> dict[str, Any]:
    ext, _status = phase_event.assemble_phase_ext(
        phase_event_parts(),
        event=phase_event.phase_event_id(phase, macro_cycle),
    )
    return ext["actions"]


def _runs(session_dir) -> list[dict[str, Any]]:
    events = [event for event in read_timeline_events(session_dir) if event.get("type") == "framework_agent"]
    assert len(events) == 1, f"expected one framework event, got {len(events)}"
    return events[0]["ext"]["runs"]


# --- the route with no framework event open -------------------------------


def test_a_prelude_round_lands_on_the_action_that_dispatched_it(tmp_path) -> None:
    """PRELUDE scouts run before any framework event exists."""
    phase_event.record_dispatch(action="specialist", task_id="t-1", phase="PRELUDE", macro_cycle=0, dispatched_unix=4.0)
    phase_event.record_specialist_round(
        task_id="t-1",
        phase="PRELUDE",
        macro_cycle=0,
        domain="attention",
        summary="two kernels worth attacking",
        proposals_total=2,
        empty=False,
        confidence=0.75,
        new_findings=["fa2 is the hot path"],
    )

    actions = _actions("PRELUDE")
    assert actions["count"] == 1, "the product must merge onto the dispatch, not append a row"
    (row,) = actions["rows"]
    assert row["task_id"] == "t-1"
    assert row["action"] == "specialist"
    assert row["domain"] == "attention"
    assert row["summary"] == "two kernels worth attacking"
    assert row["proposals_total"] == 2
    assert row["empty"] is False
    assert row["confidence"] == pytest.approx(0.75)
    assert row["new_findings"] == ["fa2 is the hot path"]


def test_an_empty_round_says_so_rather_than_going_missing(tmp_path) -> None:
    phase_event.record_dispatch(action="specialist", task_id="t-2", phase="EXPLORE", macro_cycle=2, dispatched_unix=9.0)
    phase_event.record_specialist_round(
        task_id="t-2",
        phase="EXPLORE",
        macro_cycle=2,
        proposals_total=0,
        empty=True,
        residual_questions=["is the tokenizer the bound?"],
    )

    (row,) = _actions("EXPLORE", 2)["rows"]
    assert row["proposals_total"] == 0
    assert row["empty"] is True
    assert row["residual_questions"] == ["is the tokenizer the bound?"]


def test_a_round_settled_outside_the_dispatcher_still_gets_a_row(tmp_path) -> None:
    """Nothing opened a dispatch, so the product opens one rather than dropping."""
    phase_event.record_specialist_round(
        task_id="t-3",
        phase="PLATEAU",
        macro_cycle=1,
        summary="trajectory review",
        proposals_total=1,
    )

    actions = _actions("PLATEAU", 1)
    assert actions["count"] == 1
    (row,) = actions["rows"]
    assert row["task_id"] == "t-3"
    assert row["summary"] == "trajectory review"


def test_re_recording_the_same_round_does_not_duplicate_it(tmp_path) -> None:
    """Resume re-walks the round list; the key must fold the second write in."""
    phase_event.record_dispatch(action="specialist", task_id="t-4", phase="EXPLORE", macro_cycle=0, dispatched_unix=1.0)
    phase_event.record_specialist_round(task_id="t-4", phase="EXPLORE", macro_cycle=0, proposals_total=1)
    phase_event.record_specialist_round(task_id="t-4", phase="EXPLORE", macro_cycle=0, proposals_total=3)

    actions = _actions("EXPLORE")
    assert actions["count"] == 1
    assert actions["rows"][0]["proposals_total"] == 3


def test_an_unapproved_field_cannot_smuggle_itself_into_the_row(tmp_path) -> None:
    """The round entry carries bookkeeping the timeline row has no shape for."""
    phase_event.record_dispatch(action="specialist", task_id="t-5", phase="EXPLORE", macro_cycle=0, dispatched_unix=1.0)
    phase_event.record_specialist_round(
        task_id="t-5",
        phase="EXPLORE",
        macro_cycle=0,
        summary="kept",
        parallelism=4,
        task_domains={"t-5": "gemm"},
    )

    (row,) = _actions("EXPLORE")["rows"]
    assert row["summary"] == "kept"
    assert "parallelism" not in row
    assert "task_domains" not in row


# --- the seam that picks between the two routes ---------------------------


class _StubTask:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id


class _StubState:
    def __init__(self, phase: str, macro_cycle: int = 0) -> None:
        self.phase = phase
        self.macro_cycle = macro_cycle


class _Seam:
    """The writeback method under test, with only what it reads on it."""

    _record_specialist_round_product = WritebackCollaborator._record_specialist_round_product

    def __init__(self, *, phase: str, framework_recorder: Any = None) -> None:
        self.shared_state = _StubState(phase)
        self._framework_timeline_recorder = framework_recorder


def test_a_round_that_names_no_phase_is_charged_to_the_running_one(tmp_path) -> None:
    """The entry's ``source_phase`` is optional; the live phase is the fallback."""
    seam = _Seam(phase="KERNEL_AGENT")
    seam._record_specialist_round_product(
        task=_StubTask("t-6"),
        round_entry={"round_id": "r-6", "domain": "gemm", "proposals_total": 1},
    )

    (row,) = _actions("KERNEL_AGENT")["rows"]
    assert row["task_id"] == "t-6"
    assert row["domain"] == "gemm"


def test_a_framework_round_is_not_also_written_to_the_phase(tmp_path) -> None:
    """One dispatch, one row: the two routes must not both fire."""
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_run("t-7", role=ROLE_DISCOVERY, arm=ARM_SOURCE, status="succeeded")
    seam = _Seam(phase="FRAMEWORK_AGENT", framework_recorder=recorder)
    seam._record_specialist_round_product(
        task=_StubTask("t-7"),
        round_entry={"source_phase": "FRAMEWORK_AGENT", "proposals_total": 4},
    )
    recorder.finish(exit_reason="both_arms_plateaued")

    (run,) = _runs(tmp_path)
    assert run["proposals_total"] == 4
    assert _actions("FRAMEWORK_AGENT")["count"] == 0


# --- the route inside FRAMEWORK_AGENT --------------------------------------


def test_a_framework_round_lands_on_the_run_that_dispatched_it(tmp_path) -> None:
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_run("t-9", role=ROLE_DISCOVERY, arm=ARM_SOURCE, domain="gemm", status="succeeded")
    recorder.record_run(
        "t-9",
        summary="two gemm shapes mis-tuned",
        proposals_total=2,
        empty=False,
        confidence=0.5,
        new_findings=["m=8192 falls off the tile"],
        notes=["ran short of budget"],
        ensemble_scores={"agree": 0.8},
    )
    recorder.finish(exit_reason="both_arms_plateaued")

    (run,) = _runs(tmp_path)
    assert run["domain"] == "gemm", "the first write must survive the merge"
    assert run["status"] == "succeeded"
    assert run["summary"] == "two gemm shapes mis-tuned"
    assert run["proposals_total"] == 2
    assert run["empty"] is False
    assert run["confidence"] == pytest.approx(0.5)
    assert run["new_findings"] == ["m=8192 falls off the tile"]
    assert run["notes"] == ["ran short of budget"]
    assert run["ensemble_scores"] == {"agree": 0.8}
