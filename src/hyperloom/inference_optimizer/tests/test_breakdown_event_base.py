# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The base layer every SBD V6 timeline event type records through.

These cover the invariants the layer exists to enforce rather than the shape of
any one event type: that ids are recomputable after a resume, that a row cannot
be written without both halves of its event attribution, that assembly order
does not depend on write order, and that a killed session leaves a state
finalize can tell apart from a finished one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown import recorder as rec
from hyperloom.inference_optimizer.session.session_binding import (
    SessionNotBoundError,
    bound_session,
    session_is_bound,
    session_scope,
)

_EID = "kernel_agent:3:kernel"


def _timeline_files(session_dir: Path) -> list[Path]:
    """Return the V6 timeline event files of a session, in sequence order."""
    root = session_dir / "reports" / "sbd_v6" / "timeline"
    return sorted(root.glob("*.json")) if root.is_dir() else []


def _fragments(session_dir: Path) -> list[Path]:
    """Return the recorder fragments of a session, by filename."""
    root = session_dir / "runtime" / "breakdown" / "parts"
    return sorted(root.glob("*.json")) if root.is_dir() else []


# --- ids -------------------------------------------------------------------


def test_an_event_id_is_three_readable_segments_with_the_phase_case_normalized():
    assert rec.event_id("KERNEL_AGENT", 3, "kernel") == _EID
    assert rec.event_id("prelude", 0, "roofline") == "prelude:0:roofline"


def test_recomputing_an_event_id_from_the_same_state_gives_the_same_value():
    # The resume guarantee: nothing in the id comes from a clock, a random
    # source or a process counter, so a new process continuing the same phase
    # run writes into the same fragments instead of a second set.
    assert rec.event_id("kernel_agent", 3, "kernel") == rec.event_id("kernel_agent", 3, "kernel")


@pytest.mark.parametrize(
    "phase,component",
    [
        ("kernel agent", "kernel"),  # a space would fold into the filename slug
        ("kernel:agent", "kernel"),  # a separator would split the id
        ("", "kernel"),
        ("kernel_agent", ""),
    ],
)
def test_an_event_id_segment_outside_the_narrow_alphabet_is_refused(phase, component):
    with pytest.raises(ValueError):
        rec.event_id(phase, 3, component)


def test_an_event_id_refuses_a_macro_cycle_that_is_not_a_non_negative_integer():
    with pytest.raises(ValueError):
        rec.event_id("kernel_agent", -1, "kernel")
    with pytest.raises(ValueError):
        rec.event_id("kernel_agent", "later", "kernel")


def test_a_fragment_key_puts_the_event_id_in_front_of_the_row_identity():
    assert rec.fragment_key(_EID, "lane", "att-7") == f"{_EID}:lane:att-7"
    assert rec.fragment_key(_EID, "profile", "task-1", "0") == f"{_EID}:profile:task-1:0"


def test_the_event_level_fragment_is_keyed_by_the_bare_event_id():
    assert rec.fragment_key(_EID, "") == _EID


def test_a_fragment_key_refuses_an_empty_natural_id():
    # Tolerating it would give two distinct rows one key, and the second would
    # merge into the first rather than land beside it.
    with pytest.raises(ValueError):
        rec.fragment_key(_EID, "lane", "")


# --- the sink --------------------------------------------------------------


def test_a_sink_writes_the_event_id_into_both_the_key_and_the_payload(tmp_path):
    # Both halves are load-bearing and protect against different failures, so
    # the sink supplies them together rather than trusting the caller.
    with session_scope(tmp_path):
        sink = rec.make_sink(_EID, producer="orchestrator")
        path = sink.record("kernel_lane_run", {"attempt_id": "att-7"}, row_type="lane", natural_ids="att-7")

    assert path is not None
    fragment = json.loads(path.read_text(encoding="utf-8"))
    assert fragment["payload"]["event_id"] == _EID
    assert path.name.startswith("kernel_lane_run__orchestrator__kernel_agent-3-kernel-lane-att-7-")


def test_a_sink_drops_a_payload_that_claims_a_different_event_and_says_so_loudly(tmp_path, caplog):
    # The core is meant to be ignorant of its event id, so a payload that names
    # one is a leak rather than a value to be trusted or silently overridden.
    with session_scope(tmp_path), caplog.at_level(logging.WARNING):
        sink = rec.make_sink(_EID, producer="orchestrator")
        written = sink.record(
            "kernel_lane_run",
            {"attempt_id": "att-7", "event_id": "prelude:0:roofline"},
            row_type="lane",
            natural_ids="att-7",
        )

    assert written is None
    assert not _fragments(tmp_path)
    assert "dropped a kernel_lane_run row" in caplog.text


def test_a_failing_row_is_dropped_loudly_rather_than_breaking_the_phase(tmp_path, caplog):
    # An empty natural id is data-dependent, not just an author slip: a task id
    # can come back empty on some paths. Recording must not take the run down
    # with it, but the missing fact cannot be quiet either.
    with session_scope(tmp_path), caplog.at_level(logging.WARNING):
        sink = rec.make_sink(_EID, producer="orchestrator")
        written = sink.record("kernel_lane_run", {"attempt_id": ""}, row_type="lane", natural_ids="")

    assert written is None
    assert not _fragments(tmp_path)
    assert "will be missing this fact" in caplog.text


def test_two_sinks_over_one_section_keep_their_rows_in_separate_files(tmp_path):
    # This is why the key carries the prefix: the spool is per session, so two
    # events holding a row with the same natural id would otherwise upsert into
    # one file and the first event's row would be gone.
    with session_scope(tmp_path):
        rec.make_sink(_EID, producer="orchestrator").record(
            "kernel_lane_run", {"attempt_id": "att-7", "lane": "kernel_rewrites"}, row_type="lane", natural_ids="att-7"
        )
        rec.make_sink("prelude:0:roofline", producer="orchestrator").record(
            "kernel_lane_run", {"attempt_id": "att-7", "lane": "collective_runs"}, row_type="lane", natural_ids="att-7"
        )

    assert len(_fragments(tmp_path)) == 2


def test_a_second_write_on_one_key_merges_instead_of_replacing(tmp_path):
    # Three writers touch one lane row at three different moments, none of them
    # knowing the other two's fields; whole-object write-back would erase them.
    with session_scope(tmp_path):
        sink = rec.make_sink(_EID, producer="orchestrator")
        sink.record("kernel_lane_run", {"attempt_id": "att-7", "run_id": "r-1"}, row_type="lane", natural_ids="att-7")
        sink.record("kernel_lane_run", {"e2e": {"speedup": 1.07}}, row_type="lane", natural_ids="att-7")
        path = sink.record("kernel_lane_run", {"rebench_ref": "idem-9"}, row_type="lane", natural_ids="att-7")

    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
    assert payload["run_id"] == "r-1"
    assert payload["e2e"] == {"speedup": 1.07}
    assert payload["rebench_ref"] == "idem-9"
    assert len(_fragments(tmp_path)) == 1


# --- assembly primitives ---------------------------------------------------


def test_filtering_keeps_only_the_rows_of_the_event_being_closed():
    rows = [
        {"event_id": _EID, "attempt_id": "att-7"},
        {"event_id": "prelude:0:roofline", "attempt_id": "att-7"},
        {"attempt_id": "att-9"},
        "not a row",
    ]
    assert rec.rows_for_event(rows, _EID) == [{"event_id": _EID, "attempt_id": "att-7"}]


def test_rows_sort_by_their_own_fields_regardless_of_the_order_they_were_read_in():
    early = {"event_id": _EID, "attempt_id": "att-8", "started_at": "10:00:00", "run_id": "r-2"}
    late = {"event_id": _EID, "attempt_id": "att-7", "started_at": "10:00:01", "run_id": "r-1"}
    keys = ("started_at", "run_id")
    assert rec.sort_rows([late, early], keys=keys) == rec.sort_rows([early, late], keys=keys)
    assert [row["attempt_id"] for row in rec.sort_rows([late, early], keys=keys)] == ["att-8", "att-7"]


def test_a_row_with_no_primary_key_sorts_last_rather_than_first():
    # Sorting it first would read as "this happened before everything else",
    # which is a claim the missing field cannot support.
    rows = [
        {"attempt_id": "no-ts"},
        {"attempt_id": "has-ts", "started_at": "10:00:00"},
    ]
    assert [row["attempt_id"] for row in rec.sort_rows(rows, keys=("started_at",))] == ["has-ts", "no-ts"]


def test_rows_that_agree_on_every_declared_key_still_sort_deterministically():
    # Assembling the same fragments twice has to produce the same array, even
    # when the caller's declared keys do not separate two rows.
    a = {"attempt_id": "a", "started_at": "10:00:00"}
    b = {"attempt_id": "b", "started_at": "10:00:00"}
    assert rec.sort_rows([a, b], keys=("started_at",)) == rec.sort_rows([b, a], keys=("started_at",))


def test_grouping_splits_one_section_into_its_wire_positions():
    rows = [
        {"lane": "kernel_rewrites", "attempt_id": "att-7"},
        {"lane": "fusion_runs", "attempt_id": "att-8"},
        {"lane": "kernel_rewrites", "attempt_id": "att-9"},
    ]
    grouped = rec.group_rows(rows, "lane")
    assert [row["attempt_id"] for row in grouped["kernel_rewrites"]] == ["att-7", "att-9"]
    assert [row["attempt_id"] for row in grouped["fusion_runs"]] == ["att-8"]


def test_recording_side_bookkeeping_does_not_reach_the_wire():
    row = {"event_id": _EID, "ordinal": 4, "attempt_id": "att-7"}
    assert rec.wire_row(row) == {"attempt_id": "att-7"}


# --- the two timeline writes ----------------------------------------------


def test_opening_an_event_makes_it_visible_as_running_before_it_finishes(tmp_path):
    with session_scope(tmp_path):
        sequence = rec.open_event(
            event_type="kernel",
            event=_EID,
            event_section="kernel_event",
            producer="orchestrator",
            start_time="2026-09-02T10:00:00Z",
            ext={"in_flight_stage": "forge"},
        )

    assert sequence == 1
    files = _timeline_files(tmp_path)
    assert len(files) == 1
    event = json.loads(files[0].read_text(encoding="utf-8"))
    assert event["status"] == rec.EVENT_STATUS_RUNNING
    assert event["id"] == _EID
    assert event["ext"]["in_flight_stage"] == "forge"


def test_the_sequence_of_an_opened_event_lands_on_its_event_level_fragment(tmp_path):
    # The fragment is the event's only durable identity, so the sequence has to
    # be there or the closing write cannot find the entry to update.
    with session_scope(tmp_path):
        sequence = rec.open_event(
            event_type="kernel",
            event=_EID,
            event_section="kernel_event",
            producer="orchestrator",
        )
        rows = rec.kernel_event_parts()["kernel_event"]

    assert [row[rec.TIMELINE_SEQUENCE_FIELD] for row in rows] == [sequence]


def test_closing_an_event_updates_the_same_entry_instead_of_appending_one(tmp_path):
    with session_scope(tmp_path):
        sequence = rec.open_event(
            event_type="kernel",
            event=_EID,
            event_section="kernel_event",
            producer="orchestrator",
            start_time="2026-09-02T10:00:00Z",
        )
        rec.finish_event(
            event_type="kernel",
            event=_EID,
            sequence=sequence,
            status="succeeded",
            ext={"macro_cycle": 3},
            start_time="2026-09-02T10:00:00Z",
            end_time="2026-09-02T10:05:00Z",
        )

    files = _timeline_files(tmp_path)
    assert len(files) == 1
    event = json.loads(files[0].read_text(encoding="utf-8"))
    assert event["status"] == "succeeded"
    assert event["end_time"] == "2026-09-02T10:05:00Z"


def test_the_envelope_carries_the_event_id_as_id_and_never_repeats_it_in_ext():
    envelope = rec.build_envelope(event_type="kernel", event=_EID, status="running", ext={"macro_cycle": 3})
    assert envelope["id"] == _EID
    assert "id" not in envelope["ext"]
    assert "event_id" not in envelope["ext"]


# --- what a killed session leaves ------------------------------------------


def test_an_event_killed_after_it_opened_is_residual_with_its_sequence_recoverable(tmp_path):
    with session_scope(tmp_path):
        sequence = rec.open_event(
            event_type="kernel",
            event=_EID,
            event_section="kernel_event",
            producer="orchestrator",
        )
        residual = rec.residual_events(rec.kernel_event_parts()["kernel_event"], event_type="kernel")

    assert residual == [rec.ResidualEvent(event_id=_EID, sequence=sequence, state=rec.RESIDUAL_RUNNING)]


def test_fragments_with_no_event_behind_them_are_residual_with_no_sequence(tmp_path):
    # The shell write failed, or rows were recorded before the event opened.
    # Finalize has to allocate a sequence for these rather than reuse one.
    with session_scope(tmp_path):
        rec.make_sink(_EID, producer="orchestrator").record("kernel_event", {"macro_cycle": 3})
        residual = rec.residual_events(rec.kernel_event_parts()["kernel_event"], event_type="kernel")

    assert residual == [rec.ResidualEvent(event_id=_EID, sequence=None, state=rec.RESIDUAL_NO_EVENT)]


def test_an_event_that_reached_a_terminal_status_is_not_residual(tmp_path):
    with session_scope(tmp_path):
        sequence = rec.open_event(
            event_type="kernel",
            event=_EID,
            event_section="kernel_event",
            producer="orchestrator",
        )
        rec.finish_event(
            event_type="kernel",
            event=_EID,
            sequence=sequence,
            status="succeeded",
            ext={},
        )
        assert rec.residual_events(rec.kernel_event_parts()["kernel_event"], event_type="kernel") == []


def test_a_recovered_event_is_marked_interrupted_rather_than_guessed_complete(tmp_path):
    # Nothing judged the run: the closing status is derived at assembly, and an
    # event whose closing write never ran has no verdict to report.
    with session_scope(tmp_path):
        sequence = rec.open_event(
            event_type="kernel",
            event=_EID,
            event_section="kernel_event",
            producer="orchestrator",
        )
        rec.finish_event(
            event_type="kernel",
            event=_EID,
            sequence=sequence,
            status=rec.EVENT_STATUS_INTERRUPTED,
            ext={"macro_cycle": 3},
        )

    event = json.loads(_timeline_files(tmp_path)[0].read_text(encoding="utf-8"))
    assert event["status"] == rec.EVENT_STATUS_INTERRUPTED
    assert rec.EVENT_STATUS_INTERRUPTED in rec.TERMINAL_EVENT_STATUSES


# --- session binding -------------------------------------------------------


def test_recording_without_a_bound_session_fails_instead_of_guessing_a_path():
    # This is also the backstop for a subprocess: the ContextVar is unset
    # inside a Ray actor, and two processes upserting one fragment lose writes.
    assert not session_is_bound()
    with pytest.raises(SessionNotBoundError):
        rec.get_recorder(producer="orchestrator")
    with pytest.raises(SessionNotBoundError):
        bound_session()


def test_a_session_scope_does_not_outlive_its_block(tmp_path):
    with session_scope(tmp_path):
        assert session_is_bound()
    assert not session_is_bound()


def test_two_spellings_of_one_session_directory_bind_to_one_recorder(tmp_path):
    # The recorder cache is keyed by the derived spool path, so two spellings
    # must not produce two recorders, each holding its own lock over the same
    # fragments.
    with session_scope(tmp_path):
        first = rec.get_recorder(producer="orchestrator")
    with session_scope(str(tmp_path) + "/./"):
        second = rec.get_recorder(producer="orchestrator")
    assert first is second
