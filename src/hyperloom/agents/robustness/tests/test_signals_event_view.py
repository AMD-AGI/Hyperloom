# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the shared event view (event_view.py)."""

from __future__ import annotations

from hyperloom.agents.robustness.role.prompt_inputs import InboxItem
from hyperloom.agents.robustness.signals.event_view import build_event_view, family_of


def _coord(seq: int, topic: str, payload: dict | None = None, *, ts: str = "") -> dict:
    return {
        "id": seq,
        "agent": "coordinator",
        "topic": topic,
        "payload": payload or {},
        "ts": ts,
    }


def _inbox(seq: int, topic: str, payload: dict | None = None) -> InboxItem:
    return InboxItem(
        seq=seq,
        msg_id=f"m-{seq}",
        from_agent="coordinator",
        topic=topic,
        payload=payload or {},
    )


def test_desc_input_is_straightened_to_asc():
    coord = [_coord(3, "c"), _coord(2, "b"), _coord(1, "a")]
    view = build_event_view([], coord)
    assert [r.seq for r in view] == [1, 2, 3]


def test_dedup_coord_wins_over_inbox():
    coord = [_coord(5, "from_coord", {"source": "coord"})]
    inbox = [_inbox(5, "from_inbox", {"source": "inbox"})]
    view = build_event_view(inbox, coord)
    assert len(view) == 1
    assert view[0].payload["source"] == "coord"


def test_no_seq_coord_rows_preserve_input_order():
    coord_no_seq = [
        {"agent": "coordinator", "topic": "c1", "payload": {}, "ts": ""},
        {"agent": "coordinator", "topic": "c2", "payload": {}, "ts": ""},
    ]
    view = build_event_view([], coord_no_seq)
    assert [r.topic for r in view] == ["c1", "c2"]


def test_no_seq_inbox_rows_appear_after_no_seq_coord_rows():
    coord_no_seq = [
        {"agent": "coordinator", "topic": "coord_first", "payload": {}, "ts": ""},
    ]
    inbox_item = InboxItem(seq=99, msg_id="i1", from_agent="orchestration", topic="inbox_last", payload={})
    view = build_event_view([inbox_item], coord_no_seq)
    seqless = [r.topic for r in view if r.seq is None]
    assert seqless == ["coord_first"]
    seqed = [r.topic for r in view if r.seq is not None]
    assert seqed == ["inbox_last"]


def test_rows_with_seq_sort_before_rows_without():
    coord = [
        {"id": None, "agent": "a", "topic": "no_seq", "payload": {}, "ts": ""},
        _coord(1, "has_seq"),
    ]
    view = build_event_view([], coord)
    assert view[0].seq == 1
    assert view[1].seq is None


def test_inbox_ts_taken_from_payload():
    item = InboxItem(
        seq=10,
        msg_id="m-10",
        from_agent="orchestration",
        topic="heartbeat",
        payload={"ts": "2026-08-24T10:00:00"},
    )
    view = build_event_view([item], [])
    assert view[0].ts == "2026-08-24T10:00:00"


def test_empty_inputs_return_empty():
    assert build_event_view([], []) == []


def test_family_of_prefers_family_key():
    assert family_of({"family": "sweep", "kind": "other"}) == "sweep"


def test_family_of_falls_back_to_kind():
    assert family_of({"kind": "baseline"}) == "baseline"


def test_family_of_returns_empty_string_when_absent():
    assert family_of({}) == ""
    assert family_of({"family": "  "}) == ""
