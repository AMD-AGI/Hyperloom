# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Focused unit tests for ``coordinator`` module-level helpers and static utilities."""

from __future__ import annotations


from hyperloom.orchestrator.loop.conversation import _first_present, _format_inbox_event
from hyperloom.orchestrator.loop.kernel_request import lifecycle_paths
from hyperloom.orchestrator.bus.message_bus import Message

from ._booked_results import announced


def test_first_present_non_dict_and_branches():
    """``_first_present`` ignores non-dicts and skips None values."""
    assert _first_present("not-a-dict", ("a",)) is None  # type: ignore[arg-type]
    assert _first_present({}, ("a", "b")) is None
    assert _first_present({"a": None, "b": 2}, ("a", "b")) == 2
    assert _first_present({"x": 0}, ("x",)) == 0


def test_lifecycle_paths_filters_and_types():
    """``lifecycle_paths`` only keeps non-empty string values for known keys."""
    assert lifecycle_paths(None) == {}
    assert lifecycle_paths({"workspace": "  "}) == {}
    assert lifecycle_paths({"workspace": "/data/ws", "report_path": "/r.md"}) == {
        "workspace": "/data/ws",
        "report_path": "/r.md",
    }


def test_format_inbox_delegated_result_names_its_seq_and_the_outcome():
    """delegated_result lines carry the outcome keys of the result on the announced task's row.

    The line names the row by ``seq``, which the held-after line and ``get_inbox(since_seq)`` take; the row's
    ``msg_id`` is an id nothing the agent can send refers to, so it is not printed.
    """
    m, outcomes = announced({"task_id": "t1", "kind": "integrate", "state": "done"}, {"status": "ok", "gain_pct": 3.5})
    m.msg_id = "mid"
    line = _format_inbox_event(m, outcomes)
    assert line.startswith("seq=7 from=")
    assert "msg_id" not in line
    assert "status='ok'" in line and "gain=3.5" in line


def test_format_inbox_delegated_result_of_a_task_with_no_recorded_outcome():
    """An event whose task recorded nothing renders what the event itself says, and nothing the event carries beside it."""
    m = Message(
        msg_id="1",
        from_agent="a",
        to_agent="b",
        topic="delegated_result",
        payload={"task_id": "gone", "kind": "k", "state": "s", "result": {"status": "raw"}, "error": "raw"},
        seq=1,
    )
    line = _format_inbox_event(m, {})
    assert line == "seq=1 from=a topic=delegated_result kind='k' state='s'"


def test_format_inbox_delegated_result_with_error():
    m, outcomes = announced({"task_id": "t1", "kind": "k", "state": "failed"}, {}, error="boom" * 50)
    line = _format_inbox_event(m, outcomes)
    assert "error=" in line


def test_format_inbox_policy_denied_observation_shows_the_rule_and_hint():
    denied = Message(
        msg_id="p",
        from_agent="coordinator",
        to_agent="orchestration",
        topic="observation",
        payload={
            "kind": "policy_denied",
            "intent_type": "delegate",
            "rule": "phase_incompatible",
            "hint": "KERNEL_AGENT proposes: integrate",
            "reason": "action='explore' is not proposable while the phase is 'KERNEL_AGENT'",
        },
        seq=3,
    )
    line = _format_inbox_event(denied, {})
    assert "topic=observation" in line and "kind='policy_denied'" in line
    assert "phase_incompatible" in line and "KERNEL_AGENT proposes: integrate" in line


def test_format_inbox_observation_generic():
    m = Message(
        msg_id="o",
        from_agent="a",
        to_agent="b",
        topic="observation",
        payload={"kind": "metric", "v": 1},
        seq=1,
    )
    assert "kind='metric'" in _format_inbox_event(m, {})


def test_format_inbox_fallback():
    m = Message(
        msg_id="f",
        from_agent="a",
        to_agent="b",
        topic="other",
        payload={"z": 1},
        seq=0,
    )
    assert "topic=other" in _format_inbox_event(m, {}) and "payload=" in _format_inbox_event(m, {})
