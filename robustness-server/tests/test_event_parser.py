"""Tests for the NATS event envelope parser."""

from __future__ import annotations

from datetime import datetime, timezone

from robustness_server.models import EventKind, parse_event_envelope


def test_parse_returns_none_when_session_and_subject_are_unidentifiable() -> None:
    out = parse_event_envelope(subject="other.subject", body={"type": "x"})
    assert out is None


def test_parse_returns_none_when_type_is_missing() -> None:
    out = parse_event_envelope(
        subject="events.s1",
        body={"sessionId": "s1"},
    )
    assert out is None


def test_subject_fallback_for_session_id() -> None:
    out = parse_event_envelope(
        subject="events.s-42",
        body={"type": "session_start"},
    )
    assert out is not None
    assert out.session_id == "s-42"
    assert out.kind == EventKind.SESSION_START


def test_normalises_known_terminal_kind() -> None:
    out = parse_event_envelope(
        subject="events.s1",
        body={"sessionId": "s1", "type": "EXEC_COMPLETE"},
    )
    assert out is not None
    assert out.kind == EventKind.EXEC_COMPLETE
    assert out.raw_type == "EXEC_COMPLETE"


def test_unknown_type_collapses_to_other_but_preserves_raw() -> None:
    out = parse_event_envelope(
        subject="events.s1",
        body={"sessionId": "s1", "type": "rare-event"},
    )
    assert out is not None
    assert out.kind == EventKind.OTHER
    assert out.raw_type == "rare-event"


def test_iso_timestamp_with_z_is_parsed_as_utc() -> None:
    out = parse_event_envelope(
        subject="events.s1",
        body={
            "sessionId": "s1",
            "type": "session_start",
            "timestamp": "2026-04-28T12:00:00Z",
        },
    )
    assert out is not None
    assert out.occurred_at.tzinfo is not None
    assert out.occurred_at == datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc)


def test_naive_timestamp_is_assumed_utc() -> None:
    out = parse_event_envelope(
        subject="events.s1",
        body={
            "sessionId": "s1",
            "type": "session_start",
            "timestamp": "2026-04-28T12:00:00",
        },
    )
    assert out is not None
    assert out.occurred_at.tzinfo == timezone.utc


def test_epoch_seconds_accepted() -> None:
    out = parse_event_envelope(
        subject="events.s1",
        body={"sessionId": "s1", "type": "session_start", "timestamp": 1700000000},
    )
    assert out is not None
    assert out.occurred_at == datetime.fromtimestamp(1700000000, tz=timezone.utc)


def test_pod_and_namespace_are_picked_from_alternative_keys() -> None:
    out = parse_event_envelope(
        subject="events.s1",
        body={
            "sessionId": "s1",
            "type": "sandbox_create",
            "pod": "hands-1",
            "podNamespace": "claw",
        },
    )
    assert out is not None
    assert out.pod_name == "hands-1"
    assert out.pod_namespace == "claw"


def test_body_is_preserved_for_audit() -> None:
    body = {
        "sessionId": "s1",
        "type": "session_start",
        "extra": {"nested": [1, 2, 3]},
    }
    out = parse_event_envelope(subject="events.s1", body=body)
    assert out is not None
    assert out.body == body
