"""Tests for orchestrator/multi_cli/envelope.py — A2A v0 line shape."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.multi_cli.envelope import (
    Envelope,
    EnvelopeError,
    EnvelopeKind,
    _SEQ_ALLOCATOR,
    envelopes_since_seq,
    iter_new_envelopes,
    read_cursor,
    read_envelopes,
    write_cursor,
    write_envelope,
)


@pytest.fixture(autouse=True)
def _reset_seq_allocator():
    _SEQ_ALLOCATOR.reset()
    yield
    _SEQ_ALLOCATOR.reset()


# ---------------------------------------------------------------------------
# Constructors / serialisation
# ---------------------------------------------------------------------------
def test_intent_constructor_defaults():
    env = Envelope.intent(
        from_agent="executor",
        intent_type="delegate",
        payload={"action_name": "baseline"},
    )
    assert env.kind is EnvelopeKind.INTENT
    assert env.from_agent == "executor"
    assert env.to_agent == "conductor"
    assert env.intent_type == "delegate"
    assert env.payload == {"action_name": "baseline"}
    assert env.topic is None
    assert env.msg_id  # uuid was generated
    assert env.ts


def test_message_constructor_keeps_topic():
    env = Envelope.message(
        msg_id="m1",
        seq=42,
        from_agent="conductor",
        to_agent="executor",
        topic="proposal",
        payload={"foo": 1},
        priority=2,
        in_reply_to="m0",
    )
    assert env.kind is EnvelopeKind.MESSAGE
    assert env.topic == "proposal"
    assert env.intent_type is None
    assert env.priority == 2


def test_to_json_drops_irrelevant_fields_for_intent():
    import json as _json

    env = Envelope.intent(from_agent="executor", intent_type="send_message",
                          payload={"topic": "heartbeat"})
    decoded = _json.loads(env.to_json())
    assert "topic" not in decoded  # top-level "topic" only set on MESSAGE
    assert "priority" not in decoded
    # Payload-level "topic" stays untouched.
    assert decoded["payload"] == {"topic": "heartbeat"}


def test_to_json_drops_irrelevant_fields_for_message():
    import json as _json

    env = Envelope.message(msg_id="m1", seq=1, from_agent="conductor",
                           to_agent="*", topic="event", payload={})
    decoded = _json.loads(env.to_json())
    assert "intent_type" not in decoded


def test_round_trip_intent_envelope():
    env = Envelope.intent(
        from_agent="watchdog",
        intent_type="alert",
        payload={"severity": "high", "summary": "OOM"},
        in_reply_to="x1",
    )
    decoded = Envelope.from_json(env.to_json())
    assert decoded.kind is EnvelopeKind.INTENT
    assert decoded.from_agent == "watchdog"
    assert decoded.intent_type == "alert"
    assert decoded.payload == {"severity": "high", "summary": "OOM"}
    assert decoded.in_reply_to == "x1"


def test_round_trip_message_envelope():
    env = Envelope.message(msg_id="m1", seq=7, from_agent="conductor",
                           to_agent="executor", topic="event",
                           payload={"k": "v"}, priority=0)
    decoded = Envelope.from_json(env.to_json())
    assert decoded.kind is EnvelopeKind.MESSAGE
    assert decoded.topic == "event"
    assert decoded.priority == 0


def test_from_json_rejects_unknown_kind():
    bad = '{"kind":"???","msg_id":"x","seq":1,"ts":"t","from_agent":"a","to_agent":"b"}'
    with pytest.raises(EnvelopeError, match="kind"):
        Envelope.from_json(bad)


def test_from_json_rejects_missing_fields():
    bad = '{"kind":"intent","seq":1,"ts":"t","from_agent":"a","to_agent":"b"}'
    with pytest.raises(EnvelopeError, match="msg_id"):
        Envelope.from_json(bad)


def test_from_json_rejects_invalid_json():
    with pytest.raises(EnvelopeError, match="invalid"):
        Envelope.from_json("not-json")


# ---------------------------------------------------------------------------
# write_envelope / read_envelopes / iter_new_envelopes
# ---------------------------------------------------------------------------
def test_write_envelope_assigns_monotonic_seq(tmp_path: Path):
    box = tmp_path / "outbox.jsonl"
    a = Envelope.intent(from_agent="executor", intent_type="send_message",
                        payload={"topic": "hello"})
    b = Envelope.intent(from_agent="executor", intent_type="send_message",
                        payload={"topic": "world"})
    seq_a = write_envelope(box, a)
    seq_b = write_envelope(box, b)
    assert seq_a == 1
    assert seq_b == 2
    assert a.seq == 1
    assert b.seq == 2


def test_write_envelope_resumes_seq_from_existing_file(tmp_path: Path):
    box = tmp_path / "outbox.jsonl"
    write_envelope(box, Envelope.intent(from_agent="executor",
                                        intent_type="send_message",
                                        payload={"i": 1}))
    # Reset the in-process allocator to simulate a fresh writer process.
    _SEQ_ALLOCATOR.reset(box)
    written = write_envelope(box, Envelope.intent(from_agent="executor",
                                                  intent_type="send_message",
                                                  payload={"i": 2}))
    assert written == 2


def test_write_envelope_keeps_caller_provided_seq(tmp_path: Path):
    box = tmp_path / "outbox.jsonl"
    env = Envelope.intent(from_agent="executor", intent_type="send_message",
                          payload={}, seq=42)
    written = write_envelope(box, env)
    assert written == 42


def test_read_envelopes_round_trip(tmp_path: Path):
    box = tmp_path / "outbox.jsonl"
    for topic in ("a", "b", "c"):
        write_envelope(box, Envelope.intent(from_agent="executor",
                                            intent_type="send_message",
                                            payload={"topic": topic}))
    envs = read_envelopes(box)
    assert [e.payload["topic"] for e in envs] == ["a", "b", "c"]
    assert [e.seq for e in envs] == [1, 2, 3]


def test_read_envelopes_ignores_blank_and_garbage_lines(tmp_path: Path):
    box = tmp_path / "outbox.jsonl"
    write_envelope(box, Envelope.intent(from_agent="executor",
                                        intent_type="send_message",
                                        payload={}))
    with box.open("a") as fh:
        fh.write("\n")  # empty
        fh.write("notjson\n")
    envs = read_envelopes(box)
    assert len(envs) == 1


def test_iter_new_envelopes_advances_offset(tmp_path: Path):
    box = tmp_path / "outbox.jsonl"
    write_envelope(box, Envelope.intent(from_agent="executor",
                                        intent_type="send_message",
                                        payload={"i": 1}))
    pairs = list(iter_new_envelopes(box, after_offset=0))
    assert len(pairs) == 1
    new_offset, env = pairs[0]
    assert env.payload["i"] == 1
    # Following call from new_offset returns nothing yet.
    assert list(iter_new_envelopes(box, after_offset=new_offset)) == []
    # New write picked up.
    write_envelope(box, Envelope.intent(from_agent="executor",
                                        intent_type="send_message",
                                        payload={"i": 2}))
    next_pairs = list(iter_new_envelopes(box, after_offset=new_offset))
    assert len(next_pairs) == 1
    assert next_pairs[0][1].payload["i"] == 2


def test_iter_new_envelopes_skips_garbage_but_keeps_advancing(tmp_path: Path):
    box = tmp_path / "outbox.jsonl"
    box.write_text("not-json\n", encoding="utf-8")
    pairs = list(iter_new_envelopes(box, after_offset=0))
    assert pairs == []
    # Append a real envelope and ensure we now read it.
    write_envelope(box, Envelope.intent(from_agent="executor",
                                        intent_type="send_message",
                                        payload={"ok": True}))
    pairs2 = list(iter_new_envelopes(box, after_offset=0))
    # 1 valid envelope (the bad line is skipped silently).
    assert len(pairs2) == 1


def test_iter_new_envelopes_holds_position_on_partial_line(tmp_path: Path):
    box = tmp_path / "outbox.jsonl"
    # Write a complete line + an unterminated trailing partial.
    write_envelope(box, Envelope.intent(from_agent="executor",
                                        intent_type="send_message",
                                        payload={"x": 1}))
    with box.open("a", encoding="utf-8") as fh:
        fh.write('{"kind":"intent","msg_id":"x"')  # no newline
    pairs = list(iter_new_envelopes(box, after_offset=0))
    assert len(pairs) == 1  # partial line ignored


def test_envelopes_since_seq(tmp_path: Path):
    box = tmp_path / "outbox.jsonl"
    for i in range(5):
        write_envelope(box, Envelope.intent(from_agent="executor",
                                            intent_type="send_message",
                                            payload={"i": i}))
    later = envelopes_since_seq(box, after_seq=2)
    assert [e.payload["i"] for e in later] == [2, 3, 4]


# ---------------------------------------------------------------------------
# Cursor file
# ---------------------------------------------------------------------------
def test_read_write_cursor_round_trip(tmp_path: Path):
    cur = tmp_path / "agent.cursor"
    assert read_cursor(cur) == 0
    write_cursor(cur, 1234)
    assert read_cursor(cur) == 1234


def test_read_cursor_handles_garbage(tmp_path: Path):
    cur = tmp_path / "agent.cursor"
    cur.write_text("not-a-number")
    assert read_cursor(cur) == 0
