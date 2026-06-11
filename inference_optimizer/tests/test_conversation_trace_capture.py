# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Conversation-capture coverage for the production specialist / critic paths.

The token ledger (``llm_calls.jsonl``) already captured every component, but
``conversations.jsonl`` only carried the in-process orchestration / kernel
turns: the production-default specialist (subprocess) and the critic recovered
their *tokens* but never their *text*. These tests pin the gap fix:

* :func:`parse_claude_stream_json_response` recovers the assistant reply from
  the same stream-json ``process.log`` the usage parser reads (full-trace B1).
* the recovered reply, paired with the parent-held prompt, lands a
  ``component=specialist`` row in ``conversations.jsonl``.
* the critic reasoning loop lands a ``component=critic`` row.
"""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.orchestrator.trace.conversation_trace import (
    ConversationRecord,
    append_conversation,
)
from inference_optimizer.orchestrator.trace.parse_usage import (
    parse_claude_stream_json_response,
)
from inference_optimizer.session_paths import conversations_path


# ---------------------------------------------------------------------------
# parse_claude_stream_json_response
# ---------------------------------------------------------------------------
def _write_stream_json(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_response_prefers_result_row(tmp_path: Path) -> None:
    """The terminal ``result`` row is the authoritative consolidated reply."""
    log = tmp_path / "process.log"
    _write_stream_json(log, [
        {"type": "system", "subtype": "init", "session_id": "s"},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "partial streamed chunk"},
        ]}},
        {"type": "result", "subtype": "success",
         "result": "FINAL consolidated answer", "usage": {"output_tokens": 5}},
    ])
    assert parse_claude_stream_json_response(log) == "FINAL consolidated answer"


def test_response_falls_back_to_assistant_text(tmp_path: Path) -> None:
    """Without a ``result`` row, concatenate assistant ``text`` blocks in order."""
    log = tmp_path / "process.log"
    _write_stream_json(log, [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "first"},
        ]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "second"},
        ]}},
    ])
    assert parse_claude_stream_json_response(log) == "first\nsecond"


def test_response_skips_thinking_and_tool_use(tmp_path: Path) -> None:
    """Only externally-visible ``text`` blocks count; thinking / tool_use drop."""
    log = tmp_path / "process.log"
    _write_stream_json(log, [
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "secret scratch reasoning"},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
            {"type": "text", "text": "visible reply"},
        ]}},
    ])
    out = parse_claude_stream_json_response(log)
    assert out == "visible reply"
    assert "scratch reasoning" not in out
    assert "Bash" not in out


def test_response_missing_file_returns_none(tmp_path: Path) -> None:
    assert parse_claude_stream_json_response(tmp_path / "nope.log") is None


def test_response_no_text_returns_none(tmp_path: Path) -> None:
    """A stream that only ran tools (no text, no result) yields None."""
    log = tmp_path / "process.log"
    _write_stream_json(log, [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        ]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "ok"},
        ]}},
    ])
    assert parse_claude_stream_json_response(log) is None


def test_response_tolerates_malformed_lines(tmp_path: Path) -> None:
    """Garbage lines are skipped; valid result still recovered."""
    log = tmp_path / "process.log"
    log.write_text(
        "not json at all\n"
        + json.dumps({"type": "result", "result": "recovered"})
        + "\n{ broken json\n",
        encoding="utf-8",
    )
    assert parse_claude_stream_json_response(log) == "recovered"


def test_response_blank_result_falls_back(tmp_path: Path) -> None:
    """An empty/whitespace ``result`` does not mask real assistant text."""
    log = tmp_path / "process.log"
    _write_stream_json(log, [
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "real reply"},
        ]}},
        {"type": "result", "result": "   "},
    ])
    assert parse_claude_stream_json_response(log) == "real reply"


# ---------------------------------------------------------------------------
# conversations.jsonl round trip for specialist + critic components
# ---------------------------------------------------------------------------
def _read_rows(session_dir: Path) -> list[dict]:
    path = conversations_path(session_dir)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_specialist_conversation_row_round_trip(tmp_path: Path) -> None:
    """A specialist subprocess turn (prompt paired with parsed reply) lands a
    schema-valid ``component=specialist`` row."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    append_conversation(
        session_dir=session_dir,
        record=ConversationRecord(
            session_id=session_dir.name,
            component="specialist",
            task_id="task-abc",
            turn=1,
            model="claude-opus-4-7",
            prompt="SYSTEM\n---\nUSER PROMPT",
            response="ranked 3 proposals",
        ),
    )
    rows = _read_rows(session_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["component"] == "specialist"
    assert row["task_id"] == "task-abc"
    assert row["prompt"] == "SYSTEM\n---\nUSER PROMPT"
    assert row["response"] == "ranked 3 proposals"
    assert "ts" in row


def test_critic_conversation_row_round_trip(tmp_path: Path) -> None:
    """A critic reasoning loop lands a schema-valid ``component=critic`` row."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    append_conversation(
        session_dir=session_dir,
        record=ConversationRecord(
            session_id=session_dir.name,
            component="critic",
            role="critic",
            model="gpt-5.4",
            prompt="SYS\n---\njudge bundle",
            response='{"review_verdicts": []}',
        ),
    )
    rows = _read_rows(session_dir)
    assert len(rows) == 1
    assert rows[0]["component"] == "critic"
    assert rows[0]["role"] == "critic"
    assert rows[0]["response"] == '{"review_verdicts": []}'


def test_secret_redaction_applies_to_recovered_text(tmp_path: Path) -> None:
    """Recovered prompt/response still passes through secret redaction before
    hitting disk (a leaked key in the model's text never lands raw)."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    append_conversation(
        session_dir=session_dir,
        record=ConversationRecord(
            session_id=session_dir.name,
            component="specialist",
            prompt="export OPENAI_API_KEY=sk-secretvalue123456",
            response="Authorization: Bearer abcdef1234567890",
        ),
    )
    raw = conversations_path(session_dir).read_text(encoding="utf-8")
    assert "sk-secretvalue123456" not in raw
    assert "abcdef1234567890" not in raw
    assert "[REDACTED]" in raw
