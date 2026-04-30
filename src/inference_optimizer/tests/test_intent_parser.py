"""Tests for ``orchestrator.intent_parser``."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from inference_optimizer.orchestrator.intent_parser import (
    EMIT_INTENT_TOOL_SCHEMA,
    INTENT_ENVELOPE_SCHEMA,
    Intent,
    IntentType,
    IntentValidationError,
    NoIntentEmitted,
    parse_claude_trajectory,
    parse_codex_validated_json,
    validate_envelope,
)


# ---------------------------------------------------------------------------
# validate_envelope
# ---------------------------------------------------------------------------
def test_envelope_schema_lists_all_intent_types():
    enum_in_schema = INTENT_ENVELOPE_SCHEMA["properties"]["intents"]["items"]["properties"]["intent_type"]["enum"]
    enum_in_python = [t.value for t in IntentType]
    assert sorted(enum_in_schema) == sorted(enum_in_python)


def test_emit_intent_tool_has_required_fields():
    assert EMIT_INTENT_TOOL_SCHEMA["name"] == "emit_intent"
    assert EMIT_INTENT_TOOL_SCHEMA["input_schema"]["required"] == ["intent_type", "payload"]


def test_validate_envelope_round_trip():
    envelope = {
        "intents": [
            {
                "intent_type": "send_message",
                "payload": {"topic": "heartbeat", "body_md": "hi"},
            }
        ]
    }
    intents = validate_envelope(envelope)
    assert len(intents) == 1
    assert intents[0].type == IntentType.SEND_MESSAGE
    assert intents[0].payload["topic"] == "heartbeat"


def test_validate_envelope_unknown_intent_type():
    with pytest.raises(IntentValidationError) as exc:
        validate_envelope(
            {"intents": [{"intent_type": "telepathy", "payload": {}}]}
        )
    assert "unknown intent_type" in str(exc.value)


def test_validate_envelope_missing_required_payload_fields():
    # propose_action requires both action_name and predicted_gain_pct
    with pytest.raises(IntentValidationError) as exc:
        validate_envelope(
            {
                "intents": [
                    {
                        "intent_type": "propose_action",
                        "payload": {"action_name": "kernel-opt"},
                    }
                ]
            }
        )
    assert "predicted_gain_pct" in str(exc.value)


def test_validate_envelope_empty_intents_array():
    with pytest.raises(IntentValidationError):
        validate_envelope({"intents": []})


def test_validate_envelope_extra_top_level_keys():
    with pytest.raises(IntentValidationError):
        validate_envelope(
            {
                "intents": [
                    {"intent_type": "alert", "payload": {"severity": "info", "summary": "x"}}
                ],
                "extra": 1,
            }
        )


def test_validate_envelope_payload_must_be_object():
    with pytest.raises(IntentValidationError):
        validate_envelope(
            {"intents": [{"intent_type": "alert", "payload": "string"}]}
        )


def test_validate_envelope_collects_multiple_intents():
    envelope = {
        "intents": [
            {"intent_type": "send_message", "payload": {"topic": "heartbeat"}},
            {"intent_type": "alert", "payload": {"severity": "info", "summary": "x"}},
        ]
    }
    intents = validate_envelope(envelope)
    assert len(intents) == 2
    assert intents[1].type == IntentType.ALERT


# ---------------------------------------------------------------------------
# parse_codex_validated_json
# ---------------------------------------------------------------------------
def test_codex_parses_plain_json():
    text = json.dumps(
        {"intents": [{"intent_type": "alert", "payload": {"severity": "info", "summary": "hello"}}]}
    )
    intents = parse_codex_validated_json(text)
    assert len(intents) == 1


def test_codex_parses_fenced_json_block():
    text = (
        "Here's my analysis:\n"
        "```json\n"
        '{"intents": [{"intent_type": "alert", "payload": {"severity": "warn", "summary": "x"}}]}\n'
        "```\n"
        "Done."
    )
    intents = parse_codex_validated_json(text)
    assert len(intents) == 1
    assert intents[0].type == IntentType.ALERT


def test_codex_parses_brace_fragment_after_prose():
    text = (
        "let me think...\n\n"
        '{"intents": [{"intent_type": "alert", "payload": {"severity": "high", "summary": "x"}}]}\n\n'
        "ok done"
    )
    intents = parse_codex_validated_json(text)
    assert len(intents) == 1
    assert intents[0].type == IntentType.ALERT


def test_codex_no_json_raises():
    with pytest.raises(IntentValidationError):
        parse_codex_validated_json("just regular prose, nothing structured")


def test_codex_empty_string_raises():
    with pytest.raises(IntentValidationError):
        parse_codex_validated_json("")


# ---------------------------------------------------------------------------
# parse_claude_trajectory
# ---------------------------------------------------------------------------
@dataclass
class _ToolUseBlock:
    name: str
    input: dict


@dataclass
class _AssistantMessage:
    content: list


def test_claude_parses_tool_use_blocks():
    trajectory = [
        _AssistantMessage(content=[
            _ToolUseBlock(
                name="emit_intent",
                input={"intent_type": "send_message", "payload": {"topic": "heartbeat"}},
            ),
            _ToolUseBlock(
                name="emit_intent",
                input={"intent_type": "alert", "payload": {"severity": "info", "summary": "hi"}},
            ),
        ]),
    ]
    intents = parse_claude_trajectory(trajectory)
    assert len(intents) == 2
    assert {i.type for i in intents} == {IntentType.SEND_MESSAGE, IntentType.ALERT}


def test_claude_accepts_dict_shaped_blocks():
    """Dict-shaped tool_use blocks (e.g. JSONL replay) are also parsed."""
    trajectory = [
        {
            "content": [
                {
                    "type": "tool_use",
                    "name": "emit_intent",
                    "input": {
                        "intent_type": "kill_task",
                        "payload": {"task_id": "abc123", "reason": "stuck"},
                    },
                }
            ]
        }
    ]
    intents = parse_claude_trajectory(trajectory)
    assert intents[0].type == IntentType.KILL_TASK
    assert intents[0].payload["task_id"] == "abc123"


def test_claude_unpacks_full_envelope_in_single_block():
    """Some prompts cause Claude to dump the whole envelope in one call."""
    trajectory = [
        _AssistantMessage(content=[
            _ToolUseBlock(
                name="emit_intent",
                input={
                    "intents": [
                        {"intent_type": "send_message", "payload": {"topic": "heartbeat"}},
                        {"intent_type": "alert", "payload": {"severity": "warn", "summary": "x"}},
                    ]
                },
            ),
        ]),
    ]
    intents = parse_claude_trajectory(trajectory)
    assert len(intents) == 2


def test_claude_falls_back_to_text_when_no_tool_use():
    fallback = json.dumps(
        {"intents": [{"intent_type": "alert", "payload": {"severity": "info", "summary": "x"}}]}
    )
    intents = parse_claude_trajectory([], fallback_text=fallback)
    assert len(intents) == 1


def test_claude_no_tool_use_no_fallback_raises():
    with pytest.raises(NoIntentEmitted):
        parse_claude_trajectory([])


def test_claude_ignores_non_emit_intent_tool_uses():
    trajectory = [
        _AssistantMessage(content=[
            _ToolUseBlock(name="Read", input={"path": "/tmp"}),
            _ToolUseBlock(
                name="emit_intent",
                input={"intent_type": "alert", "payload": {"severity": "info", "summary": "x"}},
            ),
        ]),
    ]
    intents = parse_claude_trajectory(trajectory)
    assert len(intents) == 1
    assert intents[0].type == IntentType.ALERT


# ---------------------------------------------------------------------------
# Qualified-name parsing — the Claude SDK rewrites in-process MCP tool names
# as ``mcp__<server>__<tool>`` before exposing them to the model, so the
# trajectory we get back carries that qualified shape. The parser must accept
# both forms (regression for the v0.7 silent-no-intents bug).
# ---------------------------------------------------------------------------
def test_claude_parses_mcp_qualified_tool_name():
    """Object-shape blocks with the ``mcp__<server>__emit_intent`` name."""
    trajectory = [
        _AssistantMessage(content=[
            _ToolUseBlock(
                name="mcp__inference_optimizer__emit_intent",
                input={"intent_type": "send_message", "payload": {"topic": "ack"}},
            ),
            _ToolUseBlock(
                name="mcp__inference_optimizer__emit_intent",
                input={
                    "intent_type": "propose_action",
                    "payload": {"action_name": "run_baseline", "predicted_gain_pct": 0},
                },
            ),
        ]),
    ]
    intents = parse_claude_trajectory(trajectory)
    assert {i.type for i in intents} == {
        IntentType.SEND_MESSAGE, IntentType.PROPOSE_ACTION,
    }


def test_claude_parses_mcp_qualified_tool_name_dict_shape():
    """Dict-shape replay of qualified MCP tool calls."""
    trajectory = [
        {
            "content": [
                {
                    "type": "tool_use",
                    "name": "mcp__inference_optimizer__emit_intent",
                    "input": {
                        "intent_type": "alert",
                        "payload": {"severity": "warn", "summary": "x"},
                    },
                }
            ]
        }
    ]
    intents = parse_claude_trajectory(trajectory)
    assert len(intents) == 1
    assert intents[0].type == IntentType.ALERT


def test_claude_parses_mcp_qualified_tool_name_top_level_dict():
    """Top-level dict with ``type=tool_use`` (no nested content)."""
    trajectory = [
        {
            "type": "tool_use",
            "name": "mcp__inference_optimizer__emit_intent",
            "input": {"intent_type": "send_message", "payload": {"topic": "hi"}},
        }
    ]
    intents = parse_claude_trajectory(trajectory)
    assert len(intents) == 1
    assert intents[0].type == IntentType.SEND_MESSAGE


def test_claude_qualified_name_independent_of_server_name():
    """Any ``mcp__<server>__emit_intent`` is accepted — defends against future
    server-name renames (the parser must not hard-code the server name)."""
    trajectory = [
        _AssistantMessage(content=[
            _ToolUseBlock(
                name="mcp__some_other_server__emit_intent",
                input={"intent_type": "send_message", "payload": {"topic": "hi"}},
            ),
        ]),
    ]
    intents = parse_claude_trajectory(trajectory)
    assert len(intents) == 1
    assert intents[0].type == IntentType.SEND_MESSAGE


def test_claude_rejects_lookalike_qualified_names():
    """Names that merely *contain* ``emit_intent`` substring must NOT match —
    only the bare name and the ``mcp__<server>__emit_intent`` shape do."""
    trajectory = [
        _AssistantMessage(content=[
            _ToolUseBlock(name="my_emit_intent", input={}),
            _ToolUseBlock(name="emit_intent_v2", input={}),
            _ToolUseBlock(name="mcp__server__other_tool", input={}),
        ]),
    ]
    with pytest.raises(NoIntentEmitted):
        parse_claude_trajectory(trajectory)


def test_claude_mixed_qualified_and_bare_names_in_one_trajectory():
    """Real-world case: SDK upgrade lands mid-session and the trajectory
    contains both shapes. Both should be picked up."""
    trajectory = [
        _AssistantMessage(content=[
            _ToolUseBlock(
                name="emit_intent",
                input={"intent_type": "send_message", "payload": {"topic": "a"}},
            ),
            _ToolUseBlock(
                name="mcp__inference_optimizer__emit_intent",
                input={"intent_type": "alert", "payload": {"severity": "info", "summary": "b"}},
            ),
        ]),
    ]
    intents = parse_claude_trajectory(trajectory)
    assert {i.type for i in intents} == {IntentType.SEND_MESSAGE, IntentType.ALERT}


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE — agent-to-agent RPC (kernel agent contract)
# ---------------------------------------------------------------------------
def test_validate_request_envelope_round_trip():
    envelope = {
        "intents": [
            {
                "intent_type": "request",
                "payload": {
                    "target_agent": "kernel",
                    "kind": "select_kernels",
                    "params": {"trace_path": "/tmp/trace.json.gz"},
                },
            }
        ]
    }
    intents = validate_envelope(envelope)
    assert len(intents) == 1
    assert intents[0].type == IntentType.REQUEST
    assert intents[0].payload["target_agent"] == "kernel"
    assert intents[0].payload["kind"] == "select_kernels"


def test_validate_response_envelope_round_trip():
    envelope = {
        "intents": [
            {
                "intent_type": "response",
                "payload": {
                    "in_reply_to": "abc123",
                    "kind": "select_kernels_done",
                    "status": "succeeded",
                    "result": {"candidates": ["foo.py", "bar.py"]},
                },
            }
        ]
    }
    intents = validate_envelope(envelope)
    assert len(intents) == 1
    assert intents[0].type == IntentType.RESPONSE
    assert intents[0].payload["in_reply_to"] == "abc123"


def test_request_missing_target_agent_rejected():
    with pytest.raises(IntentValidationError) as exc:
        validate_envelope(
            {"intents": [{"intent_type": "request", "payload": {"kind": "select_kernels"}}]}
        )
    assert "target_agent" in str(exc.value)


def test_request_missing_kind_rejected():
    with pytest.raises(IntentValidationError) as exc:
        validate_envelope(
            {"intents": [{"intent_type": "request", "payload": {"target_agent": "kernel"}}]}
        )
    assert "kind" in str(exc.value)


def test_response_missing_in_reply_to_rejected():
    with pytest.raises(IntentValidationError) as exc:
        validate_envelope(
            {"intents": [{"intent_type": "response", "payload": {"kind": "done"}}]}
        )
    assert "in_reply_to" in str(exc.value)


def test_emit_intent_tool_description_mentions_request_response():
    """Description text steers Claude to emit the right payload shape."""
    desc = EMIT_INTENT_TOOL_SCHEMA["input_schema"]["properties"]["payload"]["description"]
    assert "request" in desc and "target_agent" in desc
    assert "response" in desc and "in_reply_to" in desc


# ---------------------------------------------------------------------------
# KILL_TASK — v0.4 MVP triage-only intent
# ---------------------------------------------------------------------------
def test_validate_kill_task_envelope_round_trip():
    envelope = {
        "intents": [
            {
                "intent_type": "kill_task",
                "payload": {
                    "task_id": "abc123def456",
                    "reason": "stuck for 4x lease_ttl",
                    "scope": "task",
                    "force": False,
                },
            }
        ]
    }
    intents = validate_envelope(envelope)
    assert len(intents) == 1
    assert intents[0].type == IntentType.KILL_TASK
    assert intents[0].payload["task_id"] == "abc123def456"


def test_kill_task_missing_task_id_rejected():
    with pytest.raises(IntentValidationError) as exc:
        validate_envelope(
            {"intents": [{"intent_type": "kill_task", "payload": {"reason": "x"}}]}
        )
    assert "task_id" in str(exc.value)


def test_kill_task_missing_reason_rejected():
    with pytest.raises(IntentValidationError) as exc:
        validate_envelope(
            {"intents": [{"intent_type": "kill_task", "payload": {"task_id": "abc"}}]}
        )
    assert "reason" in str(exc.value)


def test_objection_vote_no_longer_in_intent_type_enum():
    """v0.4 MVP — parliament removed; OBJECTION/VOTE deleted from enum."""
    values = {t.value for t in IntentType}
    assert "objection" not in values
    assert "vote" not in values
    # KILL_TASK was added in v0.4
    assert "kill_task" in values


def test_emit_intent_tool_description_mentions_kill_task():
    desc = EMIT_INTENT_TOOL_SCHEMA["input_schema"]["properties"]["payload"]["description"]
    assert "kill_task" in desc
