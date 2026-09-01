# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for out-of-process token-usage parsers."""

from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.trace import parse_usage as pu


# ---- coerce_optional_int ----


def test_coerce_optional_int():
    assert pu.coerce_optional_int(None) is None
    assert pu.coerce_optional_int("5") == 5
    assert pu.coerce_optional_int(7) == 7
    assert pu.coerce_optional_int("x") is None


# ---- reasoning_output_tokens ----


def test_reasoning_output_tokens_reads_every_provider_shape():
    """One helper, so the same model bills the same on every call path."""
    # Codex CLI / codex-agent metadata spelling.
    assert pu.reasoning_output_tokens({"reasoning_output_tokens": "2048"}) == 2048
    # OpenAI HTTP shape: nested under completion_tokens_details, as a mapping...
    assert pu.reasoning_output_tokens({"completion_tokens_details": {"reasoning_tokens": 512}}) == 512

    # ...and as an SDK object.
    class _Details:
        reasoning_tokens = 96

    class _Usage:
        completion_tokens = 40
        completion_tokens_details = _Details()

    assert pu.reasoning_output_tokens(_Usage()) == 96


def test_reasoning_output_tokens_absent_is_none():
    """No reasoning concept must stay None, not become 0."""
    assert pu.reasoning_output_tokens(None) is None
    assert pu.reasoning_output_tokens({}) is None
    assert pu.reasoning_output_tokens({"output_tokens": 10}) is None
    assert pu.reasoning_output_tokens({"completion_tokens_details": {}}) is None


# ---- normalize_usage ----


def test_normalize_usage_none_and_empty():
    assert pu.normalize_usage(None) is None
    assert pu.normalize_usage({}) is None
    assert pu.normalize_usage("nope") is None


def test_normalize_usage_all_none():
    assert pu.normalize_usage({"unrelated": 1}) is None


def test_normalize_usage_valid():
    out = pu.normalize_usage({"input_tokens": 10, "output_tokens": "20", "extra": 1})
    assert out == {
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": None,
    }


# ---- parse_forge_usage ----


def test_parse_forge_usage_none_without_marker():
    assert pu.parse_forge_usage("") is None
    assert pu.parse_forge_usage("forge done: baseline=1 best=1") is None


def test_parse_forge_usage_extracts_last_marker():
    stdout = (
        "noise\n"
        'FORGE_LLM_USAGE {"input_tokens": 1, "output_tokens": 2}\n'
        "more noise\n"
        'FORGE_LLM_USAGE {"input_tokens": 100, "output_tokens": 40, '
        '"cache_creation_input_tokens": 5, "cache_read_input_tokens": 9, '
        '"total_cost_usd": 3.2, "calls": 4}\n'
    )
    out = pu.parse_forge_usage(stdout)
    # Last marker wins; extra keys (cost/calls) dropped.
    assert out == {
        "input_tokens": 100,
        "output_tokens": 40,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 9,
    }


def test_parse_forge_usage_skips_malformed_marker():
    stdout = 'FORGE_LLM_USAGE not-json\nFORGE_LLM_USAGE {"input_tokens": 7}\n'
    assert pu.parse_forge_usage(stdout)["input_tokens"] == 7


# ---- parse_forge_steps ----


def test_parse_forge_steps_none_without_marker():
    assert pu.parse_forge_steps("") is None
    assert pu.parse_forge_steps("forge done") is None


def test_parse_forge_steps_extracts_timeline_and_summary():
    payload = {
        "steps": [
            {"iteration": 1, "decision": "KEEP", "wall_ms": 88.1, "snr_db": 35.0, "rationale": "fuse epilogue"},
            {"iteration": 2, "decision": "REVERT", "wall_ms": 90.0},
        ],
        "summary": {"iterations": 2, "kept": 1, "speedup": 1.05, "termination_reason": "plateaued"},
    }
    stdout = "noise\nFORGE_STEPS " + json.dumps(payload) + "\ntail\n"
    out = pu.parse_forge_steps(stdout)
    assert [s["iteration"] for s in out["steps"]] == [1, 2]
    assert out["steps"][0]["decision"] == "KEEP"
    assert out["summary"]["termination_reason"] == "plateaued"


def test_parse_forge_steps_last_marker_wins_and_skips_malformed():
    stdout = 'FORGE_STEPS not-json\nFORGE_STEPS {"steps": [{"iteration": 1}], "summary": {"iterations": 1}}\n'
    out = pu.parse_forge_steps(stdout)
    assert out["summary"]["iterations"] == 1


# ---- parse_claude_stream_json_turn_usages ----


def test_parse_turn_usages_missing(tmp_path):
    assert pu.parse_claude_stream_json_turn_usages(tmp_path / "no.log") == []


def test_parse_turn_usages_one_row_per_response_in_order(tmp_path):
    log = tmp_path / "p.log"
    log.write_text(
        '{"type": "assistant", "message": {"id": "msg_1", "usage": {"input_tokens": 10, "output_tokens": 1}}}\n'
        "garbled\n"
        '{"type": "assistant", "message": {"id": "msg_2", "usage": {"input_tokens": 20, "output_tokens": 2}}}\n'
        '{"type": "result", "usage": {"input_tokens": 30, "output_tokens": 500}}\n',
        encoding="utf-8",
    )
    usages = pu.parse_claude_stream_json_turn_usages(log)
    assert len(usages) == 2
    assert usages[0]["input_tokens"] == 10 and usages[1]["input_tokens"] == 20
    # The start-of-stream placeholders (1, 2) give way to the result row's 500,
    # which lands on the final turn.
    assert usages[0]["output_tokens"] is None
    assert usages[1]["output_tokens"] == 500


def test_parse_turn_usages_collapses_the_content_blocks_of_one_response(tmp_path):
    """One API response streams as one line per content block, sharing a usage."""
    block = (
        '{{"type": "assistant", "message": {{"id": "msg_1", "content": [{{"type": "{kind}"}}], '
        '"usage": {{"input_tokens": 10, "output_tokens": 1, '
        '"cache_creation_input_tokens": 5, "cache_read_input_tokens": 7}}}}}}\n'
    )
    log = tmp_path / "p.log"
    log.write_text(
        block.format(kind="thinking")
        + block.format(kind="text")
        + block.format(kind="tool_use")
        + '{"type": "result", "usage": {"output_tokens": 400}}\n',
        encoding="utf-8",
    )
    usages = pu.parse_claude_stream_json_turn_usages(log)
    assert len(usages) == 1
    assert usages[0]["input_tokens"] == 10
    assert usages[0]["cache_creation_input_tokens"] == 5
    assert usages[0]["cache_read_input_tokens"] == 7
    assert usages[0]["output_tokens"] == 400


def test_parse_turn_usages_prefers_model_usage_over_result_usage(tmp_path):
    """``usage`` omits Task sub-agent output; ``modelUsage`` is what cost uses."""
    log = tmp_path / "p.log"
    log.write_text(
        '{"type": "assistant", "message": {"id": "msg_1", "usage": {"output_tokens": 1}}}\n'
        '{"type": "assistant", "message": {"id": "msg_2", "usage": {"output_tokens": 1}}}\n'
        '{"type": "result", "usage": {"output_tokens": 300}, "modelUsage": {'
        '"claude-opus-4-8": {"outputTokens": 400}, "claude-haiku-4": {"outputTokens": 50}}}\n',
        encoding="utf-8",
    )
    usages = pu.parse_claude_stream_json_turn_usages(log)
    assert [u["output_tokens"] for u in usages] == [None, 450]


def test_parse_turn_usages_falls_back_to_result_usage_without_model_usage(tmp_path):
    log = tmp_path / "p.log"
    log.write_text(
        '{"type": "assistant", "message": {"id": "msg_1", "usage": {"output_tokens": 1}}}\n'
        '{"type": "assistant", "message": {"id": "msg_2", "usage": {"output_tokens": 1}}}\n'
        '{"type": "result", "usage": {"output_tokens": 300}, "modelUsage": {}}\n',
        encoding="utf-8",
    )
    usages = pu.parse_claude_stream_json_turn_usages(log)
    assert [u["output_tokens"] for u in usages] == [None, 300]


def test_parse_turn_usages_defers_to_cumulative_row_without_message_ids(tmp_path):
    """Unidentifiable rows could be duplicates; the caller must use the total."""
    log = tmp_path / "p.log"
    log.write_text(
        '{"type": "assistant", "message": {"usage": {"input_tokens": 10}}}\n'
        '{"type": "assistant", "message": {"usage": {"input_tokens": 10}}}\n'
        '{"type": "result", "usage": {"input_tokens": 10, "output_tokens": 9}}\n',
        encoding="utf-8",
    )
    assert pu.parse_claude_stream_json_turn_usages(log) == []


def test_parse_turn_usages_keeps_per_turn_output_that_reconciles(tmp_path):
    """A CLI reporting true per-response output keeps its finer attribution."""
    log = tmp_path / "p.log"
    log.write_text(
        '{"type": "assistant", "message": {"id": "msg_1", "usage": {"output_tokens": 4}}}\n'
        '{"type": "assistant", "message": {"id": "msg_2", "usage": {"output_tokens": 6}}}\n'
        '{"type": "result", "usage": {"output_tokens": 10}}\n',
        encoding="utf-8",
    )
    usages = pu.parse_claude_stream_json_turn_usages(log)
    assert [u["output_tokens"] for u in usages] == [4, 6]


def test_parse_turn_usages_drops_placeholder_output_without_a_result_row(tmp_path):
    log = tmp_path / "p.log"
    log.write_text(
        '{"type": "assistant", "message": {"id": "msg_1", "usage": {"input_tokens": 10, "output_tokens": 1}}}\n'
        '{"type": "assistant", "message": {"id": "msg_2", "usage": {"input_tokens": 20, "output_tokens": 2}}}\n',
        encoding="utf-8",
    )
    usages = pu.parse_claude_stream_json_turn_usages(log)
    assert [u["input_tokens"] for u in usages] == [10, 20]
    assert [u["output_tokens"] for u in usages] == [None, None]


def test_parse_turn_usages_none_when_no_per_message_usage(tmp_path):
    log = tmp_path / "p.log"
    log.write_text(
        '{"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}\n'
        '{"type": "result", "usage": {"input_tokens": 5}}\n',
        encoding="utf-8",
    )
    assert pu.parse_claude_stream_json_turn_usages(log) == []


# ---- parse_claude_stream_json_tool_calls ----


def test_parse_tool_calls_missing(tmp_path):
    assert pu.parse_claude_stream_json_tool_calls(tmp_path / "no.log") == []


def test_parse_tool_calls_extracts_in_order(tmp_path):
    log = tmp_path / "p.log"
    log.write_text(
        '{"type": "assistant", "message": {"content": ['
        '{"type": "text", "text": "thinking"},'
        '{"type": "tool_use", "name": "WebSearch", "input": {"query": "rocm flash attn"}},'
        '{"type": "tool_use", "name": "mcp__recipe_kb__lookup", "input": {"q": "x"}}'
        "]}}\n"
        "garbled\n"
        '{"type": "assistant", "message": {"content": ['
        '{"type": "tool_use", "name": "Read", "input": {"path": "/a/b.py"}}'
        "]}}\n"
        '{"type": "result", "result": "done"}\n',
        encoding="utf-8",
    )
    calls = pu.parse_claude_stream_json_tool_calls(log)
    assert [c["tool"] for c in calls] == [
        "WebSearch",
        "mcp__recipe_kb__lookup",
        "Read",
    ]
    assert calls[0]["query"] == "rocm flash attn"
    assert calls[2]["query"] == "/a/b.py"
    # No recognised query key -> compact JSON fallback.
    assert '"q"' in calls[1]["query"]


def test_parse_tool_calls_none_when_no_tools(tmp_path):
    log = tmp_path / "p.log"
    log.write_text(
        '{"type": "assistant", "message": {"content": [{"type": "text", "text": "no tools here"}]}}\n',
        encoding="utf-8",
    )
    assert pu.parse_claude_stream_json_tool_calls(log) == []


def test_summarize_tool_input_clips_long():
    out = pu._summarize_tool_input({"query": "a" * 500})
    assert out.endswith("…")
    assert len(out) <= 241


def test_summarize_tool_input_redacts_bearer_and_assignment():
    """Shell commands in the intel ledger must not keep credential values."""
    bearer_header = ": ".join(("Authorization", "Bearer secret-token-value"))
    bearer = pu._summarize_tool_input({"command": f"{bearer_header} https://x"})
    assert "secret-token-value" not in bearer
    assert "[REDACTED]" in bearer

    assigned = pu._summarize_tool_input({"command": "OPENAI_API_KEY=sk-live-abcdef curl https://x"})
    assert "sk-live-abcdef" not in assigned
    assert "[REDACTED]" in assigned

    # Quoting the value is the common shell shape, so it must mask too.
    quoted = pu._summarize_tool_input({"command": 'export MYAPP_PASSWORD="hunter2" && ./run.sh'})
    assert "hunter2" not in quoted
    assert "[REDACTED]" in quoted


def test_summarize_tool_input_redacts_before_clipping():
    """A secret near the clip boundary is masked on the full string first."""
    bearer_header = ": ".join(("Authorization", "Bearer secret-token-value"))
    cmd = f"{bearer_header} " + ("x" * 300)
    out = pu._summarize_tool_input({"command": cmd})
    assert "secret-token-value" not in out
    assert "[REDACTED]" in out
    assert out.endswith("…")
    assert len(out) <= 241


def test_summarize_tool_input_redacts_past_clip_inside_scan_window():
    """A secret past the 240 clip but inside the 4096 scan window is still masked."""
    cmd = ("x" * 300) + " OPENAI_API_KEY=sk-live-abcdef"
    out = pu._summarize_tool_input({"command": cmd}, limit=400)
    assert "sk-live-abcdef" not in out
    assert "[REDACTED]" in out


def test_summarize_tool_input_bounds_scan_on_huge_write_dump():
    """A megabyte Write input is still a 240-char summary, with leading secrets masked."""
    out = pu._summarize_tool_input({"content": "OPENAI_API_KEY=sk-live-abcdef " + ("a" * 50_000)})
    assert "sk-live-abcdef" not in out
    assert "[REDACTED]" in out
    assert out.endswith("…")
    assert len(out) <= 241


# ---- parse_claude_stream_json_usage ----


def test_parse_claude_usage_missing(tmp_path):
    assert pu.parse_claude_stream_json_usage(tmp_path / "no.log") is None


def test_parse_claude_usage_result_row(tmp_path):
    log = tmp_path / "p.log"
    log.write_text(
        '{"type": "assistant", "usage": {"input_tokens": 1}}\n'
        "garbled line\n"
        '{"type": "result", "usage": {"input_tokens": 5, "output_tokens": 9}}\n',
        encoding="utf-8",
    )
    out = pu.parse_claude_stream_json_usage(log)
    assert out["input_tokens"] == 5
    assert out["output_tokens"] == 9


def test_parse_claude_usage_no_usage(tmp_path):
    log = tmp_path / "p.log"
    log.write_text('{"type": "result"}\n', encoding="utf-8")
    assert pu.parse_claude_stream_json_usage(log) is None


# ---- parse_claude_stream_json_response ----


def test_parse_claude_response_result_wins(tmp_path):
    log = tmp_path / "p.log"
    log.write_text(
        '{"type": "assistant", "message": {"content": [{"type": "text", "text": "chunk"}]}}\n'
        '{"type": "result", "result": "final answer"}\n',
        encoding="utf-8",
    )
    assert pu.parse_claude_stream_json_response(log) == "final answer"


def test_parse_claude_response_assistant_fallback(tmp_path):
    log = tmp_path / "p.log"
    log.write_text(
        '{"type": "assistant", "message": {"content": [{"type": "text", "text": "a"}]}}\n'
        '{"type": "assistant", "message": {"content": [{"type": "text", "text": "b"}]}}\n',
        encoding="utf-8",
    )
    assert pu.parse_claude_stream_json_response(log) == "a\nb"


def test_parse_claude_response_missing(tmp_path):
    assert pu.parse_claude_stream_json_response(tmp_path / "no.log") is None


def test_parse_claude_response_no_text(tmp_path):
    log = tmp_path / "p.log"
    log.write_text('{"type": "system"}\n', encoding="utf-8")
    assert pu.parse_claude_stream_json_response(log) is None


# ---- parse_codex_jsonl_error ----


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            {"type": "turn.failed", "error": {"message": "The requested model is not supported"}},
            "The requested model is not supported",
        ),
        (
            {"type": "error", "message": "401 Unauthorized: invalid API key"},
            "401 Unauthorized: invalid API key",
        ),
        (
            {
                "type": "item.completed",
                "item": {
                    "id": "item_7",
                    "type": "error",
                    "message": "Your account has reached its usage limit",
                },
            },
            "Your account has reached its usage limit",
        ),
    ],
)
def test_parse_codex_error_reads_official_exec_event_shapes(tmp_path, event, expected):
    """Pin all three error shapes from the installed Codex 0.144.4 schema."""
    log = tmp_path / "codex.jsonl"
    log.write_text(json.dumps(event) + "\n", encoding="utf-8")

    assert pu.parse_codex_jsonl_error(log) == expected


def test_parse_codex_error_is_separate_from_existing_parsers(tmp_path):
    log = tmp_path / "codex.jsonl"
    log.write_text(
        '{"type":"turn.failed","error":{"message":"gateway authentication failed"}}\n',
        encoding="utf-8",
    )

    assert pu.parse_codex_jsonl_usage(log) is None
    assert pu.parse_codex_jsonl_response(log) is None
    assert pu.parse_codex_jsonl_tool_calls(log) == []
    assert pu.parse_codex_jsonl_error(log) == "gateway authentication failed"


def test_parse_codex_error_prefers_turn_failure_over_later_lower_authority_events(tmp_path):
    log = tmp_path / "codex.jsonl"
    events = [
        {
            "type": "item.completed",
            "item": {"id": "item_1", "type": "error", "message": "retrying gateway request"},
        },
        {"type": "error", "message": "gateway request failed"},
        {"type": "turn.failed", "error": {"message": "model access is not permitted"}},
        {"type": "error", "message": "late transport shutdown"},
    ]
    log.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

    assert pu.parse_codex_jsonl_error(log) == "model access is not permitted"


def test_parse_codex_error_uses_last_failure_at_same_authority(tmp_path):
    log = tmp_path / "codex.jsonl"
    log.write_text(
        '{"type":"turn.failed","error":{"message":"first turn failed"}}\n'
        '{"type":"turn.failed","error":{"message":"second turn failed"}}\n',
        encoding="utf-8",
    )

    assert pu.parse_codex_jsonl_error(log) == "second turn failed"


def test_parse_codex_error_tolerates_sdk_error_wrapper(tmp_path):
    log = tmp_path / "codex.jsonl"
    log.write_text(
        '{"type":"error","error":{"message":"nested gateway timeout"}}\n',
        encoding="utf-8",
    )

    assert pu.parse_codex_jsonl_error(log) == "nested gateway timeout"


def test_parse_codex_error_skips_malformed_lines_and_ordinary_messages(tmp_path):
    log = tmp_path / "codex.jsonl"
    log.write_text(
        '{"type":"item.completed","item":{"id":"item_1","type":"reasoning","text":"Could be an auth error"}}\n'
        '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"Error: try another model"}}\n'
        '{"type":"error","message":"gateway unavailable"}\n'
        '{"type":"turn.failed","error":\n'
        "not-json\n",
        encoding="utf-8",
    )

    assert pu.parse_codex_jsonl_error(log) == "gateway unavailable"


def test_parse_codex_error_returns_none_for_missing_file(tmp_path):
    assert pu.parse_codex_jsonl_error(tmp_path / "missing.jsonl") is None


def test_parse_codex_error_redacts_credentials_and_ignores_request_payload(tmp_path):
    message_secret = "sk-proj-messageSecret123456"
    payload_secret = "sk-proj-requestSecret987654"
    bearer_secret = "gatewayBearerSecret123456"
    event = {
        "type": "turn.failed",
        "error": {
            "message": (f"Gateway rejected OPENAI_API_KEY={message_secret}; Authorization: Bearer {bearer_secret}"),
            "request": {
                "headers": {"Authorization": f"Bearer {payload_secret}"},
                "config": {"api_key": payload_secret},
            },
        },
    }
    log = tmp_path / "codex.jsonl"
    log.write_text(json.dumps(event) + "\n", encoding="utf-8")

    parsed = pu.parse_codex_jsonl_error(log)

    assert parsed is not None
    assert "[REDACTED]" in parsed
    assert message_secret not in parsed
    assert bearer_secret not in parsed
    assert payload_secret not in parsed
    assert "request" not in parsed


def test_parse_codex_error_is_exported_from_trace_package(tmp_path):
    from hyperloom.orchestrator.trace import parse_codex_jsonl_error

    assert parse_codex_jsonl_error(tmp_path / "missing.jsonl") is None
