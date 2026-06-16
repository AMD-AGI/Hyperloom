# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for out-of-process token-usage parsers."""

from __future__ import annotations

import json

from inference_optimizer.orchestrator.trace import parse_usage as pu


# ---- _coerce_optional_int ----

def test_coerce_optional_int():
    assert pu._coerce_optional_int(None) is None
    assert pu._coerce_optional_int("5") == 5
    assert pu._coerce_optional_int(7) == 7
    assert pu._coerce_optional_int("x") is None


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
        "input_tokens": 10, "output_tokens": 20,
        "cache_creation_input_tokens": None, "cache_read_input_tokens": None,
    }


# ---- parse_claude_stream_json_turn_usages ----

def test_parse_turn_usages_missing(tmp_path):
    assert pu.parse_claude_stream_json_turn_usages(tmp_path / "no.log") == []


def test_parse_turn_usages_per_message_in_order(tmp_path):
    log = tmp_path / "p.log"
    log.write_text(
        '{"type": "assistant", "message": {"usage": {"input_tokens": 10, "output_tokens": 3}}}\n'
        "garbled\n"
        '{"type": "assistant", "message": {"usage": {"input_tokens": 20, "output_tokens": 7}}}\n'
        # The terminal cumulative result row is intentionally ignored here.
        '{"type": "result", "usage": {"input_tokens": 30, "output_tokens": 10}}\n',
        encoding="utf-8",
    )
    usages = pu.parse_claude_stream_json_turn_usages(log)
    assert len(usages) == 2
    assert usages[0]["input_tokens"] == 10 and usages[0]["output_tokens"] == 3
    assert usages[1]["input_tokens"] == 20 and usages[1]["output_tokens"] == 7


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
        '{"type": "tool_use", "name": "mcp__cortex_kb__lookup", "input": {"q": "x"}}'
        ']}}\n'
        "garbled\n"
        '{"type": "assistant", "message": {"content": ['
        '{"type": "tool_use", "name": "Read", "input": {"path": "/a/b.py"}}'
        ']}}\n'
        '{"type": "result", "result": "done"}\n',
        encoding="utf-8",
    )
    calls = pu.parse_claude_stream_json_tool_calls(log)
    assert [c["tool"] for c in calls] == [
        "WebSearch", "mcp__cortex_kb__lookup", "Read",
    ]
    assert calls[0]["query"] == "rocm flash attn"
    assert calls[2]["query"] == "/a/b.py"
    # No recognised query key -> compact JSON fallback.
    assert '"q"' in calls[1]["query"]


def test_parse_tool_calls_none_when_no_tools(tmp_path):
    log = tmp_path / "p.log"
    log.write_text(
        '{"type": "assistant", "message": {"content": ['
        '{"type": "text", "text": "no tools here"}]}}\n',
        encoding="utf-8",
    )
    assert pu.parse_claude_stream_json_tool_calls(log) == []


def test_summarize_tool_input_clips_long():
    out = pu._summarize_tool_input({"query": "a" * 500})
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


# ---- parse_oob_json_usage ----

def test_parse_oob_empty():
    assert pu.parse_oob_json_usage("") is None
    assert pu.parse_oob_json_usage("   ") is None


def test_parse_oob_whole_doc():
    out = pu.parse_oob_json_usage(json.dumps({"usage": {"input_tokens": 3}}))
    assert out["input_tokens"] == 3


def test_parse_oob_jsonl_fallback():
    stdout = "log noise\n" + json.dumps({"result": {"usage": {"output_tokens": 8}}})
    out = pu.parse_oob_json_usage(stdout)
    assert out["output_tokens"] == 8


def test_parse_oob_nothing():
    assert pu.parse_oob_json_usage("no json here") is None


# ---- parse_geak_usage ----

def test_parse_geak_none_and_empty():
    assert pu.parse_geak_usage(None) is None
    assert pu.parse_geak_usage("") is None
    assert pu.parse_geak_usage("{bad json") is None


def test_parse_geak_translates_openai_names():
    out = pu.parse_geak_usage({"usage": {"prompt_tokens": 11, "completion_tokens": 22}})
    assert out["input_tokens"] == 11
    assert out["output_tokens"] == 22


def test_parse_geak_json_string():
    out = pu.parse_geak_usage(json.dumps({"usage": {"input_tokens": 4}}))
    assert out["input_tokens"] == 4


def test_parse_geak_no_usage():
    assert pu.parse_geak_usage({"foo": 1}) is None


# ---- _find_usage_in_obj ----

def test_find_usage_token_usage_key():
    assert pu._find_usage_in_obj({"token_usage": {"x": 1}}) == {"x": 1}


def test_find_usage_nested_list():
    obj = {"choices": [{"message": {"usage": {"input_tokens": 1}}}]}
    assert pu._find_usage_in_obj(obj) == {"input_tokens": 1}


def test_find_usage_depth_limit():
    assert pu._find_usage_in_obj({"usage": {"x": 1}}, _depth=5) is None


def test_find_usage_non_dict():
    assert pu._find_usage_in_obj("string") is None
