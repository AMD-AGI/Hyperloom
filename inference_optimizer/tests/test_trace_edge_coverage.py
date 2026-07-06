# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Edge-path coverage for the trace parsers and Langfuse mapping helpers.

These exercise the tolerant/degenerate branches (blank lines, malformed JSON,
non-dict rows, unreadable paths, and non-numeric fields) that the happy-path
unit tests don't reach, so a parse miss provably degrades to an empty/``None``
result instead of raising on the best-effort trace path.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hyperloom.orchestrator.trace import langfuse_mapping as lm
from hyperloom.orchestrator.trace import parse_usage as pu


# ---- parse_usage: stream-json usage parser edge paths ----


def test_stream_json_usage_skips_blank_and_nondict_lines(tmp_path):
    p = tmp_path / "process.log"
    p.write_text(
        "\n"  # blank line -> skipped
        "not-json\n"  # malformed -> skipped
        "[1, 2, 3]\n"  # valid JSON but not a dict -> skipped
        '{"type": "result", "usage": {"input_tokens": 5, "output_tokens": 7}}\n'
    )
    out = pu.parse_claude_stream_json_usage(p)
    assert out["input_tokens"] == 5
    assert out["output_tokens"] == 7


def test_stream_json_usage_dir_path_is_oserror(tmp_path):
    # A directory triggers IsADirectoryError (an OSError) -> tolerant None.
    assert pu.parse_claude_stream_json_usage(tmp_path) is None


# ---- parse_usage: stream-json response parser edge paths ----


def test_stream_json_response_skips_blank_nondict_and_bad_message(tmp_path):
    p = tmp_path / "process.log"
    p.write_text(
        "\n"  # blank -> skipped
        "not-json\n"  # malformed -> skipped
        "42\n"  # non-dict -> skipped
        '{"type": "assistant", "message": "oops"}\n'  # message not a dict
        '{"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}\n'
    )
    assert pu.parse_claude_stream_json_response(p) == "hi"


def test_stream_json_response_dir_path_is_oserror(tmp_path):
    assert pu.parse_claude_stream_json_response(tmp_path) is None


# ---- parse_usage: per-turn usage parser edge paths ----


def test_stream_json_turn_usages_skips_blank_lines(tmp_path):
    p = tmp_path / "process.log"
    p.write_text(
        "\n"
        '{"type": "assistant", "message": {"usage": {"input_tokens": 1}}}\n'
    )
    out = pu.parse_claude_stream_json_turn_usages(p)
    assert len(out) == 1 and out[0]["input_tokens"] == 1


def test_stream_json_turn_usages_dir_path_is_oserror(tmp_path):
    assert pu.parse_claude_stream_json_turn_usages(tmp_path) == []


# ---- parse_usage: tool-call parser edge paths ----


def test_stream_json_tool_calls_skips_blank_bad_message_and_unnamed(tmp_path):
    p = tmp_path / "process.log"
    p.write_text(
        "\n"  # blank -> skipped
        '{"type": "assistant", "message": "oops"}\n'  # message not a dict
        '{"type": "assistant", "message": {"content": '
        '[{"type": "tool_use", "name": ""}]}}\n'  # empty name -> skipped
        '{"type": "assistant", "message": {"content": '
        '[{"type": "tool_use", "name": "Grep", "input": {"pattern": "x"}}]}}\n'
    )
    out = pu.parse_claude_stream_json_tool_calls(p)
    assert out == [{"tool": "Grep", "query": "x"}]


def test_stream_json_tool_calls_dir_path_is_oserror(tmp_path):
    assert pu.parse_claude_stream_json_tool_calls(tmp_path) == []


# ---- parse_usage: tool-input summarizer fallbacks ----


def test_summarize_tool_input_unserializable_dict_falls_back_to_str():
    # No query-ish key + an unserializable value -> json.dumps raises -> str().
    out = pu._summarize_tool_input({"obj": object()})
    assert isinstance(out, str) and out


def test_summarize_tool_input_non_dict_values():
    assert pu._summarize_tool_input(None) == ""
    assert pu._summarize_tool_input(123) == "123"


# ---- parse_usage: oob + forge marker edge paths ----


def test_oob_usage_line_by_line_skips_blank(tmp_path):
    # Whole-document parse fails, then the JSONL scan skips a blank line.
    stdout = 'preamble not json\n\n{"usage": {"input_tokens": 3}}\n'
    out = pu.parse_oob_json_usage(stdout)
    assert out["input_tokens"] == 3


def test_forge_usage_marker_with_empty_blob_is_none():
    assert pu.parse_forge_usage("FORGE_LLM_USAGE") is None


def test_forge_steps_marker_with_empty_blob_is_none():
    assert pu.parse_forge_steps("FORGE_STEPS") is None


# ---- langfuse_mapping: pure helper edge paths ----


def test_span_key():
    assert lm.span_key({"phase": "EXPLORE", "component": "critic"}) == ("EXPLORE", "critic")


def test_parse_ts_missing_and_unparseable():
    assert lm.parse_ts(None) is None
    assert lm.parse_ts("") is None
    assert lm.parse_ts("not-a-timestamp") is None


def test_generation_start_fallbacks():
    end = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert lm.generation_start(None, 100) is None  # no end
    assert lm.generation_start(end, "bad") == end  # unparseable latency
    assert lm.generation_start(end, 0) == end  # non-positive latency
    assert lm.generation_start(end, -5) == end
    assert lm.generation_start(end, 10**25) == end  # timedelta overflow


def test_utc_second_key_unparseable_is_empty():
    assert lm.utc_second_key(None) == ""
    assert lm.utc_second_key("garbage") == ""


def test_decision_to_scores_non_numeric_gain_and_predicted():
    scores = lm.decision_to_scores({
        "decision": {
            "outcome": "KEEP",
            "change": "x",
            "gain_pct": "not-a-number",
            "predicted_gain_pct": "also-bad",
        }
    })
    # Only the categorical outcome score survives; bad numerics are dropped.
    assert [s["name"] for s in scores] == ["decision_outcome"]


def test_mean_proposal_score_edge_cases():
    assert lm._mean_proposal_score("not-a-list") is None
    assert lm._mean_proposal_score([]) is None
    # Non-dict entries and non-numeric scores are skipped -> no values -> None.
    assert lm._mean_proposal_score([123, {"score": "abc"}]) is None
    # A mix with one valid numeric score yields its mean.
    assert lm._mean_proposal_score([{"score": "8"}, {"nope": 1}]) == 8.0
