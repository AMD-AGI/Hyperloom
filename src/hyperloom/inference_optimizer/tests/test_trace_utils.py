# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Edge-path coverage for the trace parsers and Langfuse mapping helpers.

Exercises the tolerant/degenerate branches (blank lines, malformed JSON,
non-dict rows, unreadable paths, and non-numeric fields) so a parse miss
degrades to an empty/``None`` result instead of raising.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hyperloom.orchestrator.trace import _row_utils as ru
from hyperloom.orchestrator.trace import conversation_trace as ct
from hyperloom.orchestrator.trace import langfuse_mapping as lm
from hyperloom.orchestrator.trace import parse_usage as pu


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


def test_summarize_tool_input_unserializable_dict_falls_back_to_str():
    # No query-ish key + an unserializable value -> json.dumps raises -> str().
    out = pu._summarize_tool_input({"obj": object()})
    assert isinstance(out, str) and out


def test_summarize_tool_input_non_dict_values():
    assert pu._summarize_tool_input(None) == ""
    assert pu._summarize_tool_input(123) == "123"


def test_forge_usage_marker_with_empty_blob_is_none():
    assert pu.parse_forge_usage("FORGE_LLM_USAGE") is None


def test_forge_steps_marker_with_empty_blob_is_none():
    assert pu.parse_forge_steps("FORGE_STEPS") is None


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


# --- _row_utils: coercion + closed-schema validation -----------------------

_FIELDS = frozenset({"session_id", "component"})
_COMPONENTS = frozenset({"orchestration"})


def _row(**over):
    base = {"session_id": "sess-1", "component": "orchestration"}
    base.update(over)
    return base


def test_coerce_optional_str_variants():
    assert ru.coerce_optional_str(None) is None
    assert ru.coerce_optional_str("   ") is None  # empty after strip
    assert ru.coerce_optional_str("  hi  ") == "hi"
    assert ru.coerce_optional_str(42) == "42"


def test_coerce_optional_int_variants():
    assert ru.coerce_optional_int(None) is None
    assert ru.coerce_optional_int("7") == 7
    assert ru.coerce_optional_int("nan") is None  # bad type -> None, distinct from 0
    assert ru.coerce_optional_int(object()) is None


def test_validate_closed_row_extra_or_missing_field_raises():
    with pytest.raises(ValueError, match="closed schema"):
        ru.validate_closed_row(
            _row(unexpected=1), fields=_FIELDS, valid_components=_COMPONENTS,
            error_cls=ValueError, label="llm_calls",
        )
    with pytest.raises(ValueError, match="closed schema"):
        ru.validate_closed_row(
            {"component": "orchestration"}, fields=_FIELDS,
            valid_components=_COMPONENTS, error_cls=ValueError, label="llm_calls",
        )


def test_validate_closed_row_bad_session_id_raises():
    with pytest.raises(KeyError, match="non-empty 'session_id'"):
        ru.validate_closed_row(
            _row(session_id="  "), fields=_FIELDS, valid_components=_COMPONENTS,
            error_cls=KeyError, label="conversations",
        )


def test_validate_closed_row_bad_component_raises():
    with pytest.raises(ValueError, match="not one of"):
        ru.validate_closed_row(
            _row(component="nope"), fields=_FIELDS, valid_components=_COMPONENTS,
            error_cls=ValueError, label="conversations",
        )


def test_validate_closed_row_accepts_valid_row():
    # Happy path returns None without raising.
    assert ru.validate_closed_row(
        _row(), fields=_FIELDS, valid_components=_COMPONENTS,
        error_cls=ValueError, label="llm_calls",
    ) is None


# --- conversation_trace: redaction + tolerant I/O --------------------------

def test_redact_secrets_empty_returns_unchanged():
    assert ct.redact_secrets("") == ""


def test_redact_secrets_strips_bearer_and_env_shapes():
    out = ct.redact_secrets("Authorization: Bearer abcd1234efgh and API_KEY=supersecretvalue")
    assert "[REDACTED]" in out
    assert "supersecretvalue" not in out
    assert "abcd1234efgh" not in out


def test_coerce_text_none_and_non_str():
    assert ct._coerce_text(None) == ""
    assert ct._coerce_text(123) == "123"
    assert ct._coerce_text("hi") == "hi"


def test_append_conversation_swallows_oserror(tmp_path, monkeypatch):
    # append_jsonl raising OSError must be logged and swallowed (no raise).
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ct, "append_jsonl", _boom)
    rec = ct.ConversationRecord(session_id="s1", component="orchestration", prompt="p", response="r")
    # target set -> skips the Langfuse mirror branch; OSError swallowed.
    ct.append_conversation(session_dir=tmp_path, record=rec, target=tmp_path / "conv.jsonl")


def test_append_conversation_langfuse_mirror_failure_swallowed(tmp_path, monkeypatch):
    written = {}

    def _ok(dest, row, **k):
        written["row"] = row

    monkeypatch.setattr(ct, "append_jsonl", _ok)

    # Force the Langfuse mirror import/get_emitter to blow up; must be swallowed.
    import hyperloom.orchestrator.trace.langfuse_emitter as le

    def _boom(_dir):
        raise RuntimeError("langfuse down")

    monkeypatch.setattr(le, "get_emitter", _boom)

    rec = ct.ConversationRecord(session_id="s1", component="orchestration", prompt="p", response="r")
    ct.append_conversation(session_dir=tmp_path, record=rec)  # target=None -> mirror path
    assert written["row"]["component"] == "orchestration"
