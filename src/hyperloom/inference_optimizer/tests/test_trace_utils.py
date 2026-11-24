# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Edge-path coverage for the trace parsers and Langfuse mapping helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hyperloom.inference_optimizer.trace import _row_utils as ru
from hyperloom.inference_optimizer.trace import conversation_trace as ct
from hyperloom.inference_optimizer.trace import langfuse_mapping as lm
from hyperloom.inference_optimizer.trace import parse_usage as pu


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


@pytest.mark.parametrize(
    ("parse", "empty"),
    [
        pytest.param(pu.parse_claude_stream_json_usage, None, id="usage"),
        pytest.param(pu.parse_claude_stream_json_response, None, id="response"),
        pytest.param(pu.parse_claude_stream_json_turn_usages, [], id="turn-usages"),
        pytest.param(pu.parse_claude_stream_json_tool_calls, [], id="tool-calls"),
    ],
)
def test_stream_json_dir_path_is_oserror(tmp_path, parse, empty):
    # A directory triggers IsADirectoryError (an OSError) -> tolerant empty result.
    assert parse(tmp_path) == empty


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


def test_stream_json_turn_usages_skips_blank_lines(tmp_path):
    p = tmp_path / "process.log"
    p.write_text('\n{"type": "assistant", "message": {"usage": {"input_tokens": 1}}}\n')
    out = pu.parse_claude_stream_json_turn_usages(p)
    assert len(out) == 1 and out[0]["input_tokens"] == 1


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


def test_stream_json_tool_calls_summarize_a_missing_or_scalar_input(tmp_path):
    p = tmp_path / "process.log"
    p.write_text(
        '{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"}]}}\n'
        '{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": 123}]}}\n'
    )
    assert pu.parse_claude_stream_json_tool_calls(p) == [
        {"tool": "Read", "query": ""},
        {"tool": "Bash", "query": "123"},
    ]


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
    scores = lm.decision_to_scores(
        {
            "decision": {
                "outcome": "KEEP",
                "change": "x",
                "gain_pct": "not-a-number",
                "predicted_gain_pct": "also-bad",
            }
        }
    )
    # Only the categorical outcome score survives; bad numerics are dropped.
    assert [s["name"] for s in scores] == ["decision_outcome"]


def test_recipe_audit_is_write_defaults_to_read():
    """Absence of ``op`` (pre-write-audit rows) must not read as a write."""
    assert lm.recipe_audit_is_write({"op": "write"}) is True
    assert lm.recipe_audit_is_write({"op": "read"}) is False
    assert lm.recipe_audit_is_write({"method": "get_recipe"}) is False
    assert lm.recipe_audit_is_write({"op": ""}) is False


def test_recipe_write_span_flattens_only_nonzero_deltas():
    name, meta = lm.recipe_write_span(
        {
            "generator": "coordinator",
            "phase": "close_finalize",
            "result": {"canonical_id": "cid-x", "version": 3, "created": False},
            "delta": {"lessons": 2, "pitfalls": 0, "sessions": 1},
        }
    )
    assert name == "kb:recipe_write:coordinator"
    assert meta["kind"] == "recipe_write"
    assert meta["canonical_id"] == "cid-x"
    assert meta["version"] == 3
    assert meta["lessons_delta"] == 2
    assert meta["sessions_delta"] == 1
    # Zero deltas are omitted rather than reported as noise.
    assert "pitfalls_delta" not in meta


def test_recipe_write_span_tolerates_a_malformed_row():
    """A truncated/garbled row still yields a usable span rather than raising."""
    name, meta = lm.recipe_write_span({})
    assert name == "kb:recipe_write:unknown"
    assert meta["canonical_id"] == ""
    assert meta["created"] is False
    name, meta = lm.recipe_write_span({"result": "not-a-dict", "delta": None})
    assert name == "kb:recipe_write:unknown"
    assert not any(k.endswith("_delta") for k in meta)


def test_recipe_read_span_defaults_method():
    name, meta = lm.recipe_read_span({})
    assert name == "kb:recipe_snapshot:read"
    assert meta["kind"] == "recipe_snapshot"
    assert meta["hit"] is False


# --- _row_utils: closed-schema validation ----------------------------------

_FIELDS = frozenset({"session_id", "component"})
_COMPONENTS = frozenset({"orchestration"})


def _row(**over):
    base = {"session_id": "sess-1", "component": "orchestration"}
    base.update(over)
    return base


def test_validate_closed_row_extra_or_missing_field_raises():
    with pytest.raises(ValueError, match="closed schema"):
        ru.validate_closed_row(
            _row(unexpected=1),
            fields=_FIELDS,
            valid_components=_COMPONENTS,
            error_cls=ValueError,
            label="llm_calls",
        )
    with pytest.raises(ValueError, match="closed schema"):
        ru.validate_closed_row(
            {"component": "orchestration"},
            fields=_FIELDS,
            valid_components=_COMPONENTS,
            error_cls=ValueError,
            label="llm_calls",
        )


def test_validate_closed_row_bad_session_id_raises():
    with pytest.raises(KeyError, match="non-empty 'session_id'"):
        ru.validate_closed_row(
            _row(session_id="  "),
            fields=_FIELDS,
            valid_components=_COMPONENTS,
            error_cls=KeyError,
            label="conversations",
        )


def test_validate_closed_row_bad_component_raises():
    with pytest.raises(ValueError, match="not one of"):
        ru.validate_closed_row(
            _row(component="nope"),
            fields=_FIELDS,
            valid_components=_COMPONENTS,
            error_cls=ValueError,
            label="conversations",
        )


def test_validate_closed_row_accepts_valid_row():
    # Happy path returns None without raising.
    assert (
        ru.validate_closed_row(
            _row(),
            fields=_FIELDS,
            valid_components=_COMPONENTS,
            error_cls=ValueError,
            label="llm_calls",
        )
        is None
    )


# --- conversation_trace: tolerant I/O ---------------------------------------


@pytest.mark.parametrize(
    ("prompt", "stored"),
    [
        pytest.param(None, "", id="missing"),
        pytest.param(123, "123", id="non-text"),
        pytest.param("hi", "hi", id="text"),
    ],
)
def test_a_conversation_row_stores_the_prompt_as_text(prompt, stored):
    row = ct.ConversationRecord(session_id="s1", component="orchestration", prompt=prompt).to_row()
    assert row["prompt"] == stored


def test_append_conversation_swallows_oserror(tmp_path, monkeypatch):
    # append_jsonl raising OSError must be logged and swallowed (no raise).
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ct, "append_jsonl", _boom)
    rec = ct.ConversationRecord(session_id="s1", component="orchestration", prompt="p", response="r")
    ct.append_conversation(session_dir=tmp_path, record=rec)


def test_append_conversation_langfuse_mirror_defect_raises_after_the_ledger_write(tmp_path, monkeypatch):
    written = {}

    def _ok(dest, row, **k):
        written["row"] = row

    monkeypatch.setattr(ct, "append_jsonl", _ok)

    import hyperloom.inference_optimizer.trace.langfuse_emitter as le

    def _boom(_dir):
        raise RuntimeError("emitter defect")

    monkeypatch.setattr(le, "get_emitter", _boom)

    rec = ct.ConversationRecord(session_id="s1", component="orchestration", prompt="p", response="r")
    with pytest.raises(RuntimeError, match="emitter defect"):
        ct.append_conversation(session_dir=tmp_path, record=rec)
    assert written["row"]["component"] == "orchestration"
