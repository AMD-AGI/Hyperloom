# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
    p.write_text('\n{"type": "assistant", "message": {"usage": {"input_tokens": 1}}}\n')
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


# --- conversation_trace: structured intents --------------------------------


def test_conversation_row_carries_intents_structured():
    # A tool-calling backend's decision must survive as data, not as prose.
    rec = ct.ConversationRecord(
        session_id="s1",
        component="orchestration",
        prompt="p",
        response="Standing by.",
        intents=[{"intent_type": "delegate", "payload": {"task_id": "t1", "action": "profile"}}],
    )
    row = rec.to_row()
    assert row["intents"] == [{"intent_type": "delegate", "payload": {"task_id": "t1", "action": "profile"}}]


def test_conversation_row_intents_absent_and_empty_both_null():
    # The four text-only writers pass nothing; an empty list must not read
    # differently from an absent one.
    assert ct.ConversationRecord(session_id="s1", component="critic").to_row()["intents"] is None
    assert ct.ConversationRecord(session_id="s1", component="critic", intents=[]).to_row()["intents"] is None


def test_conversation_row_redacts_secrets_inside_intent_payloads():
    # An intent payload can carry server args, so it goes through the same
    # no-credentials-on-disk contract as the text fields.
    rec = ct.ConversationRecord(
        session_id="s1",
        component="orchestration",
        intents=[
            {
                "intent_type": "delegate",
                "payload": {"envs": {"note": "API_KEY=supersecretvalue"}, "tags": ["Bearer abcd1234efgh"]},
            }
        ],
    )
    payload = rec.to_row()["intents"][0]["payload"]
    assert "supersecretvalue" not in payload["envs"]["note"]
    assert "abcd1234efgh" not in payload["tags"][0]
    assert "[REDACTED]" in payload["envs"]["note"]


def test_redact_json_passes_through_non_string_leaves():
    assert ct._redact_json({"n": 1, "f": 1.5, "b": True, "z": None}) == {"n": 1, "f": 1.5, "b": True, "z": None}


def test_reactor_conversation_records_emitted_intents(tmp_path, monkeypatch):
    # End-to-end for the fill side: the coordinator hands over prompt/response
    # in metadata and the intents on the result, and both halves must land.
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
    from hyperloom.orchestrator.loop.conversation import ConversationCollaborator
    from hyperloom.orchestrator.roles.base import BackendTurnResult

    rows: list[dict] = []
    monkeypatch.setattr(ct, "append_jsonl", lambda dest, row, **k: rows.append(row))

    class _State:
        tick = 4
        phase = "EXPLORE"

    class _Coord:
        session_dir = tmp_path / "s1"
        shared_state = _State()

    collaborator = ConversationCollaborator(_Coord())
    result = BackendTurnResult(
        intents=[Intent(type=IntentType.DELEGATE, payload={"task_id": "t1"})],
        raw_text="Delegated.",
        metadata={"prompt": "P", "response": "Delegated.", "model": "claude-opus-5"},
    )
    collaborator._record_reactor_conversation("orchestration", result)

    assert len(rows) == 1
    assert rows[0]["intents"] == [{"intent_type": "delegate", "payload": {"task_id": "t1"}}]
    assert rows[0]["tick"] == 4 and rows[0]["model"] == "claude-opus-5"


def test_reactor_conversation_records_intent_only_turn(tmp_path, monkeypatch):
    # A turn that emitted a decision but narrated nothing is the case the old
    # text-only guard dropped, and it is the most valuable row of all.
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
    from hyperloom.orchestrator.loop.conversation import ConversationCollaborator
    from hyperloom.orchestrator.roles.base import BackendTurnResult

    rows: list[dict] = []
    monkeypatch.setattr(ct, "append_jsonl", lambda dest, row, **k: rows.append(row))

    class _Coord:
        session_dir = tmp_path / "s1"
        shared_state = type("S", (), {"tick": 1, "phase": None})()

    collaborator = ConversationCollaborator(_Coord())
    result = BackendTurnResult(
        intents=[Intent(type=IntentType.ALERT, payload={"severity": "high"})],
        raw_text="",
        metadata={},
    )
    collaborator._record_reactor_conversation("orchestration", result)

    assert len(rows) == 1
    assert rows[0]["intents"] == [{"intent_type": "alert", "payload": {"severity": "high"}}]


def test_reactor_conversation_skips_fully_empty_turn(tmp_path, monkeypatch):
    from hyperloom.orchestrator.loop.conversation import ConversationCollaborator
    from hyperloom.orchestrator.roles.base import BackendTurnResult

    rows: list[dict] = []
    monkeypatch.setattr(ct, "append_jsonl", lambda dest, row, **k: rows.append(row))

    class _Coord:
        session_dir = tmp_path / "s1"
        shared_state = type("S", (), {"tick": 1, "phase": None})()

    collaborator = ConversationCollaborator(_Coord())
    collaborator._record_reactor_conversation(
        "orchestration", BackendTurnResult(intents=[], raw_text="", metadata={})
    )
    assert rows == []
