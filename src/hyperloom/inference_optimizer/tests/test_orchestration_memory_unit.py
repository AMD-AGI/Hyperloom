# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for orchestration-memory checkpoint helpers."""

from __future__ import annotations

from hyperloom.orchestrator.state.orchestration_memory import (
    CheckpointPolicy,
    CheckpointTracker,
    build_memory_record,
    context_window_for_model,
    deterministic_memory_fallback,
    is_degenerate_checkpoint,
    parse_checkpoint_reply,
    render_memory_for_seed,
)
from hyperloom.orchestrator.state import orchestration_memory as om


# ---- CheckpointPolicy ----


def test_policy_phase_boundary():
    p = CheckpointPolicy()
    assert p.should_checkpoint(
        ticks_since_last=0,
        minutes_since_last=0,
        chars_since_last=0,
        phase_changed=True,
    )


def test_policy_phase_boundary_disabled():
    p = CheckpointPolicy(on_phase_boundary=False, every_ticks=0, every_minutes=0, char_budget=0)
    assert not p.should_checkpoint(
        ticks_since_last=100,
        minutes_since_last=100,
        chars_since_last=10**9,
        phase_changed=True,
    )


def test_policy_ticks_trigger():
    p = CheckpointPolicy(on_phase_boundary=False)
    assert p.should_checkpoint(
        ticks_since_last=20,
        minutes_since_last=0,
        chars_since_last=0,
        phase_changed=False,
    )


def test_policy_minutes_trigger():
    p = CheckpointPolicy(on_phase_boundary=False, every_ticks=0)
    assert p.should_checkpoint(
        ticks_since_last=0,
        minutes_since_last=30,
        chars_since_last=0,
        phase_changed=False,
    )


def test_policy_char_budget_trigger():
    p = CheckpointPolicy(on_phase_boundary=False, every_ticks=0, every_minutes=0)
    assert p.should_checkpoint(
        ticks_since_last=0,
        minutes_since_last=0,
        chars_since_last=om.DEFAULT_CHECKPOINT_CHAR_BUDGET,
        phase_changed=False,
    )


def test_policy_no_trigger():
    p = CheckpointPolicy()
    assert not p.should_checkpoint(
        ticks_since_last=1,
        minutes_since_last=1,
        chars_since_last=1,
        phase_changed=False,
    )


# ---- parse_checkpoint_reply / _extract_json_object ----


def test_parse_fenced_json():
    raw = '```json\n{"current_plan": "go", "hypotheses": ["a", "b"]}\n```'
    out = parse_checkpoint_reply(raw)
    assert out["current_plan"] == "go"
    assert out["hypotheses"] == ["a", "b"]
    assert out["pending"] == []


def test_parse_bare_object():
    out = parse_checkpoint_reply('noise {"current_plan": "x", "pending": "single"} trailing')
    assert out["current_plan"] == "x"
    assert out["pending"] == ["single"]


def test_parse_no_json():
    out = parse_checkpoint_reply("just prose, no object")
    assert out["parse_error"]
    assert out["current_plan"] == "just prose, no object"


def test_parse_empty_string():
    out = parse_checkpoint_reply("")
    assert out["parse_error"]


def test_parse_list_filters_blanks():
    out = parse_checkpoint_reply('{"learnings": ["keep", "", "  "]}')
    assert out["learnings"] == ["keep"]


def test_extract_json_invalid_returns_none():
    assert om._extract_json_object("{not valid json}") is None


def test_extract_json_non_dict_returns_none():
    assert om._extract_json_object("[1, 2, 3]") is None


def test_extract_json_empty():
    assert om._extract_json_object("") is None


# ---- build_memory_record ----


def test_build_memory_record_accumulates_learnings():
    prev = {"learnings": ["old"], "checkpoint_count": 2}
    parsed = {"current_plan": "p", "learnings": ["old", "new"]}
    rec = build_memory_record(parsed, seq=5, tick=10, previous=prev)
    assert rec["learnings"] == ["old", "new"]
    assert rec["checkpoint_count"] == 3
    assert rec["last_checkpoint_seq"] == 5
    assert rec["last_checkpoint_tick"] == 10
    assert rec["last_checkpoint_ts"]


def test_build_memory_record_caps_learnings():
    parsed = {"learnings": [f"L{i}" for i in range(60)]}
    rec = build_memory_record(parsed, seq=1, tick=1)
    assert len(rec["learnings"]) == 50


def test_build_memory_record_no_previous():
    rec = build_memory_record({"current_plan": "x"}, seq=0, tick=0)
    assert rec["checkpoint_count"] == 1


# ---- render_memory_for_seed ----


def test_render_empty():
    assert render_memory_for_seed({}) == ""


def test_render_full():
    mem = {
        "current_plan": "drive X",
        "hypotheses": ["h1"],
        "tried_and_why": ["t1"],
        "pending": ["p1"],
        "learnings": ["l1"],
        "checkpoint_count": 4,
    }
    text = render_memory_for_seed(mem)
    assert "current_plan: drive X" in text
    assert "  - h1" in text
    assert "(checkpoint #4)" in text


def test_render_plan_only():
    text = render_memory_for_seed({"current_plan": "only"})
    assert "current_plan: only" in text
    assert "hypotheses" not in text


# ---- CheckpointTracker ----


def test_tracker_chars_add_clamps():
    t = CheckpointTracker()
    t.chars_add(5)
    t.chars_add(-3)
    assert t.chars_since_last == 5


def test_tracker_reset():
    t = CheckpointTracker(chars_since_last=99)
    t.reset(tick=7, minute_mark=1.5, phase="EXPLORE")
    assert t.last_tick == 7
    assert t.last_minute_mark == 1.5
    assert t.chars_since_last == 0
    assert t.last_phase == "EXPLORE"


# ---- context-token guardrail ----


def test_context_window_known_and_unknown(monkeypatch):
    assert context_window_for_model("claude-opus-4-8") == 200_000
    monkeypatch.setattr(om, "DEFAULT_MODEL_CONTEXT_WINDOW", 123_456)
    monkeypatch.setitem(om.MODEL_CONTEXT_WINDOWS, "unit-known-window", 234_567)
    assert context_window_for_model("unit-known-window") == 234_567
    assert context_window_for_model("") == 123_456
    assert context_window_for_model("totally-unknown") == 123_456


def test_policy_context_token_soft_trigger():
    p = CheckpointPolicy(
        on_phase_boundary=False,
        every_ticks=0,
        every_minutes=0,
        char_budget=0,
        context_token_soft=140_000,
    )
    assert p.should_checkpoint(
        ticks_since_last=0,
        minutes_since_last=0,
        chars_since_last=0,
        phase_changed=False,
        context_tokens_now=140_000,
    )
    assert not p.should_checkpoint(
        ticks_since_last=0,
        minutes_since_last=0,
        chars_since_last=0,
        phase_changed=False,
        context_tokens_now=139_999,
    )


def test_policy_context_token_soft_disabled_by_default():
    # Default soft=0 → token trigger off; huge token count alone does not fire.
    p = CheckpointPolicy(on_phase_boundary=False, every_ticks=0, every_minutes=0, char_budget=0)
    assert not p.should_checkpoint(
        ticks_since_last=0,
        minutes_since_last=0,
        chars_since_last=0,
        phase_changed=False,
        context_tokens_now=10**9,
    )


def test_policy_is_hard_compaction():
    p = CheckpointPolicy(context_token_hard=170_000)
    assert p.is_hard_compaction(170_000)
    assert not p.is_hard_compaction(169_999)
    # Disabled (0) never fires.
    assert not CheckpointPolicy().is_hard_compaction(10**9)


def test_tracker_set_context_tokens_and_reset_preserves_level():
    t = CheckpointTracker()
    t.set_context_tokens(123_456)
    t.set_context_tokens(-5)  # clamped to 0
    assert t.context_tokens_now == 0
    t.set_context_tokens(123_456)
    t.reset(tick=1, minute_mark=2.0, phase="EXPLORE")
    # reset clears cadence counters but NOT the absolute token water level.
    assert t.chars_since_last == 0
    assert t.context_tokens_now == 123_456


# ---- degenerate detection + non-empty-inherit + fallback ----


def test_is_degenerate_parse_error():
    assert is_degenerate_checkpoint({"parse_error": "boom", "current_plan": "x"})


def test_is_degenerate_all_empty():
    assert is_degenerate_checkpoint(
        {"current_plan": "", "hypotheses": [], "pending": [], "tried_and_why": []}
    )


def test_is_degenerate_false_with_plan():
    assert not is_degenerate_checkpoint(
        {"current_plan": "go", "hypotheses": [], "pending": [], "tried_and_why": []}
    )


def test_is_degenerate_false_with_list():
    assert not is_degenerate_checkpoint(
        {"current_plan": "", "hypotheses": ["h"], "pending": [], "tried_and_why": []}
    )


def test_build_memory_record_inherits_empty_fields():
    prev = {
        "current_plan": "old",
        "hypotheses": ["h1"],
        "pending": ["p1"],
        "tried_and_why": ["t1"],
        "learnings": ["L1"],
        "checkpoint_count": 2,
    }
    rec = build_memory_record(
        {"current_plan": "", "hypotheses": [], "pending": [], "tried_and_why": [], "learnings": []},
        seq=5,
        tick=9,
        previous=prev,
    )
    assert rec["current_plan"] == "old"
    assert rec["hypotheses"] == ["h1"]
    assert rec["pending"] == ["p1"]
    assert rec["tried_and_why"] == ["t1"]
    assert rec["learnings"] == ["L1"]
    assert rec["checkpoint_count"] == 3


def test_build_memory_record_new_value_wins():
    prev = {"current_plan": "old", "hypotheses": ["h1"], "pending": ["p1"], "learnings": ["L1"]}
    rec = build_memory_record(
        {"current_plan": "new", "hypotheses": ["h2"], "learnings": ["L2"]},
        seq=6,
        tick=10,
        previous=prev,
    )
    assert rec["current_plan"] == "new"
    assert rec["hypotheses"] == ["h2"]
    assert rec["pending"] == ["p1"]  # inherited
    assert rec["learnings"] == ["L1", "L2"]


class _FakeState:
    phase = "EXPLORE"
    macro_cycle = 2
    current_best = {"tput": 1234.5}
    optimization_stack = [{"a": 1}, {"b": 2}]
    cumulative_gain_validated = 12.34


def test_deterministic_memory_fallback_is_non_degenerate():
    fb = deterministic_memory_fallback(_FakeState())
    assert "EXPLORE" in fb["current_plan"]
    assert "cycle=2" in fb["current_plan"]
    assert "best_tput=1234.5" in fb["current_plan"]
    assert "validated_gain=12.34%" in fb["current_plan"]
    assert fb["tried_and_why"] == ["stack has 2 accepted change(s)"]
    assert not is_degenerate_checkpoint(fb)


def test_deterministic_memory_fallback_missing_attrs():
    class _Empty:
        pass

    fb = deterministic_memory_fallback(_Empty())
    # Never raises; yields a usable record.
    assert fb["current_plan"].startswith("[auto]")
    assert not is_degenerate_checkpoint(fb)
