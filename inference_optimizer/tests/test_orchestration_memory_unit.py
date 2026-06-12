# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for orchestration-memory checkpoint helpers."""

from __future__ import annotations

from inference_optimizer.orchestrator.orchestration_memory import (
    CheckpointPolicy,
    CheckpointTracker,
    build_memory_record,
    parse_checkpoint_reply,
    render_memory_for_seed,
)
from inference_optimizer.orchestrator import orchestration_memory as om


# ---- CheckpointPolicy ----

def test_policy_phase_boundary():
    p = CheckpointPolicy()
    assert p.should_checkpoint(
        ticks_since_last=0, minutes_since_last=0, chars_since_last=0, phase_changed=True,
    )


def test_policy_phase_boundary_disabled():
    p = CheckpointPolicy(on_phase_boundary=False, every_ticks=0, every_minutes=0, char_budget=0)
    assert not p.should_checkpoint(
        ticks_since_last=100, minutes_since_last=100, chars_since_last=10**9, phase_changed=True,
    )


def test_policy_ticks_trigger():
    p = CheckpointPolicy(on_phase_boundary=False)
    assert p.should_checkpoint(
        ticks_since_last=20, minutes_since_last=0, chars_since_last=0, phase_changed=False,
    )


def test_policy_minutes_trigger():
    p = CheckpointPolicy(on_phase_boundary=False, every_ticks=0)
    assert p.should_checkpoint(
        ticks_since_last=0, minutes_since_last=30, chars_since_last=0, phase_changed=False,
    )


def test_policy_char_budget_trigger():
    p = CheckpointPolicy(on_phase_boundary=False, every_ticks=0, every_minutes=0)
    assert p.should_checkpoint(
        ticks_since_last=0, minutes_since_last=0, chars_since_last=400_000, phase_changed=False,
    )


def test_policy_no_trigger():
    p = CheckpointPolicy()
    assert not p.should_checkpoint(
        ticks_since_last=1, minutes_since_last=1, chars_since_last=1, phase_changed=False,
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
