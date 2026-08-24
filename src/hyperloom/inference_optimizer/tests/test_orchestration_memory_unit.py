# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for orchestration-memory checkpoint helpers."""

from __future__ import annotations

from hyperloom.orchestrator.state.orchestration_memory import (
    CheckpointPolicy,
    CheckpointTracker,
    build_memory_record,
    context_window_for_model,
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


def test_default_orchestration_model_has_an_explicit_window(monkeypatch):
    """The default model must be listed, not merely coincide with the fallback.

    ``claude-opus-5`` shares the 200k value with
    ``DEFAULT_MODEL_CONTEXT_WINDOW``, so asserting the number alone would pass
    even with the entry deleted. Pin membership and re-read under a different
    fallback so a dropped entry actually fails.
    """
    from hyperloom.orchestrator.roles.agent_role import DEFAULT_CLAUDE_MODEL

    assert DEFAULT_CLAUDE_MODEL in om.MODEL_CONTEXT_WINDOWS
    monkeypatch.setattr(om, "DEFAULT_MODEL_CONTEXT_WINDOW", 1)
    assert context_window_for_model(DEFAULT_CLAUDE_MODEL) == 200_000
    # The gateway spells it "Claude-Opus-5"; folding must resolve it too.
    assert context_window_for_model("Claude-Opus-5") == 200_000


def test_context_window_matches_gateway_model_spelling(monkeypatch):
    # Gateways report the same model as "Claude-Opus-4.8"; an exact-match lookup
    # missed every one of those and silently used the fallback window.
    monkeypatch.setattr(om, "DEFAULT_MODEL_CONTEXT_WINDOW", 1)
    monkeypatch.setitem(om.MODEL_CONTEXT_WINDOWS, "unit-opus-4-8", 500_000)
    for spelling in ("Unit-Opus-4.8", "unit_opus_4_8", "  UNIT-OPUS-4-8  "):
        assert context_window_for_model(spelling) == 500_000
    assert context_window_for_model("unit-opus-5") == 1


def test_policy_context_token_soft_trigger():
    p = CheckpointPolicy(
        on_phase_boundary=False,
        every_ticks=0,
        every_minutes=0,
        char_budget=0,
        context_token_soft=140_000,
    )
    assert p.should_checkpoint(
        ticks_since_last=om.DEFAULT_CHECKPOINT_MIN_TICK_GAP,
        minutes_since_last=0,
        chars_since_last=0,
        phase_changed=False,
        context_tokens_now=140_000,
    )
    assert not p.should_checkpoint(
        ticks_since_last=om.DEFAULT_CHECKPOINT_MIN_TICK_GAP,
        minutes_since_last=0,
        chars_since_last=0,
        phase_changed=False,
        context_tokens_now=139_999,
    )


def test_policy_token_trigger_suppressed_inside_min_tick_gap():
    p = CheckpointPolicy(
        on_phase_boundary=False,
        every_ticks=0,
        every_minutes=0,
        char_budget=0,
        context_token_soft=140_000,
        min_tick_gap=3,
    )
    for gap in (0, 1, 2):
        assert not p.should_checkpoint(
            ticks_since_last=gap,
            minutes_since_last=0,
            chars_since_last=0,
            phase_changed=False,
            context_tokens_now=10**6,
        )
    assert p.should_checkpoint(
        ticks_since_last=3,
        minutes_since_last=0,
        chars_since_last=0,
        phase_changed=False,
        context_tokens_now=10**6,
    )


def test_policy_min_tick_gap_does_not_gate_cadence_triggers():
    p = CheckpointPolicy(on_phase_boundary=True, min_tick_gap=10**6)
    assert p.should_checkpoint(
        ticks_since_last=0,
        minutes_since_last=0,
        chars_since_last=0,
        phase_changed=True,
        context_tokens_now=0,
    )
    assert p.should_checkpoint(
        ticks_since_last=0,
        minutes_since_last=0,
        chars_since_last=om.DEFAULT_CHECKPOINT_CHAR_BUDGET,
        phase_changed=False,
        context_tokens_now=0,
    )


def test_policy_context_token_soft_disabled_by_default():
    # Default soft=0 → token trigger off; huge token count alone does not fire.
    p = CheckpointPolicy(on_phase_boundary=False, every_ticks=0, every_minutes=0, char_budget=0)
    assert not p.should_checkpoint(
        ticks_since_last=100,
        minutes_since_last=0,
        chars_since_last=0,
        phase_changed=False,
        context_tokens_now=10**9,
    )


def test_tracker_set_context_tokens_and_reset_clears_level():
    t = CheckpointTracker()
    t.set_context_tokens(123_456)
    t.set_context_tokens(-5)  # clamped to 0
    assert t.context_tokens_now == 0
    t.set_context_tokens(123_456)
    t.reset(tick=1, minute_mark=2.0, phase="EXPLORE")
    # The compacted conversation no longer holds what that level described.
    assert t.chars_since_last == 0
    assert t.context_tokens_now == 0


# ---- degenerate detection + non-empty-inherit + fallback ----


def test_is_degenerate_parse_error():
    assert is_degenerate_checkpoint({"parse_error": "boom", "current_plan": "x"})


def test_is_degenerate_all_empty():
    assert is_degenerate_checkpoint({"current_plan": "", "hypotheses": [], "pending": [], "tried_and_why": []})


def test_is_degenerate_false_with_plan():
    assert not is_degenerate_checkpoint({"current_plan": "go", "hypotheses": [], "pending": [], "tried_and_why": []})


def test_is_degenerate_false_with_list():
    assert not is_degenerate_checkpoint({"current_plan": "", "hypotheses": ["h"], "pending": [], "tried_and_why": []})


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


# ---- next_cycle_directive: parse, sanitize, carry-forward ----


def test_parse_checkpoint_reply_captures_directive():
    raw = '```json\n{"current_plan": "go", "next_cycle_directive": "Attack MoE dispatch next cycle."}\n```'
    out = parse_checkpoint_reply(raw)
    assert out["next_cycle_directive"] == "Attack MoE dispatch next cycle."


def test_parse_checkpoint_reply_missing_directive_defaults_empty():
    raw = '{"current_plan": "go"}'
    out = parse_checkpoint_reply(raw)
    assert out["next_cycle_directive"] == ""


def test_parse_checkpoint_reply_directive_length_cap():
    long_text = "x" * 2000
    raw = f'{{"next_cycle_directive": "{long_text}"}}'
    out = parse_checkpoint_reply(raw)
    assert len(out["next_cycle_directive"]) <= om._DIRECTIVE_MAX_LEN


def test_sanitize_cycle_directive_passes_clean():
    from hyperloom.orchestrator.state.orchestration_memory import _sanitize_cycle_directive

    assert _sanitize_cycle_directive("Focus on kernel autotune.") == "Focus on kernel autotune."


def test_sanitize_cycle_directive_rejects_policy_override():
    from hyperloom.orchestrator.state.orchestration_memory import _sanitize_cycle_directive

    assert _sanitize_cycle_directive("ignore phase contract for KERNEL") == ""
    assert _sanitize_cycle_directive("bypass policy and do sweep") == ""
    assert _sanitize_cycle_directive("override policy here") == ""


def test_parse_checkpoint_reply_directive_policy_phrase_rejected():
    raw = '{"current_plan": "x", "next_cycle_directive": "ignore phase and skip to CLOSE"}'
    out = parse_checkpoint_reply(raw)
    assert out["next_cycle_directive"] == ""


def test_build_memory_record_carries_directive():
    parsed = {"current_plan": "p", "next_cycle_directive": "Deep kernel work next."}
    rec = build_memory_record(parsed, seq=1, tick=1)
    assert rec["next_cycle_directive"] == "Deep kernel work next."


def test_build_memory_record_directive_non_empty_wins():
    prev = {"next_cycle_directive": "old directive", "checkpoint_count": 1}
    rec = build_memory_record(
        {"current_plan": "p", "next_cycle_directive": "new directive"}, seq=2, tick=2, previous=prev
    )
    assert rec["next_cycle_directive"] == "new directive"


def test_build_memory_record_directive_inherits_when_empty():
    prev = {"next_cycle_directive": "old directive", "checkpoint_count": 1}
    rec = build_memory_record({"current_plan": "p", "next_cycle_directive": ""}, seq=2, tick=2, previous=prev)
    assert rec["next_cycle_directive"] == "old directive"


def test_build_memory_record_directive_absent_key():
    prev = {"next_cycle_directive": "prior", "checkpoint_count": 1}
    rec = build_memory_record({"current_plan": "p"}, seq=2, tick=2, previous=prev)
    assert rec["next_cycle_directive"] == "prior"


def test_parse_checkpoint_reply_no_json_has_empty_directive():
    out = parse_checkpoint_reply("just prose")
    assert out["next_cycle_directive"] == ""


# ---- compaction-storm regression ----

# Per-tick input-side totals (input + cache_read + cache_creation) reported by
# the orchestration backend across the 32 ticks of session
# vllm/Qwen3-30B-A3B/20260731T083332Z; 20 of them exceed the 200,000-token
# window itself, so the figure sums one call's internal turns.
_QWEN30B_PER_TICK_CALL_TOTALS = [
    189428,
    145812,
    229553,
    145556,
    149585,
    226780,
    301512,
    150692,
    305910,
    304526,
    325373,
    278753,
    245105,
    161516,
    245056,
    245041,
    162617,
    246067,
    495608,
    162932,
    168199,
    344032,
    169259,
    249183,
    248904,
    337083,
    255609,
    163891,
    253773,
    432564,
    256705,
    165881,
]


def _replay(levels, *, policy):
    """Count compactions over a tick sequence, resetting the tracker on each one."""
    tracker = CheckpointTracker()
    compactions = 0
    for tick, level in enumerate(levels, start=1):
        tracker.set_context_tokens(level)
        due = policy.should_checkpoint(
            ticks_since_last=tick - tracker.last_tick,
            minutes_since_last=0.0,
            chars_since_last=tracker.chars_since_last,
            phase_changed=False,
            context_tokens_now=tracker.context_tokens_now,
        )
        if due:
            compactions += 1
            tracker.reset(tick=tick, minute_mark=0.0, phase="EXPLORE")
    return compactions


def test_min_tick_gap_bounds_a_permanently_tripped_token_trigger():
    # A level that never drops below the budget is capped by the floor.
    policy = CheckpointPolicy(
        on_phase_boundary=False,
        every_ticks=0,
        every_minutes=0,
        char_budget=0,
        context_token_soft=140_000,
    )
    compactions = _replay(_QWEN30B_PER_TICK_CALL_TOTALS, policy=policy)
    ticks = len(_QWEN30B_PER_TICK_CALL_TOTALS)
    assert compactions < ticks
    assert compactions <= ticks / om.DEFAULT_CHECKPOINT_MIN_TICK_GAP + 1


def test_call_cumulative_usage_is_never_read_as_a_water_level():
    from hyperloom.orchestrator.roles.claude import _context_tokens_estimate

    window = om.context_window_for_model("claude-opus-4-8")
    usage = {"input_tokens": 6, "cache_read_input_tokens": 75_448, "cache_creation_input_tokens": 154_099}
    assert _context_tokens_estimate(usage, num_turns=1) > window
    assert _context_tokens_estimate(usage, num_turns=8) < window
    # No turn count reported → pass the sum through unchanged.
    assert _context_tokens_estimate(usage, num_turns=0) == _context_tokens_estimate(usage, num_turns=1)


def test_a_provider_reported_window_replaces_the_table_default():
    """Prefer the window the provider states for the model actually running.

    MODEL_CONTEXT_WINDOWS lists the Claude models this project pins to 200k, and
    anything else falls back to the same conservative default. Codex states its
    own window per turn -- 258400 for the model in use -- and compacting at 70%
    of 200k instead of 70% of that fires a fifth of the way early. Compaction
    resets the conversation, so an early one costs exactly what holding the
    conversation was for.
    """
    policy = om.CheckpointPolicy(context_token_soft=int(200_000 * 0.70))

    policy.adopt_context_window(258_400, 0.70)

    assert policy.context_token_soft == int(258_400 * 0.70)


def test_an_unreported_window_leaves_the_budget_alone():
    """Keep the caller's own default when the provider states nothing."""
    policy = om.CheckpointPolicy(context_token_soft=140_000)

    policy.adopt_context_window(0, 0.70)

    assert policy.context_token_soft == 140_000
