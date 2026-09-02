# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for prompt -> ReactorContext parser."""

from __future__ import annotations

import textwrap

from hyperloom.agents.robustness.role.prompt_inputs import (
    ReactorContext,
    from_coordinator_prompt,
)


def _prompt(shared: str, inbox: str, *, time_budget: str | None = None) -> str:
    parts = ["=== Shared session state ===", shared.rstrip()]
    if time_budget is not None:
        parts.extend(["=== Time budget ===", time_budget.rstrip()])
    parts.append(inbox.rstrip())
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Empty / no-messages cases
# ---------------------------------------------------------------------------


def test_empty_prompt_returns_empty_context():
    ctx = from_coordinator_prompt("")
    assert isinstance(ctx, ReactorContext)
    assert ctx.inbox == []
    assert ctx.parse_warnings == ["empty prompt"]


def test_no_new_messages_yields_empty_inbox():
    prompt = _prompt(
        textwrap.dedent(
            """\
            session_id=sess-1
            model=qwen3-8b  class=qwen3
            baseline_tput=10.5  baseline_acc=0.8
            cumulative_gain_validated=12.5%
            crash_count=0
            current_action=(idle)
            """
        ),
        textwrap.dedent(
            """\
            === Inbox for robustness ===
            (no new messages)
            """
        ),
    )
    ctx = from_coordinator_prompt(prompt, tick_index=3, now_unix=100.0)
    assert ctx.tick_index == 3
    assert ctx.now_unix == 100.0
    assert ctx.shared_state.model_name == "qwen3-8b"
    assert ctx.shared_state.model_class == "qwen3"
    assert ctx.shared_state.baseline_tput == 10.5
    assert ctx.shared_state.cumulative_gain_validated == 12.5
    assert ctx.shared_state.crash_count == 0
    assert ctx.shared_state.current_action == ""
    assert ctx.inbox == []
    assert ctx.parse_warnings == []


def test_inbox_parses_multiple_messages_with_python_repr_payload():
    prompt = _prompt(
        textwrap.dedent(
            """\
            session_id=sess-2
            model=(unset)  class=(unset)
            baseline_tput=0  baseline_acc=0
            crash_count=2
            current_action=baseline
            """
        ),
        textwrap.dedent(
            """\
            === Inbox for robustness (newest last) ===
              seq=1 msg_id=abc123 from=orchestration topic=proposal payload={'action_name': 'baseline', 'predicted_gain_pct': 0.0}
              seq=2 msg_id=def456 from=critic topic=review_verdict payload={'verdict': 'approve', 'target_proposal_msg_id': 'abc123'}
              seq=3 msg_id=ghi789 from=coordinator topic=event payload={'kind': 'task_queued', 'task_id': 't-1', 'action': 'baseline'}
            """
        ),
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.model_name == ""
    assert ctx.shared_state.crash_count == 2
    assert ctx.shared_state.current_action == "baseline"
    assert len(ctx.inbox) == 3
    first, second, third = ctx.inbox
    assert first.seq == 1 and first.msg_id == "abc123"
    assert first.from_agent == "orchestration" and first.topic == "proposal"
    assert first.payload == {"action_name": "baseline", "predicted_gain_pct": 0.0}
    assert second.payload["verdict"] == "approve"
    assert third.payload["task_id"] == "t-1"
    assert ctx.parse_warnings == []


def test_inbox_parses_per_topic_summary_fields_without_a_payload():
    """``delegated_result`` renders ``kind=/state=/error=`` and no ``payload=``;
    the repeated-failure signal reads those keys, so they must survive."""
    prompt = _prompt(
        "session_id=sess-2b\ncrash_count=0\n",
        textwrap.dedent(
            """\
            === Inbox for robustness (newest last) ===
              seq=1 msg_id=abc from=coordinator topic=delegated_result kind='baseline' state='succeeded' gain=4.875 kept=True
              seq=2 from=coordinator topic=delegated_result kind='explore' state='failed' error='exited -8: run_1stage = False, k=v inside'
              seq=3 msg_id=ghi from=coordinator topic=observation kind='retry' payload={'kind': 'retry', 'task_id': 't-1'}
            """
        ),
    )
    ctx = from_coordinator_prompt(prompt)
    first, second, third = ctx.inbox
    assert first.payload == {"kind": "baseline", "state": "succeeded", "gain": 4.875, "kept": True}
    # A quoted value owns its inner ``k=v``; splitting there would corrupt the
    # error text and invent a bogus ``k`` key.
    assert second.msg_id == ""
    assert second.payload == {
        "kind": "explore",
        "state": "failed",
        "error": "exited -8: run_1stage = False, k=v inside",
    }
    assert third.payload == {"kind": "retry", "task_id": "t-1"}
    assert ctx.parse_warnings == []


def test_unparsable_inbox_line_is_warned_not_raised():
    prompt = _prompt(
        "session_id=sess-3\ncrash_count=0\n",
        textwrap.dedent(
            """\
            === Inbox for robustness (newest last) ===
              seq=1 msg_id=ok from=x topic=y payload={'a': 1}
              this line is broken
              seq=2 msg_id=ok2 from=y topic=z payload={'b': 2}
            """
        ),
    )
    ctx = from_coordinator_prompt(prompt)
    assert len(ctx.inbox) == 2
    assert ctx.inbox[0].seq == 1
    assert ctx.inbox[1].seq == 2
    assert any("unparsable" in w for w in ctx.parse_warnings)


def test_payload_with_non_dict_repr_is_preserved_as_raw():
    prompt = _prompt(
        "session_id=sess-4\n",
        textwrap.dedent(
            """\
            === Inbox for robustness ===
              seq=10 msg_id=m1 from=x topic=y payload=this is not a dict
            """
        ),
    )
    ctx = from_coordinator_prompt(prompt)
    assert len(ctx.inbox) == 1
    item = ctx.inbox[0]
    assert item.payload["raw"] == "this is not a dict"
    assert any("not a python literal" in w for w in ctx.parse_warnings)


def test_kb_section_is_ignored_for_robustness_role():
    prompt = (
        "=== Shared session state ===\n"
        "crash_count=4\n"
        "=== Knowledge base hints ===\n"
        "kb-hint-do-not-parse\n"
        "=== Inbox for robustness ===\n"
        "(no new messages)\n"
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.crash_count == 4
    assert ctx.inbox == []


def test_baseline_tput_is_coerced_to_float_safely_on_garbage():
    prompt = _prompt(
        "session_id=sess-bad\nbaseline_tput=??\ncrash_count=NaNa\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.baseline_tput == 0.0
    assert ctx.shared_state.crash_count == 0


def test_payload_with_topic_substring_does_not_break_topic_field():
    prompt = _prompt(
        "session_id=s\ncrash_count=0\n",
        textwrap.dedent(
            """\
            === Inbox for robustness ===
              seq=7 msg_id=m1 from=x topic=alert payload={'summary': 'oops topic=alert again', 'severity': 'high'}
            """
        ),
    )
    ctx = from_coordinator_prompt(prompt)
    assert len(ctx.inbox) == 1
    item = ctx.inbox[0]
    assert item.topic == "alert"
    assert item.payload["summary"] == "oops topic=alert again"


# ---------------------------------------------------------------------------
# Time-budget section (consumed by signals/budget.py)
# ---------------------------------------------------------------------------


def test_time_budget_section_parses_into_snapshot():
    prompt = _prompt(
        "session_id=sess-tb\ncrash_count=0\ncumulative_gain_validated=2.5%\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
        time_budget="elapsed=120.5min  remaining=239.5min  budget=360min  closing_phase=False\n",
    )
    ctx = from_coordinator_prompt(prompt)
    snap = ctx.shared_state
    assert snap.elapsed_minutes == 120.5
    assert snap.remaining_minutes == 239.5
    assert snap.budget_minutes == 360.0
    assert snap.closing_phase is False
    assert snap.cumulative_gain_validated == 2.5


def test_time_budget_section_handles_closing_phase_true():
    prompt = _prompt(
        "session_id=sess-tb2\ncrash_count=0\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
        time_budget="elapsed=355.0min  remaining=5.0min  budget=360min  closing_phase=True\n",
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.closing_phase is True
    assert ctx.shared_state.remaining_minutes == 5.0


def test_time_budget_section_absent_leaves_defaults():
    prompt = _prompt(
        "session_id=sess-no-budget\ncrash_count=0\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
    )
    ctx = from_coordinator_prompt(prompt)
    snap = ctx.shared_state
    assert snap.elapsed_minutes == 0.0
    assert snap.remaining_minutes == 0.0
    assert snap.budget_minutes == 0.0
    assert snap.closing_phase is False


def test_time_budget_section_malformed_line_keeps_defaults():
    prompt = _prompt(
        "session_id=sess-tb-bad\ncrash_count=0\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
        time_budget="elapsed=??? min closing_phase=Yes\n",
    )
    ctx = from_coordinator_prompt(prompt)
    snap = ctx.shared_state
    assert snap.elapsed_minutes == 0.0
    assert snap.budget_minutes == 0.0
    assert snap.closing_phase is False


def test_cumulative_gain_validated_strips_parenthetical():
    prompt = _prompt(
        "session_id=s\ncumulative_gain_validated=8.65% (stack_len_at_validation=2, ts=2026-05-18T03:46:00)\ncrash_count=0\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.cumulative_gain_validated == 8.65


def test_tick_and_stop_reason_parse_into_snapshot():
    prompt = _prompt(
        textwrap.dedent(
            """\
            session_id=s
            tick=42  target_gap_pct=12.50
            macro_cycle=3
            stop_reason=(none)
            crash_count=0
            """
        ),
        "=== Inbox for robustness ===\n(no new messages)\n",
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.tick == 42
    assert ctx.shared_state.macro_cycle == 3
    assert ctx.shared_state.stop_reason == ""


def test_stop_reason_real_value_parses():
    prompt = _prompt(
        "session_id=s\ntick=10  target_gap_pct=0.0\nstop_reason=time_exhausted\ncrash_count=0\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.stop_reason == "time_exhausted"


def test_optimization_stack_size_counts_list_repr():
    prompt = _prompt(
        "session_id=s\noptimization_stack=['baseline:v1', 'integrate:k7']\ncrash_count=0\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.optimization_stack_size == 2


def test_optimization_stack_size_zero_when_none():
    prompt = _prompt(
        "session_id=s\noptimization_stack=(none)\ncrash_count=0\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.optimization_stack_size == 0


# ---------------------------------------------------------------------------
# explore_started — flips True iff an explore-family ``last_*`` line is non-``(none)``.
# ---------------------------------------------------------------------------


def test_explore_started_false_when_all_last_explore_keys_are_none():
    """Pre-explore/sweep the lines are all ``(none)`` → flag stays False so ``no_levers_found`` defers."""
    prompt = _prompt(
        "session_id=s\nlast_explore=(none)\ncrash_count=0\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.explore_started is False


def test_explore_started_true_when_any_last_explore_key_is_set():
    """A set ``last_explore`` flips ``explore_started`` to True."""
    prompt = _prompt(
        "session_id=s\n"
        "last_explore=status=succeeded decision=promoted "
        "tput=600.50 err=- ws=/runs/explore/abc ts=2026-05-22T05:00:00Z\n"
        "crash_count=0\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.explore_started is True


def test_explore_started_true_for_each_individual_explore_family():
    """Each explore-family key flips the flag independently."""
    for key in ("last_explore",):
        prompt = _prompt(
            f"session_id=s\n{key}=status=succeeded decision=promoted\ncrash_count=0\n",
            "=== Inbox for robustness ===\n(no new messages)\n",
        )
        ctx = from_coordinator_prompt(prompt)
        assert ctx.shared_state.explore_started is True, f"{key} should flip explore_started"


def test_explore_started_default_false_when_keys_absent():
    """Partial prompts omitting the explore-family lines must default to False so the cold-start guard stays conservative."""
    prompt = _prompt(
        "session_id=s\ncrash_count=0\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.explore_started is False


# ---------------------------------------------------------------------------
# Phase block parsing
# ---------------------------------------------------------------------------


def test_phase_block_parsed_correctly():
    prompt = (
        "=== Phase ===\n"
        "phase     : EXPLORE\n"
        "cycle     : 2\n"
        "entered   : 2026-08-04T00:00:00\n"
        "budget    : pct=0.40 elapsed_sec=123 remaining_sec=456\n"
        "allowed   : specialist, explore, integrate_patch\n"
        "=== Shared session state ===\n"
        "session_id=sess\n"
        "crash_count=0\n"
        "=== Inbox for robustness ===\n"
        "(no new messages)\n"
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.phase == "EXPLORE"


def test_phase_block_absent_gives_empty_string():
    prompt = "=== Shared session state ===\nsession_id=sess\n=== Inbox for robustness ===\n(no new messages)\n"
    ctx = from_coordinator_prompt(prompt)
    assert ctx.phase == ""


# ---------------------------------------------------------------------------
# Phase budget telemetry parsing
# ---------------------------------------------------------------------------


def test_phase_budget_parsed_correctly():
    prompt = (
        "=== Phase budget telemetry ===\n"
        "  PRELUDE: elapsed=120s cap=600s used=20%\n"
        "  EXPLORE: elapsed=300s cap=unlimited used=0%\n"
        "=== Shared session state ===\n"
        "session_id=s\n"
        "=== Inbox for robustness ===\n"
        "(no new messages)\n"
    )
    ctx = from_coordinator_prompt(prompt)
    assert len(ctx.phase_budget) == 2
    prelude = ctx.phase_budget[0]
    assert prelude.phase == "PRELUDE"
    assert prelude.elapsed_sec == 120
    assert prelude.cap_sec == 600
    assert prelude.used_pct == 20.0
    explore = ctx.phase_budget[1]
    assert explore.phase == "EXPLORE"
    assert explore.cap_sec == -1


def test_phase_budget_no_history_sentinel():
    prompt = (
        "=== Phase budget telemetry ===\n"
        "(no phase history yet)\n"
        "=== Shared session state ===\n"
        "session_id=s\n"
        "=== Inbox for robustness ===\n"
        "(no new messages)\n"
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.phase_budget == []


def test_phase_budget_absent_gives_empty_list():
    prompt = "=== Shared session state ===\nsession_id=s\n=== Inbox for robustness ===\n(no new messages)\n"
    ctx = from_coordinator_prompt(prompt)
    assert ctx.phase_budget == []


# ---------------------------------------------------------------------------
# Conversation progress parsing
# ---------------------------------------------------------------------------


def test_conversation_progress_parsed_correctly():
    prompt = (
        "=== Conversation progress ===\n"
        "ticks_without_progress=5 threshold=12 severity=ok last_progress_tick=42\n"
        "=== Shared session state ===\n"
        "session_id=s\n"
        "=== Inbox for robustness ===\n"
        "(no new messages)\n"
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.conversation_progress is not None
    cp = ctx.conversation_progress
    assert cp.ticks_without_progress == 5
    assert cp.threshold == 12
    assert cp.severity == "ok"
    assert cp.last_progress_tick == 42


def test_conversation_progress_high_severity():
    prompt = (
        "=== Conversation progress ===\n"
        "ticks_without_progress=15 threshold=12 severity=high last_progress_tick=3\n"
        "WARNING: no observable progress ...\n"
        "=== Shared session state ===\n"
        "session_id=s\n"
        "=== Inbox for robustness ===\n"
        "(no new messages)\n"
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.conversation_progress is not None
    assert ctx.conversation_progress.severity == "high"
    assert ctx.conversation_progress.ticks_without_progress == 15


def test_conversation_progress_absent_gives_none():
    prompt = "=== Shared session state ===\nsession_id=s\n=== Inbox for robustness ===\n(no new messages)\n"
    ctx = from_coordinator_prompt(prompt)
    assert ctx.conversation_progress is None
