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
    assert ctx.shared_state.session_id == ""
    assert ctx.inbox == []
    assert ctx.parse_warnings == ["empty prompt"]


def test_no_new_messages_yields_empty_inbox():
    prompt = _prompt(
        textwrap.dedent(
            """\
            session_id=sess-1
            model=qwen3-8b  class=qwen3
            baseline_tput=10.5  baseline_acc=0.8
            cumulative_gain=12.5%
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
    assert ctx.shared_state.session_id == "sess-1"
    assert ctx.shared_state.model_name == "qwen3-8b"
    assert ctx.shared_state.model_class == "qwen3"
    assert ctx.shared_state.baseline_tput == 10.5
    assert ctx.shared_state.cumulative_gain == 12.5
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
    assert ctx.shared_state.session_id == "sess-2"
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
        "session_id=sess-kb\n"
        "crash_count=0\n"
        "=== Knowledge base hints ===\n"
        "kb-hint-do-not-parse\n"
        "=== Inbox for robustness ===\n"
        "(no new messages)\n"
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.session_id == "sess-kb"
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
            stop_reason=(none)
            crash_count=0
            """
        ),
        "=== Inbox for robustness ===\n(no new messages)\n",
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.tick == 42
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
# explore_started — flips True iff at least one of the four explore
# family ``last_*`` lines is non-``(none)``. Used by the
# ``no_levers_found`` signal to defer until exploration has actually
# been attempted (cold-start guard).
# ---------------------------------------------------------------------------

def test_explore_started_false_when_all_last_explore_keys_are_none():
    """Coordinator default rendering: pre-backends/params/sweep/
    validate_stack the four lines are all ``(none)``. The flag must
    stay False so ``no_levers_found`` defers."""
    prompt = _prompt(
        "session_id=s\n"
        "last_backends=(none)\n"
        "last_params=(none)\n"
        "last_sweep=(none)\n"
        "last_validate_stack=(none)\n"
        "crash_count=0\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.explore_started is False


def test_explore_started_true_when_any_last_explore_key_is_set():
    """First successful backends round renders e.g.
    ``last_backends=status=succeeded decision=promoted ...``. That
    flips ``explore_started`` to True even if last_params/sweep/
    validate_stack are still ``(none)``."""
    prompt = _prompt(
        "session_id=s\n"
        "last_backends=status=succeeded decision=promoted "
        "tput=600.50 err=- ws=/runs/backends/abc ts=2026-05-22T05:00:00Z\n"
        "last_params=(none)\n"
        "last_sweep=(none)\n"
        "last_validate_stack=(none)\n"
        "crash_count=0\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.explore_started is True


def test_explore_started_true_for_each_individual_explore_family():
    """Sanity: each of the four explore family keys flips the flag
    independently. Iterate so a future schema rename in shared_state.py
    fails one assertion rather than silently downgrading the gate."""
    for key in (
        "last_backends",
        "last_params",
        "last_sweep",
        "last_validate_stack",
    ):
        prompt = _prompt(
            f"session_id=s\n{key}=status=succeeded decision=promoted\n"
            "crash_count=0\n",
            "=== Inbox for robustness ===\n(no new messages)\n",
        )
        ctx = from_coordinator_prompt(prompt)
        assert ctx.shared_state.explore_started is True, (
            f"{key} should flip explore_started"
        )


def test_explore_started_default_false_when_keys_absent():
    """Legacy / partial prompts (e.g. tests that omit the explore
    family lines entirely) must default to False so the cold-start
    guard stays conservative."""
    prompt = _prompt(
        "session_id=s\ncrash_count=0\n",
        "=== Inbox for robustness ===\n(no new messages)\n",
    )
    ctx = from_coordinator_prompt(prompt)
    assert ctx.shared_state.explore_started is False
