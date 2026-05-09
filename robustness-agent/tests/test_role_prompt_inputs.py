"""Unit tests for prompt -> ReactorContext parser."""

from __future__ import annotations

import textwrap

from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    from_coordinator_prompt,
)


def _prompt(shared: str, inbox: str) -> str:
    parts = ["=== Shared session state ===", shared.rstrip(), inbox.rstrip()]
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
