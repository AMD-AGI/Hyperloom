"""Tests for :mod:`runtime.inbox_parser`.

The corpus mirrors what ``Coordinator._compose_prompt`` actually emits, plus a
handful of degraded inputs that we want to handle without crashing.
"""

from __future__ import annotations

import textwrap

from hyperloom.agents.critic.runtime.inbox_parser import (
    InboxRow,
    ParsedPrompt,
    parse_inbox_prompt,
)


def _build_prompt(*, shared: str, inbox_title: str, inbox_body: str) -> str:
    return (
        "=== Shared session state ===\n"
        f"{shared.rstrip()}\n"
        f"=== {inbox_title} ===\n"
        f"{inbox_body.rstrip()}\n"
    )


def test_parse_minimal_proposal_inbox():
    text = _build_prompt(
        shared="session_id=sess_1 model=Qwen3-14B framework=sglang baseline_tput=1200",
        inbox_title="Inbox for critic (newest last)",
        inbox_body=(
            "  seq=12 msg_id=abc1 from=orchestration topic=proposal "
            "payload={'action_name': 'kernel_opt', 'predicted_gain_pct': 4.2}"
        ),
    )
    parsed = parse_inbox_prompt(text)
    assert isinstance(parsed, ParsedPrompt)
    assert parsed.agent_name == "critic"
    assert parsed.shared_state["model"] == "Qwen3-14B"
    assert parsed.shared_state["framework"] == "sglang"
    assert parsed.shared_state["baseline_tput"] == "1200"
    assert len(parsed.inbox) == 1
    row = parsed.inbox[0]
    assert isinstance(row, InboxRow)
    assert row.seq == 12
    assert row.msg_id == "abc1"
    assert row.topic == "proposal"
    assert row.payload == {"action_name": "kernel_opt", "predicted_gain_pct": 4.2}
    assert len(parsed.proposals) == 1
    assert parsed.proposals[0].action_name == "kernel_opt"
    assert parsed.proposals[0].predicted_gain_pct == 4.2


def test_parse_multiple_topics_only_extracts_proposals():
    text = _build_prompt(
        shared="model=qwen framework=vllm",
        inbox_title="Inbox for critic (newest last)",
        inbox_body=(
            "  seq=1 msg_id=aaa1 from=orchestration topic=proposal payload={'action_name':'baseline'}\n"
            "  seq=2 msg_id=bbb2 from=orchestration topic=proposal payload={'action_name':'sweep'}\n"
            "  seq=3 msg_id=ccc3 from=robustness topic=alert payload={'severity':'low'}\n"
            "  seq=4 msg_id=ddd4 from=kernel topic=request payload={'kind':'kernel_opt'}"
        ),
    )
    parsed = parse_inbox_prompt(text)
    assert [r.topic for r in parsed.inbox] == ["proposal", "proposal", "alert", "request"]
    assert sorted(p.msg_id for p in parsed.proposals) == ["aaa1", "bbb2"]


def test_parse_inbox_with_no_messages_returns_empty():
    text = (
        "=== Shared session state ===\n"
        "model=qwen\n"
        "=== Inbox for critic ===\n"
        "(no new messages)\n"
    )
    parsed = parse_inbox_prompt(text)
    assert parsed.agent_name == "critic"
    assert parsed.inbox == []
    assert parsed.proposals == []


def test_unparseable_payload_does_not_emit_proposal_but_keeps_row():
    text = _build_prompt(
        shared="model=qwen",
        inbox_title="Inbox for critic",
        inbox_body=(
            "  seq=1 msg_id=zzz from=orchestration topic=proposal payload=NOT_A_DICT_AT_ALL"
        ),
    )
    parsed = parse_inbox_prompt(text)
    assert len(parsed.inbox) == 1
    assert parsed.inbox[0].payload is None
    assert parsed.inbox[0].raw_payload == "NOT_A_DICT_AT_ALL"
    assert parsed.proposals == []


def test_malformed_inbox_line_lands_in_extras():
    text = _build_prompt(
        shared="model=qwen",
        inbox_title="Inbox for critic",
        inbox_body=(
            "  seq=1 msg_id=ok from=o topic=proposal payload={'action_name':'baseline'}\n"
            "  totally bogus line"
        ),
    )
    parsed = parse_inbox_prompt(text)
    assert len(parsed.inbox) == 1
    assert "malformed_inbox" in parsed.extras
    assert "totally bogus line" in parsed.extras["malformed_inbox"]


def test_unknown_section_preserved_in_extras():
    text = (
        "=== Shared session state ===\n"
        "model=qwen\n"
        "=== Some Future Section ===\n"
        "anything goes here\n"
        "=== Inbox for critic ===\n"
        "(no new messages)\n"
    )
    parsed = parse_inbox_prompt(text)
    assert parsed.extras["Some Future Section"] == "anything goes here"


def test_kb_hints_section_preserved_for_orchestration_role():
    text = textwrap.dedent(
        """\
        === Shared session state ===
        model=qwen framework=sglang
        === Knowledge base hints ===
        - prefer dispatch fix before kernel rewrite
        === Inbox for orchestration (newest last) ===
        (no new messages)
        """
    )
    parsed = parse_inbox_prompt(text)
    assert parsed.agent_name == "orchestration"
    assert "dispatch" in parsed.kb_hints_text


def test_payload_with_nested_quotes_round_trips():
    payload_repr = "{'reason': \"can't compare\"}"
    text = _build_prompt(
        shared="model=qwen",
        inbox_title="Inbox for critic",
        inbox_body=f"  seq=1 msg_id=h1 from=o topic=proposal payload={payload_repr}",
    )
    parsed = parse_inbox_prompt(text)
    assert parsed.inbox[0].payload == {"reason": "can't compare"}


def test_payload_json_form_also_supported():
    text = _build_prompt(
        shared="model=qwen",
        inbox_title="Inbox for critic",
        inbox_body=(
            "  seq=1 msg_id=h1 from=o topic=proposal "
            'payload={"action_name": "baseline", "predicted_gain_pct": 0}'
        ),
    )
    parsed = parse_inbox_prompt(text)
    assert parsed.proposals[0].action_name == "baseline"
    assert parsed.proposals[0].predicted_gain_pct == 0.0


def test_shared_state_preserves_value_with_spaces():
    text = (
        "=== Shared session state ===\n"
        "model=Qwen3-14B objective=throughput baseline_label=Baseline before patch\n"
        "=== Inbox for critic ===\n"
        "(no new messages)\n"
    )
    parsed = parse_inbox_prompt(text)
    assert parsed.shared_state["baseline_label"] == "Baseline before patch"


def test_multiple_inbox_sections_are_treated_independently():
    text = (
        "=== Inbox for critic (newest last) ===\n"
        "  seq=1 msg_id=a from=o topic=proposal payload={'action_name':'baseline'}\n"
    )
    parsed = parse_inbox_prompt(text)
    assert parsed.agent_name == "critic"
    assert len(parsed.proposals) == 1
