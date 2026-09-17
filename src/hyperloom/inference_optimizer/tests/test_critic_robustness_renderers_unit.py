# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the critic and robustness breakdown renderers.

Both agents watch the session from outside the optimization loop, and both have
a quiet mode that must not read as absence: a critic pass that ruled on nothing
still says the critic was asked, and a robustness turn that produced no
envelope still says the agent was reached.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.breakdown.reporters._renderers import critic as cr
from hyperloom.inference_optimizer.breakdown.reporters._renderers import robustness as rb


def _critic(*iterations: dict) -> dict:
    return {"critic": {"iterations": list(iterations)}}


def _robustness(*turns: dict) -> dict:
    return {"robustness": {"turns": list(turns)}}


# ---- critic ----


def test_a_session_without_a_critic_is_skipped():
    assert cr.render({}).skipped is True
    assert cr.render(_critic()).skipped is True


def test_rulings_are_totalled_across_passes():
    out = cr.render(
        _critic(
            {"iter": 1, "verdict_counts": {"approve": 2, "reject": 1}},
            {"iter": 2, "verdict_counts": {"approve": 1}},
        )
    )
    assert "3 approve, 1 reject" in " ".join(out.key_facts)


def test_a_pass_that_ruled_on_nothing_is_still_reported():
    out = cr.render(_critic({"iter": 1, "topic": "heartbeat", "summary": "ok", "verdict_counts": {}}))
    assert out.skipped is False
    assert "never ruled on a proposal" in " ".join(out.key_facts)
    assert "heartbeat" in out.markdown_block


def test_framework_rulings_show_the_effective_verdict_only_when_it_differs():
    held = cr.render(_critic({"iter": 1, "framework_reviews": [{"verdict": "reject", "effective_verdict": "advise"}]}))
    assert "advise" in held.markdown_block

    agreed = cr.render(
        _critic({"iter": 1, "framework_reviews": [{"verdict": "approve", "effective_verdict": "approve"}]})
    )
    rulings = agreed.markdown_block.split("**Framework rulings**", 1)[1]
    assert rulings.count("approve") == 1


def test_knowledge_base_use_is_reported_only_when_it_was_wired_up():
    off = cr.render(_critic({"iter": 1, "verdict_counts": {}}))
    assert not any("Knowledge base" in f for f in off.key_facts)

    on = cr.render(
        _critic({"iter": 1, "kb_priors": {"configured": True, "prior_count": 3, "referenced_in_verdict": True}})
    )
    assert any("supplied 3 prior(s), and was cited in 1 verdict(s)" in f for f in on.key_facts)


def test_a_malformed_verdict_count_does_not_lose_the_section():
    out = cr.render(_critic({"iter": 1, "verdict_counts": {"approve": "many"}}))
    assert out.skipped is False


# ---- robustness ----


def test_a_session_without_a_robustness_agent_is_skipped():
    assert rb.render({}).skipped is True
    assert rb.render(_robustness()).skipped is True


def test_a_turn_that_produced_no_envelope_is_distinguished_from_a_quiet_one():
    out = rb.render(
        _robustness(
            {"turn_idx": 0, "outcome": "intents", "intents": []},
            {"turn_idx": 1, "outcome": "no_envelope", "intents": []},
        )
    )
    assert "1 turn(s) produced no usable envelope" in " ".join(out.key_facts)


def test_intents_are_counted_by_severity():
    out = rb.render(
        _robustness(
            {"turn_idx": 0, "intents": [{"type": "raise", "severity": "warn"}, {"type": "raise", "severity": "warn"}]},
            {"turn_idx": 1, "intents": [{"type": "raise", "severity": "error"}]},
        )
    )
    assert "1 error, 2 warn" in " ".join(out.key_facts)


def test_what_the_agent_said_reaches_the_table():
    out = rb.render(
        _robustness({"turn_idx": 0, "intents": [{"type": "send_message", "payload": {"body_md": "disk is filling"}}]})
    )
    assert "disk is filling" in out.markdown_block


def test_parse_warnings_are_surfaced():
    out = rb.render(_robustness({"turn_idx": 0, "intents": [], "parse_warnings": ["truncated json"]}))
    assert "1 turn(s) carried parse warnings" in " ".join(out.key_facts)
    assert "truncated json" in out.markdown_block
