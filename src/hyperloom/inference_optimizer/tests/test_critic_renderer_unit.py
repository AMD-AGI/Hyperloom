# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the critic breakdown renderer.

The critic watches the session from outside the optimization loop, and its quiet
mode must not read as absence: a critic pass that ruled on nothing still says
the critic was asked.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.protocol.intent import DEFAULT_IDLE_BODY
from hyperloom.inference_optimizer.breakdown.reporters._renderers import critic as cr


def _critic(*iterations: dict) -> dict:
    return {"critic": {"iterations": list(iterations)}}


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


def test_pass_rows_are_numbered_in_order_even_when_the_recorded_counter_repeats():
    out = cr.render(_critic({"iter": 1, "topic": "a"}, {"iter": 2, "topic": "b"}, {"iter": 1, "topic": "c"}))
    rows = [line for line in out.markdown_block.splitlines() if line.startswith("| ") and "macro_cycle" not in line]

    assert [row.split(" | ")[0] for row in rows[:3]] == ["| 1", "| 2", "| 3"]


def test_the_idle_placeholder_is_not_shown_as_something_the_critic_said():
    out = cr.render(_critic({"iter": 1, "summary": DEFAULT_IDLE_BODY}, {"iter": 2, "summary": "slow gemm"}))
    rows = [line for line in out.markdown_block.splitlines() if line.startswith("| ") and "macro_cycle" not in line]

    assert [row.split(" | ")[-1] for row in rows[:2]] == ["— |", "slow gemm |"]


def test_both_tables_label_the_sequence_column_as_a_pass_number():
    out = cr.render(_critic({"iter": 7, "request_path": "r.json", "framework_reviews": [{"verdict": "approve"}]}))
    headers = [line for line in out.markdown_block.splitlines() if line.startswith("| #")]

    assert headers == [
        "| # | phase | macro_cycle | topic | rulings | said |",
        "| # | proposal | candidate | arm | verdict | effective | conf | failure_code | reasoning |",
    ]
    assert "**Artifacts of the last pass** (pass 1)" in out.markdown_block
