# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Substream composition in the breakdown assembler.

Producers write one fragment per row and the assembler folds each substream into
the view its readers see. These pin what order the rows land in, and what a
session that produced none looks like.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.breakdown.recorder import assembler as asm


def test_critic_iterations_fold_in_the_order_the_agent_ran_them():
    out = {"critic_iteration": [{"iter": 2}, {"iter": 1}]}
    asm._compose_critic(out)

    assert "critic_iteration" not in out
    assert [r["iter"] for r in out["critic"]["iterations"]] == [1, 2]


def test_a_session_the_critic_never_reviewed_gets_no_section():
    out = {}
    asm._compose_critic(out)

    assert "critic" not in out


def test_robustness_turns_fold_in_turn_order():
    out = {
        "robustness_turn": [
            {"turn_idx": 2, "outcome": "intents"},
            {"turn_idx": 1, "outcome": "no_envelope"},
        ]
    }
    asm._compose_robustness(out)

    assert "robustness_turn" not in out
    assert [t["turn_idx"] for t in out["robustness"]["turns"]] == [1, 2]


def test_a_session_with_no_robustness_turns_gets_no_section():
    out = {}
    asm._compose_robustness(out)

    assert "robustness" not in out
