# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Substream composition in the breakdown assembler.

Producers write one fragment per row and the assembler folds each substream into
the view its readers see. These pin what order the rows land in, and what a
session that produced none looks like.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.breakdown.recorder import assembler as asm


def test_critic_iterations_fold_in_timestamp_order_whatever_the_resumed_counter_says():
    out = {
        "critic_iteration": [
            {"iter": 1, "ts": "2026-01-01T00:02:00Z", "leg": "resumed"},
            {"iter": 2, "ts": "2026-01-01T00:01:00Z"},
            {"iter": 3},
            {"iter": 3, "ts": "2026-01-01T00:00:00Z"},
            {"iter": 0},
        ]
    }
    asm._compose_critic(out)

    assert "critic_iteration" not in out
    assert [(r["iter"], r.get("ts")) for r in out["critic"]["iterations"]] == [
        (3, "2026-01-01T00:00:00Z"),
        (2, "2026-01-01T00:01:00Z"),
        (1, "2026-01-01T00:02:00Z"),
        (3, None),
        (0, None),
    ]


def test_a_session_the_critic_never_reviewed_gets_no_section():
    out = {}
    asm._compose_critic(out)

    assert "critic" not in out
