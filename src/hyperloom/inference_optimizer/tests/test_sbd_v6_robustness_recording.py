# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``robustness`` key records what the agent raised, when it raised it.

The section this replaces cannot be checked against: it rebuilt each turn from
``signal.json`` / ``action.json``, filenames nothing writes, so every row it
produced was blank and "equal to the old value" would only prove both are
empty. These tests are therefore positive -- a turn that raised intents must
carry them, and a turn the agent could not complete must say which way it
failed rather than leaving no trace at all.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.inference_optimizer.breakdown.collectors.v6_robustness import collect_v6_robustness
from hyperloom.inference_optimizer.breakdown.recorder.assembler import assemble_parts
from hyperloom.inference_optimizer.breakdown.recorder.robustness_out import (
    OUTCOME_INTENTS,
    OUTCOME_INVALID_ENVELOPE,
    OUTCOME_NO_ENVELOPE,
    record_robustness_turn,
)


class _Intent:
    """An intent in the shape ``validate_envelope`` returns."""

    def __init__(self, kind: str, payload: dict) -> None:
        self.type = type("_Kind", (), {"value": kind})()
        self.payload = payload


def _turns(session_dir: Path) -> list[dict]:
    return collect_v6_robustness(assemble_parts(session_dir, warnings=[]).get("robustness"))["turns"]


def test_a_raised_intent_carries_its_type_and_severity(tmp_path: Path) -> None:
    record_robustness_turn(
        tmp_path,
        turn_idx=3,
        outcome=OUTCOME_INTENTS,
        tick_index=41,
        intents=[_Intent("alert", {"severity": "high", "topic": "crash_rate"})],
        parse_warnings=["truncated tail"],
        workdir=tmp_path / "robustness-workdir" / "003",
    )

    (turn,) = _turns(tmp_path)
    assert turn["turn_idx"] == 3
    assert turn["outcome"] == OUTCOME_INTENTS
    assert turn["tick_index"] == 41
    assert turn["intents"] == [
        {"type": "alert", "severity": "high", "topic": "crash_rate", "payload": {"severity": "high", "topic": "crash_rate"}}
    ]
    assert turn["parse_warnings"] == ["truncated tail"]
    assert turn["workdir"].endswith("003")


def test_a_mute_agent_is_distinguishable_from_a_quiet_session(tmp_path: Path) -> None:
    """The two used to look identical: both produced blank rows or none."""
    record_robustness_turn(
        tmp_path,
        turn_idx=1,
        outcome=OUTCOME_NO_ENVELOPE,
        detail="emit.json missing intent_envelope",
    )
    record_robustness_turn(
        tmp_path,
        turn_idx=2,
        outcome=OUTCOME_INVALID_ENVELOPE,
        detail="intent 0: unknown type 'escalate_now'",
    )

    outcomes = [(t["turn_idx"], t["outcome"], t["detail"]) for t in _turns(tmp_path)]
    assert outcomes == [
        (1, OUTCOME_NO_ENVELOPE, "emit.json missing intent_envelope"),
        (2, OUTCOME_INVALID_ENVELOPE, "intent 0: unknown type 'escalate_now'"),
    ]


def test_a_session_the_agent_never_spoke_in_reports_no_turns(tmp_path: Path) -> None:
    assert _turns(tmp_path) == []


def test_re_recording_a_turn_overwrites_only_its_own_row(tmp_path: Path) -> None:
    record_robustness_turn(tmp_path, turn_idx=1, outcome=OUTCOME_NO_ENVELOPE)
    record_robustness_turn(tmp_path, turn_idx=2, outcome=OUTCOME_INTENTS)
    record_robustness_turn(
        tmp_path,
        turn_idx=1,
        outcome=OUTCOME_INTENTS,
        intents=[_Intent("heartbeat", {})],
    )

    turns = _turns(tmp_path)
    assert [t["turn_idx"] for t in turns] == [1, 2]
    assert turns[0]["intents"] == [{"type": "heartbeat"}]


def test_an_intent_without_a_type_is_not_a_row(tmp_path: Path) -> None:
    record_robustness_turn(
        tmp_path,
        turn_idx=1,
        outcome=OUTCOME_INTENTS,
        intents=[_Intent("", {"severity": "low"}), {"type": "alert"}],
    )

    (turn,) = _turns(tmp_path)
    assert turn["intents"] == [{"type": "alert"}]
