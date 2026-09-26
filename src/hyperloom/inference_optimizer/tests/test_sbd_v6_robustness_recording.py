# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""V6 compatibility for historical robustness fragments and empty new sessions."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.inference_optimizer.breakdown.collectors.v6_robustness import collect_v6_robustness
from hyperloom.inference_optimizer.breakdown.recorder.assembler import assemble_parts
from hyperloom.inference_optimizer.breakdown.reporters._renderers.robustness import render


def test_new_sessions_keep_an_empty_fixed_wire_block(tmp_path: Path) -> None:
    assert collect_v6_robustness(assemble_parts(tmp_path, warnings=[]).get("robustness")) == {"turns": []}


def test_historical_turn_fragments_preserve_outcomes_and_payloads(tmp_path: Path) -> None:
    directory = tmp_path / "runtime" / "breakdown" / "parts"
    directory.mkdir(parents=True)
    turns = [
        {"turn_idx": 2, "outcome": "no_envelope", "detail": "missing envelope"},
        {"turn_idx": 1, "outcome": "intents", "intents": [{"type": "alert", "severity": "high"}]},
    ]
    for row in turns:
        (directory / f"turn-{row['turn_idx']}.json").write_text(
            json.dumps(
                {
                    "section": "robustness_turn",
                    "producer": "robustness",
                    "kind": "item",
                    "seq": row["turn_idx"],
                    "ts": "2026-08-01T00:00:00Z",
                    "payload": row,
                }
            ),
            encoding="utf-8",
        )
    block = collect_v6_robustness(assemble_parts(tmp_path, warnings=[]).get("robustness"))
    assert block == {"turns": list(reversed(turns))}


def test_empty_robustness_section_is_hidden() -> None:
    section = render({"robustness": {"turns": []}})
    assert section.skipped is True


def test_full_report_does_not_advertise_empty_robustness() -> None:
    from hyperloom.inference_optimizer.breakdown.reporters.compose import render_session_report

    result = render_session_report({"critic": {"iterations": [{"iter": 1}]}, "robustness": {"turns": []}})
    assert "### Critic" in result.markdown
    assert "Robustness" not in result.markdown


def test_historical_robustness_section_remains_visible() -> None:
    section = render({"robustness": {"turns": [{"turn_idx": 1, "outcome": "no_envelope"}]}})
    assert section.skipped is False
