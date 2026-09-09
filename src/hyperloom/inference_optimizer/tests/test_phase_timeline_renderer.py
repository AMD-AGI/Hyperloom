# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the phase-timeline renderer."""

from __future__ import annotations

from typing import Any

from hyperloom.inference_optimizer.breakdown.reporters._renderers.phase_timeline import render as render_phase_timeline


def _phase_event(phase: str, macro_cycle: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "phase", "ext": {"phase": phase, "macro_cycle": macro_cycle, "actions": {"rows": rows}}}


def test_phase_timeline_renderer_renders_capped_histogram() -> None:
    # Rows arrive spread over the phase events that dispatched them, and the
    # renderer's job is to read them back as one run-long sequence.
    rows = [
        {
            "settled_at": f"t{i}",
            "action": f"action-{i}",
            "status": "settled",
            "decision": "KEEP" if i % 2 == 0 else "REVERT",
            "task_id": f"task-{i}",
            "error_class": "RuntimeError" if i == 30 else "",
        }
        for i in range(31)
    ]
    breakdown = {
        "timeline": [
            _phase_event("EXPLORE", 1, rows[:10]),
            {"type": "kernel", "ext": {}},  # a non-phase event contributes nothing
            _phase_event("EXPLOIT", 2, rows[10:]),
        ]
    }

    sec = render_phase_timeline(breakdown)

    assert not sec.skipped
    assert any("Recorded 31 action(s); newest = `action-30` (KEEP)." in fact for fact in sec.key_facts)
    assert any("KEEP=16" in fact and "REVERT=15" in fact for fact in sec.key_facts)
    assert "_Showing last 30 of 31 actions._" in sec.markdown_block
    assert "action-0 " not in sec.markdown_block
    assert "action-30" in sec.markdown_block
    assert "RuntimeError" in sec.markdown_block
    # The phase that dispatched a row travels with it.
    assert "EXPLOIT/2" in sec.markdown_block


def test_a_dispatch_that_never_settled_is_reported_as_still_running() -> None:
    # The gap this renderer exists to show: the flat projection it replaces held
    # settled rows only, so an action killed mid-flight read as one that never
    # happened.
    breakdown = {
        "timeline": [
            _phase_event(
                "EXPLORE",
                1,
                [
                    {"dispatched_at": "t0", "action": "explore", "status": "settled", "decision": "KEEP"},
                    {"dispatched_at": "t1", "action": "kernel_opt"},
                ],
            )
        ]
    }

    sec = render_phase_timeline(breakdown)

    assert not sec.skipped
    assert any("1 action(s) were dispatched and never settled." in fact for fact in sec.key_facts)


def test_a_run_with_no_action_rows_is_skipped_not_rendered_empty() -> None:
    sec = render_phase_timeline({"timeline": [_phase_event("EXPLORE", 1, [])]})

    assert sec.skipped
    assert sec.markdown_block == ""
    assert sec.warnings
