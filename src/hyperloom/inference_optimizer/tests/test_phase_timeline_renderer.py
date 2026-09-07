# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the phase-timeline renderer."""

from __future__ import annotations

from hyperloom.inference_optimizer.breakdown.reporters._renderers.phase_timeline import render as render_phase_timeline


def test_phase_timeline_renderer_renders_capped_histogram() -> None:
    events = ["bootstrap"]
    events.extend(
        {
            "ts": f"t{i}",
            "action": f"action-{i}",
            "decision": "KEEP" if i % 2 == 0 else "REVERT",
            "task_id": f"task-{i}",
            "error_class": "RuntimeError" if i == 30 else "",
        }
        for i in range(31)
    )

    sec = render_phase_timeline({"phase_timeline": events})

    assert not sec.skipped
    assert any("Recorded 32 phase event(s); newest = `action-30` (KEEP)." in fact for fact in sec.key_facts)
    assert any("KEEP=16" in fact and "REVERT=15" in fact and "(none)=1" in fact for fact in sec.key_facts)
    assert "_Showing last 30 of 32 events._" in sec.markdown_block
    assert "action-0" not in sec.markdown_block
    assert "action-30" in sec.markdown_block
    assert "RuntimeError" in sec.markdown_block
