# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The execution-budget section states the deadline and asks for blocking waits, not clock or poll turns."""

from __future__ import annotations

from hyperloom.orchestrator.prompts import specialist_prompt_builder as spb


def _section(**kw) -> str:
    inp = spb.SpecialistPromptInputs(task_id="t", domain="freeform_specialist", workspace_path="/ws", **kw)
    return "\n".join(spb._section_execution_budget(inp))


def test_the_deadline_is_stated_so_no_turn_has_to_compute_it():
    text = _section(wall_budget_sec=2700, started_at_iso="2026-09-25T09:42:12Z")
    assert "hard deadline **2026-09-25T10:27:12Z** (UTC)" in text
    assert "vs the start above" not in text
    assert "never in a turn of its own" in text


def test_waits_are_one_blocking_command_that_keeps_the_heartbeat_fresh():
    text = _section(wall_budget_sec=600, started_at_iso="2026-09-25T09:42:12+00:00")
    assert "ONE blocking" in text
    assert "> /ws/heartbeat.json" in text
    assert "``sleep`` turn" in text


def test_an_unparseable_start_or_no_budget_degrades_quietly():
    assert spb._deadline_iso("not-a-time", 60) == ""
    assert "hard deadline" not in _section(wall_budget_sec=600, started_at_iso="yesterday")
    assert _section(wall_budget_sec=0) == ""
