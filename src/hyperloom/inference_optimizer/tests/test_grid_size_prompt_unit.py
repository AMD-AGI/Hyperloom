# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The grid author's two prompt surfaces agree on where a grid comes from and how big it gets."""

from __future__ import annotations


def _grid_prompt_surfaces() -> list[str]:
    from hyperloom.orchestrator.prompts.prompt_builder import (
        _idea_generation_lines,
        _format_grid_injection_hint,
    )

    return [_format_grid_injection_hint("explore") or "", "\n".join(_idea_generation_lines())]


def test_both_grid_surfaces_state_the_same_target_and_ceiling():
    for surface in _grid_prompt_surfaces():
        assert "4" in surface
        assert "maximum 6" in surface
    assert all("Untested proposals (current cycle)" in s for s in _grid_prompt_surfaces()[1:])


def test_the_stale_proposal_set_wording_is_gone():
    from pathlib import Path

    from hyperloom.orchestrator.prompts import prompt_builder

    stale = "proposal_set drives the next"
    assert stale not in Path(prompt_builder.__file__).read_text(encoding="utf-8")
    md = Path(prompt_builder.__file__).parent / "orchestration.md"
    assert "specialist proposal_set" not in md.read_text(encoding="utf-8")
