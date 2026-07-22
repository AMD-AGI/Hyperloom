# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for per-macro-cycle orchestration-prompt reseeding.

Covers the deterministic fallback render and _reseed_orch_prompt_for_cycle:
LLM directive wins when present, deterministic fallback otherwise, the
cycle_directive_history ring appends/caps, and a user --orch-prompt is never
clobbered. All offline; the prompt rebuild is stubbed.
"""

from __future__ import annotations

from types import SimpleNamespace

from hyperloom.orchestrator.phases.explore import ExplorePhase
from hyperloom.orchestrator.state.shared_state import SharedState


def _explore_with_stub_coordinator(
    *,
    macro_cycle: int = 1,
    next_cycle_directive: str = "",
    user_supplied: bool = False,
    plan_focus: dict | None = None,
) -> tuple[ExplorePhase, SimpleNamespace, list[dict]]:
    """Build an ExplorePhase over a minimal coordinator stub.

    Returns (phase, coord, rebuild_calls) where rebuild_calls records the kwargs
    passed to the stubbed prompt rebuilder.
    """
    st = SharedState(session_id="t", macro_cycle=macro_cycle)
    st.orchestration_memory = {"next_cycle_directive": next_cycle_directive}
    rebuild_calls: list[dict] = []

    def _rebuild(**kwargs) -> str:
        rebuild_calls.append(kwargs)
        return f"PROMPT[cycle={kwargs.get('macro_cycle')}|{kwargs.get('cycle_directive')}]"

    coord = SimpleNamespace(
        shared_state=st,
        system_prompt_overrides={"orchestration": "ORIGINAL"},
        _rebuild_orch_prompt=_rebuild,
        _orch_prompt_is_user_supplied=user_supplied,
    )
    phase = ExplorePhase(coord)
    if plan_focus is not None:
        phase._plan_cycle_focus = lambda: plan_focus  # type: ignore[method-assign]
    return phase, coord, rebuild_calls


def test_fallback_renders_focus_line():
    phase, _coord, _ = _explore_with_stub_coordinator(
        plan_focus={
            "focus": "comm_specialist",
            "rationale": "all_reduce dominates",
            "bottleneck_at_start": "all_reduce",
            "saturated_at_start": ["serving_specialist"],
        }
    )
    line = phase._cycle_directive_fallback()
    assert "focus=comm_specialist" in line
    assert "all_reduce dominates" in line
    assert "bottleneck=all_reduce" in line
    assert "deprioritize saturated=['serving_specialist']" in line


def test_fallback_empty_when_no_focus():
    phase, _coord, _ = _explore_with_stub_coordinator(plan_focus={"focus": ""})
    assert phase._cycle_directive_fallback() == ""


def test_reseed_llm_directive_wins():
    phase, coord, calls = _explore_with_stub_coordinator(
        macro_cycle=2,
        next_cycle_directive="Attack MoE dispatch; drop config sweeps.",
        plan_focus={"focus": "serving_specialist"},
    )
    assert phase._reseed_orch_prompt_for_cycle() is True
    assert calls[0]["macro_cycle"] == 2
    assert calls[0]["cycle_directive"] == "Attack MoE dispatch; drop config sweeps."
    assert "Attack MoE dispatch" in coord.system_prompt_overrides["orchestration"]
    hist = coord.shared_state.cycle_directive_history
    assert hist[-1]["source"] == "llm"
    assert hist[-1]["cycle"] == 2


def test_reseed_uses_deterministic_fallback_when_empty():
    phase, coord, calls = _explore_with_stub_coordinator(
        macro_cycle=3,
        next_cycle_directive="",
        plan_focus={"focus": "comm_specialist", "rationale": "rccl hot"},
    )
    assert phase._reseed_orch_prompt_for_cycle() is True
    assert "focus=comm_specialist" in calls[0]["cycle_directive"]
    hist = coord.shared_state.cycle_directive_history
    assert hist[-1]["source"] == "deterministic"


def test_reseed_skipped_for_user_supplied_prompt():
    phase, coord, calls = _explore_with_stub_coordinator(
        next_cycle_directive="ignored",
        user_supplied=True,
        plan_focus={"focus": "serving_specialist"},
    )
    assert phase._reseed_orch_prompt_for_cycle() is False
    assert calls == []
    assert coord.system_prompt_overrides["orchestration"] == "ORIGINAL"
    assert coord.shared_state.cycle_directive_history == []


def test_reseed_history_ring_caps_at_10():
    phase, coord, _ = _explore_with_stub_coordinator(
        next_cycle_directive="d",
        plan_focus={"focus": "serving_specialist"},
    )
    for i in range(15):
        coord.shared_state.macro_cycle = i
        phase._reseed_orch_prompt_for_cycle()
    hist = coord.shared_state.cycle_directive_history
    assert len(hist) == 10
    # Newest kept; oldest dropped.
    assert hist[-1]["cycle"] == 14
    assert hist[0]["cycle"] == 5
