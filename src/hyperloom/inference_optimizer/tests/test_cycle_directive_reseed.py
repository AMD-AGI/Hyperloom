# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for per-macro-cycle orchestration-prompt reseeding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.loop.cycle_memory import CycleMemoryCollaborator as ExplorePhase
from hyperloom.orchestrator.state.shared_state import SharedState


def _explore_with_stub_coordinator(
    *,
    session_dir: Path | None = None,
    macro_cycle: int = 1,
    next_cycle_directive: str = "",
    user_supplied: bool = False,
    plan_focus: dict | None = None,
) -> tuple[ExplorePhase, SimpleNamespace, list[dict]]:
    """Build an ExplorePhase over a minimal coordinator stub."""
    st = SharedState(session_id="t", macro_cycle=macro_cycle)
    st.orchestration_memory = {"next_cycle_directive": next_cycle_directive}
    rebuild_calls: list[dict] = []

    def _rebuild(**kwargs) -> str:
        rebuild_calls.append(kwargs)
        return f"PROMPT[cycle={kwargs.get('macro_cycle')}|{kwargs.get('cycle_directive')}]"

    coord = SimpleNamespace(
        shared_state=st,
        session_dir=session_dir,
        system_prompt_overrides={"orchestration": "ORIGINAL"},
        _rebuild_orch_prompt=_rebuild,
        _orch_prompt_is_user_supplied=user_supplied,
    )
    phase = ExplorePhase(coord)
    if plan_focus is not None:
        phase._plan_cycle_focus = lambda: plan_focus  # type: ignore[method-assign]
    return phase, coord, rebuild_calls


def test_reseed_llm_directive_passed_through(tmp_path):
    phase, coord, calls = _explore_with_stub_coordinator(
        session_dir=tmp_path,
        macro_cycle=2,
        next_cycle_directive="Attack MoE dispatch; drop config sweeps.",
        plan_focus={"focus": "serving_specialist"},
    )
    assert phase._reseed_orch_prompt_for_cycle() is True
    assert calls[0]["macro_cycle"] == 2
    assert calls[0]["cycle_directive"] == "Attack MoE dispatch; drop config sweeps."
    assert "Attack MoE dispatch" in coord.system_prompt_overrides["orchestration"]


def test_reseed_passes_cycle_strategy_when_no_directive(tmp_path):
    phase, coord, calls = _explore_with_stub_coordinator(
        session_dir=tmp_path,
        macro_cycle=3,
        next_cycle_directive="",
        plan_focus={"focus": "comm_specialist", "rationale": "rccl hot"},
    )
    assert phase._reseed_orch_prompt_for_cycle() is True
    assert calls[0]["cycle_directive"] == ""
    assert calls[0].get("cycle_strategy") is not None
    assert calls[0]["cycle_strategy"]["focus"] == "comm_specialist"


def test_reseed_skipped_for_user_supplied_prompt():
    phase, coord, calls = _explore_with_stub_coordinator(
        next_cycle_directive="ignored",
        user_supplied=True,
        plan_focus={"focus": "serving_specialist"},
    )
    assert phase._reseed_orch_prompt_for_cycle() is False
    assert calls == []
    assert coord.system_prompt_overrides["orchestration"] == "ORIGINAL"


def _explore_with_memory_backend(*, raw_text: str, previous: dict | None = None):
    """An ExplorePhase whose orchestration backend replies with ``raw_text``."""
    st = SharedState(session_id="t")
    st.orchestration_memory = dict(previous or {})

    class _Backend:
        async def run(self, **_kwargs):
            return SimpleNamespace(raw_text=raw_text)

    phase = ExplorePhase(
        SimpleNamespace(
            shared_state=st,
            session_dir=None,
            backends={"orchestration": _Backend()},
        )
    )

    async def _stub(_agent: str) -> str:
        return "STUB"

    phase._compose_prompt = _stub  # type: ignore[method-assign]
    phase._load_system_prompt = _stub  # type: ignore[method-assign]
    return phase, st


@pytest.mark.asyncio
async def test_capture_warns_when_the_reply_carries_no_json(caplog):
    phase, st = _explore_with_memory_backend(
        raw_text="I could not produce JSON, sorry.",
        previous={
            "current_plan": "drive down decode latency",
            "next_cycle_directive": "attack the KV cache",
        },
    )

    with caplog.at_level("WARNING"):
        assert await phase._capture_cycle_memory() is True

    assert "no JSON object found" in caplog.text
    # The directive steers the next cycle, so it is the one that must survive.
    assert st.orchestration_memory["next_cycle_directive"] == "attack the KV cache"
    # An unparseable reply is salvaged as prose rather than discarded.
    assert st.orchestration_memory["current_plan"] == "I could not produce JSON, sorry."


@pytest.mark.asyncio
async def test_capture_is_quiet_when_the_reply_parses(caplog):
    phase, st = _explore_with_memory_backend(
        raw_text='```json\n{"current_plan": "new plan", "next_cycle_directive": "go deep on attention"}\n```'
    )

    with caplog.at_level("WARNING"):
        assert await phase._capture_cycle_memory() is True

    assert "_capture_cycle_memory" not in caplog.text
    assert st.orchestration_memory["next_cycle_directive"] == "go deep on attention"
