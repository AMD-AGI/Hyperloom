# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for per-macro-cycle orchestration-prompt reseeding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.loop.cycle_memory import CycleMemoryCollaborator
from hyperloom.orchestrator.state.shared_state import SharedState


def _memory_with_stub_coordinator(
    *,
    session_dir: Path | None = None,
    macro_cycle: int = 1,
    next_cycle_directive: str = "",
    user_supplied: bool = False,
    plan_focus: dict | None = None,
) -> tuple[CycleMemoryCollaborator, list[dict]]:
    """Build a CycleMemoryCollaborator over a minimal coordinator stub."""
    st = SharedState(session_id="t", macro_cycle=macro_cycle)
    st.orchestration_memory = {"next_cycle_directive": next_cycle_directive}
    rebuild_calls: list[dict] = []

    def _rebuild(**kwargs) -> str:
        rebuild_calls.append(kwargs)
        return f"PROMPT[cycle={kwargs.get('macro_cycle')}|{kwargs.get('cycle_directive')}]"

    memory = CycleMemoryCollaborator()
    vars(memory).update(
        shared_state=st,
        session_dir=session_dir,
        system_prompt_overrides={"orchestration": "ORIGINAL"},
        _rebuild_orch_prompt=_rebuild,
        _orch_prompt_is_user_supplied=user_supplied,
    )
    if plan_focus is not None:
        memory._plan_cycle_focus = lambda: plan_focus  # type: ignore[method-assign]
    return memory, rebuild_calls


def test_fallback_renders_focus_line():
    memory, _ = _memory_with_stub_coordinator(
        plan_focus={
            "focus": "comm_specialist",
            "rationale": "all_reduce dominates",
            "bottleneck_at_start": "all_reduce",
            "saturated_at_start": ["serving_specialist"],
        }
    )
    line = memory._cycle_directive_fallback()
    assert "focus=comm_specialist" in line
    assert "all_reduce dominates" in line
    assert "bottleneck=all_reduce" in line
    assert "deprioritize saturated=['serving_specialist']" in line


def test_fallback_empty_when_no_focus():
    memory, _ = _memory_with_stub_coordinator(plan_focus={"focus": ""})
    assert memory._cycle_directive_fallback() == ""


def test_reseed_llm_directive_wins(tmp_path):
    memory, calls = _memory_with_stub_coordinator(
        session_dir=tmp_path,
        macro_cycle=2,
        next_cycle_directive="Attack MoE dispatch; drop config sweeps.",
        plan_focus={"focus": "serving_specialist"},
    )
    assert memory._reseed_orch_prompt_for_cycle() is True
    assert calls[0]["macro_cycle"] == 2
    assert calls[0]["cycle_directive"] == "Attack MoE dispatch; drop config sweeps."
    assert "Attack MoE dispatch" in memory.system_prompt_overrides["orchestration"]
    hist = memory.shared_state.cycle_directive_history
    assert hist[-1]["source"] == "llm"
    assert hist[-1]["cycle"] == 2


def test_reseed_uses_deterministic_fallback_when_empty(tmp_path):
    memory, calls = _memory_with_stub_coordinator(
        session_dir=tmp_path,
        macro_cycle=3,
        next_cycle_directive="",
        plan_focus={"focus": "comm_specialist", "rationale": "rccl hot"},
    )
    assert memory._reseed_orch_prompt_for_cycle() is True
    assert "focus=comm_specialist" in calls[0]["cycle_directive"]
    hist = memory.shared_state.cycle_directive_history
    assert hist[-1]["source"] == "deterministic"


def test_reseed_skipped_for_user_supplied_prompt():
    memory, calls = _memory_with_stub_coordinator(
        next_cycle_directive="ignored",
        user_supplied=True,
        plan_focus={"focus": "serving_specialist"},
    )
    assert memory._reseed_orch_prompt_for_cycle() is False
    assert calls == []
    assert memory.system_prompt_overrides["orchestration"] == "ORIGINAL"
    assert memory.shared_state.cycle_directive_history == []


def test_reseed_history_ring_caps_at_10(tmp_path):
    memory, _ = _memory_with_stub_coordinator(
        session_dir=tmp_path,
        next_cycle_directive="d",
        plan_focus={"focus": "serving_specialist"},
    )
    for i in range(15):
        memory.shared_state.macro_cycle = i
        memory._reseed_orch_prompt_for_cycle()
    hist = memory.shared_state.cycle_directive_history
    assert len(hist) == 10
    # Newest kept; oldest dropped.
    assert hist[-1]["cycle"] == 14
    assert hist[0]["cycle"] == 5


def _memory_with_backend(*, raw_text: str, previous: dict | None = None):
    """A CycleMemoryCollaborator whose orchestration backend replies with ``raw_text``."""
    st = SharedState(session_id="t")
    st.orchestration_memory = dict(previous or {})

    class _Backend:
        async def run(self, **_kwargs):
            return SimpleNamespace(raw_text=raw_text)

    memory = CycleMemoryCollaborator()
    vars(memory).update(
        shared_state=st,
        session_dir=None,
        backends={"orchestration": _Backend()},
    )

    async def _stub(_agent: str) -> str:
        return "STUB"

    memory._compose_prompt = _stub  # type: ignore[method-assign]
    memory._load_system_prompt = _stub  # type: ignore[method-assign]
    return memory, st


@pytest.mark.asyncio
async def test_capture_warns_when_the_reply_carries_no_json(caplog):
    memory, st = _memory_with_backend(
        raw_text="I could not produce JSON, sorry.",
        previous={
            "current_plan": "drive down decode latency",
            "next_cycle_directive": "attack the KV cache",
        },
    )

    with caplog.at_level("WARNING"):
        assert await memory._capture_cycle_memory() is True

    assert "no JSON object found" in caplog.text
    # The directive steers the next cycle, so it is the one that must survive.
    assert st.orchestration_memory["next_cycle_directive"] == "attack the KV cache"
    # An unparseable reply is salvaged as prose rather than discarded.
    assert st.orchestration_memory["current_plan"] == "I could not produce JSON, sorry."


@pytest.mark.asyncio
async def test_capture_is_quiet_when_the_reply_parses(caplog):
    memory, st = _memory_with_backend(
        raw_text='```json\n{"current_plan": "new plan", "next_cycle_directive": "go deep on attention"}\n```'
    )

    with caplog.at_level("WARNING"):
        assert await memory._capture_cycle_memory() is True

    assert "_capture_cycle_memory" not in caplog.text
    assert st.orchestration_memory["next_cycle_directive"] == "go deep on attention"
