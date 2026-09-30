# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for orchestration-memory capture."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.loop.cycle_memory import CycleMemoryCollaborator as ExplorePhase
from hyperloom.orchestrator.state.shared_state import SharedState


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
