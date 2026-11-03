# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lane prompts and the shared prompt cache: the prefix every lane shares and when sibling lanes start."""

from __future__ import annotations

import asyncio
import json

import pytest

from kernelforge.workflows.forge_loop.roles import lane_prompts as lane_prompts_module
from kernelforge.workflows.forge_loop.roles.lane_prompts import lane_last_prompt
from kernelforge.agent_backends import AgentCapabilities, AgentRunResult
from kernelforge.workflows.forge_loop.roles.orchestration import OrchestrationAgent
from kernelforge.tests.workflows.forge_loop.roles._orchestration_fakes import (
    GROUND_A,
    GROUND_B,
    _PerCallBackend,
    _compute_and_memory_round,
    _context,
    _partition,
)


# ── lanes sharing the round's cached prefix ───────────────────────────────────


class _PrefixCachingBackend:
    """Reports each session's response start, and records when every session started and ended."""

    capabilities = AgentCapabilities(shared_prompt_cache=True)

    def __init__(self, replies: list[str], *, lead_answers: bool = True, lead_silence_sec: float = 0.0) -> None:
        self.replies = list(replies)
        self.lead_answers = lead_answers
        self.lead_silence_sec = lead_silence_sec
        self.specs: list = []
        self.events: list[tuple[str, str]] = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.specs.append(spec)
        reply = self.replies[len(self.specs) - 1]
        self.events.append(("start", spec.role))
        await asyncio.sleep(0)
        if spec.on_response_started is not None:
            await asyncio.sleep(self.lead_silence_sec)
            if not self.lead_answers:
                raise RuntimeError("the lead lane ended before it answered")
            self.events.append(("response", spec.role))
            spec.on_response_started()
        await asyncio.sleep(0.01)
        self.events.append(("end", spec.role))
        return AgentRunResult(text=reply, end_reason="agent_stopped")


def test_the_lane_block_comes_last_and_everything_before_it_is_shared() -> None:
    payload = {"task": "plan", "context": {"b": 1, "a": 2}, "lane": {"ground": "mine"}, "specialist_analyses": ["x"]}

    prompt, shared = lane_last_prompt(payload)

    assert list(json.loads(prompt)) == ["context", "specialist_analyses", "task", "lane"]
    assert json.loads(prompt) == payload
    assert prompt.startswith(shared)
    assert "mine" not in shared and "specialist_analyses" in shared


@pytest.mark.asyncio
async def test_sibling_lanes_start_once_the_first_lane_is_answering() -> None:
    """Started together, every lane writes the round's prefix; started after the first answers, they read it."""
    backend = _PrefixCachingBackend([_partition(GROUND_A, GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    results = await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert [result.text for result in results] == ["# Lane A", "# Lane B"]
    lead, sibling = "orchestration lane 1 synthesis", "orchestration lane 2 synthesis"
    events = backend.events
    assert events.index(("response", lead)) < events.index(("start", sibling)) < events.index(("end", lead))
    lanes = backend.specs[1:]
    assert lanes[0].shared_prompt_prefix == lanes[1].shared_prompt_prefix
    assert all(spec.user_prompt.startswith(spec.shared_prompt_prefix) for spec in lanes)
    assert lanes[1].on_response_started is None


@pytest.mark.asyncio
async def test_siblings_still_run_when_the_first_lane_never_answers() -> None:
    backend = _PrefixCachingBackend([_partition(GROUND_A, GROUND_B), "# Lane A", "# Lane B"], lead_answers=False)
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    results = await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert [result.text for result in results] == ["# Lane B"]


@pytest.mark.asyncio
async def test_siblings_stop_waiting_for_a_first_lane_that_stays_silent(monkeypatch) -> None:
    """A lead retrying or stuck before its first event must not hold the round for its whole timeout."""
    monkeypatch.setattr(lane_prompts_module, "LEAD_RESPONSE_WAIT_SEC", 0.05)
    backend = _PrefixCachingBackend([_partition(GROUND_A, GROUND_B), "# Lane A", "# Lane B"], lead_silence_sec=0.5)
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    results = await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert [result.text for result in results] == ["# Lane A", "# Lane B"]
    lead, sibling = "orchestration lane 1 synthesis", "orchestration lane 2 synthesis"
    assert backend.events.index(("start", sibling)) < backend.events.index(("response", lead))


@pytest.mark.asyncio
async def test_a_backend_without_a_shared_prompt_cache_starts_every_lane_at_once() -> None:
    backend = _PerCallBackend([_partition(GROUND_A, GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert all(spec.on_response_started is None for spec in backend.specs)
