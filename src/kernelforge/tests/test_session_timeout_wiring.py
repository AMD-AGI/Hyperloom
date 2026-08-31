"""The implementer session's deadline reaches the backend and the prompt.

Two contracts land together here. Part 2: ``make_agent_fn`` must set the run
spec's ``timeout_sec`` from the campaign-sized session budget it is handed, NOT
from ``backend.runtime.timeout_sec`` (whose 1800s default would cut every
session at 30 min once the claude backend started honouring it). Part 3: the
session must be TOLD its own deadline and told to hand off its best candidate
before then, or a bounded session simply gets killed mid-thought with nothing
submitted. Both are exercised through the real ``agent_fn`` with the SDK and
backend stubbed out -- no LLM / GPU / gateway.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest


@pytest.fixture()
def captured_run_spec(monkeypatch):
    """Run one implementer ``agent_fn`` and return the spec handed to the backend."""
    stub = types.ModuleType("claude_agent_sdk")
    stub.ClaudeAgentOptions = object
    stub.query = None
    stub.HookMatcher = object
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", stub)

    import kernelforge.orchestrator.agent as agent_mod
    from kernelforge.agent_backends.base import (
        AgentCapabilities,
        AgentRunResult,
        AgentRuntimeConfig,
    )
    from kernelforge.config import Config

    def run(*, session_timeout_sec, kernel_path, **make_kwargs):
        runtime = AgentRuntimeConfig(provider="claude", model="m", timeout_sec=1800)
        backend = types.SimpleNamespace(
            name="claude",
            runtime=runtime,
            capabilities=AgentCapabilities(
                writable=True,
                resumable=True,
                stop_hooks=True,
                native_subagents=True,
                mcp=False,
            ),
            fallback_reason="",
        )
        seen: dict = {}

        async def fake_resume_driver(be, spec, usage=None, **kw):
            seen["spec"] = spec
            return AgentRunResult(text="PLAN: did a thing", subtype="success")

        monkeypatch.setattr(agent_mod, "create_registered_backend", lambda *a, **k: backend)
        monkeypatch.setattr(agent_mod, "run_session_with_api_resume", fake_resume_driver)

        config = Config()
        config.workspace = ""
        config.agent_runtime = lambda: runtime
        make_kwargs.setdefault("program_md", "PROGRAM BODY")
        agent_fn = agent_mod.make_agent_fn(
            config=config,
            session_timeout_sec=session_timeout_sec,
            **make_kwargs,
        )
        asyncio.run(agent_fn(kernel_path, "(history)"))
        return seen["spec"]

    return run


def test_run_spec_timeout_comes_from_the_session_budget(tmp_path, captured_run_spec):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("# kernel\n")

    spec = captured_run_spec(session_timeout_sec=4321, kernel_path=str(kernel))

    # The campaign-sized budget, not the backend runtime's 1800s default.
    assert spec.timeout_sec == 4321


def test_prompt_states_the_deadline_and_the_handoff(tmp_path, captured_run_spec):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("# kernel\n")

    spec = captured_run_spec(session_timeout_sec=5400, kernel_path=str(kernel))

    prompt = spec.user_prompt
    # The number of minutes the session actually has (5400s == 90 min).
    assert "90" in prompt
    # And that it must hand off its best candidate before the clock runs out,
    # via the clean handoff path rather than being killed with nothing.
    lowered = prompt.lower()
    assert "candidate_submitted" in prompt or "candidate" in lowered
    assert "deadline" in lowered or "time" in lowered
