"""The implementer session's deadline reaches the backend and the prompt."""

from __future__ import annotations

import asyncio
import hashlib
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture()
def captured_run_spec(monkeypatch):
    """Run one implementer ``agent_fn`` and return the spec handed to the backend."""
    stub = types.ModuleType("claude_agent_sdk")
    stub.ClaudeAgentOptions = object
    stub.query = None
    stub.HookMatcher = object
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", stub)

    import kernelforge.workflows.forge_loop.roles.agent as agent_mod
    from kernelforge.agent_backends.base import (
        AgentCapabilities,
        AgentRunResult,
        AgentRuntimeConfig,
    )
    from kernelforge.config import Config

    def run(*, session_timeout_sec, kernel_path, gpu_target="gfx942", **make_kwargs):
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

        monkeypatch.setattr(agent_mod, "agent_runtime", lambda _config: runtime)

        config = Config(gpu_target=gpu_target)
        config.workspace = ""
        make_kwargs.setdefault("program_md", "PROGRAM BODY")
        agent_fn = agent_mod.make_agent_fn(
            config,
            agent_mod.ImplementerSetup(
                agent_mod.ImplementerTask(**make_kwargs),
                session=agent_mod.SessionOptions(session_timeout_sec=session_timeout_sec),
            ),
        )
        asyncio.run(agent_fn(kernel_path, "(history)"))
        return seen["spec"]

    return run


@pytest.mark.parametrize("gpu_target", ["gfx942", "gfx950"])
def test_mi_implementer_profiling_prompt_matches_main(tmp_path, captured_run_spec, gpu_target):
    from kernelforge.config import Config

    kernel = tmp_path / "kernel.py"
    kernel.write_text("# kernel\n")
    spec = captured_run_spec(
        session_timeout_sec=60,
        kernel_path=str(kernel),
        gpu_target=gpu_target,
        profiling_enabled=True,
    )
    prompt = spec.system_prompt
    marker = "\n## Self-profiling (optional, on demand)\n"
    start = prompt.index(marker)
    end = prompt.find("\n## ", start + len(marker))
    section = prompt[start : end if end >= 0 else None]
    profiling_dir = Path(Config().local_knowledge_dir) / "common_methodology" / "profiling"
    section = section.replace(str(profiling_dir), "<PROFILE>")
    assert hashlib.sha256(section.encode()).hexdigest() == (
        "03de0ad8eb30d275579c6c0c044ebb48a6ad1702b788a63b2dd0a00bf3cb4df1"
    )


def test_rdna_implementer_profiling_prompt_uses_rocprofv3(tmp_path, captured_run_spec):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("# kernel\n")

    spec = captured_run_spec(
        session_timeout_sec=60,
        kernel_path=str(kernel),
        gpu_target="gfx1201",
        profiling_enabled=True,
    )

    assert "Use rocprofv3 kernel tracing first" in spec.system_prompt
    assert "instead of guessing rocprof-compute's CLI" not in spec.system_prompt


def test_run_spec_timeout_comes_from_the_session_budget(tmp_path, captured_run_spec):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("# kernel\n")

    spec = captured_run_spec(session_timeout_sec=4321, kernel_path=str(kernel))

    # The campaign-sized budget, not the backend runtime's 1800s default.
    assert spec.timeout_sec == 4321


def test_the_implementer_session_carries_forges_protected_path_rules(tmp_path, captured_run_spec):
    from kernelforge.config.edit_policy import FORGE_PROTECTED_PATHS

    kernel = tmp_path / "kernel.py"
    kernel.write_text("# kernel\n")

    spec = captured_run_spec(session_timeout_sec=600, kernel_path=str(kernel))

    assert spec.protected_path_rules is FORGE_PROTECTED_PATHS


def test_prompt_states_the_deadline_and_the_handoff(tmp_path, captured_run_spec):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("# kernel\n")

    spec = captured_run_spec(session_timeout_sec=5400, kernel_path=str(kernel))

    prompt = spec.user_prompt
    # The number of minutes the session actually has (5400s == 90 min).
    assert "90" in prompt
    # And that it must hand off its best candidate before the clock runs out, via the clean handoff path rather than
    # being killed with nothing.
    lowered = prompt.lower()
    assert "candidate_submitted" in prompt or "candidate" in lowered
    assert "deadline" in lowered or "time" in lowered
