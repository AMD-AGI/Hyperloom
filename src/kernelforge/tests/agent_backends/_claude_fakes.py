# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Scripted Claude SDK messages and the backend built around them, shared by the agent-backend and loop tests."""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from kernelforge.agent_backends.base import AgentRunSpec, AgentToolPolicy
from kernelforge.agent_backends.claude import ClaudeBackend
from kernelforge.agent_backends.registry import get_agent_provider
from kernelforge.utils.git import git
from kernelforge.config.edit_policy import FORGE_PROTECTED_PATHS


class _FakeOptions:
    """Stand-in for ClaudeAgentOptions that records what it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _system_prompt_text(system_prompt) -> str:
    """The system prompt a CLI would start with, whether it was handed over inline or as a file."""
    if isinstance(system_prompt, dict) and system_prompt.get("type") == "file":
        return Path(system_prompt["path"]).read_text(encoding="utf-8")
    return system_prompt or ""


def _result_message(session_id="", subtype="success"):
    return SimpleNamespace(
        content=[SimpleNamespace(text="done")],
        total_cost_usd=0.1,
        subtype=subtype,
        num_turns=3,
        session_id=session_id,
    )


def _backend(messages, captured, stream_error=None):
    """Build a ClaudeBackend whose SDK is replaced by a recording fake."""
    backend = ClaudeBackend.__new__(ClaudeBackend)
    backend.capabilities = get_agent_provider("claude").capabilities
    backend.runtime = SimpleNamespace(
        provider="claude",
        model="fake-model",
        executable="",
        timeout_sec=60,
        reasoning_effort="high",
        options={},
    )
    backend.fallback_reason = ""

    async def fake_query(prompt, options):
        captured["prompt"] = prompt
        captured["options"] = options
        # Read while the query runs, as the CLI does: a prompt handed over as a file is gone once the query ends.
        captured["system_prompt"] = _system_prompt_text(options.kwargs.get("system_prompt"))
        for message in messages:
            yield message
        if stream_error is not None:
            raise stream_error

    backend._query = fake_query
    backend._options_type = _FakeOptions
    return backend


@lru_cache(maxsize=1)
def _guarded_workspace() -> str:
    """A real worktree: a writable session is judged against one."""
    root = Path(tempfile.mkdtemp(prefix="forge-claude-resume-"))
    (root / "kernel.py").write_text("VALUE = 0\n")
    git("init", "--quiet", cwd=root)
    git("config", "user.email", "t@test", cwd=root)
    git("config", "user.name", "t", cwd=root)
    git("add", "-A", cwd=root)
    git("commit", "-m", "baseline", cwd=root)
    return str(root)


def _spec(**overrides) -> AgentRunSpec:
    base = dict(
        system_prompt="implementer system prompt",
        user_prompt="optimize the kernel",
        cwd=_guarded_workspace(),
        model="fake-model",
        timeout_sec=60,
        tool_policy=AgentToolPolicy(read=True, write=True, shell=True, max_turns=100),
        protected_path_rules=FORGE_PROTECTED_PATHS,
    )
    base.update(overrides)
    return AgentRunSpec(**base)
