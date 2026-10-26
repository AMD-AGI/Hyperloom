# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The read-only slot selector session: what it may run, and how its answer is read."""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.agent_backends import AgentCapabilities, AgentProviderError, AgentRunResult
from kernelforge.orchestrator.slot_selector import SlotSelector, read_only_git_denial


class _Backend:
    def __init__(self, text="", *, name="claude", hooks=True, error=None):
        self.name = name
        self.capabilities = AgentCapabilities(stop_hooks=hooks)
        self.text = text
        self.error = error
        self.specs = []

    async def run(self, spec, usage=None):
        self.specs.append(spec)
        if self.error is not None:
            raise self.error
        return AgentRunResult(text=self.text, end_reason="completed")


ANSWER = 'Reasoning.\n{"slots": [{"type": "BRANCH", "start": "abcdef1", "reason": "r"}]}'


@pytest.mark.parametrize(
    "command",
    ["git log --oneline -5", "git -C /w show HEAD:k.py", "git --no-pager diff a b", "git ls-tree HEAD"],
)
def test_read_only_git_commands_are_admitted(command):
    assert read_only_git_denial(command) == ""


@pytest.mark.parametrize(
    "command, reason",
    [
        ("git log | head", "one git command at a time"),
        ("git commit -m x", "is not one of"),
        ("git -c core.pager=sh log", "is not one of"),
        ("git diff --output=/tmp/x", "writes a file"),
        ("cat k.py", "only read-only git"),
        ("git log $(rm -rf /)", "one git command at a time"),
    ],
)
def test_anything_else_is_denied_with_a_reason(command, reason):
    assert reason in read_only_git_denial(command)


def test_the_answer_is_read_off_its_trailing_json():
    backend = _Backend(ANSWER)

    answer = asyncio.run(
        SlotSelector(backend=backend, timeout_sec=60, max_turns=5).select(workspace="/w", facts="FACTS", lanes=3)
    )

    assert answer.payload == {"slots": [{"type": "BRANCH", "start": "abcdef1", "reason": "r"}]}
    assert answer.failure == "" and answer.text == ANSWER
    spec = backend.specs[0]
    assert spec.user_prompt == "FACTS" and not spec.writable and "3 expansions" in spec.system_prompt


def test_a_hook_capable_provider_gets_a_shell_fenced_to_read_only_git():
    backend = _Backend(ANSWER)

    asyncio.run(SlotSelector(backend=backend, timeout_sec=60, max_turns=5).select(workspace="/w", facts="", lanes=2))

    spec = backend.specs[0]
    assert spec.tool_policy.shell and not spec.tool_policy.write
    hook = spec.hooks.pre_tool_use[0]
    denied = asyncio.run(hook.callback({"tool_name": "Bash", "tool_input": {"command": "git push"}}, None, None))
    allowed = asyncio.run(hook.callback({"tool_name": "Bash", "tool_input": {"command": "git log"}}, None, None))
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny" and allowed == {}


@pytest.mark.parametrize("name, shell", [("codex", True), ("deepseek", False)])
def test_without_hooks_only_a_provider_with_its_own_git_guard_gets_a_shell(name, shell):
    backend = _Backend(ANSWER, name=name, hooks=False)

    asyncio.run(SlotSelector(backend=backend, timeout_sec=60, max_turns=5).select(workspace="/w", facts="", lanes=2))

    assert backend.specs[0].tool_policy.shell is shell
    assert backend.specs[0].hooks is None


@pytest.mark.parametrize(
    "backend, failure",
    [
        (_Backend("no json at all"), "no readable selection"),
        (_Backend(error=AgentProviderError("gateway down")), "AgentProviderError: gateway down"),
        (_Backend(""), "returned no answer"),
    ],
)
def test_a_failed_selection_is_reported_not_raised(backend, failure):
    answer = asyncio.run(
        SlotSelector(backend=backend, timeout_sec=60, max_turns=5).select(workspace="/w", facts="", lanes=2)
    )

    assert answer.payload is None
    assert failure in answer.failure
