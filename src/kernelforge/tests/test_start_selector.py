"""Tests for the read-only start selector of a tree-strategy campaign."""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.agent_backends import AgentCapabilities, AgentProviderError, AgentRunResult
from kernelforge.orchestrator.start_selector import (
    START_SELECTOR_MAX_TURNS,
    START_SELECTOR_TIMEOUT_SEC,
    StartSelectorAgent,
    parse_start_commit,
    read_only_git_denial,
)

NODE = "0123456789abcdef0123456789abcdef01234567"


class _Backend:
    def __init__(self, answer, *, hooks=True):
        self.answer = answer
        self.specs = []
        self.capabilities = AgentCapabilities(stop_hooks=hooks)

    async def run(self, spec, usage=None):
        self.specs.append(spec)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def _resolve(token):
    return NODE if NODE.startswith(token) else ""


def _select(answer, *, hooks=True):
    backend = _Backend(answer, hooks=hooks)
    selection = asyncio.run(
        StartSelectorAgent(backend=backend).select(workspace="/ws", tree_view="## Search tree", resolve=_resolve)
    )
    return selection, backend


def _bash(spec, command):
    hook = spec.hooks.pre_tool_use[0]
    assert hook.matcher == "Bash"
    return asyncio.run(hook.callback({"tool_name": "Bash", "tool_input": {"command": command}}, None, None))


def test_the_last_start_commit_line_names_the_start():
    text = "Compare root and the branch.\nSTART_COMMIT: deadbeef\n\nOn reflection:\n**START_COMMIT:** `0123456789ab`"

    assert parse_start_commit(text) == "0123456789ab"
    assert parse_start_commit("no answer here") == ""


def test_a_named_node_is_selected_with_the_reasoning_and_the_whole_answer():
    answer = f"Branch b001 still has headroom.\nSTART_COMMIT: {NODE[:12]}"
    selection, backend = _select(AgentRunResult(text=answer))

    assert selection.commit == NODE
    assert selection.failure == ""
    assert selection.reason == "Branch b001 still has headroom."
    assert selection.answer == answer
    spec = backend.specs[0]
    assert spec.writable is False and spec.tool_policy.write is False
    assert spec.timeout_sec == START_SELECTOR_TIMEOUT_SEC == 300
    assert spec.tool_policy.max_turns == START_SELECTOR_MAX_TURNS == 30
    assert spec.user_prompt == "## Search tree"


def test_a_provider_that_runs_hooks_gets_a_shell_confined_to_read_only_git():
    _selection, backend = _select(AgentRunResult(text=f"START_COMMIT: {NODE}"))
    spec = backend.specs[0]

    assert spec.tool_policy.shell is True
    assert _bash(spec, "git log --oneline forge-optimize-b001") == {}
    assert _bash(spec, f"git -C /ws --no-pager diff {NODE[:8]} {NODE}") == {}
    denied = _bash(spec, "python3 driver.py")["hookSpecificOutput"]
    assert denied["permissionDecision"] == "deny"
    assert "read-only git" in denied["permissionDecisionReason"]


def test_a_provider_that_cannot_run_hooks_gets_no_shell_from_the_selector():
    _selection, backend = _select(AgentRunResult(text=f"START_COMMIT: {NODE}"), hooks=False)
    spec = backend.specs[0]

    assert spec.tool_policy.shell is False
    assert spec.hooks is None


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        ("git checkout main", "is not one of"),
        ("git commit -am x", "is not one of"),
        ("git log | head", "one git command at a time"),
        ("git show HEAD > out.txt", "one git command at a time"),
        ("git log; rm -rf src", "one git command at a time"),
        ("git log $(touch x)", "one git command at a time"),
        ("git -c core.pager=sh log", "is not one of"),
        ("git diff --output=patch.diff", "--output=patch.diff writes a file"),
        ("git log --ext-diff", "--ext-diff writes a file"),
        ("ls forge_experiments", "only read-only git commands"),
        ("git", "is not one of"),
    ],
)
def test_every_command_that_is_not_one_read_only_git_call_is_denied(command, reason):
    assert reason in read_only_git_denial(command)


@pytest.mark.parametrize(
    ("answer", "failure"),
    [
        (AgentRunResult(text="I would build on the root."), "named no START_COMMIT"),
        (AgentRunResult(text="START_COMMIT: fedcba9876543"), "not a node of the tree"),
        (AgentRunResult(text="", end_reason="sdk_error"), "AgentResponseInfrastructureError"),
        (AgentRunResult(text=""), "returned no answer"),
        (AgentProviderError("provider down"), "provider down"),
    ],
)
def test_every_unusable_answer_is_a_failure_the_loop_falls_back_on(answer, failure):
    selection, _backend = _select(answer)

    assert selection.commit == ""
    assert failure in selection.failure


def test_an_answer_cut_short_still_counts_when_it_named_a_node():
    selection, _backend = _select(AgentRunResult(text=f"START_COMMIT: {NODE}", end_reason="turn_cap"))

    assert selection.commit == NODE
