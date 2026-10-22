"""Tests for the read-only start selector of a tree-strategy campaign."""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.agent_backends import AgentProviderError, AgentRunResult
from kernelforge.orchestrator.start_selector import (
    START_SELECTOR_MAX_TURNS,
    START_SELECTOR_TIMEOUT_SEC,
    StartSelectorAgent,
    parse_start_commit,
)

NODE = "0123456789abcdef0123456789abcdef01234567"


class _Backend:
    def __init__(self, answer):
        self.answer = answer
        self.specs = []

    async def run(self, spec, usage=None):
        self.specs.append(spec)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def _resolve(token):
    return NODE if NODE.startswith(token) else ""


def _select(answer):
    backend = _Backend(answer)
    selection = asyncio.run(
        StartSelectorAgent(backend=backend).select(workspace="/ws", tree_view="## Search tree", resolve=_resolve)
    )
    return selection, backend


def test_the_last_start_commit_line_names_the_start():
    text = "Compare root and the branch.\nSTART_COMMIT: deadbeef\n\nOn reflection:\n**START_COMMIT:** `0123456789ab`"

    assert parse_start_commit(text) == "0123456789ab"
    assert parse_start_commit("no answer here") == ""


def test_a_named_node_is_selected_with_the_reasoning_that_chose_it():
    selection, backend = _select(AgentRunResult(text=f"Branch b001 still has headroom.\nSTART_COMMIT: {NODE[:12]}"))

    assert selection.commit == NODE
    assert selection.failure == ""
    assert selection.reason == "Branch b001 still has headroom."
    spec = backend.specs[0]
    assert spec.writable is False
    assert spec.tool_policy.shell is False and spec.tool_policy.write is False
    assert spec.timeout_sec == START_SELECTOR_TIMEOUT_SEC == 300
    assert spec.tool_policy.max_turns == START_SELECTOR_MAX_TURNS == 30
    assert spec.user_prompt == "## Search tree"


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
