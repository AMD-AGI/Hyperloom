"""The base tool set handed to the CLI, which is what the prefix is made of.

``allowed_tools`` only says what a session is *permitted* to call; the CLI still
loads and describes every built-in tool, and those schemas live in the cached
prefix that is re-read on every turn. Measured on a one-turn session: 33,374
prefix tokens with the default set against 6,148 with the four an implementer
uses. Naming the base set is therefore worth ~27k tokens per turn, and on the
72-turn implementer session observed in a live campaign that is most of its
cache_read. These tests pin what reaches the SDK; they are GPU- and SDK-free.
"""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.agent_backends.base import AgentRole, AgentRunSpec, AgentToolPolicy, StdioMcpServer
from kernelforge.agent_backends.claude import _builtin_tools
from kernelforge.tests.test_claude_resume import _backend, _result_message, _spec


def _kwargs(spec: AgentRunSpec) -> dict:
    captured: dict = {}
    backend = _backend([_result_message(session_id="s")], captured)
    asyncio.run(backend.run(spec))
    return captured["options"].kwargs


def test_the_base_set_is_named_rather_than_left_at_the_default() -> None:
    """Without this the CLI describes every built-in tool in the prefix."""
    kwargs = _kwargs(_spec(tool_policy=AgentToolPolicy(read=True, search=True, write=True, shell=True)))
    assert kwargs["tools"] == ["Read", "Grep", "Glob", "Edit", "Write", "Bash"]


def test_a_read_only_session_carries_no_write_or_shell_schema() -> None:
    """A specialist that cannot edit should not pay for Edit/Write/Bash either."""
    kwargs = _kwargs(_spec(tool_policy=AgentToolPolicy(read=True, search=True, write=False, shell=False)))
    assert kwargs["tools"] == ["Read", "Grep", "Glob"]
    assert kwargs["allowed_tools"] == ["Read", "Grep", "Glob"]


def test_an_mcp_tool_is_permitted_but_is_not_a_built_in() -> None:
    """``--tools`` selects from the built-in set and rejects anything else.

    The specialist probe reaches a session as an MCP server plus an entry in the
    permission list. Forwarding its name into the base set would make the CLI
    reject the whole invocation, so the probe must appear in ``allowed_tools``
    and nowhere else.
    """
    probe = "mcp__specialist_probe__probe_variant"
    kwargs = _kwargs(
        _spec(
            tool_policy=AgentToolPolicy(read=True, search=False, write=False, shell=True, extra_tools=(probe,)),
            mcp_servers={"specialist_probe": StdioMcpServer(command="python", args=["-m", "probe"], tools=(probe,))},
        )
    )
    assert probe in kwargs["allowed_tools"]
    assert probe not in kwargs["tools"]
    assert kwargs["tools"] == ["Read", "Bash"]


def test_subagents_keep_the_task_tool_reachable() -> None:
    """Declaring subagents adds Task to the permission list; the base set must follow.

    A restricted base set that omits Task would leave the roles declared right
    beside it uncallable.
    """
    kwargs = _kwargs(
        _spec(
            tool_policy=AgentToolPolicy(read=True, search=False, write=False, shell=False),
            subagents={"reviewer": AgentRole(description="reviews", instructions="review it")},
        )
    )
    assert "Task" in kwargs["allowed_tools"]
    assert "Task" in kwargs["tools"]


@pytest.mark.parametrize(
    "names, expected",
    [
        (["Read", "Read", "Bash"], ["Read", "Bash"]),
        (["mcp__a__b"], []),
        ([], []),
    ],
)
def test_the_filter_dedupes_and_drops_mcp_names(names: list[str], expected: list[str]) -> None:
    assert _builtin_tools(names) == expected


# ── an SDK too old for the field ──────────────────────────────────────────────


def test_an_sdk_without_the_field_still_builds_its_options() -> None:
    """The base set is a saving, not a requirement.

    ``tools`` reached ``ClaudeAgentOptions`` only in newer SDKs. Passing it to an
    older one would raise before the session ever started, turning a token
    optimization into a hard failure, so it is dropped when the field is absent.
    """
    import dataclasses

    from kernelforge.agent_backends.claude import _options_accept

    @dataclasses.dataclass
    class _Old:
        allowed_tools: list

    @dataclasses.dataclass
    class _New:
        allowed_tools: list
        tools: list

    assert _options_accept(_New, "tools") is True
    assert _options_accept(_Old, "tools") is False

    class _Fake:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # A test double takes anything, and must be treated as the type it replaces.
    assert _options_accept(_Fake, "tools") is True
