# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The names the runtime registers its in-process MCP tools under.

The agent can only call a tool by the name the SDK registered, so every
surface that shows one -- the prompts, the reference documents, the PolicyGate
allow-list and its denial hints, the per-tick projections -- has to spell it
the same way the server that mounts it does. That name is owned here, in a
module that imports no sibling, so ``prompts``, ``policy`` and ``roles`` can
all read it without any of them depending on another.
"""

from __future__ import annotations

# The two in-process MCP servers Claude mounts: emit_intent and the read-only context pulls.
MCP_SERVER_NAME = "inference_optimizer"
CONTEXT_MCP_SERVER_NAME = "inference_optimizer_context"

EMIT_INTENT_TOOL_NAME = "emit_intent"


def qualified_tool_name(server_name: str, tool_name: str) -> str:
    """Return the ``mcp__<server>__<tool>`` name the SDK registers a tool under."""
    return f"mcp__{server_name}__{tool_name}"


EMIT_INTENT_TOOL_QUALIFIED = qualified_tool_name(MCP_SERVER_NAME, EMIT_INTENT_TOOL_NAME)

# The context tools, in the order ``roles.mcp_context_tools`` mounts them.
CONTEXT_TOOL_NAMES: tuple[str, ...] = (
    "get_mission_status",
    "get_shared_state",
    "get_gaps",
    "get_warm_start",
    "get_proposal_scores",
    "get_intervention_mix",
    "why_denied",
    "show_analysis_md",
    "get_runtime_findings",
    "get_inbox",
    "get_recent_outcomes",
    "get_running_tasks",
    "read_reference",
    "get_failure",
    "get_variant_failures",
)
CONTEXT_TOOL_QUALIFIED_NAMES: tuple[str, ...] = tuple(
    qualified_tool_name(CONTEXT_MCP_SERVER_NAME, name) for name in CONTEXT_TOOL_NAMES
)


def qualified_context_tool_name(tool_name: str) -> str:
    """Return the name the runtime registers for a context tool, for prompts.

    Raises:
        KeyError: if ``tool_name`` is not one of :data:`CONTEXT_TOOL_NAMES`.
    """
    if tool_name not in CONTEXT_TOOL_NAMES:
        raise KeyError(f"unknown context tool: {tool_name!r}")
    return qualified_tool_name(CONTEXT_MCP_SERVER_NAME, tool_name)


# The built-in tools orchestration gets on top of the MCP ones: a sandboxed read and web search.
ORCHESTRATION_BUILTIN_TOOL_NAMES: tuple[str, ...] = ("Read", "WebSearch", "WebFetch")


def agent_tool_grant(agent_name: str) -> list[str]:
    """Return the registered tool names an agent with tools may call.

    Every reactor gets ``emit_intent``; orchestration also gets the read-only
    context pulls and the built-in tools. The caller decides whether the role
    has tools at all; this is only what the names are.
    """
    tools = [EMIT_INTENT_TOOL_QUALIFIED]
    if agent_name == "orchestration":
        tools.extend(CONTEXT_TOOL_QUALIFIED_NAMES)
        tools.extend(ORCHESTRATION_BUILTIN_TOOL_NAMES)
    return tools


__all__ = [
    "CONTEXT_MCP_SERVER_NAME",
    "CONTEXT_TOOL_NAMES",
    "CONTEXT_TOOL_QUALIFIED_NAMES",
    "EMIT_INTENT_TOOL_NAME",
    "EMIT_INTENT_TOOL_QUALIFIED",
    "MCP_SERVER_NAME",
    "ORCHESTRATION_BUILTIN_TOOL_NAMES",
    "agent_tool_grant",
    "qualified_context_tool_name",
    "qualified_tool_name",
]
