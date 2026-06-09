# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""LLM backend wrappers exposing a uniform :class:`Backend` protocol."""

from .base import Backend, BackendError, BackendTurnResult
from .claude import ClaudeBackend
from .codex import CodexBackend
from .critic_agent import CriticAgentBackend, RuntimeCall, RuntimeCaller
from .critic_mock import MockCriticBackend
from .kernel_mock import MockKernelBackend
from .mcp_context_tools import (
    CONTEXT_TOOL_NAMES,
    CONTEXT_TOOL_QUALIFIED_NAMES,
    ContextProvider,
    build_context_tools_server,
)
from .mcp_emit_intent import (
    EMIT_INTENT_TOOL_NAME,
    EMIT_INTENT_TOOL_QUALIFIED,
    MCP_SERVER_NAME,
    build_emit_intent_server,
    validate_emit_intent_input,
)
from .mock_backend import MockBackend, MockTurn, ScriptedPlan
from .robustness_agent import RobustnessAgentBackend
from .robustness_mock import MockRobustnessBackend

__all__ = [
    "Backend",
    "BackendError",
    "BackendTurnResult",
    "CONTEXT_TOOL_NAMES",
    "CONTEXT_TOOL_QUALIFIED_NAMES",
    "ClaudeBackend",
    "CodexBackend",
    "ContextProvider",
    "CriticAgentBackend",
    "EMIT_INTENT_TOOL_NAME",
    "EMIT_INTENT_TOOL_QUALIFIED",
    "MCP_SERVER_NAME",
    "MockBackend",
    "MockCriticBackend",
    "MockKernelBackend",
    "MockRobustnessBackend",
    "MockTurn",
    "RobustnessAgentBackend",
    "RuntimeCall",
    "RuntimeCaller",
    "ScriptedPlan",
    "build_context_tools_server",
    "build_emit_intent_server",
    "validate_emit_intent_input",
]
