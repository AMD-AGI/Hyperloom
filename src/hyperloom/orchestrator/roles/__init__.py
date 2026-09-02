# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLM backend wrappers exposing a uniform :class:`Backend` protocol."""

from .base import Backend, BackendError, BackendTurnResult
from .claude import ClaudeBackend
from .codex import CodexBackend
from .hermes import HermesBackend
from .critic_agent import CriticAgentBackend, RuntimeCall, RuntimeCaller
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
from .mock_backend import (
    MockBackend,
    MockRowScanBackend,
    MockTurn,
    ScriptedPlan,
    auto_approve_critic,
)
from .robustness_agent import RobustnessAgentBackend
from .robustness_mock import MockRobustnessBackend

# Public name for the row-scan critic mock (formerly the standalone
# ``MockCriticBackend`` class). Kept importable so out-of-scope callers
# (e.g. ``cli/backends.py``) and tests keep working.
MockCriticBackend = auto_approve_critic

__all__ = [
    "Backend",
    "BackendError",
    "BackendTurnResult",
    "CONTEXT_TOOL_NAMES",
    "CONTEXT_TOOL_QUALIFIED_NAMES",
    "ClaudeBackend",
    "CodexBackend",
    "HermesBackend",
    "ContextProvider",
    "CriticAgentBackend",
    "EMIT_INTENT_TOOL_NAME",
    "EMIT_INTENT_TOOL_QUALIFIED",
    "MCP_SERVER_NAME",
    "MockBackend",
    "MockCriticBackend",
    "MockRobustnessBackend",
    "MockRowScanBackend",
    "MockTurn",
    "RobustnessAgentBackend",
    "RuntimeCall",
    "RuntimeCaller",
    "ScriptedPlan",
    "auto_approve_critic",
    "build_context_tools_server",
    "build_emit_intent_server",
    "validate_emit_intent_input",
]
