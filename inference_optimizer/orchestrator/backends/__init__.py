"""LLM backend wrappers (DESIGN v0.6 §15).

Each backend exposes a uniform :class:`Backend` protocol so the Coordinator
can swap implementations (real Claude / Codex SDK, mock for tests, future
multi-CLI bridge) without touching the reactor loop.

Mock backends (P0):

* :class:`MockBackend` — generic scripted-turn playback for any agent.
* :class:`MockCriticBackend` — always-approve Critic adapter (auto-extracts
  proposal msg_id from inbox prompt).
* :class:`MockRobustnessBackend` — heartbeat-only Robustness adapter.
"""

from .base import Backend, BackendError, BackendTurnResult
from .claude import ClaudeBackend
from .codex import CodexBackend
from .critic_mock import MockCriticBackend
from .kernel_mock import MockKernelBackend
from .mcp_emit_intent import (
    EMIT_INTENT_TOOL_NAME,
    EMIT_INTENT_TOOL_QUALIFIED,
    MCP_SERVER_NAME,
    build_emit_intent_server,
    validate_emit_intent_input,
)
from .mock_backend import MockBackend, MockTurn, ScriptedPlan
from .robustness_mock import MockRobustnessBackend

__all__ = [
    "Backend",
    "BackendError",
    "BackendTurnResult",
    "ClaudeBackend",
    "CodexBackend",
    "EMIT_INTENT_TOOL_NAME",
    "EMIT_INTENT_TOOL_QUALIFIED",
    "MCP_SERVER_NAME",
    "MockBackend",
    "MockCriticBackend",
    "MockKernelBackend",
    "MockRobustnessBackend",
    "MockTurn",
    "ScriptedPlan",
    "build_emit_intent_server",
    "validate_emit_intent_input",
]
