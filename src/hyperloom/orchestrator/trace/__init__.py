# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Full-trace subsystem: unified token + decision timeline.

Feeds two parallel sinks: an always-written local ``reports/trace/*.jsonl``
ledger and an opt-in live Langfuse push (gated on ``HYPERLOOM_LANGFUSE_ENABLE``).

Key modules: :mod:`llm_trace`, :mod:`conversation_trace`, :mod:`langfuse_emitter`,
:mod:`parse_usage`, :mod:`trace_env`, :mod:`orchestration_trace`.
"""

from .conversation_trace import (
    ConversationRecord,
    ConversationRowError,
    append_conversation,
    redact_secrets,
)
from .llm_trace import (
    LLMCallRecord,
    LLMTraceRowError,
    append_llm_call,
)
from .orchestration_trace import (
    write_mcp_setup_once,
)
from .langfuse_emitter import flush_session, get_emitter
from .parse_usage import (
    normalize_usage,
    parse_claude_stream_json_usage,
    parse_codex_jsonl_error,
    parse_codex_jsonl_usage,
)
from .trace_env import langfuse_live_enabled

__all__ = [
    "ConversationRecord",
    "ConversationRowError",
    "LLMCallRecord",
    "LLMTraceRowError",
    "append_conversation",
    "append_llm_call",
    "flush_session",
    "get_emitter",
    "langfuse_live_enabled",
    "normalize_usage",
    "parse_claude_stream_json_usage",
    "parse_codex_jsonl_error",
    "parse_codex_jsonl_usage",
    "redact_secrets",
    "write_mcp_setup_once",
]
