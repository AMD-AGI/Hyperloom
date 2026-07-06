# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Full-trace subsystem: unified token + decision timeline.

This package owns the writers and parsers that let a Hyperloom run
reconstruct, after the fact, a single timeline of
``phase -> tick -> decision -> per-component token spend`` entirely from
local LLM-client responses (no litellm gateway).

Modules:

* :mod:`llm_trace` — :class:`LLMCallRecord` closed-schema dataclass and
  :func:`append_llm_call`, the best-effort atomic appender every in-process
  component calls (orchestration / kernel / specialist in-process fallback /
  codex / critic / proposal_scorer).
* :mod:`parse_usage` — parsers that recover ``usage`` token counts from
  out-of-process child output (Claude CLI ``stream-json``) so the parent can
  fold them into the same ledger.

The collector that joins this ledger with the decision streams lives in
``inference_optimizer/breakdown/collectors.py`` (``collect_decision_trace``).
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
from .langfuse_emitter import flush_session, get_emitter
from .parse_usage import (
    normalize_usage,
    parse_claude_stream_json_usage,
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
    "redact_secrets",
]
