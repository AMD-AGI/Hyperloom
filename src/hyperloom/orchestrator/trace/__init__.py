# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Full-trace subsystem: unified token + decision timeline.

Feeds two parallel sinks: an always-written local ``reports/trace/*.jsonl``
ledger that reconstructs, after the fact, a single timeline of
``phase -> tick -> decision -> per-component token spend``, and an opt-in
live Langfuse push (gated on ``HYPERLOOM_LANGFUSE_ENABLE`` + the
``LANGFUSE_*`` credentials + an importable SDK).

Modules:

* :mod:`llm_trace` — :class:`LLMCallRecord` closed-schema dataclass and
  :func:`append_llm_call`, the best-effort single-line appender.
* :mod:`parse_usage` — parsers that recover ``usage`` token counts from
  out-of-process child output (Claude CLI ``stream-json``).
* :mod:`conversation_trace` — :class:`ConversationRecord` rows plus
  :func:`redact_secrets` for the conversation ledger.
* :mod:`langfuse_emitter` — the live push sink (:func:`get_emitter`,
  :func:`flush_session`).
* :mod:`langfuse_mapping` — projection of local rows onto Langfuse
  traces / spans / generations / scores.
* :mod:`trace_env` — the env-var knobs and credential resolution.

The collector that joins this ledger with the decision streams lives in
``src/hyperloom/inference_optimizer/breakdown/collectors/decision.py`` (``collect_decision_trace``).
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
