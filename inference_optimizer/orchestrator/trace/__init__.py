# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Full-trace subsystem: unified token + decision timeline.

This package owns the writers and parsers that let a Hyperloom run
reconstruct, after the fact, a single timeline of
``phase -> tick -> decision -> per-component token spend`` entirely from
local LLM-client responses (no litellm gateway).

Modules:

* :mod:`llm_trace` — :class:`LLMCallRecord` closed-schema dataclass and
  :func:`append_llm_call`, the best-effort atomic appender every in-process
  component calls (orchestration / kernel / dynamic_action / specialist
  in-process fallback / codex / critic / proposal_scorer).
* :mod:`parse_usage` — parsers that recover ``usage`` token counts from
  out-of-process child output (Claude CLI ``stream-json``, ``oob run
  --json``, GEAK / litellm output) so the parent can fold them into the
  same ledger.

The collector that joins this ledger with the decision streams lives in
``inference_optimizer/breakdown/collectors.py`` (``collect_decision_trace``).
"""

from .llm_trace import (
    LLMCallRecord,
    LLMTraceRowError,
    append_llm_call,
)
from .parse_usage import (
    normalize_usage,
    parse_claude_stream_json_usage,
    parse_geak_usage,
    parse_oob_json_usage,
)

__all__ = [
    "LLMCallRecord",
    "LLMTraceRowError",
    "append_llm_call",
    "normalize_usage",
    "parse_claude_stream_json_usage",
    "parse_geak_usage",
    "parse_oob_json_usage",
]
