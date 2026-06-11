# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Recover token ``usage`` from out-of-process LLM-client output.

In-process backends hand us a ``BackendTurnResult.metadata`` dict directly,
but the production-critical paths (the specialist subprocess, GEAK / OOB
kernel candidates, robustness RCA) run in *child processes* whose token
counts only survive as text in a log / stdout / result JSON. These parsers
let the parent fold those counts into the same ledger.

All parsers are **tolerant**: a missing file, truncated JSON, or an absent
``usage`` block returns an empty / ``None`` result instead of raising. The
caller is always on a best-effort trace path where a parse miss must
degrade to "no token data for this call", never to a crashed optimization
loop.

Output shape: every parser returns the canonical four-key token dict (or
``None`` when nothing could be recovered):

    {"input_tokens", "output_tokens",
     "cache_creation_input_tokens", "cache_read_input_tokens"}

mirroring :data:`..backends` metadata and :class:`.llm_trace.LLMCallRecord`
counters. Backends with no prompt-cache concept (OpenAI / GEAK) leave the
two ``cache_*`` values ``None`` so the collector can tell "no cache" from
"zero cache hits".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# The four canonical counters, in stable order. ``cache_*`` default to
# ``None`` so a source that does not report caching is distinguishable from
# one reporting zero.
_TOKEN_KEYS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int | None] | None:
    """Project an arbitrary ``usage`` dict onto the canonical four keys.

    Returns ``None`` when ``usage`` is falsy or carries none of the four
    recognized counters (so a stray ``{}`` or an unrelated dict doesn't
    masquerade as a real measurement). Unknown keys are dropped; absent
    counters become ``None``.
    """
    if not isinstance(usage, dict) or not usage:
        return None
    projected: dict[str, int | None] = {
        k: _coerce_optional_int(usage.get(k)) for k in _TOKEN_KEYS
    }
    if all(v is None for v in projected.values()):
        return None
    return projected


def parse_claude_stream_json_usage(
    log_path: str | Path,
) -> dict[str, int | None] | None:
    """Extract the final ``usage`` from a Claude CLI ``stream-json`` log.

    ``claude --print --output-format stream-json --verbose`` writes one
    JSON object per line; the terminal line is ``{"type": "result", ...,
    "usage": {...}}`` carrying the cumulative session usage (same dict shape
    as the SDK's ``ResultMessage.usage`` already consumed in ``claude.py``).
    This is the key to recovering token spend for the **production default**
    specialist path (B1), whose tokens are otherwise invisible to the parent.

    Strategy: scan all lines, keep the ``usage`` from the *last* object that
    carries one (a ``type=="result"`` row is preferred, but we accept any
    line with a ``usage`` block so a CLI schema tweak that moves ``usage``
    onto a different terminal message still works). Malformed lines are
    skipped. Returns ``None`` if the file is missing or no ``usage`` is
    found.
    """
    path = Path(log_path)
    last_usage: dict[str, Any] | None = None
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(obj, dict):
                    continue
                usage = obj.get("usage")
                if isinstance(usage, dict) and usage:
                    # A result-typed row is authoritative; let it win over
                    # any earlier assistant-message usage on the same stream.
                    if obj.get("type") == "result" or last_usage is None:
                        last_usage = usage
    except FileNotFoundError:
        return None
    except OSError as exc:
        log.warning(
            "parse_usage: failed reading stream-json log %s: %r", path, exc
        )
        return None
    return normalize_usage(last_usage)


__all__ = [
    "normalize_usage",
    "parse_claude_stream_json_usage",
]
