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


def parse_oob_json_usage(stdout: str) -> dict[str, int | None] | None:
    """Extract ``usage`` from ``oob run --json`` stdout.

    OOB emits a JSON document on stdout that already carries a ``usage``
    block (see ``kernel-agent/tools/backends/oob_submit.py``). The exact
    envelope can vary, so we search defensively:

    1. parse the whole stdout as one JSON object and look for a top-level
       or nested ``usage`` (common keys: ``usage``, ``token_usage``);
    2. failing that, scan line-by-line for the last JSON object carrying a
       ``usage`` block (covers JSONL-style streamed output).

    OpenAI-style OOB has no prompt-cache split, so ``cache_*`` stay
    ``None``. Returns ``None`` when nothing parseable is found.
    """
    if not stdout or not stdout.strip():
        return None
    # Attempt 1: whole-document parse.
    try:
        obj = json.loads(stdout)
        found = _find_usage_in_obj(obj)
        if found is not None:
            return normalize_usage(found)
    except (json.JSONDecodeError, ValueError):
        pass
    # Attempt 2: line-by-line (JSONL / mixed log output).
    last_usage: dict[str, Any] | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        found = _find_usage_in_obj(obj)
        if found is not None:
            last_usage = found
    return normalize_usage(last_usage)


def parse_geak_usage(
    payload: dict[str, Any] | str | None,
) -> dict[str, int | None] | None:
    """Extract ``usage`` from a GEAK / litellm result.

    GEAK uses litellm as a *library* (local, not a gateway); a litellm
    completion response exposes ``usage`` with ``prompt_tokens`` /
    ``completion_tokens`` (OpenAI shape). ``payload`` may be the already
    parsed response dict or a JSON string. We map the OpenAI counter names
    onto our canonical keys (``prompt_tokens`` → ``input_tokens``,
    ``completion_tokens`` → ``output_tokens``) and also honor a usage block
    that already uses the canonical names.

    No prompt-cache split, so ``cache_*`` stay ``None``. Returns ``None``
    when nothing parseable is found.
    """
    obj: Any = payload
    if isinstance(payload, str):
        if not payload.strip():
            return None
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return None
    usage = _find_usage_in_obj(obj)
    if not isinstance(usage, dict) or not usage:
        return None
    # Translate OpenAI counter names if the canonical ones are absent.
    translated: dict[str, Any] = dict(usage)
    if "input_tokens" not in translated and "prompt_tokens" in translated:
        translated["input_tokens"] = translated.get("prompt_tokens")
    if "output_tokens" not in translated and "completion_tokens" in translated:
        translated["output_tokens"] = translated.get("completion_tokens")
    return normalize_usage(translated)


def _find_usage_in_obj(obj: Any, _depth: int = 0) -> dict[str, Any] | None:
    """Best-effort search for a ``usage``/``token_usage`` dict in ``obj``.

    Looks at the object itself, then a shallow set of well-known container
    keys, then recurses one extra level into nested dicts. Bounded depth
    keeps this cheap and avoids pathological deep structures; out-of-process
    envelopes nest ``usage`` at most a couple of levels down in practice.
    """
    if _depth > 4 or not isinstance(obj, dict):
        return None
    for key in ("usage", "token_usage"):
        candidate = obj.get(key)
        if isinstance(candidate, dict) and candidate:
            return candidate
    # Recurse into nested dicts (e.g. {"result": {"usage": {...}}},
    # {"response": {...}}, {"choices": [...]} is handled below).
    for value in obj.values():
        if isinstance(value, dict):
            found = _find_usage_in_obj(value, _depth + 1)
            if found is not None:
                return found
        elif isinstance(value, list):
            for item in value:
                found = _find_usage_in_obj(item, _depth + 1)
                if found is not None:
                    return found
    return None


__all__ = [
    "normalize_usage",
    "parse_claude_stream_json_usage",
    "parse_geak_usage",
    "parse_oob_json_usage",
]
