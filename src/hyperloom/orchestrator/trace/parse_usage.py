# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Recover token ``usage`` from out-of-process LLM-client output.

In-process backends hand us a ``BackendTurnResult.metadata`` dict directly,
but child-process paths (specialist subprocess, forge kernel candidates,
robustness RCA) leave their token counts only as text in a log/stdout/result
JSON. These parsers fold those counts into the same ledger.

All parsers are **tolerant**: a missing file, truncated JSON, or an absent
``usage`` block returns an empty / ``None`` result instead of raising.

Output shape: every parser returns the canonical four-key token dict (or
``None`` when nothing could be recovered):

    {"input_tokens", "output_tokens",
     "cache_creation_input_tokens", "cache_read_input_tokens"}

Backends with no prompt-cache concept (OpenAI / GEAK) leave the two ``cache_*``
values ``None`` so the collector can tell "no cache" from "zero cache hits".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ._row_utils import coerce_optional_int

log = logging.getLogger(__name__)


# The four canonical counters, in stable order.
_TOKEN_KEYS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int | None] | None:
    """Project an arbitrary ``usage`` dict onto the canonical four keys.

    Returns ``None`` when ``usage`` is falsy or carries none of the four
    recognized counters. Unknown keys are dropped; absent counters become
    ``None``.

    Args:
        usage: Arbitrary usage dict, or ``None``.

    Returns:
        The canonical four-key token dict, or ``None`` when nothing usable.
    """
    if not isinstance(usage, dict) or not usage:
        return None
    projected: dict[str, int | None] = {k: coerce_optional_int(usage.get(k)) for k in _TOKEN_KEYS}
    if all(v is None for v in projected.values()):
        return None
    return projected


def parse_claude_stream_json_usage(
    log_path: str | Path,
) -> dict[str, int | None] | None:
    """Extract the final ``usage`` from a Claude CLI ``stream-json`` log.

    ``claude --print --output-format stream-json --verbose`` writes one JSON
    object per line; the terminal ``{"type": "result", ..., "usage": {...}}``
    carries the cumulative session usage. Recovers token spend for the
    production-default specialist path (B1), otherwise invisible to the parent.

    Scans all lines, keeping the ``usage`` from the last object that carries one
    (``type=="result"`` preferred, but any line with a ``usage`` block is
    accepted). Malformed lines are skipped. Returns ``None`` if the file is
    missing or no ``usage`` is found.

    Args:
        log_path: Path to the Claude CLI ``stream-json`` log.

    Returns:
        The canonical token dict, or ``None`` when no usage was found.
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
                    # A result-typed row is authoritative over earlier usage.
                    if obj.get("type") == "result" or last_usage is None:
                        last_usage = usage
    except FileNotFoundError:
        return None
    except OSError as exc:
        log.warning("parse_usage: failed reading stream-json log %s: %r", path, exc)
        return None
    return normalize_usage(last_usage)


def parse_claude_stream_json_response(
    log_path: str | Path,
) -> str | None:
    """Recover the assistant's full reply text from a Claude CLI stream-json log.

    Sibling of :func:`parse_claude_stream_json_usage` (which reads token counts
    off the same log); this reads the conversation response so the
    production-default specialist path (B1) can land its completion in
    ``conversations.jsonl``. Only the response is recovered; the caller already
    holds the prompt.

    Reconstructs the reply from two sources, preferring the authoritative one:

    1. the terminal ``{"type": "result", ..., "result": "<text>"}`` row, whose
       ``result`` is the consolidated final answer; when non-empty it wins;
    2. otherwise concatenate the ``text`` blocks from every
       ``{"type": "assistant"}`` message in order — covering a truncated run
       that never emitted a ``result`` row. ``thinking`` and ``tool_use`` blocks
       are dropped.

    Tolerant by contract: a missing file, malformed lines, or no recoverable
    text returns ``None``.

    Args:
        log_path: Path to the Claude CLI ``stream-json`` log.

    Returns:
        The recovered response text, or ``None`` when none could be read.
    """
    path = Path(log_path)
    result_text: str | None = None
    assistant_chunks: list[str] = []
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
                obj_type = obj.get("type")
                if obj_type == "result":
                    res = obj.get("result")
                    if isinstance(res, str) and res.strip():
                        result_text = res
                elif obj_type == "assistant":
                    message = obj.get("message")
                    if not isinstance(message, dict):
                        continue
                    for block in message.get("content") or []:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text")
                            if isinstance(text, str) and text:
                                assistant_chunks.append(text)
    except FileNotFoundError:
        return None
    except OSError as exc:
        log.warning("parse_usage: failed reading stream-json log %s: %r", path, exc)
        return None
    if result_text is not None:
        return result_text
    if assistant_chunks:
        return "\n".join(assistant_chunks)
    return None


def parse_claude_stream_json_turn_usages(
    log_path: str | Path,
) -> list[dict[str, int | None]]:
    """Recover *per-assistant-turn* usage from a Claude CLI stream-json log.

    Unlike :func:`parse_claude_stream_json_usage` (which returns one cumulative
    row), this returns the per-message usage on each
    ``{"type":"assistant", "message": {..., "usage": {...}}}`` line, in order,
    so a multi-turn specialist subprocess can be traced as one row per turn.

    Each assistant message carries its OWN (non-cumulative) usage, so the rows
    sum to the session total. Lines without a usage block are skipped; the
    terminal cumulative ``result`` row is ignored to avoid double counting.

    Returns the normalized four-key dicts in stream order, or ``[]`` when the
    file is missing/truncated/carries no per-message usage.
    """
    path = Path(log_path)
    usages: list[dict[str, int | None]] = []
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
                if not isinstance(obj, dict) or obj.get("type") != "assistant":
                    continue
                message = obj.get("message")
                usage = message.get("usage") if isinstance(message, dict) else None
                normalized = normalize_usage(usage if isinstance(usage, dict) else None)
                if normalized is not None:
                    usages.append(normalized)
    except FileNotFoundError:
        return []
    except OSError as exc:
        log.warning(
            "parse_usage: failed reading stream-json log %s: %r", path, exc
        )
        return []
    return usages


def parse_claude_stream_json_tool_calls(
    log_path: str | Path,
) -> list[dict[str, Any]]:
    """Recover the intel/tool calls a specialist made from its stream-json log.

    Recovers every ``tool_use`` block the agent emitted, so the trace can
    surface what the specialist actually read (WebSearch / WebFetch / Grep /
    Read / ...). This is the data behind the per-call ``intel:<tool>`` spans.

    Each returned entry is ``{"tool": <name>, "query": <short input summary>}``,
    in call order. The input summary prefers common query-ish keys
    (``query`` / ``url`` / ``pattern`` / ``path`` / ``prompt``) and otherwise
    falls back to a compact clipped JSON of the input.

    Tolerant by contract: a missing file, malformed lines, or no tool calls
    returns ``[]``.
    """
    path = Path(log_path)
    calls: list[dict[str, Any]] = []
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
                if not isinstance(obj, dict) or obj.get("type") != "assistant":
                    continue
                message = obj.get("message")
                if not isinstance(message, dict):
                    continue
                for block in message.get("content") or []:
                    if (
                        not isinstance(block, dict)
                        or block.get("type") != "tool_use"
                    ):
                        continue
                    name = str(block.get("name") or "").strip()
                    if not name:
                        continue
                    calls.append({
                        "tool": name,
                        "query": _summarize_tool_input(block.get("input")),
                    })
    except FileNotFoundError:
        return []
    except OSError as exc:
        log.warning(
            "parse_usage: failed reading stream-json log %s: %r", path, exc
        )
        return []
    return calls


def _summarize_tool_input(value: Any, *, limit: int = 240) -> str:
    """Compact, clipped one-line summary of a tool_use ``input`` block."""
    if isinstance(value, dict):
        for key in ("query", "url", "pattern", "path", "prompt", "command"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                s = v.strip()
                return s if len(s) <= limit else (s[:limit] + "…")
        try:
            s = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            s = str(value)
    else:
        s = "" if value is None else str(value)
    return s if len(s) <= limit else (s[:limit] + "…")


def parse_forge_usage(stdout: str) -> dict[str, int | None] | None:
    """Extract the run's LLM usage from a Kernel-Forge backend's stdout log.

    ``forge_submit`` aggregates the per-query ``ResultMessage`` token spend and
    prints one canonical marker line::

        FORGE_LLM_USAGE {"input_tokens": ..., "output_tokens": ...,
                         "cache_creation_input_tokens": ..., ...}

    Recovers the last such marker (the authoritative run total). Returns
    ``None`` when no marker is present.
    """
    if not stdout or "FORGE_LLM_USAGE" not in stdout:
        return None
    last_usage: dict[str, Any] | None = None
    for line in stdout.splitlines():
        marker = line.partition("FORGE_LLM_USAGE")
        if not marker[1]:
            continue
        blob = marker[2].strip()
        if not blob:
            continue
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and obj:
            last_usage = obj
    return normalize_usage(last_usage)


def parse_forge_steps(stdout: str) -> dict[str, Any] | None:
    """Extract the Kernel-Forge loop's key-step timeline from its stdout log.

    ``forge_submit`` prints one canonical marker carrying the per-iteration step
    timeline plus a run summary::

        FORGE_STEPS {"steps": [{"iteration": 1, "decision": "KEEP", ...}, ...],
                     "summary": {"iterations": ..., "termination_reason": ...}}

    Returns the parsed ``{"steps": [...], "summary": {...}}`` dict from the last
    marker, or ``None`` when no marker is present.
    """
    if not stdout or "FORGE_STEPS" not in stdout:
        return None
    last: dict[str, Any] | None = None
    for line in stdout.splitlines():
        marker = line.partition("FORGE_STEPS")
        if not marker[1]:
            continue
        blob = marker[2].strip()
        if not blob:
            continue
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
            last = obj
    return last


def parse_geak_usage(
    payload: dict[str, Any] | str | None,
) -> dict[str, int | None] | None:
    """Extract ``usage`` from a GEAK / litellm result.

    A litellm completion response exposes ``usage`` with ``prompt_tokens`` /
    ``completion_tokens`` (OpenAI shape). ``payload`` may be the parsed response
    dict or a JSON string. Maps the OpenAI counter names onto the canonical keys
    (``prompt_tokens`` → ``input_tokens``, ``completion_tokens`` →
    ``output_tokens``) and also honors a usage block already using canonical names.

    No prompt-cache split, so ``cache_*`` stay ``None``. Returns ``None`` when
    nothing parseable is found.

    Args:
        payload: A parsed litellm response dict, a JSON string, or ``None``.

    Returns:
        The canonical token dict, or ``None`` when nothing parseable.
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
    # Translate OpenAI counter names when the canonical ones are absent.
    translated: dict[str, Any] = dict(usage)
    if "input_tokens" not in translated and "prompt_tokens" in translated:
        translated["input_tokens"] = translated.get("prompt_tokens")
    if "output_tokens" not in translated and "completion_tokens" in translated:
        translated["output_tokens"] = translated.get("completion_tokens")
    return normalize_usage(translated)


def _find_usage_in_obj(obj: Any, _depth: int = 0) -> dict[str, Any] | None:
    """Best-effort search for a ``usage``/``token_usage`` dict in ``obj``.

    Looks at the object itself, then recurses into nested dicts/lists with a
    bounded depth to keep this cheap.

    Args:
        obj: Arbitrary parsed JSON value to search.
        _depth: Internal recursion depth guard.

    Returns:
        The first ``usage``/``token_usage`` dict found, or ``None``.
    """
    if _depth > 4 or not isinstance(obj, dict):
        return None
    for key in ("usage", "token_usage"):
        candidate = obj.get(key)
        if isinstance(candidate, dict) and candidate:
            return candidate
    # Recurse into nested dicts and lists.
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
    "parse_claude_stream_json_response",
    "parse_claude_stream_json_tool_calls",
    "parse_claude_stream_json_turn_usages",
    "parse_claude_stream_json_usage",
    "parse_forge_steps",
    "parse_forge_usage",
]
