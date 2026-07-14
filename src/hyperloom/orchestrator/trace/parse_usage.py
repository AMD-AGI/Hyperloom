# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Recover token ``usage`` from out-of-process LLM-client output.

In-process backends hand us a ``BackendTurnResult.metadata`` dict directly,
but the production-critical paths (the specialist subprocess, forge
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

from ._row_utils import coerce_optional_int as _coerce_optional_int

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


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int | None] | None:
    """Project an arbitrary ``usage`` dict onto the canonical four keys.

    Returns ``None`` when ``usage`` is falsy or carries none of the four
    recognized counters (so a stray ``{}`` or an unrelated dict doesn't
    masquerade as a real measurement). Unknown keys are dropped; absent
    counters become ``None``.

    Args:
        usage: Arbitrary usage dict, or ``None``.

    Returns:
        The canonical four-key token dict, or ``None`` when nothing usable.
    """
    if not isinstance(usage, dict) or not usage:
        return None
    projected: dict[str, int | None] = {k: _coerce_optional_int(usage.get(k)) for k in _TOKEN_KEYS}
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
                    # A result-typed row is authoritative; let it win over
                    # any earlier assistant-message usage on the same stream.
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

    Sibling of :func:`parse_claude_stream_json_usage`: that function reads the
    *token* counts off the same ``process.log``, this one reads the
    *conversation* response so the production-default specialist path (B1)
    can land its completion in ``conversations.jsonl``. The prompt is not
    echoed into the stream (the CLI takes it via ``-p`` / a prompt file), so
    only the response is recovered here; the caller already holds the prompt
    in memory and pairs the two.

    The ``claude --output-format stream-json`` log is one JSON object per
    line. We reconstruct the reply from two sources, preferring the
    authoritative one:

    1. the terminal ``{"type": "result", ..., "result": "<text>"}`` row,
       whose ``result`` is the consolidated final answer (mirrors the SDK's
       ``ResultMessage.result``); when present and non-empty it wins;
    2. otherwise we concatenate the ``text`` blocks from every
       ``{"type": "assistant"}`` message in order — covering a truncated or
       crashed run that never emitted a ``result`` row. ``thinking`` and
       ``tool_use`` blocks are intentionally dropped: the response field is
       the model's externally-visible answer, not its scratch reasoning or
       tool plumbing.

    Tolerant by contract: a missing file, malformed lines, or a stream with
    no recoverable text returns ``None`` so the best-effort trace path
    degrades to "no response captured" instead of raising.

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

    Sibling of :func:`parse_claude_stream_json_usage`, which collapses a
    multi-turn ``claude`` session into ONE cumulative usage row. This one
    instead returns the per-message usage carried on each
    ``{"type":"assistant", "message": {..., "usage": {...}}}`` line, in order,
    so a multi-turn specialist subprocess can be traced as one ledger row per
    model turn (B1 granularity) instead of a single ``turn=1`` lump.

    Each assistant message in the Claude Messages API carries its OWN usage
    (not cumulative), so the rows can be summed to reconstruct the session
    total. Lines without a usage block are skipped; a terminal ``result`` row
    (cumulative) is intentionally ignored here to avoid double counting.

    Returns the normalized four-key dicts in stream order, or ``[]`` when the
    file is missing / truncated / carries no per-message usage. Tolerant by
    contract: never raises into the trace path.
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

    Sibling of :func:`parse_claude_stream_json_usage` /
    :func:`parse_claude_stream_json_response`: those recover the *tokens* and
    the *reply*; this one recovers the **tool plumbing** — every ``tool_use``
    block the agent emitted — so the trace can surface what the specialist
    actually read (WebSearch / WebFetch / ``mcp__pr_monitor__*`` /
    ``mcp__cortex_kb__*`` / Grep / Read / ...) instead of only its token total.
    This is the data behind the per-call ``intel:<tool>`` spans.

    Each returned entry is ``{"tool": <name>, "query": <short input summary>}``,
    in call order. The input summary prefers the common query-ish keys
    (``query`` / ``url`` / ``pattern`` / ``path`` / ``prompt``) and otherwise
    falls back to a compact JSON of the input, clipped to keep the trace small.

    Tolerant by contract: a missing file, malformed lines, or a stream with no
    tool calls returns ``[]`` so the best-effort trace path never raises.
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

    Kernel-Forge's autonomous loop drives the claude-agent-sdk in-process, so
    (unlike GEAK) there is no single SDK ``usage`` envelope on stdout.
    Instead ``forge_submit`` aggregates the per-query ``ResultMessage`` token
    spend (via Kernel-Forge's ``UsageAccumulator``) and prints one canonical
    marker line::

        FORGE_LLM_USAGE {"input_tokens": ..., "output_tokens": ...,
                         "cache_creation_input_tokens": ..., ...}

    This parser recovers the *last* such marker (the authoritative run total).
    The four canonical counters come straight from the claude usage shape.
    Returns ``None`` when no marker is present (older Forge / no-agent run).
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
    timeline (rationale / validation / bench / keep-revert) plus a run summary::

        FORGE_STEPS {"steps": [{"iteration": 1, "decision": "KEEP", ...}, ...],
                     "summary": {"iterations": ..., "termination_reason": ...}}

    Returns the parsed ``{"steps": [...], "summary": {...}}`` dict from the last
    marker, or ``None`` when no marker is present (older Forge / no-result run).
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

    GEAK uses litellm as a *library* (local, not a gateway); a litellm
    completion response exposes ``usage`` with ``prompt_tokens`` /
    ``completion_tokens`` (OpenAI shape). ``payload`` may be the already
    parsed response dict or a JSON string. We map the OpenAI counter names
    onto our canonical keys (``prompt_tokens`` → ``input_tokens``,
    ``completion_tokens`` → ``output_tokens``) and also honor a usage block
    that already uses the canonical names.

    No prompt-cache split, so ``cache_*`` stay ``None``. Returns ``None``
    when nothing parseable is found.

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
    "parse_claude_stream_json_response",
    "parse_claude_stream_json_tool_calls",
    "parse_claude_stream_json_turn_usages",
    "parse_claude_stream_json_usage",
    "parse_forge_steps",
    "parse_forge_usage",
]
