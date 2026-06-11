# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pure mapping helpers shared by the Langfuse live emitter and backfill CLI.

The trace subsystem persists three local JSONL streams under
``reports/trace/`` (``llm_calls.jsonl`` token ledger, ``conversations.jsonl``
full text, ``decision_trace.jsonl`` KEEP/REVERT journal). Both the *live*
emitter (:mod:`.langfuse_emitter`) and the *offline* backfill
(``inference_optimizer.scripts.backfill_langfuse``) project those rows onto
the same Langfuse object model:

* session            -> Trace      (``trace_id`` derived from ``session_id``)
* phase              -> Span
* one LLM call       -> Generation (model + token usage + prompt/response)
* one decision       -> Score      (gain_pct NUMERIC / outcome CATEGORICAL)

Keeping the projection here -- as pure, SDK-free functions -- means the two
producers can never drift on how a token row becomes a Generation or how a
session id becomes a trace id. Nothing in this module imports ``langfuse``;
it only reshapes dicts and parses timestamps.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

UNPHASED = "(unphased)"
UNKNOWN_AGENT = "(unknown)"


def correlation_seed(manifest: dict[str, Any], fallback: str) -> str:
    """Pick the Langfuse correlation seed for a session.

    Prefers the PrimusClaw session id (``claw_session_id``) so every trace
    produced for one hosted sandbox session -- live push, offline backfill,
    and any future claw-side upload -- collapses onto the same Langfuse trace
    / session view. Falls back to the internal session id (e.g. the session
    dir name) for standalone/local runs where no claw id exists.
    """
    claw = str(manifest.get("claw_session_id") or "").strip()
    if claw:
        return claw
    sid = str(manifest.get("session_id") or "").strip()
    return sid or str(fallback)


def langfuse_session_id(manifest: dict[str, Any], fallback: str) -> str:
    """The value for Langfuse's ``session_id`` grouping dimension.

    Same precedence as :func:`correlation_seed` (claw id wins) -- this is the
    human-facing session grouping in the Langfuse UI, whereas the trace_id is
    its hashed form.
    """
    return correlation_seed(manifest, fallback)


def agent_of(row: dict[str, Any]) -> str:
    """The agent that produced a row: its ``component`` (role fallback).

    ``component`` is the closed producer vocabulary (orchestration / kernel /
    specialist / critic / geak / oob / robustness / proposal_scorer /
    tracelens / breakdown); it is the "which agent did this" axis used for the
    per-agent span layer.
    """
    return str(row.get("component") or row.get("role") or UNKNOWN_AGENT)


def phase_of(row: dict[str, Any]) -> str:
    """The phase a row belongs to (``(unphased)`` when absent)."""
    return str(row.get("phase") or UNPHASED)


def span_key(row: dict[str, Any]) -> tuple[str, str]:
    """(phase, agent) key identifying which agent-span a Generation nests in."""
    return (phase_of(row), agent_of(row))


def derive_trace_id(seed: str) -> str:
    """Map a session id (or any seed) to a stable 32-char lowercase hex id.

    Langfuse trace ids must be 32-char lowercase hex; deriving from the
    session id keeps re-runs / live+backfill of the same session writing to
    one trace instead of duplicating it.
    """
    return hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:32]


def parse_ts(ts: str | None) -> datetime | None:
    """Parse the two ts formats we emit: ISO+offset and ``...Z``.

    Returns ``None`` for missing / unparseable input so callers can fall
    back to a span/trace-level time without crashing.
    """
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def utc_second_key(ts: str | None) -> str:
    """Truncate a ts to whole UTC seconds, for cross-file pairing.

    ``llm_calls.jsonl`` and ``conversations.jsonl`` stamp their own ``ts`` a
    few milliseconds apart for the same logical call, so pairing is done at
    whole-second resolution.
    """
    dt = parse_ts(ts)
    if dt is None:
        return ""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def pair_key(row: dict[str, Any]) -> tuple:
    """Stable join key pairing a token row with its conversation row.

    Both streams (``llm_calls.jsonl`` / ``conversations.jsonl``) share the
    same closed schema, so we key on every per-call identity field they
    carry -- (component, task_id, dyn_id, tick, turn, role, model) -- and
    fall back to the UTC-second of ``ts`` only to disambiguate. ``turn`` /
    ``task_id`` / ``dyn_id`` keep a *burst* of calls in the same UTC second
    and same (component, tick, role) -- e.g. multi-turn specialist/critic
    within one tick -- from cross-pairing. ``model`` keeps concurrently
    scored proposals apart: ``ProposalScorer.score`` fires several models via
    ``asyncio.gather``, so multiple rows land in the same second with
    otherwise identical keys. Older rows that predate any field carry
    ``None``, so the key degrades gracefully to the previous behaviour.
    """
    return (
        str(row.get("component") or ""),
        row.get("task_id"),
        row.get("dyn_id"),
        row.get("tick"),
        row.get("turn"),
        str(row.get("role") or ""),
        str(row.get("model") or ""),
        utc_second_key(row.get("ts")),
    )


def usage_details(row: dict[str, Any]) -> dict[str, int]:
    """Project a token row's four counters onto Langfuse ``usage_details``.

    Drops ``None`` counters (so an unreported counter is absent rather than
    a misleading zero) and maps our canonical names onto the short Langfuse
    keys.
    """
    raw = {
        "input": row.get("input_tokens"),
        "output": row.get("output_tokens"),
        "cache_creation_input": row.get("cache_creation_input_tokens"),
        "cache_read_input": row.get("cache_read_input_tokens"),
    }
    return {k: int(v) for k, v in raw.items() if v is not None}


def generation_name(row: dict[str, Any]) -> str:
    """Human-friendly Generation name: component, falling back to role."""
    return str(row.get("component") or row.get("role") or "llm_call")


def generation_metadata(
    row: dict[str, Any], *, phase: str, has_text: bool,
) -> dict[str, Any]:
    """Assemble the per-Generation metadata block (join keys + flags)."""
    return {
        "phase": phase,
        "tick": row.get("tick"),
        "turn": row.get("turn"),
        "task_id": row.get("task_id"),
        "dyn_id": row.get("dyn_id"),
        "role": row.get("role"),
        "component": row.get("component"),
        "has_text": has_text,
    }


def trace_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    """Assemble the trace-level metadata block from ``manifest.json``."""
    workload = manifest.get("workload") or {}
    return {
        "model_name": manifest.get("model_name"),
        "model_class": workload.get("model_class") if isinstance(workload, dict) else None,
        "gpu_type": manifest.get("gpu_type"),
        "framework": manifest.get("framework"),
        "tp": manifest.get("tp"),
        "image": manifest.get("image"),
        "claw_session_id": manifest.get("claw_session_id"),
        "code_revision": manifest.get("code_revision"),
        "objective": manifest.get("objective"),
        "workload": workload,
        "host": manifest.get("host"),
    }


def decision_to_scores(decision_row: dict[str, Any]) -> list[dict[str, Any]]:
    """Project one ``decision_trace.jsonl`` row onto zero or more Score dicts.

    Always emits a CATEGORICAL ``decision_outcome`` score (KEEP / REVERT /
    no_promote); additionally a NUMERIC ``gain_pct`` score when the decision
    carries a measured gain. Each returned dict is transport-agnostic
    (``name`` / ``value`` / ``data_type`` / ``comment`` / ``metadata``) so the
    caller can hand it to the SDK or a REST body unchanged.
    """
    dec = decision_row.get("decision") or {}
    meta = {
        "phase": decision_row.get("phase"),
        "tick": decision_row.get("tick"),
        "change": dec.get("change"),
        "component": dec.get("component"),
        "task_id": dec.get("task_id"),
    }
    comment = str(dec.get("change") or "")
    scores: list[dict[str, Any]] = [{
        "name": "decision_outcome",
        "value": str(dec.get("outcome") or "unknown"),
        "data_type": "CATEGORICAL",
        "comment": comment,
        "metadata": meta,
    }]
    gain = dec.get("gain_pct")
    if gain is not None:
        try:
            scores.append({
                "name": "gain_pct",
                "value": float(gain),
                "data_type": "NUMERIC",
                "comment": comment,
                "metadata": meta,
            })
        except (TypeError, ValueError):
            pass
    return scores


__all__ = [
    "UNKNOWN_AGENT",
    "UNPHASED",
    "agent_of",
    "correlation_seed",
    "decision_to_scores",
    "derive_trace_id",
    "generation_metadata",
    "generation_name",
    "langfuse_session_id",
    "pair_key",
    "parse_ts",
    "phase_of",
    "span_key",
    "trace_metadata",
    "usage_details",
    "utc_second_key",
]
