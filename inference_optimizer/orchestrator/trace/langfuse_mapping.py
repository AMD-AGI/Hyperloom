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

    Keyed on (component, tick, role, UTC-second-of-ts) -- the identity a
    single logical call shares across the two streams.
    """
    return (
        str(row.get("component") or ""),
        row.get("tick"),
        str(row.get("role") or ""),
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
    "UNPHASED",
    "decision_to_scores",
    "derive_trace_id",
    "generation_metadata",
    "generation_name",
    "pair_key",
    "parse_ts",
    "trace_metadata",
    "usage_details",
    "utc_second_key",
]
