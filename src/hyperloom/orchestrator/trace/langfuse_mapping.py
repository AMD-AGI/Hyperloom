# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure mapping helpers shared by the Langfuse live emitter and backfill CLI.

Projects the local JSONL trace streams under ``reports/trace/`` onto the
Langfuse object model, shared by the live emitter and offline backfill:

* session              -> Trace       (``trace_id`` derived from ``session_id``)
* phase                -> Span
* (phase, agent)       -> Span        (``agent:<name>``, nested under the phase span)
* one LLM call         -> Generation  (model + token usage + prompt/response)
* one decision         -> ``optimization_step:<operation_kind>`` Span, plus 1-4 Scores:
                          ``decision_outcome`` CATEGORICAL (always) and, when present,
                          ``gain_pct`` / ``predicted_gain_pct`` / ``proposal_score`` NUMERIC
* recipe-KB audit row  -> Span        (``kb:recipe_write:<generator>`` / ``kb:recipe_snapshot:<method>``)

These are pure, SDK-free functions; nothing here imports ``langfuse``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

UNPHASED = "(unphased)"
UNKNOWN_AGENT = "(unknown)"

# Langfuse observation levels we emit. A failed LLM call must be ERROR so the
# error rate is queryable; everything else stays DEFAULT.
LEVEL_DEFAULT = "DEFAULT"
LEVEL_ERROR = "ERROR"

# ``llm_trace.LLM_STATUS_OK``, re-declared rather than imported: this module is
# a pure projection layer that ``llm_trace`` reaches back into via the emitter,
# so importing it here would close an import cycle.
_STATUS_OK = "ok"

# Env-var name fragments whose value is redacted before the environment
# snapshot is attached to session_start. Matched case-insensitively as a substring.
_SENSITIVE_ENV_MARKERS: tuple[str, ...] = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "PASSPHRASE",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "PRIVATEKEY",
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "SECRET_KEY",
    "AUTH",
    "SIGNATURE",
)
_REDACTED = "***redacted***"


def correlation_seed(manifest: dict[str, Any], fallback: str) -> str:
    """Pick the Langfuse correlation seed for a session.

    Prefers the PrimusClaw session id (``claw_session_id``) so every trace for
    one hosted sandbox session collapses onto the same Langfuse trace/session
    view. Falls back to the internal session id for standalone/local runs.

    Args:
        manifest: Session manifest dict carrying id fields.
        fallback: Seed used when no id is present in the manifest.

    Returns:
        The chosen correlation seed string.
    """
    claw = str(manifest.get("claw_session_id") or "").strip()
    if claw:
        return claw
    sid = str(manifest.get("session_id") or "").strip()
    return sid or str(fallback)


langfuse_session_id = correlation_seed
"""The value for Langfuse's ``session_id`` grouping dimension.

Same precedence as :func:`correlation_seed` (claw id wins) -- this is the
human-facing session grouping in the Langfuse UI, whereas the trace_id is its
hashed form.
"""


def agent_of(row: dict[str, Any]) -> str:
    """The agent that produced a row: its ``component`` (role fallback).

    ``component`` is the closed producer vocabulary used as the "which agent
    did this" axis for the per-agent span layer.

    Args:
        row: A trace row dict.

    Returns:
        The producing agent name, or ``UNKNOWN_AGENT`` when absent.
    """
    return str(row.get("component") or row.get("role") or UNKNOWN_AGENT)


def span_agent_for(proposer: str) -> str:
    """Map a resolved proposer back to a span-attachable agent name.

    ``specialist:<domain>`` collapses to ``specialist`` and ``grid`` to
    ``orchestration`` so a per-decision score lands under a real agent span.

    Args:
        proposer: The resolved proposer label from the decision metadata.

    Returns:
        The span-attachable agent name (``specialist`` / ``orchestration`` for
        the collapsed aliases, else the proposer or ``UNKNOWN_AGENT``).
    """
    p = (proposer or "").strip()
    if p.startswith("specialist:"):
        return "specialist"
    if p == "grid":
        return "orchestration"
    return p or UNKNOWN_AGENT


def phase_of(row: dict[str, Any]) -> str:
    """The phase a row belongs to (``(unphased)`` when absent).

    Args:
        row: A trace row dict.

    Returns:
        The phase name, or ``UNPHASED`` when absent.
    """
    return str(row.get("phase") or UNPHASED)


def derive_trace_id(seed: str) -> str:
    """Map a session id (or any seed) to a stable 32-char lowercase hex id.

    Langfuse trace ids must be 32-char lowercase hex; deriving from the session
    id keeps re-runs / live+backfill of the same session on one trace.

    Args:
        seed: Session id or any seed string to hash.

    Returns:
        A 32-char lowercase hex trace id.
    """
    return hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:32]


def parse_ts(ts: str | None) -> datetime | None:
    """Parse the two ts formats we emit: ISO+offset and ``...Z``.

    Returns ``None`` for missing / unparseable input so callers can fall
    back to a span/trace-level time without crashing.

    Args:
        ts: Timestamp string in ISO+offset or ``...Z`` form, or ``None``.

    Returns:
        The parsed ``datetime``, or ``None`` when missing/unparseable.
    """
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def generation_start(end: datetime | None, latency_ms: Any) -> datetime | None:
    """Backdate a generation's start by its measured call latency.

    Returns ``end - latency`` so the Langfuse generation shows a real duration.
    Falls back to ``end`` (zero-width) when either input is missing/unparseable
    or the latency is non-positive.
    """
    if end is None:
        return None
    try:
        ms = int(latency_ms)
    except (TypeError, ValueError):
        return end
    if ms <= 0:
        return end
    try:
        return end - timedelta(milliseconds=ms)
    except (OverflowError, ValueError):
        return end


def utc_second_key(ts: str | None) -> str:
    """Truncate a ts to whole UTC seconds, for cross-file pairing.

    ``llm_calls.jsonl`` and ``conversations.jsonl`` stamp their own ``ts`` a
    few ms apart for the same call, so pairing is done at whole-second resolution.

    Args:
        ts: Timestamp string, or ``None``.

    Returns:
        The UTC second key string, or ``""`` when unparseable.
    """
    dt = parse_ts(ts)
    if dt is None:
        return ""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def pair_key(row: dict[str, Any]) -> tuple:
    """Stable join key pairing a token row with its conversation row.

    A row carrying ``call_id`` keys on that alone: it identifies the one call
    both halves describe, so the pair survives a call whose two rows land either
    side of a second boundary and can never marry two calls made inside one
    second. Without it the key falls back to every per-call identity field both
    streams carry -- (component, task_id, dyn_id, tick, turn, role, model) --
    plus the UTC-second of ``ts`` to disambiguate. ``turn`` / ``task_id`` /
    ``dyn_id`` keep a burst of calls in the same second from cross-pairing;
    ``model`` keeps concurrently scored proposals apart. Missing fields degrade
    to ``None``.

    Args:
        row: A token or conversation trace row dict.

    Returns:
        A tuple join key pairing the row across streams.
    """
    call_id = str(row.get("call_id") or "").strip()
    if call_id:
        return ("call_id", call_id)
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
    """Project a token row's counters onto Langfuse ``usage_details``.

    Drops ``None`` counters and maps canonical names onto the short Langfuse
    keys. ``reasoning_output`` is reported as its own key: on a reasoning model
    it dominates the output budget while being absent from the visible reply, so
    folding it into ``output`` would misreport both.

    Args:
        row: A token trace row dict.

    Returns:
        Mapping of Langfuse usage key to integer count (``None`` dropped).
    """
    raw = {
        "input": row.get("input_tokens"),
        "output": row.get("output_tokens"),
        "cache_creation_input": row.get("cache_creation_input_tokens"),
        "cache_read_input": row.get("cache_read_input_tokens"),
        "reasoning_output": row.get("reasoning_output_tokens"),
    }
    return {k: int(v) for k, v in raw.items() if v is not None}


def generation_name(row: dict[str, Any]) -> str:
    """Human-friendly Generation name: component, falling back to role.

    Args:
        row: A token trace row dict.

    Returns:
        The Generation display name.
    """
    return str(row.get("component") or row.get("role") or "llm_call")


def generation_level(row: dict[str, Any]) -> str:
    """Map a row's terminal status onto a Langfuse observation level.

    Rows written before ``status`` existed carry no such key and are treated as
    successes, so a backfill of an old session does not turn every call red.

    Args:
        row: A token trace row dict.

    Returns:
        ``ERROR`` for a failed call, else ``DEFAULT``.
    """
    status = str(row.get("status") or _STATUS_OK).strip().lower()
    return LEVEL_DEFAULT if status == _STATUS_OK else LEVEL_ERROR


def generation_status_message(row: dict[str, Any]) -> str | None:
    """The human-readable failure reason for a row, or ``None`` on success.

    Args:
        row: A token trace row dict.

    Returns:
        ``"<error_type>: <error_message>"`` (either part may be missing), or
        ``None`` when the row is a success or carries no error detail.
    """
    if generation_level(row) == LEVEL_DEFAULT:
        return None
    parts = [str(row.get(k) or "").strip() for k in ("error_type", "error_message")]
    return ": ".join(p for p in parts if p) or None


def generation_metadata(
    row: dict[str, Any],
    *,
    phase: str,
    has_text: bool,
) -> dict[str, Any]:
    """Assemble the per-Generation metadata block (join keys + flags).

    Args:
        row: A token trace row dict.
        phase: Phase name to stamp onto the metadata.
        has_text: Whether a paired conversation text was found.

    Returns:
        The per-Generation metadata dict.
    """
    return {
        "phase": phase,
        "tick": row.get("tick"),
        "turn": row.get("turn"),
        "task_id": row.get("task_id"),
        "dyn_id": row.get("dyn_id"),
        "role": row.get("role"),
        "component": row.get("component"),
        "has_text": has_text,
        "latency_ms": row.get("latency_ms"),
        "reviewed_msg_ids": row.get("reviewed_msg_ids"),
        "status": row.get("status") or _STATUS_OK,
        "error_type": row.get("error_type"),
    }


def trace_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    """Assemble the trace-level metadata block from ``manifest.json``.

    Args:
        manifest: Parsed ``manifest.json`` dict.

    Returns:
        The trace-level metadata dict.
    """
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


def redact_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Snapshot the process environment with secret values redacted.

    Keeps every variable name but replaces the value of anything whose name
    looks like a credential (see :data:`_SENSITIVE_ENV_MARKERS`) with a
    placeholder, so session_start never ships secrets to Langfuse.

    Args:
        environ: The process environment mapping (e.g. ``os.environ``).

    Returns:
        A plain dict copy with sensitive values redacted.
    """
    out: dict[str, str] = {}
    for key, value in environ.items():
        upper = key.upper()
        if any(marker in upper for marker in _SENSITIVE_ENV_MARKERS):
            out[key] = _REDACTED
        else:
            out[key] = value
    return out


def session_start_payload(
    manifest: dict[str, Any],
    *,
    user_data_path: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the one-shot session-start document.

    Carries the entire ``manifest.json`` plus the WekaFS user-data root and a
    redacted environment snapshot, so a run that aborts before producing a
    breakdown still leaves a self-describing Langfuse trace.

    Args:
        manifest: Parsed ``manifest.json`` dict (copied verbatim into the
            payload).
        user_data_path: WekaFS user-data root (the ``USER_DATA_PATH`` env),
            passed in by the caller so this stays free of env coupling.
        env: Process environment to snapshot; redacted via :func:`redact_env`.
            Omitted from the payload when ``None``.

    Returns:
        JSON-serializable startup document.
    """
    payload: dict[str, Any] = dict(manifest or {})
    payload["user_data_path"] = user_data_path or None
    if env is not None:
        payload["env"] = redact_env(env)
    return payload


def decision_to_scores(decision_row: dict[str, Any]) -> list[dict[str, Any]]:
    """Project one ``decision_trace.jsonl`` row onto one or more Score dicts.

    Always emits a CATEGORICAL ``decision_outcome`` score (KEEP / REVERT /
    no_promote / skipped); additionally NUMERIC ``gain_pct`` (measured gain),
    ``predicted_gain_pct`` (the proposer's estimate) and ``proposal_score``
    (mean pre-decision rater score) when the decision carries them. Each
    returned dict is transport-agnostic
    (``name`` / ``value`` / ``data_type`` / ``comment`` / ``metadata``) so the
    caller can hand it to the SDK or a REST body unchanged.

    Args:
        decision_row: One ``decision_trace.jsonl`` row dict.

    Returns:
        A list of one to four transport-agnostic Score dicts.
    """
    dec = decision_row.get("decision") or {}
    meta = {
        "phase": decision_row.get("phase"),
        "tick": decision_row.get("tick"),
        "change": dec.get("change"),
        "component": dec.get("component"),
        "task_id": dec.get("task_id"),
        # Proposer attribution + filter label for slicing a trace.
        "operation_kind": dec.get("operation_kind"),
        "proposer": dec.get("component"),
        "provenance": dec.get("provenance"),
        "scope": dec.get("scope"),
        "variant_name": dec.get("variant_name"),
        "fingerprint": dec.get("fingerprint"),
        "metrics": dec.get("metrics"),
        "proposal_scores": dec.get("proposal_scores"),
        "predicted_gain_pct": dec.get("predicted_gain_pct"),
    }
    # Drop keys the decision didn't carry.
    meta = {k: v for k, v in meta.items() if v is not None}
    comment = str(dec.get("change") or "")
    scores: list[dict[str, Any]] = [
        {
            "name": "decision_outcome",
            "value": str(dec.get("outcome") or "unknown"),
            "data_type": "CATEGORICAL",
            "comment": comment,
            "metadata": meta,
        }
    ]
    gain = dec.get("gain_pct")
    if gain is not None:
        try:
            scores.append(
                {
                    "name": "gain_pct",
                    "value": float(gain),
                    "data_type": "NUMERIC",
                    "comment": comment,
                    "metadata": meta,
                }
            )
        except (TypeError, ValueError):
            pass
    # Calibration signal: the proposer's predicted gain as its own NUMERIC score.
    predicted = dec.get("predicted_gain_pct")
    if predicted is not None:
        try:
            scores.append(
                {
                    "name": "predicted_gain_pct",
                    "value": float(predicted),
                    "data_type": "NUMERIC",
                    "comment": comment,
                    "metadata": meta,
                }
            )
        except (TypeError, ValueError):
            pass
    # Calibration signal: the proposal_scorer's pre-decision rating (mean across
    # raters) as its own NUMERIC score.
    pred = _mean_proposal_score(dec.get("proposal_scores"))
    if pred is not None:
        scores.append(
            {
                "name": "proposal_score",
                "value": pred,
                "data_type": "NUMERIC",
                "comment": comment,
                "metadata": meta,
            }
        )
    return scores


def recipe_audit_is_write(row: Mapping[str, Any]) -> bool:
    """Whether one recipe-KB audit row describes a write.

    Rows written before the write-audit event carry no ``op`` field; they are
    reads, so the absence of ``op`` must not be mistaken for a write.

    Args:
        row: One row from ``runtime/recipe_snapshot/.audit.jsonl``.

    Returns:
        ``True`` for a write row, ``False`` for a read (or legacy) row.
    """
    return str(row.get("op") or "read") == "write"


def recipe_write_span(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Project a recipe-KB write audit row onto a span name + metadata.

    The generator suffix in the name separates the two write sites: the
    session-opening ``t0_anchor`` stamp and the ``coordinator``
    KEEP/REVERT/PR/CLOSE amends.

    ``delta`` is flattened into ``<field>_delta`` scalars because Langfuse
    filters on flat metadata values; the full row is attached separately as the
    span output. Only fields that actually changed are flattened, so a
    no-content write (the T0 anchor) yields no delta keys rather than a row of
    zeros.

    Args:
        row: One ``op="write"`` row from the recipe-KB audit log.

    Returns:
        A ``(span_name, metadata)`` pair.
    """
    result = row.get("result") if isinstance(row.get("result"), Mapping) else {}
    delta = row.get("delta") if isinstance(row.get("delta"), Mapping) else {}
    generator = str(row.get("generator") or "unknown")
    metadata: dict[str, Any] = {
        "kind": "recipe_write",
        "generator": generator,
        "phase": row.get("phase") or "",
        "canonical_id": result.get("canonical_id") or "",
        "version": result.get("version"),
        "created": bool(result.get("created")),
        "best_throughput": result.get("best_throughput"),
        "best_config_nonempty": bool(result.get("best_config_nonempty")),
    }
    for field, value in delta.items():
        if value:
            metadata[f"{field}_delta"] = value
    return f"kb:recipe_write:{generator}", metadata


def recipe_read_span(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Project a recipe-KB read audit row onto a span name + metadata.

    Args:
        row: One read row from the recipe-KB audit log.

    Returns:
        A ``(span_name, metadata)`` pair.
    """
    method = str(row.get("method") or "read")
    return f"kb:recipe_snapshot:{method}", {
        "kind": "recipe_snapshot",
        "method": method,
        "remote": row.get("remote"),
        "resolution": row.get("resolution"),
        "hit": bool(row.get("hit")),
    }


def _mean_proposal_score(proposal_scores: Any) -> float | None:
    """Mean of the per-rater ``score`` values in a decision's proposal_scores.

    Returns the mean rater score (0-10), or ``None`` when there are no numeric
    scores.
    """
    if not isinstance(proposal_scores, list):
        return None
    vals: list[float] = []
    for entry in proposal_scores:
        if not isinstance(entry, dict):
            continue
        try:
            vals.append(float(entry.get("score")))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


__all__ = [
    "UNKNOWN_AGENT",
    "UNPHASED",
    "agent_of",
    "correlation_seed",
    "decision_to_scores",
    "derive_trace_id",
    "generation_metadata",
    "generation_name",
    "generation_start",
    "langfuse_session_id",
    "pair_key",
    "parse_ts",
    "phase_of",
    "recipe_audit_is_write",
    "recipe_read_span",
    "recipe_write_span",
    "redact_env",
    "session_start_payload",
    "span_agent_for",
    "trace_metadata",
    "usage_details",
    "utc_second_key",
]
