# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pure mapping helpers shared by the Langfuse live emitter and backfill CLI.

The trace subsystem persists three local JSONL streams under
``reports/trace/`` (``llm_calls.jsonl`` token ledger, ``conversations.jsonl``
full text, ``decision_trace.jsonl`` KEEP/REVERT journal). Both the *live*
emitter (:mod:`.langfuse_emitter`) and the *offline* backfill
(``hyperloom.inference_optimizer.tools.backfill_langfuse``) project those rows onto
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
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

UNPHASED = "(unphased)"
UNKNOWN_AGENT = "(unknown)"

# Env-var name fragments whose *value* is redacted before the environment
# snapshot is attached to the session_start marker. The key name is kept (so
# you still see the variable existed), but the secret value never leaves the
# sandbox. Matched case-insensitively as a substring.
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

    Prefers the PrimusClaw session id (``claw_session_id``) so every trace
    produced for one hosted sandbox session -- live push, offline backfill,
    and any future claw-side upload -- collapses onto the same Langfuse trace
    / session view. Falls back to the internal session id (e.g. the session
    dir name) for standalone/local runs where no claw id exists.

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


def langfuse_session_id(manifest: dict[str, Any], fallback: str) -> str:
    """The value for Langfuse's ``session_id`` grouping dimension.

    Same precedence as :func:`correlation_seed` (claw id wins) -- this is the
    human-facing session grouping in the Langfuse UI, whereas the trace_id is
    its hashed form.

    Args:
        manifest: Session manifest dict carrying id fields.
        fallback: Seed used when no id is present in the manifest.

    Returns:
        The Langfuse session grouping id.
    """
    return correlation_seed(manifest, fallback)


def agent_of(row: dict[str, Any]) -> str:
    """The agent that produced a row: its ``component`` (role fallback).

    ``component`` is the closed producer vocabulary (orchestration / kernel /
    specialist / critic / geak / forge / robustness / proposal_scorer /
    tracelens / breakdown); it is the "which agent did this" axis used for the
    per-agent span layer.

    Args:
        row: A trace row dict.

    Returns:
        The producing agent name, or ``UNKNOWN_AGENT`` when absent.
    """
    return str(row.get("component") or row.get("role") or UNKNOWN_AGENT)


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

    Langfuse trace ids must be 32-char lowercase hex; deriving from the
    session id keeps re-runs / live+backfill of the same session writing to
    one trace instead of duplicating it.

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

    ``end`` is the call's write-time ``ts`` (≈ when the model reply landed);
    ``latency_ms`` is the wall-clock the call took. Returns ``end - latency``
    so the Langfuse generation shows a real duration. Falls back to ``end``
    (zero-width, legacy behaviour) when either input is missing/unparseable
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
    few milliseconds apart for the same logical call, so pairing is done at
    whole-second resolution.

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

    Args:
        row: A token or conversation trace row dict.

    Returns:
        A tuple join key pairing the row across streams.
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

    Keeps every variable name (so debuggers can see what was set) but replaces
    the value of anything whose name looks like a credential (see
    :data:`_SENSITIVE_ENV_MARKERS`) with a placeholder, so the session_start
    marker never ships secrets to Langfuse.

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

    Carries the *entire* ``manifest.json`` (identity, provenance, workload,
    dependency commits, stack fingerprint, host/image/pid, ...) plus the
    WekaFS user-data root and a redacted environment snapshot, so a run that
    aborts in pre-flight or is killed before producing a breakdown still leaves
    a fully self-describing Langfuse trace.

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
    """Project one ``decision_trace.jsonl`` row onto zero or more Score dicts.

    Always emits a CATEGORICAL ``decision_outcome`` score (KEEP / REVERT /
    no_promote); additionally a NUMERIC ``gain_pct`` score when the decision
    carries a measured gain. Each returned dict is transport-agnostic
    (``name`` / ``value`` / ``data_type`` / ``comment`` / ``metadata``) so the
    caller can hand it to the SDK or a REST body unchanged.

    Args:
        decision_row: One ``decision_trace.jsonl`` row dict.

    Returns:
        A list of transport-agnostic Score dicts (one or two entries).
    """
    dec = decision_row.get("decision") or {}
    meta = {
        "phase": decision_row.get("phase"),
        "tick": decision_row.get("tick"),
        "change": dec.get("change"),
        "component": dec.get("component"),
        "task_id": dec.get("task_id"),
        # Proposer attribution + filter label so a trace can be sliced by
        # "what this step did" (operation_kind) and "who proposed it".
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
    # Drop keys the decision didn't carry so the score metadata stays compact.
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
            # Malformed metric payload; skip this event.
            pass
    # Calibration signal: the proposer's predicted gain emitted as its own
    # NUMERIC score on the SAME decision so a trace can compute the
    # predicted-vs-realized error directly against ``gain_pct``.
    predicted = dec.get("predicted_gain_pct")
    if predicted is not None:
        try:
            scores.append({
                "name": "predicted_gain_pct",
                "value": float(predicted),
                "data_type": "NUMERIC",
                "comment": comment,
                "metadata": meta,
            })
        except (TypeError, ValueError):
            # Malformed metric payload; skip this event.
            pass
    # Calibration signal: the proposal_scorer's pre-decision rating (mean
    # across raters) emitted as its own NUMERIC score on the SAME decision so
    # a trace can compare predicted score vs realized ``gain_pct`` directly.
    pred = _mean_proposal_score(dec.get("proposal_scores"))
    if pred is not None:
        scores.append({
            "name": "proposal_score",
            "value": pred,
            "data_type": "NUMERIC",
            "comment": comment,
            "metadata": meta,
        })
    return scores


def _mean_proposal_score(proposal_scores: Any) -> float | None:
    """Mean of the per-rater ``score`` values in a decision's proposal_scores.

    ``proposal_scores`` is the ``[{"rater", "score", "reason"}, ...]`` list the
    collector attaches from the proposal_scorer ensemble. Returns the mean
    rater score (0-10), or ``None`` when there are no numeric scores.
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
    "redact_env",
    "session_start_payload",
    "trace_metadata",
    "usage_details",
    "utc_second_key",
]
