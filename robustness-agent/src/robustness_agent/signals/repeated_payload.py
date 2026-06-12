# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Detect same-fingerprint action retries (B1 / same_payload_loop).

Generalises the upstream ``baseline_no_param_change`` guard to the whole
action catalogue (motivated by the 2026-05 ``validate_stack`` 11-retry
loop where each OOM looked fresh because ``idempotency_key`` differed but
``params`` were identical). Hashes the action-defining subset of each
``delegated_result`` payload (from coordinator_events + inbox) and fires
``same_payload_loop`` when a family produces N consecutive same-hash
results with no intervening success. Per-family hash dimensions live in
``_FAMILY_PROJECTIONS``; unknown actions fall back to a generic ``params``
projection.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .._payload_aliases import (
    CANONICAL_KEY as _EXTRA_SERVER_ARGS_CANONICAL,
    LEGACY_KEY as _EXTRA_SERVER_ARGS_LEGACY,
    read_extra_server_args as _read_extra_server_args,
)
from ..role.prompt_inputs import InboxItem, ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity



# Per-family payload projection: dotted keys (stable order) that define
# the fingerprint; missing keys map to ``None`` so empties hash identically.
_FAMILY_PROJECTIONS: dict[str, tuple[str, ...]] = {
    "validate_stack": (
        "params.optimization_stack",
        "params.config_path",
        "params.benchmark_script",
        "params.result_dir",
    ),
    "backends": (
        "params.grid",
        "params.extra_envs",
        "params.config_path",
    ),
    "params": (
        "params.grid",
        "params.extra_envs",
        "params.config_path",
    ),
    "sweep": (
        "params.grid",
        "params.extra_envs",
        "params.config_path",
    ),
    "integrate": (
        "params.kernel_id",
        "params.patch_path",
        "params.extra_server_args",
        "params.extra_envs",
    ),
    "baseline": (
        "params.benchmark_script",
        "params.result_dir",
        "params.extra_server_args",
        "params.extra_envs",
        "params.model_path",
        "params.gpu_type",
        "params.config_path",
        "params.disable_run_eval",
    ),
}

# Generic fallback for unknown families with a ``params`` dict.
_GENERIC_PROJECTION: tuple[str, ...] = (
    "params",
)

# Per-attempt fields stripped before hashing; including them would make
# the fingerprint always-unique and the signal a no-op.
_HASH_BLACKLIST: frozenset[str] = frozenset({
    "idempotency_key",
    "task_id",
    "ts",
    "timestamp",
    "started_at",
    "finished_at",
    "submitted_at",
    "msg_id",
    "in_reply_to",
    "target_proposal_msg_id",
})


@dataclass
class RepeatedPayloadConfig:
    """Tunables for :func:`evaluate_repeated_payload_signals`.

    ``streak_threshold`` (consecutive same-hash failures before firing,
    default 3) gives one tick of warning before the deadline.
    ``lookback_events`` (default 80) caps the event walk at ~30 min.
    """

    streak_threshold: int = 3
    lookback_events: int = 80


def evaluate_repeated_payload_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: RepeatedPayloadConfig | None = None,
) -> list[Symptom]:
    """Fire ``same_payload_loop`` for action families stuck retrying one payload.

    Walks the combined inbox + coordinator event stream, groups consecutive
    same-fingerprint failures per family, and emits a symptom once a streak
    reaches the configured threshold.

    Args:
        ctx (ReactorContext): Reactor context providing the inbox.
        data (SourceData): Collected source data including coordinator events.
        config (RepeatedPayloadConfig | None): Tunables; defaults to
            :class:`RepeatedPayloadConfig` when ``None``.

    Returns:
        list[Symptom]: One ``same_payload_loop`` symptom per offending family,
            possibly empty.
    """
    cfg = config or RepeatedPayloadConfig()
    events = _gather_events(ctx.inbox, data.coordinator_events, cfg)
    if not events:
        return []
    out: list[Symptom] = []
    for family, streak_events in _walk_streaks(events).items():
        if len(streak_events) < cfg.streak_threshold:
            continue
        sym = _build_symptom(family, streak_events, cfg)
        if sym is not None:
            out.append(sym)
    return out


# ---------------------------------------------------------------------------
# Streak detection
# ---------------------------------------------------------------------------

def _walk_streaks(
    events: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group consecutive same-hash failures per family; a ``succeeded`` entry resets the streak."""
    by_family: dict[str, list[dict[str, Any]]] = {}
    current_hash: dict[str, str | None] = {}
    for ev in events:
        family = _family_of(ev)
        if not family:
            continue
        state = str(ev.get("state") or "")
        if state == "succeeded":
            by_family[family] = []
            current_hash[family] = None
            continue
        payload_hash = _hash_for(family, ev)
        if payload_hash is None:
            continue
        if current_hash.get(family) == payload_hash:
            by_family.setdefault(family, []).append(ev)
        else:
            by_family[family] = [ev]
            current_hash[family] = payload_hash
    return by_family


# ---------------------------------------------------------------------------
# Event normalisation
# ---------------------------------------------------------------------------

def _gather_events(
    inbox: list[InboxItem],
    coord_events: list[dict[str, Any]],
    cfg: RepeatedPayloadConfig,
) -> list[dict[str, Any]]:
    """Build a single time-ordered list of ``delegated_result`` rows, trimmed to ``lookback_events``."""
    inbox_rows = [
        {
            "topic": item.topic,
            "agent": item.from_agent,
            "payload": item.payload,
        }
        for item in inbox
        if item.topic == "delegated_result"
        and isinstance(item.payload, dict)
    ]
    coord_rows: list[dict[str, Any]] = []
    for ev in coord_events:
        if ev.get("topic") != "delegated_result":
            continue
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        coord_rows.append({
            "topic": "delegated_result",
            "agent": ev.get("agent", ""),
            "payload": payload,
        })

    combined: list[dict[str, Any]] = []
    for row in coord_rows + inbox_rows:
        payload = row.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        combined.append({
            "kind": payload.get("kind") or payload.get("action_name") or "",
            "family": payload.get("family") or "",
            "state": payload.get("state") or "",
            "task_id": payload.get("task_id"),
            "error": payload.get("error"),
            "error_class": payload.get("error_class"),
            "payload": payload,
        })
    if cfg.lookback_events > 0:
        combined = combined[-cfg.lookback_events:]
    return combined


def _family_of(event: dict[str, Any]) -> str:
    """Resolve the action family for an event, falling back to ``kind``.

    Args:
        event (dict[str, Any]): A normalised event row.

    Returns:
        str: The action family, or an empty string when neither ``family`` nor
            ``kind`` is set.
    """
    family = str(event.get("family") or "").strip()
    if family:
        return family
    kind = str(event.get("kind") or "").strip()
    return kind


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _hash_for(family: str, event: dict[str, Any]) -> str | None:
    """Compute the action-defining fingerprint for an event payload.

    Projects the family-specific (or generic) subset of the payload, strips
    blacklisted churn keys, and hashes the canonical JSON.

    Args:
        family (str): The action family used to pick the projection.
        event (dict[str, Any]): A normalised event row carrying the payload.

    Returns:
        str | None: A hex SHA-1 fingerprint, or ``None`` when the payload is not
            a usable dict.
    """
    payload = event.get("payload") or {}
    if not isinstance(payload, dict):
        return None
    # Normalise legacy extra-args key so legacy + canonical envelopes
    # produce identical fingerprints (else a legacy-keyed retry burst is missed).
    payload = _normalise_extra_server_args_key(payload)
    projection = _FAMILY_PROJECTIONS.get(family, _GENERIC_PROJECTION)
    subset: dict[str, Any] = {}
    for path in projection:
        value = _walk_path(payload, path)
        subset[path] = _strip_blacklisted(value)
    canonical = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _normalise_extra_server_args_key(payload: dict[str, Any]) -> dict[str, Any]:
    """Shallow copy with ``params.extra_server_args`` set from the compat
    helper (originals not mutated). No-op when no extra-args key or canonical
    already present; the shim's DeprecationWarning stays as the legacy audit channel.
    """
    params = payload.get("params")
    if not isinstance(params, dict):
        return payload
    if _EXTRA_SERVER_ARGS_CANONICAL in params:
        return payload
    if _EXTRA_SERVER_ARGS_LEGACY not in params:
        return payload
    new_params = dict(params)
    new_params[_EXTRA_SERVER_ARGS_CANONICAL] = _read_extra_server_args(params)
    new_payload = dict(payload)
    new_payload["params"] = new_params
    return new_payload


def _walk_path(payload: dict[str, Any], path: str) -> Any:
    """Walk dotted ``a.b.c`` paths against nested dicts (non-dicts short-circuit to ``None``)."""
    cur: Any = payload
    for token in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(token)
    return cur


def _strip_blacklisted(value: Any) -> Any:
    """Recursively drop ``_HASH_BLACKLIST`` keys from dicts.

    Args:
        value (Any): A value that may be a dict, list, or scalar.

    Returns:
        Any: The value with all blacklisted keys removed from nested dicts.
    """
    if isinstance(value, dict):
        return {
            k: _strip_blacklisted(v)
            for k, v in value.items()
            if k not in _HASH_BLACKLIST
        }
    if isinstance(value, list):
        return [_strip_blacklisted(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Symptom builder
# ---------------------------------------------------------------------------

def _build_symptom(
    family: str,
    streak_events: list[dict[str, Any]],
    cfg: RepeatedPayloadConfig,
) -> Symptom | None:
    """Build the ``same_payload_loop`` symptom for a detected streak.

    Args:
        family (str): The looping action family.
        streak_events (list[dict[str, Any]]): Consecutive same-hash failures.
        cfg (RepeatedPayloadConfig): Tunables (provides the streak threshold).

    Returns:
        Symptom | None: A HIGH-severity ``same_payload_loop`` symptom, or
            ``None`` when ``streak_events`` is empty.
    """
    if not streak_events:
        return None
    count = len(streak_events)
    last = streak_events[-1]
    error_classes = Counter(
        str(ev.get("error_class") or "").strip()
        for ev in streak_events
        if ev.get("error_class")
    )
    top_error = error_classes.most_common(1)[0][0] if error_classes else ""
    return Symptom(
        name="same_payload_loop",
        severity=SymptomSeverity.HIGH,
        summary=(
            f"action family {family!r} produced {count} consecutive failed "
            f"attempts with identical payload fingerprint "
            f"(>= {cfg.streak_threshold}); "
            f"top error_class={top_error or '(none)'}"
        ),
        evidence={
            "family": family,
            "count": count,
            "streak_threshold": cfg.streak_threshold,
            "last_task_id": last.get("task_id"),
            "last_error": (str(last.get("error") or ""))[:240],
            "last_error_class": last.get("error_class"),
            "top_error_class": top_error,
            "error_class_distribution": dict(error_classes),
        },
        subject={"family": family},
        source="coordinator_events",
        suggestion=(
            f"prune_branch family={family}; change params content (not just "
            f"idempotency_key) or propose `report` to wind down"
        ),
    )


__all__ = [
    "RepeatedPayloadConfig",
    "evaluate_repeated_payload_signals",
]
