"""Detect same-fingerprint action retries (B1 / same_payload_loop).

The upstream ``Coordinator._baseline_self_loop_denial`` already guards
the ``baseline`` action against same-fingerprint retries (PolicyGate
``rule="baseline_self_loop"``), but the same trap can fire for any
action the orchestration LLM proposes — most famously the 2026-05
``validate_stack`` 11-retry loop where every attempt OOMed on a leaked
GPU but the Coordinator viewed each as a fresh task because the
``idempotency_key`` differed even though the underlying ``params``
were identical.

This signal generalises the upstream guard to the rest of the action
catalogue. We accept the ``delegated_result`` stream from
:data:`SourceData.coordinator_events` (and the robustness inbox), hash
the action-defining subset of each payload, and fire
``same_payload_loop`` when an action family produces ``N`` consecutive
results sharing the same hash without an intervening success.

Hash dimensions per action family:

* ``validate_stack`` — ``optimization_stack`` content + ``config_path``
* ``backends`` / ``params`` — sorted ``grid`` + ``extra_envs``
* ``integrate`` — ``kernel_id`` + ``patch_path`` + ``extra_server_args``
* ``baseline`` — falls back to ``_BASELINE_FINGERPRINT_KEYS`` so we
  stay aligned with the upstream policy denial fingerprint.

Other actions fall back to a generic ``params`` projection so a brand
new action still benefits from the safety net.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .._payload_aliases import (
    CANONICAL_KEY as _EXTRA_SERVER_ARGS_CANONICAL,
    LEGACY_KEY as _EXTRA_SERVER_ARGS_LEGACY,
    read_extra_server_args as _read_extra_server_args,
)
from ..role.prompt_inputs import InboxItem, ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity



# Per-family payload projection. Each tuple lists the dotted keys we
# care about in stable order; missing keys map to ``None`` so empty vs
# default-empty payloads still hash identically.
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

# Generic fallback used when ``family`` is unknown but the payload looks
# action-shaped (i.e. has a ``params`` dict). Drops idempotency_key /
# timestamps so renames do not bypass the dedup.
_GENERIC_PROJECTION: tuple[str, ...] = (
    "params",
)

# Fields we always strip from any payload subtree before hashing —
# these change every attempt by design (the Coordinator and the
# orchestration LLM stamp them) and must not contribute to the
# fingerprint or the signal turns into a no-op.
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

    ``streak_threshold`` is the number of consecutive same-hash failures
    that must precede a fire. Default = 3 matches the on-call pain
    point: at four identical attempts the operator should already be
    suspicious; at three we still have one tick of warning before the
    deadline. ``lookback_events`` caps how far back we walk the event
    stream; the upstream Coordinator only emits one
    ``delegated_result`` per task so 80 covers ~30 minutes of activity
    at a typical 4-tasks-per-minute clip.
    """

    streak_threshold: int = 3
    lookback_events: int = 80


def evaluate_repeated_payload_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: RepeatedPayloadConfig | None = None,
) -> list[Symptom]:
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
    """Group consecutive same-hash entries per family.

    A successful (``state == "succeeded"``) entry resets the family's
    streak — the LLM has made a different choice that worked, so any
    prior loop has been broken. A streak of failures with the same
    payload hash, by contrast, accumulates until a different hash or a
    success appears.
    """
    by_family: dict[str, list[dict[str, Any]]] = {}
    current_hash: dict[str, str | None] = {}
    for ev in events:
        family = _family_of(ev)
        if not family:
            continue
        state = str(ev.get("state") or "")
        if state == "succeeded":
            # Reset the streak; we don't even track the success in the
            # streak buffer because the next failure is a fresh start.
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
    """Build a single, time-ordered list of ``delegated_result`` rows.

    Inbox items don't carry a timestamp but they're already ordered by
    seq; coordinator_events ride the SQLite ``seq`` PK. We concatenate
    them and trim to ``lookback_events`` so the hash compute stays O(N)
    even after a long resume.
    """
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
    family = str(event.get("family") or "").strip()
    if family:
        return family
    kind = str(event.get("kind") or "").strip()
    return kind


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _hash_for(family: str, event: dict[str, Any]) -> str | None:
    payload = event.get("payload") or {}
    if not isinstance(payload, dict):
        return None
    # Phase 4 / gap G3: legacy ``params.extra_sglang_args`` envelopes
    # would otherwise walk to ``None`` for ``params.extra_server_args``
    # and the same-fingerprint loop guard would silently miss a
    # legacy-keyed retry burst. Normalise the canonical key in a
    # payload-local copy before projection so both legacy and
    # canonical envelopes produce identical fingerprints.
    payload = _normalise_extra_server_args_key(payload)
    projection = _FAMILY_PROJECTIONS.get(family, _GENERIC_PROJECTION)
    subset: dict[str, Any] = {}
    for path in projection:
        value = _walk_path(payload, path)
        subset[path] = _strip_blacklisted(value)
    canonical = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _normalise_extra_server_args_key(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``payload`` whose ``params`` dict carries
    ``extra_server_args`` set from the canonical-or-legacy compat
    helper. Originals are NOT mutated — the caller's view is preserved.

    No-op when the params dict has no extra-args keys at all OR when
    the canonical key is already present. The single
    ``DeprecationWarning`` (stacklevel=3 from the shim) is the audit
    channel for legacy envelopes — leaving it on so the live operator
    sees one warning per legacy envelope class.
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
    """Walk dotted ``a.b.c`` paths against nested dicts.

    Lists short-circuit to ``None`` because lists rarely carry stable
    keys we'd hash on; we let the projection caller include the list
    container (e.g. ``params.grid``) and rely on JSON canonicalisation
    to detect order-preserving identity.
    """
    cur: Any = payload
    for token in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(token)
    return cur


def _strip_blacklisted(value: Any) -> Any:
    """Recursively drop ``_HASH_BLACKLIST`` keys from dicts."""
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
