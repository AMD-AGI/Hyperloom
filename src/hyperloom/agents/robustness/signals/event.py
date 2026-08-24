# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator-event-driven signals.

Watches repeated ``policy_denied`` observations from the same source
(misconfigured / systemically-rejected agent → MEDIUM alert),
``delegated_result`` failures clustering on one action family (stuck
branch → ``prune_branch``), ``recover_unsuccessful`` recovery outcomes,
and ``idempotency_replay`` (repeated idempotency keys). Reads both
``ctx.inbox`` and ``data.coordinator_events``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .event_view import EventRow, build_event_view, family_of
from .symptom import Symptom, SymptomSeverity


@dataclass
class EventConfig:
    """Knobs for :func:`evaluate_event_signals`."""

    policy_denied_threshold: int = 3
    delegated_failure_threshold: int = 2
    # Sustained-failure count at/above which repeated_failure escalates to HIGH
    # so the action ladder emits prune_branch (first hits stay advisory MEDIUM).
    delegated_failure_prune_threshold: int = 4
    # Lookback over inbox + coordinator_events for the most recent recover result.
    recover_lookback_events: int = 50
    # ``idempotency_replay``: fire when >= threshold distinct idempotency_keys
    # share the same action+payload hash within one tick.
    idempotency_replay_threshold: int = 2


def evaluate_event_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: EventConfig | None = None,
) -> list[Symptom]:
    """Evaluate all Coordinator-event-driven signals for this tick.

    Combines inbox items and ``data.coordinator_events`` via the shared event
    view (sorted by seq, cross-source duplicates removed) and runs the
    policy-denied, delegated-failure, recover-unsuccessful, and
    idempotency-replay rules.

    Args:
        ctx (ReactorContext): Reactor context providing the inbox.
        data (SourceData): Collected source data including coordinator events.
        config (EventConfig | None): Tunables; defaults to :class:`EventConfig`
            when ``None``.

    Returns:
        list[Symptom]: All event-driven symptoms found this tick, possibly
            empty.
    """
    cfg = config or EventConfig()
    view = build_event_view(ctx.inbox, data.coordinator_events)

    out: list[Symptom] = []
    out.extend(_policy_denied_symptoms(view, cfg))
    out.extend(_delegated_failure_symptoms(view, cfg))
    out.extend(_recover_unsuccessful_symptoms(view, cfg))
    out.extend(_idempotency_replay_symptoms(ctx, cfg))
    return out


def _policy_denied_symptoms(
    events: list[EventRow],
    cfg: EventConfig,
) -> list[Symptom]:
    """Fire ``repeated_policy_denied`` for sources over the denial threshold.

    Args:
        events: Shared event view for this tick.
        cfg (EventConfig): Tunables (provides the policy-denied threshold).

    Returns:
        list[Symptom]: One ``repeated_policy_denied`` symptom per offending
            source, possibly empty.
    """
    sources: Counter[str] = Counter()
    rules: Counter[str] = Counter()
    for ev in events:
        if ev.topic != "observation":
            continue
        if ev.payload.get("kind") != "policy_denied":
            continue
        target = ev.agent or ev.payload.get("source") or "unknown"
        rule = ev.payload.get("rule") or "unknown"
        sources[target] += 1
        rules[rule] += 1

    out: list[Symptom] = []
    for source, count in sources.items():
        if count < cfg.policy_denied_threshold:
            continue
        top_rule = rules.most_common(1)[0][0] if rules else "unknown"
        out.append(
            Symptom(
                name="repeated_policy_denied",
                severity=SymptomSeverity.MEDIUM,
                summary=(
                    f"agent {source} hit policy_denied {count} times "
                    f"(>= {cfg.policy_denied_threshold}); top rule={top_rule}"
                ),
                evidence={
                    "from": source,
                    "count": count,
                    "top_rule": top_rule,
                    "rule_distribution": dict(rules),
                },
                subject={"agent": source},
                source="coordinator_events",
                suggestion="escalate_strategy_change to review the action catalogue / role config",
            )
        )
    return out


def _delegated_failure_symptoms(
    events: list[EventRow],
    cfg: EventConfig,
) -> list[Symptom]:
    """Fire ``repeated_failure`` for action families over the failure threshold.

    Args:
        events: Shared event view for this tick.
        cfg (EventConfig): Tunables (provides the delegated-failure threshold).

    Returns:
        list[Symptom]: One ``repeated_failure`` symptom per offending action
            family, possibly empty.
    """
    family_counts: Counter[str] = Counter()
    last_evidence: dict[str, dict[str, Any]] = {}
    for ev in events:
        if ev.topic != "delegated_result":
            continue
        if ev.payload.get("state") != "failed":
            continue
        family = family_of(ev.payload)
        if not family:
            continue
        family_counts[family] += 1
        last_evidence[family] = {
            "task_id": ev.payload.get("task_id"),
            "kind": ev.payload.get("kind"),
            "error": ev.payload.get("error"),
        }

    out: list[Symptom] = []
    for family, count in family_counts.items():
        if count < cfg.delegated_failure_threshold:
            continue
        severity = (
            SymptomSeverity.HIGH if count >= cfg.delegated_failure_prune_threshold else SymptomSeverity.MEDIUM
        )
        out.append(
            Symptom(
                name="repeated_failure",
                severity=severity,
                summary=(f"action family {family!r} failed {count} times (>= {cfg.delegated_failure_threshold})"),
                evidence={
                    "family": family,
                    "count": count,
                    "last_failure": last_evidence.get(family, {}),
                },
                subject={"family": family},
                source="coordinator_events",
                suggestion=f"prune_branch family={family}",
            )
        )
    return out


def _idempotency_replay_symptoms(
    ctx: ReactorContext,
    cfg: EventConfig,
) -> list[Symptom]:
    """Detect distinct-key, same-payload ``delegate`` proposals in inbox.

    Coordinator dedup keys only off ``idempotency_key``, so an LLM minting fresh keys
    per attempt with identical payload slips through. Fire when one action+payload hash
    carries ``>= idempotency_replay_threshold`` distinct keys within a tick.

    Args:
        ctx: Reactor context (supplies the inbox).
        cfg: Event configuration (replay threshold).

    Returns:
        Symptoms for offending action+payload groups, possibly empty.
    """
    if not ctx.inbox:
        return []
    import hashlib  # local import
    import json

    grouped: dict[tuple[str, str], set[str]] = {}
    samples: dict[tuple[str, str], dict[str, Any]] = {}
    for item in ctx.inbox:
        if item.topic != "proposal" and item.topic != "delegate":
            continue
        payload = item.payload or {}
        if not isinstance(payload, dict):
            continue
        action_name = str(payload.get("action_name") or "").strip()
        if not action_name:
            continue
        key = str(payload.get("idempotency_key") or "").strip()
        if not key:
            # No key — Coordinator's payload-hash dedup already covers this.
            continue
        params = payload.get("params") or {}
        try:
            canonical = json.dumps(
                {"action_name": action_name, "params": params},
                sort_keys=True,
                default=str,
            )
        except (TypeError, ValueError):
            continue
        payload_hash = hashlib.sha1(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()
        bucket = (action_name, payload_hash)
        grouped.setdefault(bucket, set()).add(key)
        samples.setdefault(bucket, {"action_name": action_name, "first_key": key})
    out: list[Symptom] = []
    for (action_name, payload_hash), keys in grouped.items():
        if len(keys) < cfg.idempotency_replay_threshold:
            continue
        out.append(
            Symptom(
                name="idempotency_replay",
                severity=SymptomSeverity.MEDIUM,
                summary=(
                    f"delegate(action_name={action_name!r}) submitted with "
                    f"{len(keys)} distinct idempotency_keys but identical "
                    f"payload hash within one tick"
                ),
                evidence={
                    "action_name": action_name,
                    "payload_hash": payload_hash,
                    "distinct_keys": sorted(keys),
                    "threshold": cfg.idempotency_replay_threshold,
                },
                subject={"action_name": action_name},
                source="inbox",
                suggestion=(
                    "alert orchestration: the proposing role is bypassing "
                    "Coordinator dedup by minting fresh keys; review LLM "
                    "prompt or freeze its idempotency_key generation"
                ),
            )
        )
    return out


def _recover_unsuccessful_symptoms(
    events: list[EventRow],
    cfg: EventConfig,
) -> list[Symptom]:
    """Emit ``recover_unsuccessful`` when the latest recover needs review.

    Fires (HIGH) when the latest recover hit ``state == "needs_review"`` —
    cleanup failed to free VRAM, terminal for this budget. Inspects only the
    latest recover ``delegated_result``; earlier successes are not
    second-guessed.

    Args:
        events: Shared event view in chronological order.
        cfg: Event configuration (recover lookback window).

    Returns:
        A list with one :class:`Symptom` when it fires, else empty.
    """
    head = events[-cfg.recover_lookback_events :] if events else []
    latest: dict[str, Any] | None = None
    for ev in head:
        if ev.topic != "delegated_result":
            continue
        if not _is_recover_payload(ev.payload):
            continue
        latest = ev.payload
    if latest is None:
        return []
    state = str(latest.get("state") or "").strip()
    if state != "needs_review":
        return []
    error_class = str(latest.get("error_class") or "")
    # Any needs_review from recover is terminal.
    return [
        Symptom(
            name="recover_unsuccessful",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"recover returned state=needs_review "
                f"(error_class={error_class or 'unknown'!r}) — GPU memory "
                f"remains unhealthy after the in-loop cleanup; the "
                f"session cannot make further progress on this budget"
            ),
            evidence={
                "task_id": latest.get("task_id"),
                "kind": latest.get("kind") or "recover",
                "state": state,
                "error_class": error_class,
                "force_gpu_cleanup": latest.get("force_gpu_cleanup"),
                "mid_free_mb_per_gpu": latest.get("mid_free_mb_per_gpu"),
            },
            subject={},  # session-wide
            source="coordinator_events",
            suggestion=("propose report to finalize at the last validated gain"),
        )
    ]


def _is_recover_payload(payload: dict[str, Any]) -> bool:
    """Best-effort check that a ``delegated_result`` came from ``recover``.

    Recognized via ``kind`` / ``action_name`` / ``family`` or recover-only
    executor fields.

    Args:
        payload: A ``delegated_result`` payload.

    Returns:
        ``True`` if the payload appears to be from a recover action.
    """
    if str(payload.get("kind") or "").strip() == "recover":
        return True
    if str(payload.get("action_name") or "").strip() == "recover":
        return True
    if str(payload.get("family") or "").strip() == "recover":
        return True
    # Executor signature — recover-only fields.
    if "force_gpu_cleanup" in payload and "mid_free_mb_per_gpu" in payload:
        return True
    return False


__all__ = ["EventConfig", "evaluate_event_signals"]
