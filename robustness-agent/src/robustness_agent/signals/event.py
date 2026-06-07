# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator-event-driven signals.

Two patterns the robustness role watches per:

* Repeated ``policy_denied`` observations from the same source within
  a short window indicate either a misconfigured agent or a
  systemically-rejected action — emit a medium-severity alert
  suggesting the operator review the ActionRegistry.
* ``delegated_result`` events with ``state == "failed"`` clustering on
  the same action family hint at a stuck branch, suggesting a
  ``prune_branch`` action.

Inputs come from both ``ctx.inbox`` (rendered into the prompt) and
``data.coordinator_events`` (read directly from the local conductor.db
when robustness-server is unavailable).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import InboxItem, ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity


@dataclass
class EventConfig:
    """Knobs for :func:`evaluate_event_signals`."""

    policy_denied_threshold: int = 3
    delegated_failure_threshold: int = 2
    # Window we look back over inbox + coordinator_events for the most
    # recent recover result. The Coordinator only emits one
    # ``delegated_result`` per recover task, so a small window is enough.
    recover_lookback_events: int = 50
    # B4: ``idempotency_replay``. The Coordinator de-dupes ``delegate``
    # intents by ``idempotency_key``, so a malformed LLM that bumps the
    # key while keeping the payload identical can spam fresh tasks
    # inside one tick. We aggregate inbox-side ``delegate`` payloads per
    # tick and fire when ``>= threshold`` distinct keys share the same
    # action+payload hash.
    idempotency_replay_threshold: int = 2


def evaluate_event_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: EventConfig | None = None,
) -> list[Symptom]:
    cfg = config or EventConfig()
    inbox_view = _normalise_inbox(ctx.inbox)
    coord_view = _normalise_events(data.coordinator_events)
    combined = inbox_view + coord_view

    out: list[Symptom] = []
    out.extend(_policy_denied_symptoms(combined, cfg))
    out.extend(_delegated_failure_symptoms(combined, cfg))
    out.extend(_recover_unsuccessful_symptoms(combined, cfg))
    out.extend(_idempotency_replay_symptoms(ctx, cfg))
    return out


def _policy_denied_symptoms(
    events: list[dict[str, Any]],
    cfg: EventConfig,
) -> list[Symptom]:
    sources: Counter[str] = Counter()
    rules: Counter[str] = Counter()
    for ev in events:
        if ev.get("topic") != "observation":
            continue
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        if payload.get("kind") != "policy_denied":
            continue
        target = ev.get("agent") or payload.get("source") or "unknown"
        rule = payload.get("rule") or "unknown"
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
                suggestion="escalate_strategy_change to review ActionRegistry / role config",
            )
        )
    return out


def _delegated_failure_symptoms(
    events: list[dict[str, Any]],
    cfg: EventConfig,
) -> list[Symptom]:
    family_counts: Counter[str] = Counter()
    last_evidence: dict[str, dict[str, Any]] = {}
    for ev in events:
        if ev.get("topic") != "delegated_result":
            continue
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        if payload.get("state") != "failed":
            continue
        family = _family_of(payload)
        if not family:
            continue
        family_counts[family] += 1
        last_evidence[family] = {
            "task_id": payload.get("task_id"),
            "kind": payload.get("kind"),
            "error": payload.get("error"),
        }

    out: list[Symptom] = []
    for family, count in family_counts.items():
        if count < cfg.delegated_failure_threshold:
            continue
        out.append(
            Symptom(
                name="repeated_failure",
                severity=SymptomSeverity.MEDIUM,
                summary=(
                    f"action family {family!r} failed {count} times "
                    f"(>= {cfg.delegated_failure_threshold})"
                ),
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

    The Coordinator's de-dup is keyed off ``idempotency_key`` only. An
    LLM that mints a fresh key per attempt while keeping the payload
    identical (the smoking-gun in the 2026-05-18 GPU-leak post-mortem:
    "4 validate_stack attempts in 3 minutes") slips through unchallenged.
    We hash the projected payload + action_name and fire when the same
    hash carries ``>= idempotency_replay_threshold`` distinct keys
    within a single tick's inbox view.
    """
    if not ctx.inbox:
        return []
    import hashlib  # local — avoid module-level cost for runs without delegates
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
            # No key at all — the Coordinator's own dedup catches this
            # via ``payload`` hashing; the duplicate-payload signal
            # would emit a noisy false positive here. Skip.
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
        payload_hash = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
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
    events: list[dict[str, Any]],
    cfg: EventConfig,
) -> list[Symptom]:
    """Emit ``recover_unsuccessful`` when the latest recover hit needs_review.

    The :mod:`recover` action returns ``state == "needs_review"`` with a
    ``gpu_unhealthy_*`` ``error_class`` when its best-effort cleanup
    (SIGTERM/SIGKILL + optional ``rocm-smi --gpureset``) failed to free
    VRAM back above the healthy threshold. That outcome is terminal for
    the current session — no further recover round will fix it without
    an out-of-band pod reset — so we emit a high-severity symptom that
    the ActionLadder turns into ``delegate(report)`` to finalize at the
    last validated gain instead of burning the remaining budget on
    doomed ``validate_stack`` retries.

    The function inspects only the latest ``delegated_result`` for the
    ``recover`` action; older recover attempts that succeeded already
    cleared the leak so we don't second-guess them.
    """
    head = events[-cfg.recover_lookback_events:] if events else []
    latest: dict[str, Any] | None = None
    for ev in head:
        if ev.get("topic") != "delegated_result":
            continue
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        if not _is_recover_payload(payload):
            continue
        latest = payload  # last-write-wins → newest because events list is in order
    if latest is None:
        return []
    state = str(latest.get("state") or "").strip()
    if state != "needs_review":
        return []
    error_class = str(latest.get("error_class") or "")
    # We treat any needs_review from recover as terminal; the
    # ``error_class`` discriminator just feeds evidence to the LLM /
    # operator. ``gpu_unhealthy_after_*`` is the dominant code surfaced
    # by :class:`RecoverExecutor`.
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
                "gpureset_attempted": latest.get("gpureset_attempted"),
                # Caller may have flattened the executor result into the
                # delegated_result payload — pass through the per-GPU
                # diagnostic so the report stage has full context.
                "post_free_mb_per_gpu": latest.get("post_free_mb_per_gpu"),
            },
            subject={},  # session-wide; cooldown collapses across ticks
            source="coordinator_events",
            suggestion=(
                "delegate(report) to finalize at the last validated gain"
            ),
        )
    ]


def _is_recover_payload(payload: dict[str, Any]) -> bool:
    """Best-effort check that a ``delegated_result`` came from ``recover``.

    Coordinator may tag the action via ``kind`` (canonical), or the
    payload may carry executor-specific fields like ``force_gpu_cleanup``
    that betray the action even when ``kind`` is missing.
    """
    if str(payload.get("kind") or "").strip() == "recover":
        return True
    if str(payload.get("action_name") or "").strip() == "recover":
        return True
    if str(payload.get("family") or "").strip() == "recover":
        return True
    # Executor signature — recover-only fields the Coordinator forwards
    # verbatim when the delegated_result was emitted from the recover
    # workspace's result.json.
    if "force_gpu_cleanup" in payload and "gpureset_attempted" in payload:
        return True
    return False


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_inbox(inbox: list[InboxItem]) -> list[dict[str, Any]]:
    return [
        {
            "agent": item.from_agent,
            "topic": item.topic,
            "payload": item.payload,
            "ts": None,
        }
        for item in inbox
    ]


def _normalise_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in events:
        out.append(
            {
                "agent": ev.get("agent", ""),
                "topic": ev.get("topic", ""),
                "payload": ev.get("payload"),
                "ts": ev.get("ts") or ev.get("timestamp"),
            }
        )
    return out


def _family_of(payload: dict[str, Any]) -> str:
    """Best-effort family inference from a delegated_result payload.

    Coordinator does not always tag a family; fall back to ``kind``
    when present so the rule still groups by action name.
    """
    family = payload.get("family")
    if isinstance(family, str) and family.strip():
        return family.strip()
    kind = payload.get("kind")
    if isinstance(kind, str) and kind.strip():
        return kind.strip()
    return ""


__all__ = ["EventConfig", "evaluate_event_signals"]
