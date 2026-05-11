"""Coordinator-event-driven signals.

Two patterns the robustness role watches per DESIGN v0.6 §7.4 / §19.3:

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

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import InboxItem, ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity

log = logging.getLogger(__name__)


@dataclass
class EventConfig:
    """Knobs for :func:`evaluate_event_signals`."""

    policy_denied_threshold: int = 3
    delegated_failure_threshold: int = 2


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
