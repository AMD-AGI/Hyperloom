# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Agent-stall detection.

: the robustness role is expected to detect "agent
stalls (>3min no message processed)" and emit medium-severity alerts.

We compute the "last activity" timestamp per agent from
:attr:`SourceData.coordinator_events` (newest first) and compare to
``ctx.now_unix``. Inbox tail entries (rendered into the prompt) are
also folded in because they may be more recent than what the local
SQLite probe sees.

Activity criterion is intentionally lenient: any event from the agent
counts as activity, including heartbeats. That matches the upstream
reactor protocol where every tick must emit at least one intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import InboxItem, ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity


# Agents we care about for stall detection. The robustness role itself
# is excluded because it is the one running the rule.
_TRACKED_AGENTS: frozenset[str] = frozenset({
    "orchestration",
    "kernel",
    "critic",
})


@dataclass
class StallConfig:
    """Knobs for :func:`evaluate_stall_signals`."""

    stall_timeout_s: float = 300.0
    severity_high_after_s: float = 900.0


def evaluate_stall_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: StallConfig | None = None,
) -> list[Symptom]:
    cfg = config or StallConfig()
    last_seen = _collect_last_seen(ctx.inbox, data.coordinator_events)
    out: list[Symptom] = []
    for agent in _TRACKED_AGENTS:
        ts = last_seen.get(agent)
        if ts is None:
            # Without ground truth we cannot accuse the agent of being
            # stalled; the very-first tick will always look empty.
            continue
        idle_s = max(0.0, ctx.now_unix - ts)
        if idle_s < cfg.stall_timeout_s:
            continue
        severity = (
            SymptomSeverity.HIGH
            if idle_s >= cfg.severity_high_after_s
            else SymptomSeverity.MEDIUM
        )
        out.append(
            Symptom(
                name="agent_stall",
                severity=severity,
                summary=(
                    f"agent {agent} silent for {int(idle_s)}s "
                    f"(threshold={int(cfg.stall_timeout_s)}s)"
                ),
                evidence={
                    "agent": agent,
                    "idle_seconds": int(idle_s),
                    "last_seen_unix": int(ts),
                    "threshold_s": int(cfg.stall_timeout_s),
                },
                subject={"agent": agent},
                source="local" if data.coordinator_events else "inbox",
                suggestion=(
                    "force_dispatch the head queued task or escalate"
                    " strategy if agent remains silent"
                ),
            )
        )
    return out


def _collect_last_seen(
    inbox: list[InboxItem],
    coordinator_events: list[dict[str, Any]],
) -> dict[str, float]:
    last: dict[str, float] = {}

    for item in inbox:
        if item.from_agent not in _TRACKED_AGENTS:
            continue
        ts = _coerce_unix(item.payload.get("ts") if isinstance(item.payload, dict) else None)
        if ts is None:
            continue
        prev = last.get(item.from_agent)
        if prev is None or ts > prev:
            last[item.from_agent] = ts

    for ev in coordinator_events:
        agent = str(ev.get("agent", "")).strip()
        if agent not in _TRACKED_AGENTS:
            continue
        ts = _coerce_unix(ev.get("ts") or ev.get("timestamp"))
        if ts is None:
            continue
        prev = last.get(agent)
        if prev is None or ts > prev:
            last[agent] = ts

    return last


def _coerce_unix(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # ISO 8601 produced by SQLite WAL stores; fall back to float.
        try:
            from datetime import datetime

            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return None
    return None


__all__ = ["StallConfig", "evaluate_stall_signals"]
