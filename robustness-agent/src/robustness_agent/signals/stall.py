# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Agent-stall detection.

Computes each agent's last-activity timestamp from
:attr:`SourceData.coordinator_events` (plus inbox tail, which may be more
recent than the local SQLite probe) and alerts when idle past the
threshold. Any event counts as activity, including heartbeats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import InboxItem, ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity


# Agents tracked for stall detection; robustness excludes itself.
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
    """Emit ``agent_stall`` symptoms for tracked agents that have gone silent.

    Computes per-agent idle time from the most recent activity timestamp and
    fires MEDIUM (or HIGH past ``severity_high_after_s``) once idle time exceeds
    the stall timeout.

    Args:
        ctx (ReactorContext): Reactor context (provides inbox and current time).
        data (SourceData): Collected source data including coordinator events.
        config (StallConfig | None): Tunables; defaults to :class:`StallConfig`
            when ``None``.

    Returns:
        list[Symptom]: One ``agent_stall`` symptom per stalled agent, possibly
            empty.
    """
    cfg = config or StallConfig()
    last_seen = _collect_last_seen(ctx.inbox, data.coordinator_events)
    out: list[Symptom] = []
    for agent in _TRACKED_AGENTS:
        ts = last_seen.get(agent)
        if ts is None:
            # No ground truth (e.g. first tick) — can't accuse of a stall.
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
    """Compute the latest activity timestamp per tracked agent.

    Folds together inbox items and coordinator events, keeping the most recent
    timestamp seen for each tracked agent.

    Args:
        inbox (list[InboxItem]): Inbox items from the reactor context.
        coordinator_events (list[dict[str, Any]]): Raw coordinator events.

    Returns:
        dict[str, float]: Mapping of agent name to its latest activity unix
            timestamp.
    """
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
    """Coerce a timestamp value to unix seconds.

    Accepts numeric epoch seconds or an ISO-8601 string (``Z`` suffix
    tolerated), falling back to parsing the string as a float.

    Args:
        value (Any): The raw timestamp value.

    Returns:
        float | None: Unix seconds, or ``None`` when the value cannot be
            interpreted as a timestamp.
    """
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
