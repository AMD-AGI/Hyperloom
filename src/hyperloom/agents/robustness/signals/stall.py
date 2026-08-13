# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent-stall detection.

Computes each agent's last-activity timestamp from
:attr:`SourceData.coordinator_events` (plus inbox tail) and alerts when
idle past the threshold. Any event counts as activity, including
heartbeats.

Silence is only evidence of a stall when nothing else is moving. A phase whose
work is one multi-hour deterministic task — a baseline pair, a profile and its
roofline, an explore grid — has no LLM turn to emit, so agent silence there is
the design rather than a fault, and alerting on it trains operators to ignore
the signal. :attr:`SourceData.local_task_progress` carries the counter-evidence:
while an agent's *own* dispatched work is still reporting units, its accusation
is downgraded rather than raised. The suppression has no wall-clock ceiling,
because the work units it covers routinely run past any threshold worth setting
— a single warmup can take an hour — and a ceiling would make the alert fire on
exactly the healthy runs it was written to stay quiet about. What bounds it
instead is the freshness of the evidence: the moment the work stops reporting,
the next tick accuses. A long wait still shows up, as a withheld accusation that
rises from LOW to MEDIUM past :attr:`StallConfig.severity_high_after_s`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hyperloom.common.coerce import to_unix

from ..role.prompt_inputs import InboxItem, ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity


log = logging.getLogger(__name__)


# Agents tracked for stall detection; robustness excludes itself.
_TRACKED_AGENTS: frozenset[str] = frozenset(
    {
        "orchestration",
        "kernel_agent",
        "critic",
    }
)


@dataclass
class StallConfig:
    """Knobs for :func:`evaluate_stall_signals`.

    Attributes:
        stall_timeout_s (float): Silence past which an agent is accused, and
            the freshness a heartbeat must beat to count as counter-evidence.
        severity_high_after_s (float): Silence past which an accusation is HIGH
            rather than MEDIUM, and past which a withheld one is MEDIUM rather
            than LOW.
    """

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
    the stall timeout. An agent whose own dispatched work is still reporting is
    not accused: work units outlive the stall window by design, so suppression
    lasts as long as the evidence stays fresh, and only its own severity rises.

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
            # No ground truth yet — can't accuse of a stall.
            continue
        idle_s = max(0.0, ctx.now_unix - ts)
        if idle_s < cfg.stall_timeout_s:
            continue
        out.append(
            _stall_symptom(
                agent,
                last_seen_unix=ts,
                idle_s=idle_s,
                data=data,
                now_unix=ctx.now_unix,
                cfg=cfg,
            )
        )
    return out


def _stall_symptom(
    agent: str,
    *,
    last_seen_unix: float,
    idle_s: float,
    data: SourceData,
    now_unix: float,
    cfg: StallConfig,
) -> Symptom:
    """Build the ``agent_stall`` symptom for one agent that has gone silent.

    Args:
        agent (str): The silent agent.
        last_seen_unix (float): Its most recent activity timestamp.
        idle_s (float): Seconds of silence.
        data (SourceData): Collected source data; supplies the in-flight
            progress counter-evidence and the symptom ``source``.
        now_unix (float): Current time.
        cfg (StallConfig): Thresholds.

    Returns:
        Symptom: MEDIUM (HIGH past ``severity_high_after_s``) when accused, or
        LOW (MEDIUM past the same threshold) when the agent's own work is still
        reporting and the accusation is withheld.
    """
    work_idle_s, work_task = _agent_in_flight_work(
        data.local_task_progress,
        agent=agent,
        now_unix=now_unix,
    ) or (None, "")
    evidence: dict[str, Any] = {
        "agent": agent,
        "idle_seconds": int(idle_s),
        "last_seen_unix": int(last_seen_unix),
        "threshold_s": int(cfg.stall_timeout_s),
    }
    if work_idle_s is not None:
        evidence["in_flight_work_idle_seconds"] = int(work_idle_s)
        evidence["in_flight_work"] = work_task
    withheld = work_idle_s is not None and work_idle_s < cfg.stall_timeout_s
    if not withheld:
        severity = SymptomSeverity.HIGH if idle_s >= cfg.severity_high_after_s else SymptomSeverity.MEDIUM
        return Symptom(
            name="agent_stall",
            severity=severity,
            summary=(f"agent {agent} silent for {int(idle_s)}s (threshold={int(cfg.stall_timeout_s)}s)"),
            evidence=evidence,
            subject={"agent": agent},
            source="local" if data.coordinator_events else "inbox",
            suggestion=("escalate strategy if agent remains silent"),
        )
    evidence["accusation_withheld"] = True
    long_wait = idle_s >= cfg.severity_high_after_s
    summary = (
        f"agent {agent} silent for {int(idle_s)}s but its dispatched work "
        f"({work_task or 'unknown'}) reported {int(work_idle_s)}s ago; "
        f"accusation withheld while that work keeps reporting"
    )
    log.info("stall: %s", summary)
    return Symptom(
        name="agent_stall",
        severity=SymptomSeverity.MEDIUM if long_wait else SymptomSeverity.LOW,
        summary=summary,
        evidence=evidence,
        subject={"agent": agent},
        source="local" if data.coordinator_events else "inbox",
        suggestion=("no action while this agent's own work keeps reporting units"),
    )


def _agent_in_flight_work(
    task_progress: dict[str, Any],
    *,
    agent: str,
    now_unix: float,
) -> tuple[float, str] | None:
    """Seconds since ``agent``'s own dispatched work last reported a unit.

    Progress belonging to another agent is deliberately invisible here: one
    busy task must not vouch for an agent it has nothing to do with.

    Args:
        task_progress (dict[str, Any]): :attr:`SourceData.local_task_progress`.
        agent (str): The agent under accusation.
        now_unix (float): Current time.

    Returns:
        tuple[float, str] | None: ``(idle_seconds, task_kind)``, or ``None``
        when this agent has no attributed heartbeat — no heartbeat is no
        evidence either way, so the caller falls back to agent silence.
    """
    entry = (task_progress.get("by_agent") or {}).get(agent) if task_progress else None
    if not isinstance(entry, dict):
        return None
    ts = to_unix(entry.get("last_progress_unix"))
    if ts is None:
        return None
    return max(0.0, now_unix - ts), str(entry.get("task") or "")


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
        ts = to_unix(item.payload.get("ts") if isinstance(item.payload, dict) else None)
        if ts is None:
            continue
        prev = last.get(item.from_agent)
        if prev is None or ts > prev:
            last[item.from_agent] = ts

    for ev in coordinator_events:
        agent = str(ev.get("agent", "")).strip()
        if agent not in _TRACKED_AGENTS:
            continue
        ts = to_unix(ev.get("ts") or ev.get("timestamp"))
        if ts is None:
            continue
        prev = last.get(agent)
        if prev is None or ts > prev:
            last[agent] = ts

    return last


__all__ = ["StallConfig", "evaluate_stall_signals"]
