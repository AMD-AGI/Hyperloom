# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent-stall detection.

Computes each agent's last-activity timestamp from
:attr:`SourceData.coordinator_events` (plus inbox tail) and alerts when
idle past the threshold. Any event counts as activity, including
heartbeats.

Silence is only evidence of a stall when nothing else is moving. A phase whose
work is one multi-hour deterministic task — a baseline pair, a profile and its
roofline, a config grid — has no LLM turn to emit, so agent silence there is
the design rather than a fault, and alerting on it trains operators to ignore
the signal. :attr:`SourceData.local_task_progress` carries the counter-evidence:
while an agent's *own* dispatched work is still reporting units, the accusation
is withheld and the tick reports ``agent_quiet_work_progressing`` instead. The
suppression has no wall-clock ceiling, because the work units it covers
routinely run past any threshold worth setting — a single warmup runs 3941s —
and a ceiling would make the alert fire on exactly the healthy runs it was
written to stay quiet about. What bounds it instead is the freshness of the
evidence: the moment the work stops reporting, the next tick accuses, at full
severity.

Severity therefore follows the evidence and not the length of the wait. A phase
that keeps reporting throughout stays an observation however long it runs;
elapsed silence only decides how loud the *accusation* is, once there is no
fresh evidence left to withhold it. The two cases carry different symptom names
so RCA can tell a healthy long phase from an agent that went quiet past
:attr:`StallConfig.severity_high_after_s`.

One reporting unit is enough to withhold even when the agent owns several, since
a quiet unit is not an agent fault and has the lease watchdog behind it. It is
still named in the evidence, so the healthy sibling stops being the only thing
an operator can see.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hyperloom.common.coerce import to_unix

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .event_view import EventRow, build_event_view
from .symptom import Symptom, SymptomSeverity


log = logging.getLogger(__name__)


# Reactor roles tracked for stall detection; robustness excludes itself.
# ``kernel_agent`` is not one — it has no turn and no heartbeat, only a
# completion receipt the Coordinator signs for it.
_TRACKED_AGENTS: frozenset[str] = frozenset(
    {
        "orchestration",
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
            rather than MEDIUM. It does not grade a withheld accusation: an
            agent whose work is still reporting is not more suspect for having
            been dispatched a longer unit.
    """

    stall_timeout_s: float = 300.0
    severity_high_after_s: float = 900.0


def evaluate_stall_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: StallConfig | None = None,
) -> list[Symptom]:
    """Report each tracked agent that has gone silent past the stall timeout.

    Computes per-agent idle time from the most recent activity timestamp and
    fires ``agent_stall`` MEDIUM (or HIGH past ``severity_high_after_s``) once
    idle time exceeds the stall timeout. An agent whose own dispatched work is
    still reporting is not accused at all: work units outlive the stall window
    by design, so it reports ``agent_quiet_work_progressing`` (LOW) for as long
    as the evidence stays fresh.

    Args:
        ctx (ReactorContext): Reactor context (provides inbox and current time).
        data (SourceData): Collected source data including coordinator events.
        config (StallConfig | None): Tunables; defaults to :class:`StallConfig`
            when ``None``.

    Returns:
        list[Symptom]: One ``agent_stall`` or ``agent_quiet_work_progressing``
            symptom per silent agent, possibly empty.
    """
    cfg = config or StallConfig()
    view = build_event_view(ctx.inbox, data.coordinator_events)
    last_seen = _collect_last_seen(view)
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
    """Build the symptom for one agent that has gone silent.

    Args:
        agent (str): The silent agent.
        last_seen_unix (float): Its most recent activity timestamp.
        idle_s (float): Seconds of silence.
        data (SourceData): Collected source data; supplies the in-flight
            progress counter-evidence and the symptom ``source``.
        now_unix (float): Current time.
        cfg (StallConfig): Thresholds.

    Returns:
        Symptom: ``agent_stall`` at MEDIUM (HIGH past
        ``severity_high_after_s``) when accused, or
        ``agent_quiet_work_progressing`` at LOW while the agent's own work is
        still reporting and the accusation is withheld.
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
        evidence.update(
            _quiet_sibling_evidence(
                data.local_task_progress,
                agent=agent,
                now_unix=now_unix,
                fresh_idle_s=work_idle_s,
                cfg=cfg,
            )
        )
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
    evidence["withheld_while_work_reports_within_s"] = int(cfg.stall_timeout_s)
    summary = (
        f"agent {agent} silent for {int(idle_s)}s but its dispatched work "
        f"({work_task or 'unknown'}) reported {int(work_idle_s)}s ago; "
        f"accusation withheld while that work keeps reporting"
    )
    log.info("stall: %s", summary)
    return Symptom(
        name="agent_quiet_work_progressing",
        severity=SymptomSeverity.LOW,
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
    ts_key: str = "last_progress_unix",
    task_key: str = "task",
) -> tuple[float, str] | None:
    """Seconds since one of ``agent``'s own dispatched units reported.

    Progress belonging to another agent is deliberately invisible here: one
    busy task must not vouch for an agent it has nothing to do with.

    Args:
        task_progress (dict[str, Any]): :attr:`SourceData.local_task_progress`.
        agent (str): The agent under accusation.
        now_unix (float): Current time.
        ts_key (str): Snapshot key holding the timestamp to read — the agent's
            freshest note by default, ``"oldest_progress_unix"`` for its
            quietest.
        task_key (str): Snapshot key naming the unit ``ts_key`` belongs to.

    Returns:
        tuple[float, str] | None: ``(idle_seconds, task_kind)``, or ``None``
        when this agent has no attributed heartbeat — no heartbeat is no
        evidence either way, so the caller falls back to agent silence.
    """
    entry = (task_progress.get("by_agent") or {}).get(agent) if task_progress else None
    if not isinstance(entry, dict):
        return None
    ts = to_unix(entry.get(ts_key))
    if ts is None:
        return None
    return max(0.0, now_unix - ts), str(entry.get(task_key) or "")


def _quiet_sibling_evidence(
    task_progress: dict[str, Any],
    *,
    agent: str,
    now_unix: float,
    fresh_idle_s: float,
    cfg: StallConfig,
) -> dict[str, Any]:
    """Name the agent's quietest unit when a busier sibling is speaking for it.

    One unit reporting still answers the question this signal asks — is this
    agent's work progressing — so the accusation stays withheld. Declining to
    withhold instead would fire on healthy runs: a Ray-backed baseline round has
    no liveness callback to give it, and reports on entry and then not again
    until it returns. What must not happen is the quiet unit leaving no trace,
    which is what a snapshot carrying only the freshest heartbeat did.

    Args:
        task_progress (dict[str, Any]): :attr:`SourceData.local_task_progress`.
        agent (str): The agent under accusation.
        now_unix (float): Current time.
        fresh_idle_s (float): Idle seconds of the agent's freshest unit.
        cfg (StallConfig): Thresholds.

    Returns:
        dict[str, Any]: ``{quiet_in_flight_work,
        quiet_in_flight_work_idle_seconds}`` when a strictly quieter unit of the
        same agent is past the stall window, otherwise empty.
    """
    quietest = _agent_in_flight_work(
        task_progress,
        agent=agent,
        now_unix=now_unix,
        ts_key="oldest_progress_unix",
        task_key="oldest_task",
    )
    if quietest is None:
        return {}
    idle_s, task = quietest
    if idle_s <= fresh_idle_s or idle_s < cfg.stall_timeout_s:
        return {}
    return {
        "quiet_in_flight_work": task,
        "quiet_in_flight_work_idle_seconds": int(idle_s),
    }


def _collect_last_seen(
    view: list[EventRow],
) -> dict[str, float]:
    """Compute the latest activity timestamp per tracked agent.

    Args:
        view: Shared event view for this tick.

    Returns:
        dict[str, float]: Mapping of agent name to its latest activity unix
            timestamp.
    """
    last: dict[str, float] = {}
    for ev in view:
        if ev.agent not in _TRACKED_AGENTS:
            continue
        ts = to_unix(ev.ts)
        if ts is None:
            continue
        prev = last.get(ev.agent)
        if prev is None or ts > prev:
            last[ev.agent] = ts
    return last


__all__ = ["StallConfig", "evaluate_stall_signals"]
