# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent-stall detection from coordinator-stamped ``agent_last_active_unix``.

Silence is only evidence of a stall when nothing else is moving: a phase whose
work is one multi-hour deterministic task has no LLM turn to emit. While an
agent's own dispatched work still reports units, the accusation is withheld as
``agent_quiet_work_progressing``. That suppression has no wall-clock ceiling on
purpose -- a single warmup runs 3941s, so any ceiling would fire on exactly the
healthy runs this exists to stay quiet about; what bounds it is the freshness of
the evidence. Elapsed silence therefore sets an accusation's severity, not
whether one is made.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity


log = logging.getLogger(__name__)


# Reactor roles tracked for stall detection; robustness excludes itself.
_TRACKED_AGENTS: frozenset[str] = frozenset(
    {
        "orchestration",
        "critic",
    }
)


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
    """Report each tracked agent that has gone silent past the stall timeout."""
    cfg = config or StallConfig()
    last_seen = dict(ctx.shared_state.agent_last_active_unix or {})
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
    """Build the symptom for one agent that has gone silent."""
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
    """Seconds since one of ``agent``'s own dispatched units reported."""
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
    """Name the agent's quietest unit when a busier sibling is speaking for it."""
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


__all__ = ["StallConfig", "evaluate_stall_signals"]
