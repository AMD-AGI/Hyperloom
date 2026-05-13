"""Translate Symptoms into Coordinator Intents.

The ladder has three tiers, matching DESIGN v0.6 §13.2 / §19.3:

1. **observe** — low severity: emit ``send_message(topic="observation")``
   so the orchestration agent has visibility but no pause is triggered.
2. **diagnose** — medium severity: emit ``alert(severity="medium")``
   carrying the symptom evidence; the orchestration agent runs a
   focused RCA next tick.
3. **recommend** — high severity: emit ``alert(severity="high")`` plus
   one of the scheduling-police intents (``escalate_strategy_change``,
   ``prune_branch``, ``kill_task``) when the symptom comes with a
   concrete suggestion.

To avoid flooding the inbox the ladder maintains a per-key cooldown:
the same ``Symptom.dedup_key()`` will not produce another intent until
``cooldown_ticks`` ticks have elapsed.

Findings — one persistent record per intent batch — are emitted
alongside the intents and consumed by :class:`FindingSink` (T9).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..role.envelope import (
    Intent,
    build_alert,
    build_escalate,
    build_heartbeat,
    build_prune_branch,
    build_send_message,
)
from ..signals import Symptom, SymptomSeverity


log = logging.getLogger(__name__)


@dataclass
class Finding:
    """Persistent record describing one ladder firing.

    Stored on disk by the FindingSink for later inspection / reporting
    to robustness-server.
    """

    tick_index: int
    timestamp_unix: float
    symptom_name: str
    severity: str
    summary: str
    intents: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    rca_text: str = ""


@dataclass
class ActionLadderConfig:
    """Tunables for the ladder."""

    cooldown_ticks: int = 5


@dataclass
class _LadderResult:
    intents: list[Intent]
    findings: list[Finding]


class ActionLadder:
    """Stateful ladder that maps symptoms onto intents and findings.

    The ladder is deliberately conservative in M1: it emits intents
    only for symptoms whose dedup key is outside the cooldown window
    and falls back to a heartbeat when the symptom set is empty.
    """

    def __init__(self, *, config: ActionLadderConfig | None = None) -> None:
        self._config = config or ActionLadderConfig()
        self._last_emitted_tick: dict[tuple[str, ...], int] = {}

    async def decide(
        self,
        symptoms: list[Symptom],
        *,
        tick_index: int,
        now_unix: float,
        rca_provider: Any | None = None,
    ) -> _LadderResult:
        """Produce intents (+ findings) for this tick.

        ``rca_provider`` may be ``None`` (no RCA), or any object exposing
        ``async def summarize(symptom) -> str``. When the provider has a
        ``set_tick(int)`` hook (e.g. :class:`LlmRcaEngine`) we call it
        once per tick so per-tick budgets reset deterministically.
        """
        intents: list[Intent] = []
        findings: list[Finding] = []
        any_emit = False
        if rca_provider is not None:
            set_tick = getattr(rca_provider, "set_tick", None)
            if callable(set_tick):
                try:
                    set_tick(tick_index)
                except Exception:
                    log.exception("rca_provider.set_tick failed; ignoring")
        for sym in symptoms:
            key = sym.dedup_key()
            if not self._cooldown_elapsed(key, tick_index):
                continue
            sym_intents = self._intents_for(sym)
            if not sym_intents:
                continue
            any_emit = True
            self._last_emitted_tick[key] = tick_index
            rca_text = await _safe_rca(rca_provider, sym)
            findings.append(
                _build_finding(
                    sym,
                    sym_intents,
                    tick_index=tick_index,
                    now_unix=now_unix,
                    rca_text=rca_text,
                )
            )
            intents.extend(sym_intents)

        if not any_emit:
            intents.append(build_heartbeat())
        return _LadderResult(intents=intents, findings=findings)

    def _cooldown_elapsed(self, key: tuple[str, ...], tick_index: int) -> bool:
        cooldown = self._config.cooldown_ticks
        last = self._last_emitted_tick.get(key)
        if last is None:
            return True
        return (tick_index - last) >= cooldown

    def _intents_for(self, sym: Symptom) -> list[Intent]:
        if sym.severity is SymptomSeverity.LOW:
            return self._observe(sym)
        if sym.severity is SymptomSeverity.MEDIUM:
            return self._diagnose(sym)
        return self._recommend(sym)

    def _observe(self, sym: Symptom) -> list[Intent]:
        return [
            build_send_message(
                "observation",
                body_md=f"{sym.name}: {sym.summary}",
                extras={"detail": _detail(sym)},
            )
        ]

    def _diagnose(self, sym: Symptom) -> list[Intent]:
        return [build_alert("medium", sym.summary, detail=_detail(sym))]

    def _recommend(self, sym: Symptom) -> list[Intent]:
        intents: list[Intent] = [build_alert("high", sym.summary, detail=_detail(sym))]
        if sym.name in {"crash_count_emergency", "crash_count_high"}:
            intents.append(
                build_escalate(
                    reason=sym.name,
                    next_action_hint=sym.suggestion or "revert to known-good baseline",
                    severity="high",
                )
            )
        elif sym.name == "agent_stall" and sym.severity is SymptomSeverity.HIGH:
            intents.append(
                build_escalate(
                    reason="agent_stall_high",
                    next_action_hint=sym.suggestion or "force_dispatch head queued task",
                    severity="high",
                )
            )
        elif sym.name == "cluster_fault":
            # Wide-blast-radius cluster faults need an escalate so the
            # orchestration agent reroutes work away from the affected
            # node before the fault sweeps more sessions.
            intents.append(
                build_escalate(
                    reason="cluster_fault_high",
                    next_action_hint=(
                        sym.suggestion
                        or "drain affected node; reschedule away from fault"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "repeated_failure":
            family = sym.evidence.get("family") if isinstance(sym.evidence, dict) else None
            if isinstance(family, str) and family.strip():
                intents.append(build_prune_branch(family=family, reason="repeated_failure"))
        return intents


def _detail(sym: Symptom) -> dict[str, Any]:
    body = {
        "symptom": sym.name,
        "severity": sym.severity.value,
        "subject": sym.subject,
        "source": sym.source,
        "evidence": sym.evidence,
    }
    if sym.suggestion:
        body["suggestion"] = sym.suggestion
    return body


def _build_finding(
    sym: Symptom,
    intents: Iterable[Intent],
    *,
    tick_index: int,
    now_unix: float,
    rca_text: str,
) -> Finding:
    return Finding(
        tick_index=tick_index,
        timestamp_unix=now_unix,
        symptom_name=sym.name,
        severity=sym.severity.value,
        summary=sym.summary,
        intents=[i.to_envelope_item() for i in intents],
        evidence=dict(sym.evidence) if isinstance(sym.evidence, dict) else {},
        rca_text=rca_text,
    )


async def _safe_rca(provider: Any | None, sym: Symptom) -> str:
    if provider is None:
        return ""
    try:
        result = provider.summarize(sym)
        if hasattr(result, "__await__"):
            result = await result
        return result or ""
    except Exception:
        log.exception("rca provider raised; continuing without RCA text")
        return ""


__all__ = ["ActionLadder", "ActionLadderConfig", "Finding"]
