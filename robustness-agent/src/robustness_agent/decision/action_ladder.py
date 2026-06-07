# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Translate Symptoms into Coordinator Intents.

The ladder has three tiers, matching:

1. **observe** — low severity: emit ``send_message(topic="observation")``
   so the orchestration agent has visibility but no pause is triggered.
2. **diagnose** — medium severity: emit ``alert(severity="medium")``
   carrying the symptom evidence; the orchestration agent runs a
   focused RCA next tick.
3. **recommend** — high severity: emit ``alert(severity="high")`` plus,
   for resource-safety symptoms only, the resource recovery intent
   (``kill_task`` for ``stale_lease``, ``delegate(recover)`` for
   ``gpu_memory_leaked``, ``delegate(report)`` for the wall-clock
   deadline backstops).

Strategic suggestions (prune branches, skip a phase, wind down on
plateau) are surfaced via the ``suggestion`` field in the alert
``detail`` payload. Orchestration consumes them and decides whether
to act; the ladder no longer auto-emits ``escalate_strategy_change``
or ``prune_branch`` from a HIGH symptom.

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

from ..state_store import DetectorStateView
from ..role.envelope import (
    Intent,
    build_alert,
    build_delegate,
    build_heartbeat,
    build_kill_task,
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

    def __init__(
        self,
        *,
        config: ActionLadderConfig | None = None,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        self._config = config or ActionLadderConfig()
        self._state_view = state_view
        # Cooldown bookkeeping — persisted across subprocess restarts.
        # Without persistence the ladder re-emits the same intent every
        # tick because the in-memory dict resets, defeating the
        # cooldown contract advertised in :class:`ActionLadderConfig`.
        loaded = state_view.load() if state_view is not None else {}
        self._last_emitted_tick: dict[tuple[str, ...], int] = (
            _decode_last_emitted(loaded.get("last_emitted"))
        )
        # Updated at the top of each :meth:`decide` call so per-symptom
        # branches (notably ``gpu_memory_leaked``) can stamp a stable
        # tick-indexed ``idempotency_key`` onto the intents they emit
        # without threading the tick through every helper.
        self._last_tick_index: int = 0

    def _persist_cooldown(self) -> None:
        if self._state_view is None:
            return
        self._state_view.save({
            "last_emitted": _encode_last_emitted(self._last_emitted_tick),
        })

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
        self._last_tick_index = tick_index
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
            self._persist_cooldown()
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
        # Resource-safety branch: stale lease releases a lane that is
        # held by a dead PID. Robustness owns ``kill_task(scope='task')``
        # exclusively for this case; everything else is policy and stays
        # advisory.
        if sym.name == "stale_lease":
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            task_id = str(evidence.get("task_id") or "").strip()
            if task_id and task_id != "unknown":
                intents.append(
                    build_kill_task(task_id=task_id, reason="stale_lease")
                )
            return intents
        # Resource-safety: GPU memory leak detected by the local probe.
        # ``delegate(recover, force_gpu_cleanup=True)`` is the in-loop
        # recovery action; Robustness has explicit allowlist authority
        # for it.
        if sym.name == "gpu_memory_leaked":
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_delegate(
                    action_name="recover",
                    params={
                        "reason": "gpu_memory_leaked",
                        "force_gpu_cleanup": True,
                        "evidence": evidence,
                    },
                    idempotency_key=(
                        f"recover-gpu-leak-tick-{self._last_tick_index}"
                    ),
                )
            )
            return intents
        # Wall-clock invariant wind-down: the absolute-time deadline
        # supervisor will SIGTERM in-flight work past the wall, so
        # ``delegate(report)`` is the only way to land a deterministic
        # report inside the remaining budget. ``recover_unsuccessful``
        # is the matching finalization path when an in-loop recover has
        # already returned ``needs_review``.
        if sym.name in {
            "deadline_warning",
            "deadline_imminent",
            "deadline_hard_cutoff",
            "recover_unsuccessful",
        }:
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            slug = sym.name.replace("_", "-")
            intents.append(
                build_delegate(
                    action_name="report",
                    params={"reason": sym.name, "evidence": evidence},
                    idempotency_key=(
                        f"report-{slug}-tick-{self._last_tick_index}"
                    ),
                )
            )
            return intents
        # Every other HIGH symptom is strategic. The alert above already
        # carries the symptom evidence and the suggested hint via
        # :func:`_detail`; Orchestration consumes those and decides
        # whether to emit ``escalate_strategy_change`` / ``prune_branch``
        # itself.
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


# ---------------------------------------------------------------------------
# State-store (de)serialisation helpers
# ---------------------------------------------------------------------------

# Separator used to pack ``tuple[str, ...]`` dedup keys into a single
# JSON-safe string. A vertical-bar is unlikely to appear in symptom
# names / subject IDs and keeps the encoded key human-readable in the
# detector_state.json file (useful when debugging cooldown behaviour).
_LADDER_KEY_SEP: str = "\x1f"  # ASCII unit separator — safe inside JSON strings


def _encode_last_emitted(
    last_emitted: dict[tuple[str, ...], int],
) -> dict[str, int]:
    """Serialise a tuple-keyed cooldown dict to a JSON-safe dict.

    JSON object keys must be strings; we join the ``Symptom.dedup_key``
    tuple components with ``_LADDER_KEY_SEP`` so reads can recover the
    original tuple verbatim.
    """
    out: dict[str, int] = {}
    for key, tick in last_emitted.items():
        try:
            encoded = _LADDER_KEY_SEP.join(str(part) for part in key)
        except Exception:  # noqa: BLE001 — best-effort, skip bad keys
            continue
        try:
            out[encoded] = int(tick)
        except (TypeError, ValueError):
            continue
    return out


def _decode_last_emitted(
    payload: Any,
) -> dict[tuple[str, ...], int]:
    """Inverse of :func:`_encode_last_emitted`; tolerant of bad input."""
    if not isinstance(payload, dict):
        return {}
    out: dict[tuple[str, ...], int] = {}
    for raw_key, raw_tick in payload.items():
        if not isinstance(raw_key, str):
            continue
        try:
            tick = int(raw_tick)
        except (TypeError, ValueError):
            continue
        parts = tuple(raw_key.split(_LADDER_KEY_SEP))
        out[parts] = tick
    return out


__all__ = ["ActionLadder", "ActionLadderConfig", "Finding"]
