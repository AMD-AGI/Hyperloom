# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Reactor: the heart of the robustness role.

A single :meth:`Reactor.tick` runs the M1 pipeline: :class:`DegradeRouter` ->
:class:`Classifier` -> :class:`ActionLadder` -> :class:`PolicyAware` filter ->
:class:`FindingSink` persist -> return validated intents. The Reactor holds tick
state but no business logic (that lives in classifier + ladder) so transports
can be swapped.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from ..decision.action_ladder import ActionLadder, Finding
from ..decision.policy_aware import PolicyAware, PolicyViolation
from ..decision.rca_engine import NoopRcaEngine, RcaEngine
from ..finalize.postmortem import PostmortemFinalizer
from ..findings.sink import FindingSink
from ..signals import Classifier, Symptom
from ..sources.base import DegradeRouter
from ..state_store import DetectorStateStore
from .envelope import Intent
from .prompt_inputs import ReactorContext


import asyncio  # noqa: E402  — local import keeps the diff readable


log = logging.getLogger(__name__)


@dataclass
class ReactorComponents:
    """Aggregate constructor argument so callers do not pass 6 positional kw."""

    router: DegradeRouter
    classifier: Classifier
    ladder: ActionLadder
    policy: PolicyAware
    sink: FindingSink | None = None
    rca: RcaEngine | None = None
    # Session-end finalizer: invoked once on ``stop_reason`` empty→non-empty
    # via the ``_finalize_fired`` latch; disk-level idempotency via marker file.
    finalizer: PostmortemFinalizer | None = None
    # Cross-tick state persistence (subprocess-per-tick transport loses memory
    # otherwise, so consecutive-tick rules can't fire). ``None`` disables (tests).
    state_store: DetectorStateStore | None = None


class Reactor:
    """Stateful pipeline driver.

    Each call to :meth:`tick` advances the internal tick index. The
    reactor is single-task: callers must not run multiple ``tick`` coros
    concurrently against the same instance.
    """

    def __init__(self, components: ReactorComponents) -> None:
        """Initialise the reactor from its component bundle.

        Args:
            components (ReactorComponents): Bundle of the router,
                classifier, ladder, policy and optional sink / RCA engine /
                finalizer / state store the pipeline drives each tick.
        """
        self._router = components.router
        self._classifier = components.classifier
        self._ladder = components.ladder
        self._policy = components.policy
        self._sink = components.sink
        self._rca: RcaEngine = components.rca or NoopRcaEngine()
        self._finalizer = components.finalizer
        self._state_store = components.state_store
        self._tick_index = 0
        self._last_symptoms: list[Symptom] = []
        self._last_data_summary: dict[str, Any] = {}
        # In-memory latch: ``finalizer.finalize`` runs at most once per instance,
        # complementing the finalizer's disk marker for cross-process resume.
        self._finalize_fired: bool = False

    @property
    def tick_index(self) -> int:
        """Current in-process tick index.

        Returns:
            int: Number of ``tick`` calls served by this instance.
        """
        return self._tick_index

    @property
    def last_symptoms(self) -> list[Symptom]:
        """Symptoms classified on the most recent tick.

        Returns:
            list[Symptom]: A copy of the last tick's symptom list.
        """
        return list(self._last_symptoms)

    async def tick(self, ctx: ReactorContext) -> list[Intent]:
        """Run one pipeline tick and return the validated intents.

        Advances the tick index, collects a source snapshot, classifies
        symptoms, runs the action ladder, filters intents through the
        policy gate, persists findings, fires the finalizer once on
        stop, and flushes cross-tick state.

        Args:
            ctx (ReactorContext): Per-tick input parsed from the
                Coordinator prompt or inbox.

        Returns:
            list[Intent]: Intents that passed policy validation this tick.
        """
        self._tick_index += 1
        now_unix = ctx.now_unix or time.time()

        data = await self._router.collect(ctx)
        symptoms = self._classifier.classify(data, ctx)
        # Prefer the Coordinator's session-wide ``ctx.shared_state.tick`` so ladder
        # cooldowns and finding stamps survive subprocess restarts.
        authoritative_tick = self._resolve_authoritative_tick(ctx)
        result = await self._ladder.decide(
            symptoms,
            tick_index=authoritative_tick,
            now_unix=now_unix,
            rca_provider=self._rca,
        )

        validated_intents: list[Intent] = []
        rejected: list[tuple[str, str]] = []
        for intent in result.intents:
            try:
                self._policy.assert_payload_complete(intent)
            except PolicyViolation as exc:
                rejected.append((intent.type.value, str(exc)))
                continue
            validated_intents.append(intent)

        if rejected:
            log.warning(
                "reactor tick=%d dropped %d intents due to policy violations: %s",
                self._tick_index,
                len(rejected),
                rejected,
            )

        if self._sink is not None and result.findings:
            try:
                await self._sink.append_many(result.findings)
            except Exception:  # noqa: BLE001 — sink already swallows IO errors
                log.exception("reactor tick=%d sink.append_many failed", self._tick_index)

        # Run after the sink write so this tick's findings are in the corpus.
        await self._maybe_finalize(ctx)

        # Flush mutated detector/ladder/throttle state last; off the event loop
        # because fsync blocks.
        await self._flush_state_store()

        self._last_symptoms = symptoms
        self._last_data_summary = _summarise(data, symptoms, result.findings)
        return validated_intents

    def _resolve_authoritative_tick(self, ctx: ReactorContext) -> int:
        """Pick the most reliable tick index: prefer the Coordinator's
        session-wide ``ctx.shared_state.tick``, else the in-memory counter
        (tests / first tick before the prompt is written).
        """
        shared_tick = getattr(ctx.shared_state, "tick", 0) or 0
        if shared_tick > 0:
            return int(shared_tick)
        return self._tick_index

    async def _flush_state_store(self) -> None:
        """Flush cross-tick detector state to disk off the event loop.

        No-op when no state store is configured. Failures are logged and
        swallowed so a flush error never crashes the tick.
        """
        if self._state_store is None:
            return
        try:
            await asyncio.to_thread(self._state_store.flush_atomic)
        except Exception:  # noqa: BLE001 — best-effort, never crash tick
            log.exception(
                "reactor tick=%d state_store flush failed", self._tick_index,
            )

    async def _maybe_finalize(self, ctx: ReactorContext) -> None:
        """Fire the postmortem finalizer once on the first stop_reason.

        No-op when no finalizer is configured, the latch has already
        fired, or ``stop_reason`` is still empty. An in-memory latch plus
        the finalizer's disk marker guarantee at-most-once semantics.

        Args:
            ctx (ReactorContext): Per-tick context carrying ``stop_reason``.
        """
        if self._finalizer is None or self._finalize_fired:
            return
        stop_reason = str(ctx.shared_state.stop_reason or "").strip()
        if not stop_reason:
            return
        # Latch before the disk write so a failed write isn't retried this
        # instance (spec: fire at most once per session).
        self._finalize_fired = True
        try:
            await asyncio.to_thread(
                self._finalizer.finalize, stop_reason=stop_reason,
            )
        except Exception:  # noqa: BLE001 — finalize is best-effort
            log.exception(
                "reactor tick=%d postmortem finalizer raised", self._tick_index,
            )


def _summarise(
    data: Any,
    symptoms: list[Symptom],
    findings: list[Finding],
) -> dict[str, Any]:
    """Build a compact debug summary of a tick's pipeline outputs.

    Args:
        data (Any): The source snapshot produced by the router.
        symptoms (list[Symptom]): Symptoms classified this tick.
        findings (list[Finding]): Findings produced by the action ladder.

    Returns:
        dict[str, Any]: Summary with sources used, degraded reason, and
        symptom / finding counts.
    """
    return {
        "sources_used": list(getattr(data, "sources_used", []) or []),
        "degraded_reason": getattr(data, "degraded_reason", None),
        "symptom_count": len(symptoms),
        "finding_count": len(findings),
    }


__all__ = ["Reactor", "ReactorComponents"]
