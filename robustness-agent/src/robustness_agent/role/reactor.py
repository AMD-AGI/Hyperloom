"""Reactor: the heart of the robustness role.

A single :meth:`Reactor.tick` runs the M1 pipeline:

1. :class:`DegradeRouter` produces one :class:`SourceData` snapshot.
2. :class:`Classifier` distils the snapshot into :class:`Symptom` list.
3. :class:`ActionLadder` maps symptoms onto Intents + Findings.
4. :class:`PolicyAware` filters every intent through PolicyGate-equivalent
   checks before they leave the agent.
5. :class:`FindingSink` persists the findings to disk.
6. The reactor returns the validated intents.

The Reactor holds tick state (tick index, last data snapshot) but no
business logic; that lives in classifier + ladder so transports can be
swapped without recompiling rules.
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


# Shielded asyncio.to_thread import — only needed for the L1+L2 finalize
# hook below. Keeping it at module scope avoids re-importing per tick.
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
    # L1 + L2: session-end finalizer. When set, the reactor watches
    # ``ctx.shared_state.stop_reason`` for an empty→non-empty
    # transition and invokes the finalizer exactly once via the
    # per-instance ``_finalize_fired`` latch. Idempotency at disk
    # level is enforced by the finalizer's marker file.
    finalizer: PostmortemFinalizer | None = None
    # Cross-tick state persistence. M1 subprocess transport spawns a
    # fresh Python per tick — without this store, every detector /
    # ladder / throttle starts empty each tick and consecutive-tick
    # rules can never fire. The reactor flushes the store at the end
    # of every successful tick. ``None`` disables persistence (tests).
    state_store: DetectorStateStore | None = None


class Reactor:
    """Stateful pipeline driver.

    Each call to :meth:`tick` advances the internal tick index. The
    reactor is single-task: callers must not run multiple ``tick`` coros
    concurrently against the same instance.
    """

    def __init__(self, components: ReactorComponents) -> None:
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
        # In-memory latch — guarantees ``finalizer.finalize`` is called
        # at most once per reactor instance, complementing the disk-
        # level marker the finalizer maintains for cross-process resume.
        self._finalize_fired: bool = False

    @property
    def tick_index(self) -> int:
        return self._tick_index

    @property
    def last_symptoms(self) -> list[Symptom]:
        return list(self._last_symptoms)

    async def tick(self, ctx: ReactorContext) -> list[Intent]:
        self._tick_index += 1
        now_unix = ctx.now_unix or time.time()

        data = await self._router.collect(ctx)
        symptoms = self._classifier.classify(data, ctx)
        # ``ctx.shared_state.tick`` (set by the Coordinator prompt
        # parser) is the authoritative tick index across subprocess
        # restarts; ``self._tick_index`` only counts ticks within this
        # Python process. Prefer the Coordinator value when present so
        # ladder cooldowns and finding tick_index stamps match the
        # session-wide timeline.
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

        # L1 + L2: detect stop_reason transition and trigger the
        # postmortem finalizer once. The check runs *after* the sink
        # write so the most recent findings of this tick are part of
        # the postmortem corpus.
        await self._maybe_finalize(ctx)

        # Flush any detector / ladder / throttle state mutated this
        # tick. Done last so even if downstream raises, we record
        # what we observed. Off the event loop because fsync blocks.
        await self._flush_state_store()

        self._last_symptoms = symptoms
        self._last_data_summary = _summarise(data, symptoms, result.findings)
        return validated_intents

    def _resolve_authoritative_tick(self, ctx: ReactorContext) -> int:
        """Pick the most reliable tick index for this turn.

        Order of preference:

        1. ``ctx.shared_state.tick`` — the Coordinator's session-wide
           tick counter, propagated through the prompt. Survives
           subprocess restarts so cooldowns stay coherent.
        2. ``self._tick_index`` — in-memory counter incremented per
           ``tick()`` call. Used by ad-hoc tests and the first tick of
           a session before the Coordinator has written the prompt.
        """
        shared_tick = getattr(ctx.shared_state, "tick", 0) or 0
        if shared_tick > 0:
            return int(shared_tick)
        return self._tick_index

    async def _flush_state_store(self) -> None:
        if self._state_store is None:
            return
        try:
            await asyncio.to_thread(self._state_store.flush_atomic)
        except Exception:  # noqa: BLE001 — best-effort, never crash tick
            log.exception(
                "reactor tick=%d state_store flush failed", self._tick_index,
            )

    async def _maybe_finalize(self, ctx: ReactorContext) -> None:
        if self._finalizer is None or self._finalize_fired:
            return
        stop_reason = str(ctx.shared_state.stop_reason or "").strip()
        if not stop_reason:
            return
        # In-memory latch first — even if the disk write fails we will
        # not retry within this reactor instance, matching the spec
        # ("fire at most once per session").
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
    return {
        "sources_used": list(getattr(data, "sources_used", []) or []),
        "degraded_reason": getattr(data, "degraded_reason", None),
        "symptom_count": len(symptoms),
        "finding_count": len(findings),
    }


__all__ = ["Reactor", "ReactorComponents"]
