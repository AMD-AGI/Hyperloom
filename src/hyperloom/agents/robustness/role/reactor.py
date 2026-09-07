# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reactor: the heart of the robustness role."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ..decision.action_ladder import ActionLadder
from ..decision.policy_aware import PolicyAware
from ..decision.rca_engine import NoopRcaEngine, RcaEngine
from ..signals import Classifier
from ..sources.base import DegradeRouter
from ..state_store import DetectorStateStore
from .envelope import Intent, PolicyViolation
from .findings import FindingSink
from .prompt_inputs import ReactorContext


import asyncio  # noqa: E402


log = logging.getLogger(__name__)


@dataclass
class ReactorComponents:
    """Bundle the reactor's collaborators into one constructor argument."""

    router: DegradeRouter
    classifier: Classifier
    ladder: ActionLadder
    policy: PolicyAware
    sink: FindingSink | None = None
    rca: RcaEngine | None = None
    # Cross-tick state persistence; ``None`` disables (tests).
    state_store: DetectorStateStore | None = None


class Reactor:
    """Stateful pipeline driver."""

    def __init__(self, components: ReactorComponents) -> None:
        """Initialise the reactor from its component bundle."""
        self._router = components.router
        self._classifier = components.classifier
        self._ladder = components.ladder
        self._policy = components.policy
        self._sink = components.sink
        self._rca: RcaEngine = components.rca or NoopRcaEngine()
        self._state_store = components.state_store
        self._tick_index = 0

    @property
    def tick_index(self) -> int:
        """Current in-process tick index."""
        return self._tick_index

    async def tick(self, ctx: ReactorContext) -> list[Intent]:
        """Run one pipeline tick and return the validated intents."""
        self._tick_index += 1
        now_unix = ctx.now_unix or time.time()

        data = await self._router.collect(ctx)
        symptoms = self._classifier.classify(data, ctx)
        # Prefer the session-wide tick so ladder cooldowns and finding stamps survive subprocess restarts.
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

        # Flush mutated detector/ladder/throttle state last, off the event loop.
        await self._flush_state_store()

        return validated_intents

    def _resolve_authoritative_tick(self, ctx: ReactorContext) -> int:
        """Pick the most reliable tick index for this reactor pass."""
        shared_tick = ctx.shared_state.tick or 0
        if shared_tick > 0:
            return int(shared_tick)
        return self._tick_index

    async def _flush_state_store(self) -> None:
        """Flush cross-tick detector state to disk off the event loop."""
        if self._state_store is None:
            return
        try:
            await asyncio.to_thread(self._state_store.flush_atomic)
        except Exception:  # noqa: BLE001 — best-effort, never crash tick
            log.exception(
                "reactor tick=%d state_store flush failed",
                self._tick_index,
            )


__all__ = ["Reactor", "ReactorComponents"]
