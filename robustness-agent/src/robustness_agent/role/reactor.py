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
        result = await self._ladder.decide(
            symptoms,
            tick_index=self._tick_index,
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

        self._last_symptoms = symptoms
        self._last_data_summary = _summarise(data, symptoms, result.findings)
        return validated_intents

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
