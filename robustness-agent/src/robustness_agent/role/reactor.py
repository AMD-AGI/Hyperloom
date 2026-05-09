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
from ..findings.sink import FindingSink
from ..signals import Classifier, Symptom
from ..sources.base import DegradeRouter
from .envelope import Intent
from .prompt_inputs import ReactorContext


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
        self._tick_index = 0
        self._last_symptoms: list[Symptom] = []
        self._last_data_summary: dict[str, Any] = {}

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

        self._last_symptoms = symptoms
        self._last_data_summary = _summarise(data, symptoms, result.findings)
        return validated_intents


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
