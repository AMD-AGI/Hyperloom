"""SINGLE_PROC adapter: expose :class:`Reactor` as a Coordinator Backend.

Mirrors the Backend protocol declared by
``inference_optimizer/orchestrator/backends/base.py``:

.. code-block:: python

    class Backend(Protocol):
        name: str

        async def run(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
            tools: list[str] | None = None,
            max_turns: int = 1,
        ) -> BackendTurnResult: ...

The Coordinator imports :class:`RobustnessAgentBackend` directly from
this module when the orchestration plan selects ``robustness`` and the
deployment is configured for SINGLE_PROC. Multi-CLI / Claw subsession
modes do not use this adapter — they ship the reactor inside a long-
running CLI that reads/writes JSONL files instead (M3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .envelope import BackendTurnResult
from .prompt_inputs import from_coordinator_prompt
from .reactor import Reactor


log = logging.getLogger(__name__)


@dataclass
class RobustnessAgentBackend:
    """Async :class:`Reactor` -> Coordinator Backend bridge."""

    reactor: Reactor
    name: str = "robustness"

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        ctx = from_coordinator_prompt(prompt)
        intents = await self.reactor.tick(ctx)
        if ctx.parse_warnings:
            log.debug(
                "robustness reactor parse warnings: %s",
                ctx.parse_warnings,
            )
        metadata: dict[str, Any] = {
            "tick_index": self.reactor.tick_index,
            "parse_warnings": list(ctx.parse_warnings),
        }
        return BackendTurnResult(
            intents=intents,
            raw_text="(robustness-agent)",
            metadata=metadata,
        )


__all__ = ["RobustnessAgentBackend"]
