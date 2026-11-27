# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""MockBackend — scripted-turn LLM stub for unit / e2e tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.roles.base import BackendTurnResult


@dataclass
class MockTurn:
    """One scripted turn the mock backend will play back."""

    intents: list[Intent] = field(default_factory=list)
    raw_text: str = "(mock turn)"
    raise_error: BaseException | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScriptedPlan:
    """Sequence of pre-recorded turns."""

    turns: list[MockTurn]
    loop_last: bool = False
    default_intent: Intent | None = None


class MockBackend:
    """Implements :class:`Backend` by playing back a :class:`ScriptedPlan`."""

    def __init__(self, plan: ScriptedPlan, *, name: str = "mock"):
        """Initialise the mock backend with a scripted plan."""
        self.plan = plan
        self.name = name
        self._cursor = 0
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
    ) -> BackendTurnResult:
        """Record the call and play back the next scripted turn."""
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "tools": list(tools or []),
            }
        )
        turn = self._next_turn()
        if turn.raise_error is not None:
            raise turn.raise_error
        return BackendTurnResult(
            intents=list(turn.intents),
            raw_text=turn.raw_text,
            metadata=dict(turn.metadata),
        )

    def _next_turn(self) -> MockTurn:
        """Return the next turn to play, applying loop/default/fallback rules."""
        if self._cursor < len(self.plan.turns):
            t = self.plan.turns[self._cursor]
            self._cursor += 1
            return t
        if self.plan.loop_last and self.plan.turns:
            return self.plan.turns[-1]
        if self.plan.default_intent is not None:
            return MockTurn(intents=[self.plan.default_intent])
        # Out of script and no fallback → emit an observation so the reactor keeps ticking.
        return MockTurn(
            intents=[
                Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "observation", "body_md": "ok"}),
            ]
        )
